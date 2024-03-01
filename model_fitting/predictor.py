import os, sys, json
import matplotlib.pyplot as plt
import torch
import numpy as np

print(os.getcwd())
sys.path.append(os.getcwd())
current_directory = os.path.dirname(__file__)

from model_fitting_nature.models import (
    MLP,
    Auxilary,
    LinearKoopman,
    LinearKoopmanWithInput,
    TrainableKoopmanDynamics,
    CombinedDeepKoopman,
    CombinedKoopman_withAux,
)
from model_fitting_nature.utils import (
    MLPDataset,
)
from torch.utils.data import DataLoader, Dataset


koopman_params = {  ### model params for encoder, decoder is driven accordingly
    "num_complexeigens_pairs": 2,  
    "num_realeigens": 2,  
    "sample_time": 0.0016,
}
# encoder_params = {  ### model params for encoder, decoder is driven accordingly
#     "architecture": "MLP",  # TCNMLP, MLP
#     "state_size": 2,
#     "hidden_sizes": ([80, 80, 80]),
#     "lifted_state_size": koopman_params["num_realeigens"] + 2*koopman_params["num_complexeigens_pairs"],
#     "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
#     "sample_time": None,
# }
# auxilary_params = {  ### model params for encoder, decoder is driven accordingly
#     "architecture": "MLP",
#     "num_complexeigens_pairs": koopman_params["num_complexeigens_pairs"],
#     "num_realeigens": koopman_params["num_realeigens"],
#     "state_size": koopman_params["num_realeigens"] + 2*koopman_params["num_complexeigens_pairs"],
#     "hidden_sizes": ([30, 30]),
#     "output_size": koopman_params["num_realeigens"] + 2*koopman_params["num_complexeigens_pairs"],
#     "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
# }

encoder_params = {  ### model params for encoder, decoder is driven accordingly
    "architecture": "MLP",  # TCNMLP, MLP
    "state_size": 4,
    "hidden_sizes": ([80, 80, 80, 80]),
    "lifted_state_size": koopman_params["num_realeigens"] + 2*koopman_params["num_complexeigens_pairs"],
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
    "sample_time": None,
}
auxilary_params = {  ### model params for encoder, decoder is driven accordingly
    "architecture": "MLP",
    "num_complexeigens_pairs": koopman_params["num_complexeigens_pairs"],
    "num_realeigens": koopman_params["num_realeigens"],
    "state_size": koopman_params["num_realeigens"] + 2*koopman_params["num_complexeigens_pairs"],
    "hidden_sizes": ([30, 30, 30]),
    "output_size": koopman_params["num_realeigens"] + 2*koopman_params["num_complexeigens_pairs"],
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
}




class Predictor:
    def __init__(self, model_dir, num_pred_steps=1, _seq_length=None, cuda=True):
        self._model_name = model_dir
        self._num_pred_steps = num_pred_steps
        self._seq_length = _seq_length
        self._use_gpu = cuda

        self.encoder_dict = torch.load(os.path.join(model_dir, "trained_model", "TorchTrained_Encoder.pt"), map_location=torch.device("cpu"))
        self.decoder_dict = torch.load(os.path.join(model_dir, "trained_model", "TorchTrained_Decoder.pt"), map_location=torch.device("cpu"))
        self.auxilary_dict = torch.load(os.path.join(model_dir, "trained_model", "TorchTrained_Auxilary.pt"), map_location=torch.device("cpu"))
        self.koopman_dict = torch.load(os.path.join(model_dir, "trained_model", "TorchTrained_Koopman.pt"), map_location=torch.device("cpu"))

        encoder = self.load_encoder(self.encoder_dict)
        decoder = self.load_decoder(self.decoder_dict)
        auxilary = self.load_auxilary(self.auxilary_dict)
        koopmanOperator = self.load_koopman(self.koopman_dict)
        self._model = CombinedKoopman_withAux(encoder, auxilary, koopmanOperator, decoder)

        self._model.encoder.load_state_dict(self.encoder_dict["state_dict"])
        self._model.decoder.load_state_dict(self.decoder_dict["state_dict"])
        self._model.auxilary.load_state_dict(self.auxilary_dict["state_dict"])
        self._model.koopman.load_state_dict(self.koopman_dict["state_dict"])

    def load_encoder(self, model_dict):
        model = MLP(
            input_size=model_dict["state_size"],
            hidden_sizes=model_dict["hidden_sizes"],
            output_size=model_dict["lifted_state_size"],
            activation=model_dict["activation"],
        )
        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()
        
    def load_decoder(self, model_dict):
        model = MLP(
            input_size=model_dict["lifted_state_size"],
            hidden_sizes=model_dict["hidden_sizes"],
            output_size=model_dict["state_size"],
            activation=model_dict["activation"],
        )
        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()
        
    def load_auxilary(self, model_dict):
        model = Auxilary(
            input_size=auxilary_params["state_size"],
            params=auxilary_params,
        )
        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()
    
    def load_koopman(self, model_dict):
        model = TrainableKoopmanDynamics(
            koopman_params["num_complexeigens_pairs"], 
            koopman_params["num_realeigens"], 
            koopman_params["sample_time"]
        )
        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()


    def predict(self, x0):
        Yp_list = []  ## truth value of y_plus from advanced encoded x
        Yp_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
        Xp_list = []  ## truth value of x_plus from advanced x
        Xp_pred_list = []  ## predicted value of x_plus from decoding the iterated Koopman model
        Lambdas_list = []

        ### loop over the sequnec steps
        y0 = self._model.encoder(x0)
        for j in range(self._num_pred_steps):
            Lambdas = self._model.auxilary(y0)
            Lambdas_list.append(Lambdas)
            y_adv = self._model.koopman(y0, Lambdas)
            Yp_pred_list.append(y_adv)
            Xp_pred_list.append(self._model.decoder(y_adv))
            y0 = y_adv
        return Xp_pred_list, Yp_pred_list, Lambdas_list
    
    def lift(self, x):
        Yp_list = []  ## truth value of y_plus from advanced encoded x

        ### loop over the sequnec steps
        for j in range(self._seq_length-1):
            Yp_list.append(self._model.encoder(x[:, j + 1, :]))
        return Yp_list


