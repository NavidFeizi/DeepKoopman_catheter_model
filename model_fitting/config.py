# config.py

import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DETAILS = {
    "model_name": "catheter_1T_unforced_4.3.2",
    "model_name": "test",
    "desc": "one tendon with negative - forced - mlp encoder - corrected dataset velocity - shares damping with actuated",
}

# dataset related parameters
dataset_options = {
    # "dataset_name": "catheter_1T_unforced_4.3",
    "dataset_name": "catheter_1T_forced_5.3.3",
    "add_sequence_dimension": True,  # True for TCN and LSTM
    "fit_normalization": 'X',  #'X', 'U', 'All',   'X' for unforced system - None for forced system to use the unforced params
    "prenormalization": True,
    "sample_step": 1,
    "sample_time": None,
}

training_options = {
    "transfer_from": "catheter_1T_forced_5.3.3",
    # "transfer_from": None,
    "transfer_input_aux": True,
    "transfer_states_aux": True,
    "transfer_auto_encoder": True,
    "n_epochs": 2000,
    "n_recon_epoch": 200,    # number of first epocs only with reconstruction loss
    "learning_rate": 10e-4,
    "decay_rate": 0.9985,
    "batch_size": 240,
    "lossfunc_weights": ([1e-1, 1e-7, 1e-13]),      #recond and prediction loss, infinit loss, regularizarion 
    "weights_x": (([1, 1, 1, 1])),
    "num_pred_steps": 40,
    "train_encoder": True,
    "train_decoder": True,
    "train_states_auxiliary": True,
    "train_inputs_auxiliary": True,
    "cuda": True,
}

# model params for Koopman operator
koopman_params = {
    "num_complexeigens_pairs": 1,
    "num_realeigens": 0,
    "sample_time": None,  # loads from the dataste options
    "structure": "Jordan",  # "Jordan" or "Controlable"
}

encoder_params = {
    "architecture": "MLP",  # DoubleMLP, MLP
    "state_size": 4,
    "hidden_sizes": ([128, 128, 128, 128]),
    "lifted_state_size": koopman_params["num_realeigens"]
    + 2 * koopman_params["num_complexeigens_pairs"],
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
    "sample_time_new": None,
}

decoder_params = encoder_params.copy()
decoder_params["hidden_sizes"] = [128, 128, 128, 128]
decoder_params["architecture"] = "MLP"

states_auxiliary_params = {
    "architecture": "MLP",
    "hidden_sizes": ([128, 128, 128, 128]),
    "activation": "ReLU",   # ReLU, Tanh, Sigmoid, LeakyReLU ...
}

inputs_auxiliary_params = {
    "architecture": "MLP",
    "system_input_size": 1,
    "hidden_sizes": ([128, 128, 128, 128]),
    "activation": "ReLU",    # ReLU, Tanh, Sigmoid, LeakyReLU ...
}

states_auxiliary_params["num_complexeigens_pairs"] = koopman_params[
    "num_complexeigens_pairs"
]
states_auxiliary_params["num_realeigens"] = koopman_params["num_realeigens"]
states_auxiliary_params["state_size"] = encoder_params["lifted_state_size"]
states_auxiliary_params["output_size"] = encoder_params["lifted_state_size"]
inputs_auxiliary_params["lifted_state_size"] = encoder_params["lifted_state_size"]
inputs_auxiliary_params["output_shape"] = [
    encoder_params["lifted_state_size"],
    inputs_auxiliary_params["system_input_size"],
]

model_dir = os.path.join(os.getcwd(), "trained_models", MODEL_DETAILS["model_name"])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
