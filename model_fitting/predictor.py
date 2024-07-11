import os, sys, json
import torch
from model_fitting.models import (
    MLP,
    DoubleMLP,
    States_Auxiliary,
    Input_Auxiliary,
    LinearKoopman,
    LinearKoopmanWithInput,
    TrainableKoopmanDynamics,
    CombinedDeepKoopman,
    CombinedKoopman_withAux_withoutInput,
    CombinedKoopman_withAux_withInput,
)


class Predictor:
    def __init__(
        self,
        model_dir,
        num_pred_steps=1,
        seq_length=None,
        normalizer_params=None,
        cuda=True,
        sample_time_override=None,
        use_encoder_jacobian=False,
    ):
        self._model_name = model_dir
        self._num_pred_steps = num_pred_steps
        self._seq_length = seq_length
        # self._normalizer_params = normalizer_params
        self._use_gpu = cuda
        self._x = None
        self._y = None
        self._encoder_jacobian = None
        self._use_encoder_jacobian = use_encoder_jacobian
        # self._sampling_freq_factor = sampling_freq_factor       ## this option is used for observer to use smaller sample time
        # sample_time_override       ## this option is used for observer to use smaller sample time

        #### calcualte norm factor for adjusting the encoder gradient. This is because the network was trained with pre-normalized dataset
        #### so the narmalizationeffect is not considered in the autogradient and it should be added amunally.
        if normalizer_params is not None:
            x_min = torch.tensor(
                normalizer_params["params_x"]["min"], dtype=torch.float32
            )
            x_max = torch.tensor(
                normalizer_params["params_x"]["max"], dtype=torch.float32
            )
            # self._norm_factor = 1 / (x_max - x_min)         # grad x with respect x_norm
            self._norm_factor = x_max - x_min
        else:
            self._norm_factor = 1

        """ load trained netwroks dictionaries """
        self.encoder_dict = torch.load(
            os.path.join(model_dir, "trained_model", "TorchTrained_Encoder.pt"),
            map_location=torch.device("cpu"),
        )
        self.decoder_dict = torch.load(
            os.path.join(model_dir, "trained_model", "TorchTrained_Decoder.pt"),
            map_location=torch.device("cpu"),
        )
        self.states_auxilary_dict = torch.load(
            os.path.join(model_dir, "trained_model", "TorchTrained_States_Auxilary.pt"),
            map_location=torch.device("cpu"),
        )
        self.koopman_dict = torch.load(
            os.path.join(model_dir, "trained_model", "TorchTrained_Koopman.pt"),
            map_location=torch.device("cpu"),
        )
        if os.path.exists(
            os.path.join(model_dir, "trained_model", "TorchTrained_Inputs_Auxilary.pt")
        ):
            self.inputs_auxilary_dict = torch.load(
                os.path.join(
                    model_dir, "trained_model", "TorchTrained_Inputs_Auxilary.pt"
                ),
                map_location=torch.device("cpu"),
            )

        if sample_time_override is not None:
            self._sample_time = sample_time_override
        else:
            self._sample_time = self.koopman_dict["sample_time"]

        """ load (build) the structure of the model components based on structure info saved in model dictionary """
        encoder = self.load_encoder(self.encoder_dict, normalizer_params)
        decoder = self.load_decoder(self.decoder_dict, normalizer_params)
        states_auxilary = self.load_states_auxilary(self.states_auxilary_dict)
        koopmanOperator = self.load_koopman(self.koopman_dict)
        if hasattr(self, "inputs_auxilary_dict"):
            inputs_auxilary = self.load_inputs_auxilary(self.inputs_auxilary_dict)

        """ build the whole model from the components """
        # self._model = CombinedKoopman_withAux(encoder, states_auxilary, koopmanOperator, decoder)
        self._model = CombinedKoopman_withAux_withInput(
            encoder, states_auxilary, koopmanOperator, decoder, inputs_auxilary
        )

        """ load trained weights to the model """
        self._model.encoder.load_state_dict(self.encoder_dict["state_dict"])
        self._model.decoder.load_state_dict(self.decoder_dict["state_dict"])
        self._model.states_auxilary.load_state_dict(
            self.states_auxilary_dict["state_dict"]
        )
        self._model.koopman.load_state_dict(self.koopman_dict["state_dict"])
        self._model.inputs_auxilary.load_state_dict(
            self.inputs_auxilary_dict["state_dict"]
        )


    def save_scripted_model(self):
        self._scripted_model_dir = os.path.join(self._model_name, "scripted_model")
        if not os.path.exists(self._scripted_model_dir):
            os.makedirs(self._scripted_model_dir)
        scripted_encoder = torch.jit.script(self._model.encoder)
        scripted_encoder.save(os.path.join(self._model_name, "scripted_model", "TorchScript_Encoder.pt"))
        scripted_decoder = torch.jit.script(self._model.decoder)
        scripted_decoder.save(os.path.join(self._model_name, "scripted_model", "TorchScript_Decoder.pt"))
        scripted_states_auxilary = torch.jit.script(self._model.states_auxilary)
        scripted_states_auxilary.save(os.path.join(self._model_name, "scripted_model", "TorchScript_States_Auxilary.pt"))
        # scripted_koopman = torch.jit.script(self._model.koopman)
        # scripted_koopman.save(os.path.join(self._model_name, "scripted_model", "TorchScript_Koopman.pt"))
        ##### koopman was built manually
        scripted_inputs_auxilary = torch.jit.script(self._model.inputs_auxilary)
        scripted_inputs_auxilary.save(os.path.join(self._model_name, "scripted_model", "TorchScript_Inputs_Auxilary.pt"))

    def load_script_model(self):
        model_path = os.path.join(self._model_name, "scripted_model", "TorchScript_Encoder.pt")
        scripted_encoder = torch.jit.load(model_path)
        model_path = os.path.join(self._model_name, "scripted_model", "TorchScript_Encoder.pt")
        scripted_encoder = torch.jit.load(model_path)
        return scripted_encoder


    def load_encoder(self, model_dict, normalizer_params=None):
        if model_dict["architecture"] == "MLP":
            model = MLP(
                input_size=model_dict["state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["lifted_state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=normalizer_params,
                output_normalizer_params=None,
            )
        elif model_dict["architecture"] == "DualMLP":
            model = DoubleMLP(
                input_size=model_dict["state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["lifted_state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=normalizer_params,
                output_normalizer_params=None,
            )

        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()

    def load_decoder(self, model_dict, normalizer_params=None):
        if model_dict["architecture"] == "MLP":
            model = MLP(
                input_size=model_dict["lifted_state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=None,
                output_normalizer_params=normalizer_params,
            )
        elif model_dict["architecture"] == "DoubleMLP":
            model = DoubleMLP(
                input_size=model_dict["lifted_state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=None,
                output_normalizer_params=normalizer_params,
            )

        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()

    def load_states_auxilary(self, model_dict):
        model = States_Auxiliary(
            num_complexeigens_pairs=model_dict["num_complexeigens_pairs"],
            num_realeigens=model_dict["num_realeigens"],
            hidden_sizes=model_dict["hidden_sizes"],
            activation=model_dict["activation"],
        )
        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()

    def load_koopman(self, model_dict):
        model = TrainableKoopmanDynamics(
            num_complexeigens_pairs=model_dict["num_complexeigens_pairs"],
            num_realeigens=model_dict["num_realeigens"],
            sample_time=self._sample_time,
            # structure=model_dict["structure"]
        )
        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()

    def load_inputs_auxilary(self, model_dict):
        model = Input_Auxiliary(
            input_size=model_dict["lifted_state_size"],
            hidden_sizes=model_dict["hidden_sizes"],
            output_shape=model_dict["output_shape"],
            activation=model_dict["activation"],
        )
        if self._use_gpu:
            return model.gpu()
        else:
            return model.cpu()

    def encode(self, x, requires_grad=False):
        """Transform a state in state space to the lifted space (X(i)->Y(i)) and canculate the jacobian of the encoder"""
        if requires_grad:
            x.requires_grad_()
        ### loop over the sequnec steps
        y = self._model.encoder(x)
        if requires_grad:
            output_dim = y.shape[1]
            batch_size, input_dim = x.shape
            encoder_jacobian = torch.zeros(
                batch_size, output_dim, input_dim, device=y.device
            )
            # calculate gradient of the encoder
            for i in range(output_dim):
                grad_outputs = torch.zeros_like(y)
                grad_outputs[:, i] = 1  # Select the i-th output
                grads = torch.autograd.grad(
                    outputs=y,
                    inputs=x,
                    grad_outputs=grad_outputs,
                    retain_graph=True,  # Need to retain for next iterations
                    create_graph=False,  # True if you need higher-order gradients
                    only_inputs=True,
                )[
                    0
                ]  # [0] to get the gradient tensor
                encoder_jacobian[:, i, :] = grads
            encoder_jacobian = (
                encoder_jacobian * self._norm_factor
            )  #  the effect of normalization is not considered in the autograd
        else:
            encoder_jacobian = None
        y = y.detach()
        # x0.requires_grad_(self._training_parameters["train_encoder"])
        return y, encoder_jacobian

    def decode(self, y, requires_grad=False):
        """Transform a state in lifted space to state space (X(i)->Y(i)) and canculate the jacobian of the encoder"""
        if requires_grad:
            y.requires_grad_()
        ### loop over the sequnec steps
        x = self._model.decoder(y)
        if requires_grad:
            output_dim = x.shape[1]
            batch_size, input_dim = y.shape
            decoder_jacobian = torch.zeros(
                batch_size, output_dim, input_dim, device=x.device
            )
            # calculate gradient of the encoder
            for i in range(output_dim):
                grad_outputs = torch.zeros_like(x)
                grad_outputs[:, i] = 1  # Select the i-th output
                grads = torch.autograd.grad(
                    outputs=x,
                    inputs=y,
                    grad_outputs=grad_outputs,
                    retain_graph=True,  # Need to retain for next iterations
                    create_graph=False,  # True if you need higher-order gradients
                    only_inputs=True,
                )[
                    0
                ]  # [0] to get the gradient tensor
                decoder_jacobian[:, i, :] = grads
            #  the effect of normalization is not considered in the autograd
            decoder_jacobian = decoder_jacobian * (1 / self._norm_factor).unsqueeze(1)  
        else:
            decoder_jacobian = None
        x = x.detach()
        # x0.requires_grad_(self._training_parameters["train_encoder"])
        return x, decoder_jacobian

    def get_ABL(self, y=None, encoder_jacobian=None, decoder_jacobian=None):
        """Get Koopman matrix A, input matrix B_phi, auxiliary matrix B, and eigenvalues Lambdas"""
        if y is None and self._y is not None:
            y = self._y
        if encoder_jacobian is None and self._encoder_jacobian is not None:
            encoder_jacobian = self._encoder_jacobian

        Lambdas = self._model.states_auxilary(y)
        _, G, A = self._model.koopman(y, Lambdas)

        B = self._model.inputs_auxilary(y)

        if self._use_encoder_jacobian:
            B_phi = torch.matmul(encoder_jacobian, B)
        else:
            B_phi = B
        identity_matrix = torch.eye(y.shape[1], device=y.device)
        temp = torch.matmul(torch.inverse(A), (G - identity_matrix))
        H = torch.matmul(temp, B_phi)

        return (G, H, B, Lambdas)

    def koopman_step(self, G, H, y, u):
        """Compute next lifted state using Koopman dynamics"""
        # y0 = y0.unsqueeze(-1)                 # Add an extra dimension for matmul
        # y_adv = torch.matmul(A, y0).squeeze(-1)  + torch.matmul(B_phi, u.unsqueeze(-1)).squeeze(-1)
        y_adv = torch.matmul(G, y.unsqueeze(-1)).squeeze(-1) + torch.matmul(
            H, u.unsqueeze(-1)
        ).squeeze(-1)
        return y_adv

    def lift_a_trajectory(self, x):
        """Transform a trajectory of states in state space (robot space) to the lifted space (X->Y)"""
        Yp_list = []  ## truth value of y_plus from advanced encoded x
        ### loop over the sequnec steps
        for j in range(self._num_pred_steps):
            Yp_list.append(self._model.encoder(x[:, j + 1, :]))
        return Yp_list

    def encode_and_decode_a_trajectory(self, x):
        """Transform a trajectory of states in state space (robot space) to the lifted space (X->Y)"""
        Xp_list = []  ## truth value of y_plus from advanced encoded x
        ### loop over the sequnec steps
        for j in range(self._num_pred_steps):
            Xp_list.append(self._model.decoder(self._model.encoder(x[:, j + 1, :])))
        return Xp_list

    def initialize(self, x0):
        """Call the encoder and store the states in member variables"""
        self._x = x0
        self._y, self._encoder_jacobian = self.encode(
            self._x, requires_grad=self._use_encoder_jacobian
        )
        return self._y

    def step(self, u=0):
        """Step in the lifted space and transfer to the state space | it updates the record of the states"""
        ### check if the model is initialized
        if (
            self._x is None
            or self._y is None
            or (self._encoder_jacobian is None and self._use_encoder_jacobian)
        ):
            raise AssertionError("The model is not initialized")

        G, H, B, Lambdas = self.get_ABL(self._y, self._encoder_jacobian)
        self._y = self.koopman_step(G, H, self._y, u)
        self._x = self.decode(self._y)[0]
        return self._x, self._y

    def predict_trajectory(self, x0, u, requires_grad=False):
        """Predict trajectory using Koopman model"""
        Yp_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
        Xp_pred_list = []  ## predicted value of x_plus from decoding the iterated model
        Lambdas_list = []
        B_list = []
        U_list = []
        x = x0

        y, encoder_jacobian = self.encode(x, requires_grad=requires_grad)

        for j in range(self._num_pred_steps):
            G, H, B, Lambdas = self.get_ABL(y, encoder_jacobian)
            y = self.koopman_step(G, H, y, u[:, j, :])
            x, _ = self.decode(y, requires_grad=False)
            Xp_pred_list.append(x)
            Lambdas_list.append(Lambdas)
            B_list.append(B)
            U_list.append(u[:, j, :])
            Yp_pred_list.append(y)

        return Xp_pred_list, Yp_pred_list, Lambdas_list, B_list, U_list

    def predict_ahead(self, u_h, requires_grad=True):
        """Predict trajectory using Koopman model"""
        Y_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
        X_pred_list = (
            []
        )  ## predicted value of x_plus from decoding the iterated Koopman model
        # Lambdas_list = []
        # B_list = []
        U_pred_list = []

        y, encoder_jacobian = self._y, self._encoder_jacobian

        num_pred_steps = u_h.shape[0]
        for j in range(num_pred_steps):
            A, B_phi, B, Lambdas = self.get_ABL(y, encoder_jacobian)
            y = self.koopman_step(A, B_phi, y, u_h[j, :, :])
            X_pred_list.append(self.decode(y)[0])
            # Lambdas_list.append(Lambdas)
            # B_list.append(B)
            U_pred_list.append(u_h[j, :, :])
            Y_pred_list.append(y)
        return X_pred_list, Y_pred_list, U_pred_list