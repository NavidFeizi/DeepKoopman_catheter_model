import os
import torch
from model_fitting.models import (
    MLP,
    DoubleMLP,
    States_Auxiliary,
    Input_Auxiliary,
    TrainableKoopmanDynamics,
    CombinedKoopman_withAux_withInput,
)
from typing import Optional, Dict, Any, Tuple, List
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(
        self,
        model_dir: str,
        num_pred_steps: int = 1,
        normalizer_params: Optional[int] = None,
        cuda:bool = True,
        sample_time_override: Optional[float] = None,
        use_encoder_jacobian: bool = False,
    ):
        self._model_name = model_dir
        self._num_pred_steps = num_pred_steps
        self._cuda = cuda
        self._x = None
        self._y = None
        self._encoder_jacobian = None
        self._use_encoder_jacobian = use_encoder_jacobian

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

        self._sample_time = sample_time_override if sample_time_override is not None else self.koopman_dict["sample_time"]

        self.load_trained_networks(model_dir)
        self.reconstruct_model_structure(normalizer_params)
        self.load_trained_weights()

    def load_encoder(
        self,
        model_dict: Dict[str, Any],
        normalizer_params: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        """
        Load and initialize the encoder model based on the provided architecture.

        Args:
            model_dict (dict): Dictionary containing the model architecture and parameters.
            normalizer_params (dict, optional): Normalization parameters for the input data.

        Returns:
            torch.nn.Module: Initialized encoder model, moved to GPU if available and specified.
        """
        architecture = model_dict.get("architecture")
        if architecture == "MLP":
            model = MLP(
                input_size=model_dict["state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["lifted_state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=normalizer_params,
                output_normalizer_params=None,
            )
        elif architecture == "DualMLP":
            model = DoubleMLP(
                input_size=model_dict["state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["lifted_state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=normalizer_params,
                output_normalizer_params=None,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        return model.to("cuda" if self._cuda else "cpu")

    def load_decoder(
        self,
        model_dict: Dict[str, Any],
        normalizer_params: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        """
        Load and initialize the decoder model based on the provided architecture.

        Args:
            model_dict (Dict[str, Any]): Dictionary containing the model architecture and parameters.
            normalizer_params (Optional[Dict[str, Any]]): Normalization parameters for the output data.

        Returns:
            torch.nn.Module: Initialized decoder model, moved to GPU if available and specified.
        """
        architecture = model_dict.get("architecture")
        if architecture == "MLP":
            model = MLP(
                input_size=model_dict["lifted_state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=None,
                output_normalizer_params=normalizer_params,
            )
        elif architecture == "DoubleMLP":
            model = DoubleMLP(
                input_size=model_dict["lifted_state_size"],
                hidden_sizes=model_dict["hidden_sizes"],
                output_size=model_dict["state_size"],
                activation=model_dict["activation"],
                input_normalizer_params=None,
                output_normalizer_params=normalizer_params,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        return model.to("cuda" if self._cuda else "cpu")

    def load_states_auxiliary(self, model_dict: Dict[str, Any]) -> torch.nn.Module:
        """
        Load and initialize the states auxiliary model based on the provided architecture.

        Args:
            model_dict (Dict[str, Any]): Dictionary containing the model architecture and parameters.

        Returns:
            torch.nn.Module: Initialized states auxiliary model, moved to GPU if available and specified.
        """
        model = States_Auxiliary(
            num_complexeigens_pairs=model_dict["num_complexeigens_pairs"],
            num_realeigens=model_dict["num_realeigens"],
            hidden_sizes=model_dict["hidden_sizes"],
            activation=model_dict["activation"],
        )

        return model.to("cuda" if self._cuda else "cpu")

    def load_koopman(self, model_dict: Dict[str, Any]) -> torch.nn.Module:
        """
        Load and initialize the Koopman dynamics model based on the provided architecture.

        Args:
            model_dict (Dict[str, Any]): Dictionary containing the model architecture and parameters.

        Returns:
            torch.nn.Module: Initialized Koopman dynamics model, moved to GPU if available and specified.
        """
        model = TrainableKoopmanDynamics(
            num_complexeigens_pairs=model_dict["num_complexeigens_pairs"],
            num_realeigens=model_dict["num_realeigens"],
            sample_time=self._sample_time,
            structure=model_dict["structure"],
        )

        return model.to("cuda" if self._cuda else "cpu")

    def load_inputs_auxiliary(self, model_dict: Dict[str, Any]) -> torch.nn.Module:
        """
        Load and initialize the inputs auxiliary model based on the provided architecture.

        Args:
            model_dict (Dict[str, Any]): Dictionary containing the model architecture and parameters.

        Returns:
            torch.nn.Module: Initialized inputs auxiliary model, moved to GPU if available and specified.
        """
        model = Input_Auxiliary(
            input_size=model_dict["lifted_state_size"],
            hidden_sizes=model_dict["hidden_sizes"],
            output_shape=model_dict["output_shape"],
            activation=model_dict["activation"],
        )

        return model.to("cuda" if self._cuda else "cpu")

    def load_trained_networks(self, model_dir: str) -> None:
        """
        Load trained network dictionaries from the specified directory.

        Args:
            model_dir (str): Directory containing the trained model.
        """
        logger = logging.getLogger(f"{__name__}.load_trained_networks")

        self.encoder_dict = torch.load(
            os.path.join(model_dir, "trained_model", "encoder.pt"),
            map_location=torch.device("cpu"),
        )
        self.decoder_dict = torch.load(
            os.path.join(model_dir, "trained_model", "decoder.pt"),
            map_location=torch.device("cpu"),
        )
        self.states_auxiliary_dict = torch.load(
            os.path.join(model_dir, "trained_model", "states_auxiliary.pt"),
            map_location=torch.device("cpu"),
        )
        self.koopman_dict = torch.load(
            os.path.join(model_dir, "trained_model", "koopman.pt"),
            map_location=torch.device("cpu"),
        )
        if os.path.exists(os.path.join(model_dir, "trained_model", "inputs_auxiliary.pt")):
            self.inputs_auxiliary_dict = torch.load(
                os.path.join(model_dir, "trained_model", "inputs_auxiliary.pt"),
                map_location=torch.device("cpu"),
            )
        else:
            self.inputs_auxiliary_dict = None

    def reconstruct_model_structure(self, normalizer_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Build the structure of the model components based on the loaded dictionaries.

        Args:
            normalizer_params (Optional[Dict[str, Any]]): Normalization parameters.
        """
        logger = logging.getLogger(f"{__name__}.build_model_components")

        encoder = self.load_encoder(self.encoder_dict, normalizer_params)
        decoder = self.load_decoder(self.decoder_dict, normalizer_params)
        states_auxiliary = self.load_states_auxiliary(self.states_auxiliary_dict)
        koopman_operator = self.load_koopman(self.koopman_dict)
        inputs_auxiliary = self.load_inputs_auxiliary(self.inputs_auxiliary_dict) if self.inputs_auxiliary_dict else None

        # Build the whole model from the components
        self._model = CombinedKoopman_withAux_withInput(
            encoder, states_auxiliary, koopman_operator, decoder, inputs_auxiliary
        )

        logger.info(" Model structure reconstructed successfully")

    def load_trained_weights(self) -> None:
        """
        Load trained weights into the model components.
        """
        logger = logging.getLogger(f"{__name__}.load_trained_weights")

        self._model.encoder.load_state_dict(self.encoder_dict["state_dict"])
        self._model.decoder.load_state_dict(self.decoder_dict["state_dict"])
        self._model.states_auxiliary.load_state_dict(self.states_auxiliary_dict["state_dict"])
        self._model.koopman.load_state_dict(self.koopman_dict["state_dict"])
        if self.inputs_auxiliary_dict:
            self._model.inputs_auxiliary.load_state_dict(self.inputs_auxiliary_dict["state_dict"])

        logger.info(" Trained weights and biases loaded successfully")

    def save_scripted_model(self) -> None:
        """
        Save the scripted version of the model components to the disk.

        This method scripts the encoder, decoder, states auxiliary, and inputs auxiliary
        components of the model and saves them as TorchScript files in the `scripted_model` directory.
        """
        logger = logging.getLogger(f"{__name__}.save_scripted_model")

        self._scripted_model_dir = os.path.join(self._model_name, "scripted_model")

        # Create the directory if it does not exist
        if not os.path.exists(self._scripted_model_dir):
            os.makedirs(self._scripted_model_dir)

        # Script and save the encoder
        scripted_encoder = torch.jit.script(self._model.encoder)
        scripted_encoder.save(os.path.join(self._scripted_model_dir, "encoder.pt"))

        # Script and save the decoder
        scripted_decoder = torch.jit.script(self._model.decoder)
        scripted_decoder.save(os.path.join(self._scripted_model_dir, "decoder.pt"))

        # Script and save the states auxiliary
        scripted_states_auxiliary = torch.jit.script(self._model.states_auxiliary)
        scripted_states_auxiliary.save(
            os.path.join(self._scripted_model_dir, "states_auxiliary.pt")
        )

        # Script and save the inputs auxiliary
        scripted_inputs_auxiliary = torch.jit.script(self._model.inputs_auxiliary)
        scripted_inputs_auxiliary.save(
            os.path.join(self._scripted_model_dir, "inputs_auxiliary.pt")
        )

        logger.info(" Scripted model components saved in `scripted_model` directory")

    def load_script_model(self):
        model_path = os.path.join(self._model_name, "scripted_model", "encoder.pt")
        scripted_encoder = torch.jit.load(model_path)
        return scripted_encoder

    def encode(
        self,
        x: torch.Tensor,
        requires_grad: bool = False,
        return_jacobian: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Transform a state in state space to the lifted space (X(i) -> Y(i)) and calculate the Jacobian of the encoder.

        Parameters:
        - x (torch.Tensor): Input tensor representing the state in state space.
        - requires_grad (bool): Flag indicating whether gradients are required.

        Returns:
        - Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - The transformed state in the lifted space (Y).
            - The Jacobian of the encoder (if requires_grad is True), otherwise None.
        """
        if requires_grad or return_jacobian:
            x.requires_grad_()

        y = self._model.encoder(x)

        if return_jacobian:
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
                )[0]
                encoder_jacobian[:, i, :] = grads
            # The effect of normalization is not considered in the autograd
            encoder_jacobian *= self._norm_factor
        else:
            encoder_jacobian = None

        if not requires_grad and not return_jacobian:
            y = y.detach()

        return y, encoder_jacobian

    def decode(
        self,
        y: torch.Tensor,
        requires_grad: bool = False,
        return_jacobian: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Transform a state in lifted space to state space (Y(i) -> X(i)) and calculate the Jacobian of the decoder.

        Parameters:
        - y (torch.Tensor): Input tensor representing the state in lifted space.
        - requires_grad (bool): Flag indicating whether gradients are required.

        Returns:
        - Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - The transformed state in the original space (X).
            - The Jacobian of the decoder (if requires_grad is True), otherwise None.
        """

        if requires_grad or return_jacobian:
            y.requires_grad_()

        x = self._model.decoder(y)

        if return_jacobian:
            output_dim = x.shape[1]
            batch_size, input_dim = y.shape
            decoder_jacobian = torch.zeros(
                batch_size, output_dim, input_dim, device=x.device
            )
            # Calculate gradient of the decoder
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
                )[0]
                decoder_jacobian[:, i, :] = grads
            # The effect of normalization is not considered in the autograd
            decoder_jacobian *= (1 / self._norm_factor).unsqueeze(1)
        else:
            decoder_jacobian = None

        if not requires_grad and not return_jacobian:
            x = x.detach()

        return x, decoder_jacobian

    def compute_model_matrices(
        self,
        y: Optional[torch.Tensor] = None,
        encoder_jacobian: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the discrete states evaluation matrix (G), discrete input matrix (H), continuous input matrix (B), and eigenvalues (Lambdas).

        Args:
            y (Optional[torch.Tensor]): Lifted state tensor. If None, use the stored state `self._y`.
            encoder_jacobian (Optional[torch.Tensor]): Jacobian of the encoder. If None, use the stored Jacobian `self._encoder_jacobian`.
            decoder_jacobian (Optional[torch.Tensor]): Jacobian of the decoder. This parameter is currently not used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - G (torch.Tensor): discrete states evaluation (Koopman matrix).
                - H (torch.Tensor): discrete input matrix.
                - B (torch.Tensor): continuous input matrix.
                - Lambdas (torch.Tensor): Eigenvalues.
        """
        if y is None:
            y = self._y

        if encoder_jacobian is None:
            encoder_jacobian = self._encoder_jacobian

        lambdas = self._model.states_auxiliary(y)
        y_unfroced, G, A = self._model.koopman(y, lambdas)
        B = self._model.inputs_auxiliary(y)

        if self._use_encoder_jacobian:
            B_phi = torch.matmul(encoder_jacobian, B)
        else:
            B_phi = B

        identity_matrix = torch.eye(y.shape[1], device=y.device)
        temp = torch.matmul(torch.inverse(A), (G - identity_matrix))
        H = torch.matmul(temp, B_phi)

        return G, H, A, B, lambdas

    def compute_next_lifted_state(
        self, G: torch.Tensor, H: torch.Tensor, y: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the next lifted state using Koopman dynamics.

        Args:
            G (torch.Tensor): Koopman matrix.
            H (torch.Tensor): Input matrix.
            y (torch.Tensor): Current lifted state.
            u (torch.Tensor): Control input.

        Returns:
            torch.Tensor: The next lifted state.
        """
        # y_adv = torch.matmul(A, y0).squeeze(-1)  + torch.matmul(B_phi, u.unsqueeze(-1)).squeeze(-1)
        y_unforced = torch.matmul(G, y.unsqueeze(-1)).squeeze(-1)
        y_forced = torch.matmul(H, u.unsqueeze(-1)).squeeze(-1)
        y_next = y_unforced + y_forced

        return y_next

    def lift_trajectory(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Transform a trajectory of states in state space (robot space) to the lifted space (X -> Y).

        Args:
            x (torch.Tensor): Trajectory of states in state space.

        Returns:
            List[torch.Tensor]: List of transformed states in the lifted space.
        """
        Yp_list = []  # List to store the lifted states

        # Loop over the sequence steps
        for j in range(self._num_pred_steps):
            lifted_state = self._model.encoder(x[:, j + 1, :])
            Yp_list.append(lifted_state)

        return Yp_list

    def reconstruct_trajectory(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode and decode a trajectory of states in state space to the lifted space and back.

        Args:
            x (torch.Tensor): Trajectory of states in state space.

        Returns:
            List[torch.Tensor]: List of transformed and decoded states in the state space.
        """
        Xp_list = []  # List to store the transformed and decoded states

        # Loop over the sequence steps
        for j in range(self._num_pred_steps):
            encoded_state = self._model.encoder(x[:, j + 1, :])
            decoded_state = self._model.decoder(encoded_state)
            Xp_list.append(decoded_state)

        return Xp_list

    def initialize(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Initialize the model by encoding the initial state and storing the states in member variables.

        Args:
            x0 (torch.Tensor): Initial state in the state space.

        Returns:
            torch.Tensor: Encoded initial state in the lifted space.
        """
        self._x = x0
        self._y, self._encoder_jacobian = self.encode(
            self._x, requires_grad=self._use_encoder_jacobian
        )
        return self._y

    def step(self, u: Any = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a step in the lifted space and transfer it to the state space, updating the state records.

        Args:
            u (Any): Control input. Default is 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated state in the state space and lifted space.
        """
        # Check if the model is initialized
        if (
            self._x is None
            or self._y is None
            or (self._encoder_jacobian is None and self._use_encoder_jacobian)
        ):
            raise AssertionError("The model is not initialized")

        G, H, A, B, lambdas = self.compute_koopman_matrices(
            self._y, self._encoder_jacobian
        )
        self._y = self.compute_next_lifted_state(G, H, self._y, u)
        self._x = self.decode(self._y)[0]

        return self._x, self._y

    def predict_trajectory(
        self, x0: torch.Tensor, u: torch.Tensor, requires_grad: bool = False
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Predict trajectory using the Koopman model.

        Args:
            x0 (torch.Tensor): Initial state in the state space.
            u (torch.Tensor): Control input sequence.
            requires_grad (bool): Whether to require gradients for the encoder.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
                - Xp_pred_list: Predicted states in the state space.
                - Yp_pred_list: Predicted states in the lifted space.
                - Lambdas_list: List of eigenvalues for each prediction step.
                - B_list: List of auxiliary matrices for each prediction step.
                - U_list: List of control inputs for each prediction step.
        """
        Yp_pred_list = []  # Predicted value of y_plus from iterating the Koopman model
        Xp_pred_list = []  # Predicted value of x_plus from decoding the iterated model
        Lambdas_list = []
        H_list = []
        G_list = []
        U_list = []
        x = x0

        y, encoder_jacobian = self.encode(x, requires_grad=requires_grad)

        for j in range(self._num_pred_steps):
            G, H, A, B, Lambdas = self.compute_model_matrices(y, encoder_jacobian)
            y = self.compute_next_lifted_state(G, H, y, u[:, j, :])
            x, _ = self.decode(y, requires_grad=False)
            Xp_pred_list.append(x)
            Lambdas_list.append(Lambdas)
            H_list.append(H)
            G_list.append(G)
            U_list.append(u[:, j, :])
            Yp_pred_list.append(y)

        return Xp_pred_list, Yp_pred_list, Lambdas_list, G_list, H_list, U_list

