import os, copy, torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from typing import Tuple, List, Any, Callable, Optional, Union
import logging

plt.ion()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,  # Model to be trained.
        crit: Callable,  # Loss function
        metric: Callable,  # Metric function
        optim: Callable,  # Optimizer
        lr_scheduler: Callable,  # learning rate decay
        train_dl,  # Training data set
        val_dev_dl,  # Validation (or test) data set
        cuda=False,  # Whether to use the GPU
        early_stopping_patience=None,  # The patience for early stopping
        save_dir=None,
        model_parameters=None,
        training_parameters=None,
        normalizer_scale=1,
        sequence_length=None,
    ):
        self._model = model
        self._crit = crit
        self._metric = metric
        self._optim = optim
        self._train_dl = train_dl
        self._val_dev_dl = val_dev_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self._lr_scheduler = lr_scheduler
        self._save_dir = save_dir
        self._model_parameters = model_parameters
        self._training_parameters = training_parameters
        self._normalizer_scale = normalizer_scale
        self._sequence_length = sequence_length

        logger = logging.getLogger(f"{__name__}.Trainer.__init__")
        logger.info(" Initializing Trainer")

        if cuda:
            logger.info(" Using CUDA")
            self._model = model.cuda()
            self._crit = crit.cuda()
            if self._metric is not None:
                self._metric = metric.cuda()

        self._trained_model_dir = os.path.join(self._save_dir, "trained_model")
        if not os.path.exists(self._trained_model_dir):
            os.makedirs(self._trained_model_dir)
            logger.info(
                f"Created directory {self._trained_model_dir} for saving trained models"
            )

        self._norm_factor = 1
        self._epoch = 0
        self._save_step = 10
        self._plot_step = 10

        if self._sequence_length is None:
            raise ValueError("seuqnece length cannot be None")

    def save_checkpoint(self, epoch):
        path = os.path.join(
            self._checkpoints_dir, "checkpoint_{:03d}.ckp".format(epoch)
        )
        torch.save({"state_dict": self._model.state_dict()}, path)

    def restore_checkpoint(self, epoch_n):
        if not os.path.exists(self._checkpoints_dir):
            error_message = f"[{self._checkpoints_dir}] does not exist!"
            raise FileNotFoundError(error_message)
        path = os.path.join(
            self._checkpoints_dir, "checkpoint_{:03d}.ckp".format(epoch_n)
        )
        ckp = torch.load(path, "cuda" if self._cuda else None)
        self._model.load_state_dict(ckp["state_dict"])

    def save_model(self, dir: str = None, log: bool = True) -> None:
        """
        Save the model and its components to the specified directory.

        Parameters:
        - dir (str): Directory to save the model. If None, saves to the default trained model directory.
        """

        logger = logging.getLogger(f"{__name__}.save_model")

        # Determine the save directory
        save_dir = dir if dir is not None else self._trained_model_dir

        def save_component(component_name: str, state_dict: dict, filename: str):
            path = os.path.join(save_dir, filename)
            try:
                torch.save(state_dict, path)
            except Exception as e:
                logger.error(f" Error saving {component_name} to {path}: {e}")

        # Save the whole model
        model_dict = copy.deepcopy(self._model_parameters)
        model_dict["state_dict"] = self._model.state_dict()
        save_component("Whole model", model_dict, "whole_model.pt")

        # Save encoder separately
        encoder_dict = copy.deepcopy(self._model_parameters["encoder_parameters"])
        encoder_dict["state_dict"] = self._model.encoder.state_dict()
        save_component("Encoder", encoder_dict, "encoder.pt")

        # Save decoder separately, if exists
        if hasattr(self._model, "decoder"):
            decoder_dict = copy.deepcopy(self._model_parameters["decoder_parameters"])
            decoder_dict["state_dict"] = self._model.decoder.state_dict()
            save_component("Decoder", decoder_dict, "decoder.pt")

        # Save states auxiliary separately, if exists
        if hasattr(self._model, "states_auxiliary"):
            states_aux_dict = copy.deepcopy(
                self._model_parameters["states_auxiliary_parameters"]
            )
            states_aux_dict["state_dict"] = self._model.states_auxiliary.state_dict()
            save_component("States auxiliary", states_aux_dict, "states_auxiliary.pt")

        # Save Koopman operator separately
        if hasattr(self._model, "koopman"):
            koopman_dict = copy.deepcopy(self._model_parameters["koopman_parameters"])
            koopman_dict["state_dict"] = self._model.koopman.state_dict()
            save_component("Koopman operator", koopman_dict, "koopman.pt")

        # Save inputs auxiliary separately, if exists
        if hasattr(self._model, "inputs_auxiliary"):
            inputs_aux_dict = copy.deepcopy(
                self._model_parameters["inputs_auxiliary_parameters"]
            )
            inputs_aux_dict["state_dict"] = self._model.inputs_auxiliary.state_dict()
            save_component("Inputs auxiliary", inputs_aux_dict, "inputs_auxiliary.pt")
        
        # Save inputs auxiliary separately, if exists
        if hasattr(self._model, "states_matrix"):
            states_matrix_dict = copy.deepcopy(
                self._model_parameters["koopman_parameters"]
            )
            states_matrix_dict["state_dict"] = self._model.states_matrix.state_dict()
            save_component("States matrix", states_matrix_dict, "states_matrix.pt")

        # Save inputs auxiliary separately, if exists
        if hasattr(self._model, "actuation_matrix"):
            actuation_matrix_dict = copy.deepcopy(
                self._model_parameters["inputs_parameters"]
            )
            actuation_matrix_dict["state_dict"] = self._model.actuation_matrix.state_dict()
            save_component("Actuation matrix", actuation_matrix_dict, "actuation_matrix.pt")


        if log:
            logger.info(" Trained model saved successfully")

    def save_model_structure(self, model: nn.Module) -> None:
        """
        Save the model structure to a file.

        Parameters:
        - model (nn.Module): The PyTorch model.
        - file_path (str): Path to the file where the structure will be saved.
        """
        logger = logging.getLogger(f"{__name__}.save_model_structure")
        file_path = os.path.join(self._save_dir, "model_summary.txt")
        try:
            with open(file_path, "w") as f:
                f.write(str(model))
            logger.info(f" Model summary saved successfully")
        except Exception as e:
            logger.error(f" Error saving model summary to {file_path}: {e}")

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
        logger = logging.getLogger(f"{__name__}.encode")
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
        logger = logging.getLogger(f"{__name__}.decode")

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

    def deep_koopman_step(self, X: torch.Tensor, u: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Perform a step in the deep Koopman process.

        Parameters:
        - X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        - u (torch.Tensor): Control input tensor of shape (batch_size, sequence_length, control_dim).

        Returns:
        - Tuple containing:
            - x0: Initial state.
            - x0_pred: Predicted initial state.
            - Xp_list: List of true future states.
            - Xp_pred_list: List of predicted future states.
            - Yp_list: List of true future outputs in lifted space.
            - Yp_pred_list: List of predicted future outputs in lifted space.
        """
        logger = logging.getLogger(f"{__name__}.deep_koopman_step")

        Yp_list, Yp_pred_list, Xp_list, Xp_pred_list = [], [], [], []

        # Reconstruct the first state using the autoencoder
        len_history = self._model_parameters["encoder_parameters"]["len_history"]
        x0_h = X[:, 0:len_history, :]
        x0 = X[:, len_history-1, :]
        if not self._model_parameters["encoder_parameters"]["extend_lifted_state"]:
            x0_pred = self._model.decoder(self._model.encoder(x0))
        else:
            x0_pred = self._model.encoder(x0_h)[:, 0:self._model_parameters["encoder_parameters"]["state_size"]].unsqueeze(1)  ## this line is temporary and must be fixed professionaly
            x0_pred = x0  ## temporarly

        x = X[:, 0:len_history, :]
        x.requires_grad_(self._training_parameters["train_encoder"])
        # y, encoder_jacobian = self.encode(x=x, requires_grad=True)
        y = self._model.encoder(x)

        # Loop over the sequence steps
        
        for j in range(len_history - 1, self._sequence_length - 1, 1):
            if hasattr(self._model, 'states_auxiliary'):        ## if the model is space varying
                lambdas = self._model.states_auxiliary(y)
                y_unfroced, G, A = self._model.koopman(y, lambdas)  ## unforced part
                B = self._model.inputs_auxiliary(y)

                #### this is for direct use of estimated B as B_Phi - may need to be cahnges according to the structure
                B_phi = B
                # B_phi = torch.matmul(encoder_jacobian, B)
                identity_matrix = torch.eye(y.shape[1], device=y.device)
                temp = torch.matmul(torch.inverse(A), (G - identity_matrix))
                H = torch.matmul(temp, B_phi)

                # control input part
                y_froced = torch.matmul(H, u[:, j, :].unsqueeze(-1)).squeeze(-1)
                y = y_unfroced + y_froced

            else:
                y = self._model.states_matrix(y) + self._model.actuation_matrix(u[:, j, :]) 

            Yp_pred_list.append(y)
            Yp_list.append(self._model.encoder(X[:, j-len_history+1:j+1, :]))
            if j < self._training_parameters["num_pred_steps"]:
                if not self._model_parameters["encoder_parameters"]["extend_lifted_state"]:
                    x = self._model.decoder(y)
                else:
                    x = y[:, 0:self._model_parameters["encoder_parameters"]["state_size"]]  ## this line is temporary and must be fixed professionaly

                Xp_pred_list.append(x)
                Xp_list.append(X[:, j + 1, :])

        return x0, x0_pred, Xp_list, Xp_pred_list, Yp_list, Yp_pred_list

    def train_step(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single training step.

        Parameters:
        - x (torch.Tensor): Input tensor representing the state in state space.
        - u (torch.Tensor): Control input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]:
            - loss: The computed loss for the current training step.
            - metric: The metric to evaluate the model's performance.
        """
        logger = logging.getLogger(f"{__name__}.train_step")

        self._optim.zero_grad()
        (
            x0,
            x0_pred,
            Xp_list,
            Xp_pred_list,
            Yp_list,
            Yp_pred_list,
        ) = self.deep_koopman_step(x, u)

        # Calculate losses
        loss_recon = self._crit.recon(x0, x0_pred)
        loss_pred = self._crit.pred(Xp_list, Xp_pred_list)
        loss_lin = self._crit.lin(Yp_list, Yp_pred_list)
        loss_inf = self._crit.inf_norm(x0, x0_pred, Xp_list[0], Xp_pred_list[0])

        # Determine which loss function to use
        # In the begining only use reconstuction loss for faster initial convergence, then switch to total loss
        if self._epoch < self._training_parameters["n_recon_epoch"]:
            loss = self._crit.forward_rec(loss_recon, self._model.parameters())
        else:
            loss = self._crit.forward(
                loss_recon, loss_pred, loss_lin, loss_inf, self._model.parameters()
            )
        metric = loss

        # Backpropagate and update
        loss.backward()
        self._optim.step()
        return loss, metric

    def validation_step(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Perform a single validation step.

        Parameters:
        - x (torch.Tensor): Input tensor representing the state in state space.
        - u (torch.Tensor): Control input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            - loss: The computed loss for the current validation step.
            - metric: The metric to evaluate the model's performance.
            - x0: Initial state.
            - x0_pred: Predicted initial state.
            - Xp_list: List of true future states.
            - Xp_pred_list: List of predicted future states.
            - Yp_list: List of true future outputs in lifted space.
            - Yp_pred_list: List of predicted future outputs in lifted space.
        """
        logger = logging.getLogger(f"{__name__}.validation_step")

        (
            x0,
            x0_pred,
            Xp_list,
            Xp_pred_list,
            Yp_list,
            Yp_pred_list,
        ) = self.deep_koopman_step(x, u)

        # Calculate losses
        loss_recon = self._crit.recon(x0, x0_pred)
        loss_pred = self._crit.pred(Xp_list, Xp_pred_list)
        loss_lin = self._crit.lin(Yp_list, Yp_pred_list)
        loss_inf = self._crit.inf_norm(x0, x0_pred, Xp_list[0], Xp_pred_list[0])

        # Determine which loss function to use
        # In the begining only use reconstuction loss for faster initial convergence, then switch to total loss
        if self._epoch < self._training_parameters["n_recon_epoch"]:
            loss = self._crit.forward_rec(loss_recon, self._model.parameters())
        else:
            loss = self._crit.forward(
                loss_recon, loss_pred, loss_lin, loss_inf, self._model.parameters()
            )

        # Calculate metrics
        metric_recon = self._metric.recon(x0, x0_pred)
        metric_pred = self._metric.pred(Xp_list, Xp_pred_list)
        metric_lin = self._metric.lin(Yp_list, Yp_pred_list)
        metric = self._metric.forward(metric_recon, metric_pred, metric_lin)

        return loss, metric, x0, x0_pred, Xp_list, Xp_pred_list, Yp_list, Yp_pred_list

    def train_epoch(self) -> Tuple[float, float]:
        """
        Perform a single epoch of training.

        Returns:
        - Tuple[float, torch.Tensor]:
            - loss: The average loss for the epoch.
            - metric: The average metric for the epoch.
        """
        logger = logging.getLogger(f"{__name__}.train_epoch")

        ## set training mode
        self._model.train()

        # Enable gradients for relevant parts of the model
        self._model.encoder.requires_grad_(self._training_parameters["train_encoder"])
        if hasattr(self._model, 'decoder'):
            self._model.decoder.requires_grad_(self._training_parameters["train_decoder"])
        if hasattr(self._model, 'states_auxiliary'):
            self._model.states_auxiliary.requires_grad_(
                self._training_parameters["train_states_auxiliary"]
            )
        if hasattr(self._model, 'inputs_auxiliary'):
            self._model.inputs_auxiliary.requires_grad_(
                self._training_parameters["train_inputs_auxiliary"]
            )
        if hasattr(self._model, 'states_matrix'):
            self._model.states_matrix.requires_grad_(
                self._training_parameters["train_A"]
            )
        if hasattr(self._model, 'actuation_matrix'):
            self._model.actuation_matrix.requires_grad_(
                self._training_parameters["train_B"]
            )

        counter = 0
        running_loss = torch.zeros(1)
        running_metric = torch.zeros(3)
        if self._cuda:
            running_loss = torch.zeros(1).cuda()
            running_metric = torch.zeros(3).cuda()

        # Iterate through the training set
        for x, u, y in self._train_dl:
            # Transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x, u, y = x.cuda(), u.cuda(), y.cuda()

            # Perform a training step
            loss_batch, metric_batch = self.train_step(x, u)
            running_loss += loss_batch
            running_metric += metric_batch
            counter += 1

        # Step the learning rate scheduler
        self._lr_scheduler.step()

        # Calculate the average loss and metric for the epoch
        loss = running_loss / counter
        metric = running_metric / counter
        return loss.detach().cpu().numpy(), metric.detach().cpu().numpy()

    def evaluate_epoch(
        self,
    ) -> Tuple[
        float,
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Perform validation or development evaluation.

        Returns:
        - Tuple containing:
            - loss: The average loss for the validation set.
            - metric: The average metric for the validation set.
            - X0_list: Tensor of initial states.
            - X0_pred_list: Tensor of predicted initial states.
            - Xp_list: Tensor of true future states.
            - Xp_pred_list: Tensor of predicted future states.
            - Yp_list: Tensor of true future outputs in lifted space.
            - Yp_pred_list: Tensor of predicted future outputs in lifted space.
        """

        X0_list, X0_pred_list, Yp_list, Yp_pred_list = [], [], [], []
        Xp_list, Xp_pred_list = [], []

        # Set evaluation mode and disable gradient computation
        self._model.eval()
        self._model.requires_grad_(False)

        counter = 0
        running_loss = torch.zeros(1, device="cuda" if self._cuda else "cpu")
        running_metric = torch.zeros(3, device="cuda" if self._cuda else "cpu")

        # Iterate through the validation set
        for batch_idx, (x, u, y) in enumerate(self._val_dev_dl, 1):
            # Transfer the batch to the GPU if given
            if self._cuda:
                x, u, y = x.cuda(), u.cuda(), y.cuda()

            # Perform a validation step
            loss_batch, metric_batch, x0, x0p, xp, xpp, yp, ypp = self.validation_step(
                x, u
            )

            # Stack lists of the tensors and append for real-time visualization of estimations
            X0_list += [x0[i].view(1, x0.size(1)) for i in range(x0.size(0))]
            X0_pred_list += [x0p[i].view(1, x0p.size(1)) for i in range(x0p.size(0))]
            Xp_list += self.list_to_tensor(xp)
            Xp_pred_list += self.list_to_tensor(xpp)
            Yp_list += self.list_to_tensor(yp)
            Yp_pred_list += self.list_to_tensor(ypp)

            running_loss += loss_batch
            running_metric += metric_batch
            counter += 1

        # Calculate the average loss and average metrics
        loss = running_loss / counter
        metric = (running_metric / counter) * self._normalizer_scale

        return (
            loss.detach().cpu().numpy(),
            metric.detach().cpu().numpy(),
            torch.stack(X0_list, dim=0).detach().cpu(),
            torch.stack(X0_pred_list, dim=0).detach().cpu(),
            torch.stack(Xp_list, dim=0).detach().cpu(),
            torch.stack(Xp_pred_list, dim=0).detach().cpu(),
            torch.stack(Yp_list, dim=0).detach().cpu(),
            torch.stack(Yp_pred_list, dim=0).detach().cpu(),
        )

    def fit(
        self, epochs: int = -1, plot_learning_history: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Train the model for a specified number of epochs.

        Parameters:
        - epochs (int): Number of epochs to train. Default is -1.
        - plot_learning (bool): Flag to indicate whether to plot learning progress. Default is True.

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - train_losses: Array of training losses.
            - val_losses: Array of validation losses.
            - val_metrics: Array of validation metrics.
        """
        logger = logging.getLogger(f"{__name__}.fit")

        counter = 0
        self._epoch = 1
        train_losses = np.full(1, np.nan)
        train_metrics = np.full((3, 1), np.nan)
        val_losses = np.full(1, np.nan)
        val_metrics = np.full((1, 3), np.nan)

        num_states = self._model.encoder.get_inputsize()

        self._model.encoder.requires_grad_(self._training_parameters["train_encoder"])
        if hasattr(self._model, 'decoder'):
            self._model.decoder.requires_grad_(self._training_parameters["train_decoder"])
        if hasattr(self._model, 'states_auxiliary'):
            self._model.states_auxiliary.requires_grad_(
                self._training_parameters["train_states_auxiliary"]
            )
        if hasattr(self._model, 'inputs_auxiliary'):
            self._model.inputs_auxiliary.requires_grad_(
                self._training_parameters["train_inputs_auxiliary"]
            )
        if hasattr(self._model, 'states_matrix'):
            self._model.states_matrix.requires_grad_(
                self._training_parameters["train_A"]
            )
        if hasattr(self._model, 'actuation_matrix'):
            self._model.actuation_matrix.requires_grad_(
                self._training_parameters["train_B"]
            )
        self.save_model_structure(self._model)

        # Log model info
        logger.info(
            f" Model has {sum(p.numel() for p in self._model.parameters() if p.requires_grad)} parameters."
        )
        logger.info(f" Train dataset has {len(self._train_dl)} batches")

        # Initialize progress bar
        p_bar = tqdm(range(epochs))
        if plot_learning_history:
            self.fig, self.axs = plt.subplots(2, int(num_states / 2), figsize=(11, 7))

        for _ in p_bar:
            # Train for an epoch
            train_loss, train_metric = self.train_epoch()
            train_losses = np.append(train_losses, train_loss)
            train_metrics = np.append(
                train_metrics, np.expand_dims(train_metric, 1), axis=1
            )

            # Calculate loss and metrics on the validation set
            val_loss, val_metric, x0, x0_pred, x_plus, x_plus_pred, _, _ = (
                self.evaluate_epoch()
            )
            val_losses = np.append(val_losses, val_loss)
            val_metrics = np.append(val_metrics, np.expand_dims(val_metric, 0), axis=0)

            ## update printed status
            p_bar.set_description(
                f"Train Loss: {train_loss.item():.6f}, Val Loss {val_loss.item():.6f}"
            )
            p_bar.refresh()

            if plot_learning_history and (self._epoch % self._plot_step == 0):
                self.plot_training_history(
                    counter,
                    train_losses,
                    val_losses,
                    val_metrics,
                    x_plus,
                    x0_pred,
                    x_plus_pred,
                )

            # Save the model
            if self._epoch % self._save_step == 0:
                self.save_model(dir=None, log=False)

            # Check early stopping
            if self._early_stopping_patience is not None:
                if counter > self._early_stopping_patience:
                    diff_loss = (
                        train_losses[counter - 1]
                        - train_losses[counter - 1 - self._early_stopping_patience]
                    )
                    if diff_loss >= 0:
                        logger.info(" Early stopping enabled")
                        break

            counter += 1
            self._epoch += 1

        return (train_losses[1:], val_losses[1:], val_metrics[1:, :])

    def list_to_tensor(self, input_list):
        torchtensor = torch.stack(input_list, dim=1)
        return [torchtensor[i] for i in range(torchtensor.shape[0])]

    def plot_training_history(
        self,
        counter,
        train_losses,
        val_losses,
        val_metrices,
        x_plus,
        x0_pred,
        x_plus_pred,
    ):
        color_list = ("red", "blue", "black", "green", "cyan", "magenta", "yellow")
        self.axs[0, 0].clear()
        self.axs[0, 1].clear()
        self.axs[1, 0].clear()
        self.axs[1, 1].clear()
        x_h = range(1, counter + 2)

        # plot losses
        self.axs[0, 0].plot(
            x_h,
            train_losses[1:],
            label="train",
            linewidth=1.5,
            color="b",
            linestyle="solid",
        )
        self.axs[0, 0].plot(
            x_h,
            val_losses[1:],
            label="validation",
            linewidth=1.5,
            color="r",
            linestyle="solid",
        )
        self.axs[0, 0].set_yscale("log")
        self.axs[0, 0].legend(loc="upper right")
        self.axs[0, 0].grid(visible=True)
        self.axs[0, 0].set_xlabel("epoch")
        self.axs[0, 0].set_title("Loss")

        # plot metrics
        metric_labels = ["recon", "pred", "linear"]  # Define your labels here
        for i in range(val_metrices.shape[1]):
            self.axs[0, 1].plot(
                range(1, counter + 2),
                val_metrices[1:, i],
                label="val_" + metric_labels[i],
                linewidth=1.5,
                color=color_list[i + 2],
                linestyle="solid",
            )
        self.axs[0, 1].set_yscale("log")
        self.axs[0, 1].legend(loc="upper right")
        self.axs[0, 1].grid(visible=True)
        self.axs[0, 1].set_xlabel("epoch")
        self.axs[0, 1].set_title("Metric")

        # Plot states and predicted states
        for j in range(self.axs.shape[1]):
            for i in range(x_plus.shape[0]):
                self.axs[1, j].plot(
                    x_plus[i, :, j * 2],
                    x_plus[i, :, j * 2 + 1],
                    c="black",
                    label="truth" if i == 0 else "",
                    linewidth=0.5,
                )
                self.axs[1, j].plot(
                    x_plus_pred[i, :, j * 2],
                    x_plus_pred[i, :, j * 2 + 1],
                    c="red",
                    label="model" if i == 0 else "",
                    linewidth=0.5,
                )

            self.axs[1, j].set_xlabel(f"$x_{{{j*2+1}}}$")  # Adjusted to j*2+1
            self.axs[1, j].set_ylabel(f"$x_{{{j*2+2}}}$")  # Adjusted to j*2+2
            self.axs[1, j].minorticks_on()
            self.axs[1, j].grid(
                which="both", color="lightgray", linestyle="--", linewidth=0.5
            )
            self.axs[1, j].set_title("Normalized phase portrait - State Space")
            self.axs[1, j].legend(loc="best")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        plt.savefig(os.path.join(self._save_dir, "training_history.pdf"), format="pdf")
