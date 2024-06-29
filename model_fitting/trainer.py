import os, copy, csv, torch
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

plt.ion()

from tqdm import tqdm
from IPython.display import clear_output


class Trainer:
    def __init__(
        self,
        model,  # Model to be trained.
        crit,  # Loss function
        metric=None,  # Metric function
        optim=None,  # Optimizer
        lr_scheduler=None,  # learning rate decay
        train_dl=None,  # Training data set
        val_dev_dl=None,  # Validation (or test) data set
        cuda=False,  # Whether to use the GPU
        early_stopping_patience=None,  # The patience for early stopping
        save_dir=None,
        model_parameters=None,
        training_parameters=None,
        normalilzer_scale=1,
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
        self._normalilzer_scale = normalilzer_scale
        self._sequence_length = sequence_length

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            if self._metric is not None:
                self._metric = metric.cuda()

        self._trained_model_dir = os.path.join(self._save_dir, "trained_model")
        if not os.path.exists(self._trained_model_dir):
            os.makedirs(self._trained_model_dir)

        self._norm_factor = 1

        # self._train_animation_dir = os.path.join(self._save_dir, "train_animation")
        # if not os.path.exists(self._train_animation_dir):
        #     os.makedirs(self._train_animation_dir)

    # def save_checkpoint(self, epoch):
    #     path = os.path.join(self._checkpoints_dir, "checkpoint_{:03d}.ckp".format(epoch))
    #     torch.save({"state_dict": self._model.state_dict()}, path)

    def restore_checkpoint(self, epoch_n):
        if not os.path.exists(self._checkpoints_dir):
            error_message = f"[{self._checkpoints_dir}] does not exist!"
            raise FileNotFoundError(error_message)
        path = os.path.join(
            self._checkpoints_dir, "checkpoint_{:03d}.ckp".format(epoch_n)
        )
        ckp = torch.load(path, "cuda" if self._cuda else None)
        self._model.load_state_dict(ckp["state_dict"])

    def save_model(self, dir=None):
        if dir is not None:
            save_dir = dir
        else:
            save_dir = self._trained_model_dir

        ### save the whole model
        model_dict = copy.copy(self._model_parameters)
        model_dict["state_dict"] = self._model.state_dict()
        path = os.path.join(save_dir, "TorchTrained_WholeModel.pt")
        torch.save(model_dict, path)

        ### save encoder separately
        model_dict = copy.copy(self._model_parameters["encoder_parameters"])
        model_dict["state_dict"] = self._model.encoder.state_dict()
        path = os.path.join(save_dir, "TorchTrained_Encoder.pt")
        torch.save(model_dict, path)

        ### save decoder separately
        model_dict = copy.copy(self._model_parameters["decoder_parameters"])
        model_dict["state_dict"] = self._model.decoder.state_dict()
        path = os.path.join(save_dir, "TorchTrained_Decoder.pt")
        torch.save(model_dict, path)

        ### save auxilary separately, if exists
        if hasattr(self._model, "states_auxilary"):
            model_dict = copy.copy(self._model_parameters["states_auxilary_parameters"])
            model_dict["state_dict"] = self._model.states_auxilary.state_dict()
            path = os.path.join(save_dir, "TorchTrained_States_Auxilary.pt")
            torch.save(model_dict, path)

        ### save Koopman operator separately
        model_dict = copy.copy(self._model_parameters["koopman_parameters"])
        model_dict["state_dict"] = self._model.koopman.state_dict()
        path = os.path.join(save_dir, "TorchTrained_Koopman.pt")
        torch.save(model_dict, path)

        ### save inputs_auxilary separately, if exists
        if hasattr(self._model, "inputs_auxilary"):
            model_dict = copy.copy(self._model_parameters["inputs_auxilary_parameters"])
            model_dict["state_dict"] = self._model.inputs_auxilary.state_dict()
            path = os.path.join(save_dir, "TorchTrained_Inputs_Auxilary.pt")
            torch.save(model_dict, path)

        # A_matrix = self._model.koopman.state_dict()['A.weight'].cpu().numpy()
        # # Open a CSV file and write the weights matrix
        # with open(os.path.join(save_dir, "Koopman_A.csv"), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for row in A_matrix:
        #         writer.writerow(row)
        # B_matrix = self._model.koopman.state_dict()['B.weight'].cpu().numpy()
        # # Open a CSV file and write the weights matrix
        # with open(os.path.join(save_dir, "Koopman_B.csv"), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for row in B_matrix:
        #         writer.writerow(row)

        # print('Trained model saved in "', self._model_parameters["name"], '"')

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = torch.randn(1, 3, 120, 120, requires_grad=True)
        y = self._model(x)
        torch.onnx.export(
            m,  # model being run
            x,  # model input (or a tuple for multiple inputs)
            fn,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable lenght axes
                "output": {0: "batch_size"},
            },
        )

    """  Transform a state in state space to the lifted space (X(i)->Y(i)) and canculate the jacobian of the encoder """

    def encode(self, x, requires_grad=False):
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

    """  Transform a state in lifted space to state space (X(i)->Y(i)) and canculate the jacobian of the encoder """

    def decode(self, y, requires_grad=False):
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
            decoder_jacobian = decoder_jacobian * (1 / self._norm_factor).unsqueeze(
                1
            )  #  the effect of normalization is not considered in the autograd
        else:
            decoder_jacobian = None
            x = x.detach()
        # x0.requires_grad_(self._training_parameters["train_encoder"])
        return x, decoder_jacobian

    def deep_koopman_step(self, X, u):
        Yp_list = []  ## truth value of y_plus from advanced encoded x
        Yp_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
        Xp_list = []  ## truth value of x_plus from advanced x
        Xp_pred_list =   [] ## predicted value of x_plus from decoding the iterated Koopman model
        # Placeholder for the gradient

        if self._sequence_length is None:
            raise ValueError("seuqnece length cannot be None")
        ### reconstruct the fist states only using the auto encoder
        x0 = X[:, 0, :]
        x0_pred = self._model.decoder(self._model.encoder(x0))
        ### loop over the sequnec steps
        x = X[:, 0, :]

        # x.requires_grad_(True)
        # y = self._model.encoder(x)

        # output_dim = y.shape[1]
        # batch_size, input_dim = x.shape
        # encoder_jacobian = torch.zeros(
        #     batch_size, output_dim, input_dim, device=y.device
        # )
        # for i in range(output_dim):
        #     grad_outputs = torch.zeros_like(y)
        #     grad_outputs[:, i] = 1  # Select the i-th output
        #     grads = torch.autograd.grad(
        #         outputs=y,
        #         inputs=x,
        #         grad_outputs=grad_outputs,
        #         retain_graph=True,  # Need to retain for next iterations
        #         create_graph=False,  # True if you need higher-order gradients
        #         only_inputs=True,
        #     )[
        #         0
        #     ]  # [0] to get the gradient tensor
        #     encoder_jacobian[:, i, :] = grads
            
        x.requires_grad_(self._training_parameters["train_encoder"])

        # itterative solver loop
        y, encoder_jacobian = self.encode(x=x, requires_grad=True)

        for j in range(self._sequence_length - 1):
                Lambdas = self._model.states_auxilary(y)
                y, G, A = self._model.koopman(y, Lambdas)  ## unforced part
                B = self._model.inputs_auxilary(y)

                #### this is for direcrtr use of estimated B as B_Phi - may need to be cahnges according to the structure
                B_phi = B
                # B_phi = torch.matmul(encoder_jacobian, B)
                identity_matrix = torch.eye(y.shape[1], device=y.device)
                temp = torch.matmul(torch.inverse(A), (G - identity_matrix))
                H = torch.matmul(temp, B_phi)
                ## control input part
                y = y + torch.matmul(H, u[:, j, :].unsqueeze(-1)).squeeze(-1)
                # y_adv = y_adv + torch.matmul(B_phi, u[:,j,:].unsqueeze(-1)).squeeze(-1)    ## control input part
                # y_adv = self._model.koopman(y0, Lambdas)
                Yp_pred_list.append(y)
                Yp_list.append(self._model.encoder(X[:, j + 1, :]))
                if j < self._training_parameters["num_pred_steps"]:  
                    x = self._model.decoder(y)
                    Xp_pred_list.append(x)
                    Xp_list.append(X[:, j + 1, :])

        return x0, x0_pred, Xp_list, Xp_pred_list, Yp_list, Yp_pred_list

    def train_step(self, x, u):
        self._optim.zero_grad()
        (
            x0,
            x0_pred,
            Xp_list,
            Xp_pred_list,
            Yp_list,
            Yp_pred_list,
        ) = self.deep_koopman_step(x, u)
        loss_recon = self._crit.recon(x0, x0_pred)
        loss_pred = self._crit.pred(Xp_list, Xp_pred_list)
        loss_lin = self._crit.lin(Yp_list, Yp_pred_list)
        loss_inf = self._crit.infnorm(x0, x0_pred, Xp_list[0], Xp_pred_list[0])
        ## in the begining only use reconstuction loss for faster initial convergence, then switch to total loss
        if self._epoch < self._training_parameters["n_recon_epoch"]:
            loss = self._crit.forward_recReg(loss_recon, self._model.parameters())
        else:
            loss = self._crit.forward(
                loss_recon, loss_pred, loss_lin, loss_inf, self._model.parameters()
            )
        metric = loss
        ## backpropagate and update
        loss.backward()
        self._optim.step()
        return loss, metric

    def val_dev_step(self, x, u):
        (
            x0,
            x0_pred,
            Xp_list,
            Xp_pred_list,
            Yp_list,
            Yp_pred_list,
        ) = self.deep_koopman_step(x, u)
        loss_recon = self._crit.recon(x0, x0_pred)
        loss_pred = self._crit.pred(Xp_list, Xp_pred_list)
        loss_lin = self._crit.lin(Yp_list, Yp_pred_list)
        loss_inf = self._crit.infnorm(x0, x0_pred, Xp_list[0], Xp_pred_list[0])
        ## in the begining only use reconstuction loss for faster initial convergence, then switch to total loss
        if self._epoch < self._training_parameters["n_recon_epoch"]:
            loss = self._crit.forward_recReg(loss_recon, self._model.parameters())
        else:
            loss = self._crit.forward(
                loss_recon, loss_pred, loss_lin, loss_inf, self._model.parameters()
            )

        metric_recon = self._metric.recon(x0, x0_pred)
        metric_pred = self._metric.pred(Xp_list, Xp_pred_list)
        metric_lin = self._metric.lin(Yp_list, Yp_pred_list)
        metric = self._metric.forward(metric_recon, metric_pred, metric_lin)
        return loss, metric, x0, x0_pred, Xp_list, Xp_pred_list, Yp_list, Yp_pred_list

    def train_epoch(self):
        ## set training mode
        self._mode = "train"

        self._model.encoder.requires_grad_(self._training_parameters["train_encoder"])
        self._model.decoder.requires_grad_(self._training_parameters["train_decoder"])
        self._model.states_auxilary.requires_grad_(
            self._training_parameters["train_states_auxilary"]
        )
        # self._model.koopman.requires_grad_(self._training_parameters["train_states_auxilary"])
        self._model.inputs_auxilary.requires_grad_(
            self._training_parameters["train_inputs_auxilary"]
        )

        counter = 0
        running_loss = torch.zeros(1)
        running_metric = torch.zeros(3)
        if self._cuda:
            running_loss = torch.zeros(1).cuda()
            running_metric = torch.zeros(3).cuda()
        batch_num = len(self._train_dl)
        ## iterate through the training set
        for x, u, y in self._train_dl:
            ## transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                u = u.cuda()
                y = y.cuda()
            ## perform a training step
            loss_batch, metric_batch = self.train_step(x, u)
            running_loss += loss_batch
            running_metric += metric_batch
            counter += 1
        self._lr_scheduler.step()
        ## calculate the average loss for the epoch
        loss = running_loss / counter
        metric = running_metric / counter
        # print('train loss: %0.4f    |   ' % loss, end = " ")
        return loss.detach().cpu().numpy(), metric.detach().cpu().numpy()

    def val_dev(self):
        def list2tensor(input_list):
            torchtensor = torch.stack(input_list, dim=1)
            return [torchtensor[i] for i in range(torchtensor.shape[0])]

        # save the predictions and the labels for each batch
        X0_list = []  ## truth value of y_plus from advanced encoded x
        X0_pred_list = []  ## truth value of y_plus from advanced encoded x
        Yp_list = []  ## truth value of y_plus from advanced encoded x
        Yp_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
        Xp_list = []  ## truth value of x_plus from advanced x
        Xp_pred_list = (
            []
        )  ## predicted value of x_plus from decoding the iterated Koopman model
        # set eval mode
        self._mode = "val"
        # disable gradient computation
        self._model.requires_grad_(False)
        counter = 0
        running_loss = torch.zeros(1)
        running_metric = torch.zeros(3)
        if self._cuda:
            running_loss = torch.zeros(1).cuda()
            running_metric = torch.zeros(3).cuda()
        # iterate through the validation set
        for x, u, y in self._val_dev_dl:
            # transfer the batch to the gpu if given
            if self._cuda:
                x = x.cuda()
                u = u.cuda()
                y = y.cuda()
            # perform a validation step
            loss_batch, metric_batch, x0, x0p, xp, xpp, yp, ypp = self.val_dev_step(
                x, u
            )
            # stack lists of the tensors and append for realtime visualization of estimations
            X0_list = X0_list + [x0[i].view(1, x0.size(1)) for i in range(x0.size(0))]
            X0_pred_list = X0_pred_list + [
                x0p[i].view(1, x0.size(1)) for i in range(x0p.size(0))
            ]
            Xp_list = Xp_list + list2tensor(xp)
            Xp_pred_list = Xp_pred_list + list2tensor(xpp)
            Yp_list = Yp_list + list2tensor(yp)
            Yp_pred_list = Yp_pred_list + list2tensor(ypp)
            running_loss += loss_batch
            running_metric += metric_batch
            counter += 1
        # calculate the average loss and average metrics of your choice.
        loss = running_loss / counter
        metric = running_metric / counter * self._normalilzer_scale
        # print('val loss: %0.4f' % loss)
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

    def fit(self, epochs=-1, plot_learning=True):
        counter = 0
        self._epoch = 1
        train_losses = np.zeros(1)
        train_metrices = np.zeros((3, 1))
        val_losses = np.zeros(1)
        val_metrices = np.zeros((1, 3))
        val_losses[:] = np.nan
        val_metrices[:] = np.nan
        train_losses[:] = np.nan
        train_metrices[:] = np.nan

        self.num_states = self._model.encoder.get_inputsize()

        self._model.encoder.requires_grad_(self._training_parameters["train_encoder"])
        self._model.decoder.requires_grad_(self._training_parameters["train_decoder"])
        self._model.states_auxilary.requires_grad_(
            self._training_parameters["train_states_auxilary"]
        )
        # self._model.koopman.requires_grad_(self._training_parameters["train_states_auxilary"])
        self._model.inputs_auxilary.requires_grad_(
            self._training_parameters["train_inputs_auxilary"]
        )

        ## print model info
        print(f"Train dataset has {len(self._train_dl)} batches.")
        print(self._model)
        # summary(self._model, (1, 2))
        print(
            f"Model has {sum(p.numel() for p in self._model.parameters() if p.requires_grad)} parameters."
        )
        ## print an plot trainig status
        p_bar = tqdm(range(epochs))
        if plot_learning:
            self.fig, self.axs = plt.subplots(
                2, int(self.num_states / 2), figsize=(13, 8)
            )

        for i in p_bar:
            # train for an epoch
            train_loss, train_metric = self.train_epoch()
            train_losses = np.append(
                train_losses, train_loss, 0
            )  # append the losses to the respective lists
            train_metrices = np.append(
                train_metrices, np.expand_dims(train_metric, 1), 1
            )
            # calculate the loss and metrics on the validation set
            (
                val_loss,
                val_metric,
                x0,
                x0_pred,
                x_plus,
                x_plus_pred,
                y_pred,
                y_plus_pred,
            ) = self.val_dev()
            val_losses = np.append(val_losses, val_loss, 0)
            val_metrices = np.append(val_metrices, np.expand_dims(val_metric, 0), 0)
            ## update printed status
            p_bar.set_description(
                f"Train Loss: {train_loss.item():.6f}, Val Loss {val_loss.item():.6f}"
            )
            p_bar.refresh()

            if plot_learning and (self._epoch % 20 == 0):
                self.plot_training_progress(
                    counter,
                    train_losses,
                    val_losses,
                    val_metrices,
                    x_plus,
                    x0_pred,
                    x_plus_pred,
                )

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if self._epoch % 20 == 0:
                self.save_model()

            # check whether early stopping should be performed using the early stopping criterion
            if self._early_stopping_patience is not None:
                if counter > self._early_stopping_patience:
                    diffLoss = (
                        train_losses[counter - 1]
                        - train_losses[counter - 1 - self._early_stopping_patience]
                    )
                    if diffLoss >= 0:
                        print("===> Log: Early_stopping enabled")
                        break

            counter += 1
            self._epoch += 1
        return (train_losses[1:], val_losses[1:], val_metrices[1:, :])

    def plot_training_progress(
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
        xh = range(1, counter + 2)

        #### plot losses
        self.axs[0, 0].plot(
            xh,
            train_losses[1:],
            label="train",
            linewidth=1.5,
            color="b",
            linestyle="solid",
        )
        self.axs[0, 0].plot(
            xh, val_losses[1:], label="val", linewidth=1.5, color="r", linestyle="solid"
        )
        self.axs[0, 0].set_yscale("log")
        self.axs[0, 0].legend(loc="upper right")
        self.axs[0, 0].grid(visible=True)
        self.axs[0, 0].set_xlabel("epoch")
        self.axs[0, 0].set_title("Loss")

        #### plot metrics
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

        #### plot states and predicted states
        for j in range(self.axs.shape[1]):
            for i in range(x_plus.shape[0]):
                self.axs[1, j].plot(
                    x_plus[i, :, j * 2],
                    x_plus[i, :, j * 2 + 1],
                    c="b",
                    label="X",
                    linewidth=0.5,
                )
                self.axs[1, j].plot(
                    torch.cat((x0_pred, x_plus_pred), dim=1)[i, :, j * 2],
                    torch.cat((x0_pred, x_plus_pred), dim=1)[i, :, j * 2 + 1],
                    c="r",
                    label="X",
                    linewidth=0.5,
                )
            # self.axs[1, 1].axis('equal')
            self.axs[1, j].set_xlabel(f"$x_{{{j*2+1}}}$")  # Adjusted to j*2+1
            self.axs[1, j].set_ylabel(f"$x_{{{j*2+2}}}$")  # Adjusted to j*2+2
            self.axs[1, j].minorticks_on()
            self.axs[1, j].grid(
                which="both", color="lightgray", linestyle="--", linewidth=0.5
            )
            self.axs[1, j].set_title("Phase portraits - State Space")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        plt.savefig(os.path.join(self._save_dir, "Training_Loss.pdf"), format="pdf")
