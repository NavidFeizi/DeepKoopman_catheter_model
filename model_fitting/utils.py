import os, sys, ast
import numpy as np

print(os.getcwd())
sys.path.append(os.getcwd())

import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset

plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linewidth"] = 0.5
matplotlib.rc("font", family="serif", size=7)
matplotlib.rcParams["text.usetex"] = True

# for live history plot
# from livelossplot import PlotLossesKeras
from IPython.display import clear_output

# from Data.Model_Generated import SynthesizedData
# from Data.Experiment_Measured import MeasuredData

""" Callback to plot the learning curves of the model during training """


class PlotLearning:
    """
    Callback to plot the learning curves of the model during training.
    """

    def __init__(self):
        self.flag = True

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if "val" not in x]
        if self.flag:
            fig, self.axs = plt.subplots(1, len(metrics), figsize=(12, 5))
            self.flag = False
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            self.axs[i].plot(
                range(1, epoch + 2), self.metrics[metric], "k", label=metric
            )

            if logs["val_" + metric]:
                self.axs[i].plot(
                    range(1, epoch + 2),
                    self.metrics["val_" + metric],
                    "r",
                    label="val_" + metric,
                )

            self.axs[i].set_yscale("log")
            self.axs[i].legend([metric, "val_" + metric])
            self.axs[i].grid(visible=True)
            self.axs[i].set_xlabel("epoch")
            plt.pause(0.1)

        plt.tight_layout()
        plt.pause(1)
        plt.draw()
        plt.pause(2)


class MLPDataset(Dataset):
    def __init__(
        self,
        datasets_dir,
        filename,
        sample_time=None,
        seq_length=None,
        add_dim=True,
        normalizer=False,
        scaler_params=None,
    ):
        self._scaler_params = scaler_params
        self._seq_length = seq_length
        self._add_dim = add_dim

        input_size = 1
        state_size = 4

        rec_file_dir = os.path.join(datasets_dir, filename)
        df = pd.read_csv(rec_file_dir)
        num_smaples = len(df) 
        num_seq = int(num_smaples / seq_length)

        state = undo_jsonify(df["state"].to_numpy())
        state_seq = np.reshape(state, (num_seq, seq_length, state_size))

        input = undo_jsonify(df["input"].to_numpy())
        input_seq = np.reshape(input, (num_seq, seq_length, input_size))

        # state = undo_jsonify(df["state"].to_numpy())[0:-1]
        # if "input" in df:
        #     input = undo_jsonify(df["input"].to_numpy())[0:-1]
        # else:
        #     input = None
        # if "state_dot" in df:
        #     statedot = undo_jsonify(df["state_dot"].to_numpy())[0:-1]
        # else:
        #     statedot = None
        # stateplus = undo_jsonify(df["state"].to_numpy())[1:]

        # x_train = np.concatenate((state, input), axis=1)
        x_train = state_seq
        u_train = input_seq
        y_train = state_seq[:,1:,:]


        # #### create sequenced data if model requires
        # if seq_length is not None:
        #     in_train = x_train[seq_length:, -1 * input_size :]
        #     x_train = self.__create_sequence(x_train[:, 0 : -1 * input_size])
        #     x_train = np.concatenate((x_train, in_train), 1)
        #     y_train = y_train[seq_length:, :]

        # #### normalize the dataset
        if normalizer:
            if self._scaler_params is None:
                self._scaler_params = {
                    "params_x": _fit_normalizer(x_train),
                    "params_u": _fit_normalizer(u_train),
                    "params_y": _fit_normalizer(y_train),
                }

            x_train = normalize(x_train, self._scaler_params["params_x"])
            u_train = normalize(u_train, self._scaler_params["params_y"])
            y_train = normalize(y_train, self._scaler_params["params_y"])

        # ### convert to torch tensor
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.u_train = torch.tensor(u_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        
    def get_normalizer_params(self):
        return self._scaler_params

    def __create_sequence(self, dataset):
        input_size = dataset.shape[1]
        dataset_size = len(dataset) - self._seq_length
        if self._add_dim:
            X = np.empty((dataset_size, self._seq_length, input_size))
            X[:] = np.nan
            for i in range(dataset_size):
                X[i, :, :] = dataset[i : i + self._seq_length, :]
            X = np.transpose(X, (0, 2, 1))
        else:
            X = np.empty((dataset_size, self._seq_length * input_size))
            X[:] = np.nan
            for i in range(dataset_size):
                X[i, :] = dataset[i : i + self._seq_length, :].reshape(
                    (1, self._seq_length * input_size)
                )
            # X = np.expand_dims(X, axis=1)

        return X

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.u_train[idx], self.y_train[idx]


def _fit_normalizer(X):
    # Compute min/max across all dimensions except the last one
    min_values = X.min(axis=tuple(range(X.ndim - 1))).tolist()
    max_values = X.max(axis=tuple(range(X.ndim - 1))).tolist()
    scaler = {
        "min": min_values,
        "max": max_values,
    }
    return scaler


def normalize(X, scaler):
    min = np.array(scaler["min"])
    max = np.array(scaler["max"])
    X_std = (X - min) / (max - min)
    return X_std


def inverse_normalize(X_std, scaler):
    min = np.array(scaler["min"])
    max = np.array(scaler["max"])
    X_scaled = X_std * (max - min) + min
    return X_scaled


def jsonify(array):
    if isinstance(array, np.ndarray):
        return array.tolist()
    if isinstance(array, list):
        return array
    return array


def undo_jsonify(array):
    # Using a list comprehension for efficiency
    # Extracting the substring and converting to float in one go
    return np.array(
        [
            [
                float(num)
                for num in elem[elem.index("[") + 1 : elem.index("]")].split(",")
            ]
            for elem in array
        ]
    )

def plot_error(error_list, labels, ylim=None):
    # Data for plotting: separate box plots for error_x and error_y
    data_for_plotting = []
    combined_labels = []
    for i in range(len(error_list)):
        data_for_plotting.append(error_list[i].flatten())
        combined_labels.append(labels[i])

    # Create the figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.boxplot(data_for_plotting, patch_artist=True, notch="True", vert=1)    
    ax.set_xticklabels(combined_labels, rotation=0, ha="right")
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.title("Test dataset prediction errors")
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.draw()
    plt.pause(0.1)  # This line may be necessary in some environments
    return fig


# ### Plot errors boxplot
def plot_states(X, x_pred, Y, Y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    num_seq = X.shape[0]
    for i in range(num_seq):
        ax[0].plot(X[i, :, 0], X[i, :, 1], c="b", label="X", linewidth=1)
        ax[0].plot(x_pred[i, :, 0], x_pred[i, :, 1], c="r", label="X", linewidth=0.5)
        ax[1].plot(Y[i, :, 0], Y[i, :, 1], c="b", label="Y", linewidth=1)
        ax[1].plot(Y_pred[i, :, 0], Y_pred[i, :, 1], c="r", label="Y", linewidth=0.5)

    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].minorticks_on()
    # ax[0].set_aspect("equal", adjustable="box")
    ax[0].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    ax[1].set_xlabel("y1")
    ax[1].set_ylabel("y2")
    ax[1].minorticks_on()
    # ax[1].set_aspect("equal", adjustable="box")
    ax[1].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    plt.subplots_adjust(hspace=0.3)  # Set the spacing between subplots
    plt.tight_layout()
    
    plt.draw()
    plt.pause(0.1)  # This line may be necessary in some environments
    return fig


def plot_trend(truth, pred, labels, time=None):
    num_plots = truth.shape[1]
    if time is None:
        time = np.arange(truth.shape[0])
    fig, ax = plt.subplots(truth.shape[1], 1, figsize=(20, 12))
    for i in range(num_plots):
        ax[i].plot(
            time,
            truth[:, i],
            label="$truth$",
            linewidth=1.5,
            color="black",
            linestyle="solid",
        )
        ax[i].plot(
            time,
            pred[:, i],
            label="$pred$",
            linewidth=1.5,
            color="red",
            linestyle="solid",
        )
        ax[i].set_ylabel(labels[i])
        ax[i].set_xticklabels([])
        ax[i].minorticks_on()
        ax[i].grid(which="minor", color="lightgray", linestyle="--", linewidth=0.5)
        ax[i].legend()
        plt.legend(loc=1)
    if time is None:
        ax[i].set_xlabel("Sample")
    else:
        ax[i].set_xlabel("Time [s]")

    ax[i].set_xticklabels(ax[i].get_xticks())

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, hspace=0.1, wspace=0.0
    )
    plt.draw()
    plt.pause(0.1)
    return fig


def init_he_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.02)

def init_dl_weights(m):
    if isinstance(m, torch.nn.Linear):
        a = m.in_features  # 'a' is the dimension of the input to the layer
        s = 1.0 / (a ** 0.5)
        torch.nn.init.uniform_(m.weight, -s, s)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def init_tn_weights(m):
    scale = 0.1
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=scale)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


class Weighted_MSE_Loss(_WeightedLoss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        weight: Tensor = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sum(self.weight * (input - target) ** 2)

class DeepKoopmanExplicitLoss(nn.Module):
    # this loss function is explained in Bethany Lusch's Natura Communication paper
    def __init__(self, alpha1, alpha2, alpha_reg):
        super(DeepKoopmanExplicitLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha_reg = alpha_reg  # Regularization strength

    def recon(self, x0, x0_pred):
        loss_recon = F.mse_loss(x0, x0_pred)
        return loss_recon
    
    def pred(self, x_plus, x_plus_pred):
        loss = 0
        for i in range(len(x_plus)):
            # loss = loss + F.mse_loss(x_plus[i], x_plus_pred[i])
            loss = loss + torch.mean(torch.mean(torch.square(x_plus[i] - x_plus_pred[i]), dim=1))
        loss_pred = loss / len(x_plus)
        return loss_pred
        
    def lin(self, y_plus, y_plus_pred):
        loss = 0
        for i in range(len(y_plus)-1):
            # loss = loss + F.mse_loss(y_plus[i], y_plus_pred[i])
            loss = loss + torch.mean(torch.mean(torch.square(y_plus[i] - y_plus_pred[i]), dim=1))
        loss_lin = loss / len(y_plus)
        return loss_lin

    def infnorm(self, x0, x0_pred, x_plus, x_plus_pred):
        Linf1_penalty = torch.norm(torch.norm(x0 - x0_pred, dim=1, p=float('inf')), p=float('inf'))
        Linf2_penalty = torch.norm(torch.norm(x_plus - x_plus_pred, dim=1, p=float('inf')), p=float('inf'))
        return Linf1_penalty + Linf2_penalty
    
    def forward_recReg(self, loss_recon, model_parameters):
        # Regularization term (L2 norm of the parameters)
        l2_reg = torch.tensor(0.).to(loss_recon.device)
        for param in model_parameters:
            l2_reg += torch.norm(param)

        total_loss = self.alpha1 * loss_recon + self.alpha_reg * l2_reg
        return total_loss

    def forward(self, loss_recon, loss_pred, loss_lin, loss_inf, model_parameters):
        # Regularization term (L2 norm of the parameters)
        l2_reg = torch.tensor(0.).to(loss_recon.device)
        for param in model_parameters:
            l2_reg += torch.norm(param)

        total_loss = self.alpha1 * (loss_recon + loss_pred) + loss_lin + self.alpha2 * loss_inf + self.alpha_reg * l2_reg
        # total_loss = self.alpha1 * (loss_recon) + loss_lin + self.alpha_reg * l2_reg
        return total_loss

class DeepKoopmanExplicitMetric(nn.Module):
    # this loss function is explained in Bethany Lusch's Natura Communication paper
    def __init__(self):
        super(DeepKoopmanExplicitMetric, self).__init__()

    def recon(self, x0, x0_pred):
        metrics_recon = torch.mean(torch.abs(x0 - x0_pred))
        return metrics_recon
    
    def pred(self, x_plus, x_plus_pred):
        metrics = 0
        for i in range(len(x_plus)):
            # loss = loss + F.mse_loss(x_plus[i], x_plus_pred[i])
            metrics = metrics + torch.mean(torch.mean(torch.abs(x_plus[i] - x_plus_pred[i]), dim=1))
        metrics_pred = metrics / len(x_plus)
        return metrics_pred
        
    def lin(self, y_plus, y_plus_pred):
        metrics = 0
        for i in range(len(y_plus)-1):
            # loss = loss + F.mse_loss(y_plus[i], y_plus_pred[i])
            metrics = metrics + torch.mean(torch.mean(torch.abs(y_plus[i] - y_plus_pred[i]), dim=1))
        metrics_lin = metrics / len(y_plus)
        return metrics_lin  

    def forward(self, metric_recon, metric_pred, metric_lin):
        return torch.tensor([metric_recon, metric_pred, metric_lin]).to(metric_recon.device)

class MAE_Metrix(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "none") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.abs(input - target), dim=0)


# ##### *****************************************************************************************************************************
# print("Hi!")
# print(get_available_devices())

# """ Load synthesized datasets """
# train_dataset, valid_dataset, test_dataset = SynthesizedData.get_datasets(
#     FolderList=("1", "2", "3", "4", "5", "6")
# )

# """ Normalize train/valid and test datasets """
# scaler = MinMaxScaler(feature_range=(0, 1))
# train_dataset_norm = scaler.fit_transform(train_dataset)
# valid_dataset_norm = scaler.transform(valid_dataset)
# test_dataset_norm = scaler.transform(test_dataset)
# dump(
#     scaler, "SavedResults/scaler_preTrain.joblib"
# )  # save normalizer object for future predictions

# """ Split datasets (for MLP)"""
# train_x_norm, train_y_norm = train_dataset_norm[:, 0:6], train_dataset_norm[:, 6:9]
# valid_x_norm, valid_y_norm = valid_dataset_norm[:, 0:6], valid_dataset_norm[:, 6:9]
# test_x_norm, test_y_norm = test_dataset_norm[:, 0:6], test_dataset_norm[:, 6:9]


# """ Load measured datasets """
# (
#     train_dataset_expt,
#     valid_dataset_expt,
#     test_dataset_expt,
# ) = MeasuredData.get_datasets_expt(FolderList=("3", "4", "5", "6", "1", "2"))
# dummy, dummy, test_dataset_expt_model = MeasuredData.get_datasets_model(
#     FolderList=("3", "4", "5", "6", "1", "2")
# )

# """ Normalize train/valid and test datasets """
# scaler_2 = MinMaxScaler(feature_range=(0, 1))
# train_dataset_norm_expt = scaler_2.fit_transform(train_dataset_expt)
# valid_dataset_norm_expt = scaler_2.transform(valid_dataset_expt)
# test_dataset_norm_expt = scaler_2.transform(test_dataset_expt)
# dump(
#     scaler_2, "SavedResults/scaler_fineTune.joblib"
# )  # save normalizer object for future predictions

# """ Split datasets (for MLP)"""
# train_x_norm_expt, train_y_norm_expt = (
#     train_dataset_norm_expt[:, 0:6],
#     train_dataset_norm_expt[:, 6:9],
# )
# valid_x_norm_expt, valid_y_norm_expt = (
#     valid_dataset_norm_expt[:, 0:6],
#     valid_dataset_norm_expt[:, 6:9],
# )
# test_x_norm_expt, test_y_norm_expt = (
#     test_dataset_norm_expt[:, 0:6],
#     test_dataset_norm_expt[:, 6:9],
# )


# """ Building the MLP model """
# inputs = tf.keras.layers.Input(shape=(train_x_norm.shape[1],))
# dense1 = tf.keras.layers.Dense(units=512, activation="relu")(inputs)
# dense2 = tf.keras.layers.Dense(units=256, activation="relu")(dense1)
# dense3 = tf.keras.layers.Dense(units=128, activation="relu")(dense2)
# dense4 = tf.keras.layers.Dense(units=128, activation="relu")(dense3)
# dense5 = tf.keras.layers.Dense(units=64, activation="relu")(dense4)
# dense6 = tf.keras.layers.Dense(units=64, activation="relu")(dense5)
# dense7 = tf.keras.layers.Dense(units=32, activation="relu")(dense6)
# denseLast = tf.keras.layers.Dense(units=3)(dense7)
# # model = tf.keras.models.Model(inputs = (inputs_1, inputs_2), outputs = dense5)
# model = tf.keras.models.Model(inputs=inputs, outputs=denseLast)


# #### Plot
# # fig, ax1 = plt.subplots()
# # ax2 = ax1.twinx()
# # sampleNum = 1060
# # ax1.plot(np.arange(0,look_back,1), train_x[sampleNum, :, 0], 'g-')
# # # ax1.plot(np.arange(0,look_back,1), train_x[sampleNum, :, 1], 'b-')
# # ax2.plot(look_back+1, train_y[sampleNum], 'r-', marker='*')
# # ax1.set_ylim([0, 1])
# # ax2.set_ylim([0, 1])
# # ax1.set_xlabel('sample')
# # ax1.set_ylabel('X data', color='g')
# # ax2.set_ylabel('Y data', color='b')
# # plt.show()


# # def f1_score(y_true, y_pred):
# #     precision = precision_m(y_true, y_pred)
# #     recall = recall_m(y_true, y_pred)
# #     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# """ Compile and train the model """
# callbacks_list = [PlotLearning()]

# opt = Adam(
#     learning_rate=CustomSchedule(initial_learning_rate=0.01, decay=0.5),
#     beta_1=0.9,
#     beta_2=0.999,
# )
# model.compile(optimizer=opt, loss="mse", metrics=["mae"])
# trainHistory = model.fit(
#     x=train_x_norm,
#     y=train_y_norm,
#     epochs=50,
#     batch_size=64,
#     shuffle=True,
#     validation_data=(valid_x_norm, valid_y_norm),
#     validation_batch_size=16,
#     callbacks=callbacks_list,
# )
# # callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 50)
# # trainHistory = model.fit(x = train_input1, y = train_y_pad[::,0,0], epochs = 400, batch_size = 512, shuffle = False, validation_data=(val_x, val_y), validation_batch_size=1, callbacks=[callback])
# # trainHistory = model.fit(x = (train_input1, train_input2), y = train_y_pad, epochs = 100, batch_size = 4, shuffle = False)

# plt.savefig("SavedResults/TrainHistory_PreTrain.pdf", format="pdf")
# plt.close()


# ### summarize history for loss
# # fig, ax1 = plt.subplots()
# # ax1.plot(trainHistory.history['loss'])
# # plt.plot(trainHistory.history['val_loss'])
# # ax1.set_title('model loss')
# # ax1.set_ylabel('loss')
# # ax1.set_xlabel('epoch')
# # ax1.legend(['train', 'validation'], loc = 'upper left')
# # plt.show()
# # plt.pause(1)

# """ save the model for future preditions """
# model.save("SavedResults/saved_models/pre_trained")


# """ Predict using test dataset """
# pred_temp_norm = model.predict(x=(test_x_norm))
# pred_temp = scaler.inverse_transform(
#     np.concatenate((test_dataset_norm[::, 0:6], pred_temp_norm), axis=1)
# )
# y_NN = pred_temp[::, 6:9]

# y_truth = scaler.inverse_transform(test_dataset_norm[::, ::])
# y_model = y_truth[::, 6:9]

# # y_model = y_truth[::, 6:9]

# error = y_model - y_NN
# error_x = np.abs(error[:, 0])
# error_y = np.abs(error[:, 1])
# error_z = np.abs(error[:, 2])
# error_model = np.sqrt(np.sum((error) ** 2, axis=1))
# # error_NN = np.sqrt(np.mean((y_expt - y_NN)**2))

# data = [error_x * 1000, error_y * 1000, error_z * 1000, error_model * 1000]
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# bp = ax.boxplot(data, patch_artist=True, notch="True", vert=1)
# ax.set_xticklabels(["X", "Y", "X", "distance"])
# ax.set_ylabel("error [mm]")
# plt.title("Errors box plot")
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()

# plt.draw()
# plt.pause(2)
# plt.savefig("SavedResults/PredictionError_PreTrain.pdf", format="pdf")
# plt.close()

# ### plot
# fig = plt.figure(figsize=(18, 6))
# ax1 = fig.add_subplot(111)
# # ax2 = ax1.twinx()
# # line1 = ax1.plot(y_expt*1000, 'b-')
# line2 = ax1.plot(y_NN[:3000] * 1000)
# line3 = ax1.plot(y_model[:3000] * 1000)
# # ax1.set_ylim([0, 6])
# # ax2.set_ylim([0, 6])
# ax1.set_xlabel("sample")
# ax1.set_ylabel("position (mm)", color="k")
# ax1.legend(["model_x", "model_y", "model_z", "NN_x", "NN_y", "NN_z"])
# # ax2.set_ylabel('NDD', color='r')
# # ax3.set_ylabel('Model', color='g')
# plt.draw()
# plt.pause(2)
# plt.savefig("SavedResults/Prediction_PreTrain.pdf", format="pdf")
# plt.close()


# # Fine-tune from this layer onwards
# fine_tune_at = 4

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in model.layers[:fine_tune_at]:
#     layer.trainable = False

# # To get learning rate
# # print(K.get_value(model.optimizer.lr))

# callbacks_list_2 = [PlotLearning()]
# trainHistory = model.fit(
#     x=train_x_norm_expt,
#     y=train_y_norm_expt,
#     epochs=200,
#     batch_size=64,
#     shuffle=True,
#     validation_data=(valid_x_norm_expt, valid_y_norm_expt),
#     validation_batch_size=16,
#     callbacks=callbacks_list_2,
# )
# # callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 50)
# # trainHistory = model.fit(x = train_input1, y = train_y_pad[::,0,0], epochs = 400, batch_size = 512, shuffle = False, validation_data=(val_x, val_y), validation_batch_size=1, callbacks=[callback])
# # trainHistory = model.fit(x = (train_input1, train_input2), y = train_y_pad, epochs = 100, batch_size = 4, shuffle = False)

# plt.savefig("SavedResults/TrainHistory_FineTuned.pdf", format="pdf")
# plt.close()


# """ save the model for future preditions """
# model.save("SavedResults/saved_models/fine_tuned")


# """ Predict using test dataset """
# pred_temp_norm_expt = model.predict(x=(test_x_norm_expt))
# pred_temp = scaler_2.inverse_transform(
#     np.concatenate((test_dataset_norm_expt[::, 0:6], pred_temp_norm_expt), axis=1)
# )
# y_NN = pred_temp[::, 6:9]

# y_truth = scaler_2.inverse_transform(test_dataset_norm_expt[::, ::])
# y_truth = y_truth[::, 6:9]

# y_model = test_dataset_expt_model[::, 6:9]

# # y_model = y_truth[::, 6:9]

# error = y_truth - y_NN
# error_NN_x = np.abs(error[:, 0])
# error_NN_y = np.abs(error[:, 1])
# error_NN_z = np.abs(error[:, 2])
# error_NN = np.sqrt(np.sum((error) ** 2, axis=1))

# error = y_truth - y_model
# error_model_x = np.abs(error[:, 0])
# error_model_y = np.abs(error[:, 1])
# error_model_z = np.abs(error[:, 2])
# error_model = np.sqrt(np.sum((error) ** 2, axis=1))

# data = [
#     error_NN_x * 1000,
#     error_model_x * 1000,
#     error_NN_y * 1000,
#     error_model_y * 1000,
#     error_NN_z * 1000,
#     error_model_z * 1000,
#     error_NN * 1000,
#     error_model * 1000,
# ]
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# bp = ax.boxplot(data, patch_artist=True, notch="True", vert=1)
# ax.set_xticklabels(
#     ["NN_X", "model_X", "NN_Y", "model_Y", "NN_Z", "model_Z", "NN", "model"]
# )
# ax.set_ylabel("error [mm]")
# ax.set_ylim([0, 16])
# plt.title("Errors box plot")
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()

# plt.draw()
# plt.pause(2)
# plt.savefig("SavedResults/PredictionError_FineTuned.pdf", format="pdf")
# plt.close()

# ### plot
# fig = plt.figure(figsize=(18, 6))
# ax1 = fig.add_subplot(111)
# # ax2 = ax1.twinx()
# # line1 = ax1.plot(y_expt*1000, 'b-')
# line2 = ax1.plot(y_truth[:3000] * 1000)
# line3 = ax1.plot(y_NN[:3000] * 1000)
# line4 = ax1.plot(y_model[:3000] * 1000)
# # ax2.set_ylim([0, 6])
# ax1.set_xlabel("sample")
# ax1.set_ylabel("position (mm)", color="k")
# ax1.legend(
#     [
#         "expt_x",
#         "expt_y",
#         "expt_z",
#         "NN_x",
#         "NN_y",
#         "NN_z",
#         "model_x",
#         "model_y",
#         "model_z",
#     ]
# )
# # ax2.set_ylabel('NDD', color='r')
# # ax3.set_ylabel('Model', color='g')
# plt.draw()
# plt.pause(2)
# plt.savefig("SavedResults/Prediction_FineTuned.pdf", format="pdf")
# plt.close()


# # """ Save results """
# # time = np.arange(0, len(y_truth)*sampleTime, sampleTime)
# # ResultsTest = pd.DataFrame({'time': time, 'voltage': TestDataSet[:, 0], 'velocity': TestDataSet[:, 1], 'model': TestDataSet[:, 2], 'y_truth': y_truth, 'y_pred': y_pred})
# # ResultsTest.to_csv('SavedResults/DataforLaTeX/TestResults.csv', sep = ',')

# # ResultsTrain = pd.DataFrame({'epoch': trainHistory.epoch, 'loss': np.array(trainHistory.history['loss']), 'mae': np.array(trainHistory.history['mae']), 'test_loss': TestHistory[0], 'test_mae': TestHistory[1]})
# # ResultsTrain.to_csv('SavedResults/DataforLaTeX/TrainResults.csv', sep = ',')

# # print("Done!")
