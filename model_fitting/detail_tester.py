import os, sys, json, torch

sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Dataset
from model_fitting.utils import MLPDataset
from model_fitting.predictor import Predictor
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

logger = logging.getLogger(__name__)
logger.info(" Running on %s", device)

""" Adjust Parameters """
##################################################################################
model_name = "catheter_1T_unforced_4.3.1"
dataset_override = None #"catheter_1T_unforced_4.3.1" #"catheter_1T_forced_5.2.test"
simulation_time = 1.0
sample_time_override = 2e-3
# plot_index_list = [2, 21, 51, 61]
plot_index_list = [2, 5, 7]
# plot_index_list = [5, 10, 17]
# plot_index_list = [5, 24, 54, 64, 74]

states_name = ["$x$", "$\dot{x}$", "$z$", "$\dot{z}$"]  # Example state names
inputs_name = ["$q_0$"]  # Example state names
lifted_states_name = ["$y_1$", "$y_2$"]  # Example state names


def main():
    """Load Info Files"""
    ####################################################################################
    model_dir = os.path.join(os.getcwd(), "trained_models", model_name)

    #### create the model saving folder if not exists
    PNG_dir = os.path.join(os.getcwd(), "trained_models", model_name, "PNG")
    if not os.path.exists(PNG_dir):
        os.makedirs(PNG_dir)

    #### load model_info json file
    model_info_path = os.path.join(model_dir, "model_info.json")
    if os.path.isfile(model_info_path):
        with open(model_info_path, "r") as fp:
            model_info = json.load(fp)
    else:
        error_message = f"[{model_info_path}] does not exist!"
        raise FileNotFoundError(error_message)

    #### load dataset_info json file
    if dataset_override is not None:
        datasets_dir = os.path.join(
            os.getcwd(), "datasets", dataset_override
        )
    else:
        datasets_dir = os.path.join(
            os.getcwd(), "datasets", model_info["dataset_options"]["dataset_name"]
        )
    dataset_info_path = os.path.join(datasets_dir, "dataset_info.json")
    if os.path.isfile(dataset_info_path):
        with open(dataset_info_path, "r") as fp:
            dataset_info = json.load(fp)
    else:
        error_message = f"[{dataset_info_path}] does not exist!"
        raise FileNotFoundError(error_message)

    #### some preparations
    if (sample_time_override % dataset_info["sample_time"]) == 0:
        STF = int(
            sample_time_override / dataset_info["sample_time"]
        )  # sample_time_factor
        num_pred_steps = int(simulation_time / sample_time_override) - 1
        # num_dataset_steps = int(simulation_time / dataset_info["sample_time"]) - 1
    else:
        raise Exception(
            "sample_time_override ("
            + sample_time_override
            + ") must be a multiplied factor of dataset sample time ("
            + dataset_info["sample_time"]
            + ")"
        )

    """ Load Test Dataset """
    ####################################################################################
    dataset_test = MLPDataset(
        datasets_dir=datasets_dir,
        filename="test.txt",
        sample_time=dataset_info["sample_time"],
        seq_length=dataset_info["test_sequence_length"],
        add_dim=model_info["dataset_options"]["add_sequence_dimension"],
        fit_normalize=False,
        prenomalize=False,
        scaler_params=None,
    )
    dataloader_test = DataLoader(dataset=dataset_test, shuffle=False)
    num_traj = 1
    num_pred_steps = 10000
    # num_traj = 20
    logger.info(" Loading test datasets finished")

    """ Initialize simulation variables """
    ####################################################################################
    Yp_list = []  ## truth value of y_plus from advanced encoded x
    Xrecon_list = []  ## truth value of y_plus from advanced encoded x
    Yp_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
    Xp_pred_list = []  ## predicted value of x_plus from decoding the iterated Koopman
    Yobs_list = []  ## predicted value of y_plus from iterating the Koopman model
    Xobs_list = []  ## predicted value of y_plus from iterating the Koopman model
    Xp = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["state_size"],
        )
    )
    Xrecon = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["state_size"],
        )
    )
    Xp_pred = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["state_size"],
        )
    )
    Yp = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["lifted_state_size"],
        )
    )
    Yp_pred = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["lifted_state_size"],
        )
    )
    Lambdas = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["lifted_state_size"],
        )
    )
    U = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["inputs_auxiliary_params"]["system_input_size"],
        )
    )
    A = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["lifted_state_size"],
            model_info["encoder_params"]["lifted_state_size"],
        )
    )
    B = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["lifted_state_size"],
            model_info["inputs_auxiliary_params"]["system_input_size"],
        )
    )
    Yobs = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["lifted_state_size"],
        )
    )
    Xobs = np.empty(
        (
            num_traj,
            num_pred_steps,
            model_info["encoder_params"]["state_size"],
        )
    )

    """ Initiate Predictor and Observer Objects """
    ####################################################################################
    pred = Predictor(
        model_dir,
        num_pred_steps=num_pred_steps,
        normalizer_params=model_info["normalizer_params"],
        cuda=False,
        sample_time_override=sample_time_override,
        use_encoder_jacobian=False,
    )

    pred.save_scripted_model()

    """ Predict """
    ####################################################################################
    print("Predicting...")
    
    
    # if i >= 20:
    #     break
    Xp_list = []  ## truth value of x_plus from advanced x
    Yp_list = []
    Yp_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
    ## predicted value of x_plus from decoding the iterated Koopman model
    Xp_pred_list = []

    Lambdas_list = []
    A_list = []
    B_list = []
    U_list = []
    Yobs_list = []
    Xobs_list = []

    # y = torch.tensor([[x[:, 0, 0], x[:, 0, 2]]])
    # y = x[:, 0, :]
    # u = u[:, 0::STF, :]  ## downsample u

    # ## predict trajectory at once
    # Xp_pred_list, Yp_pred_list, Lambdas_list, A_list, B_list, U_list = (
    #     pred.predict_trajectory(x0=x[:, 0, :], u=u, requires_grad=True)
    # )
    for i in range(num_pred_steps):
        print("data number: ", i)
        ### predict trajectory in a loop
        # X0 = x[:, 0, :]
        # y, encoder_jacobian = pred.encode(x=X0, requires_grad=True)
        y =  torch.zeros(1,2)
        y[0,0], y[0,1] = random_point_in_circle(0.12)
        # pred.load_script_model(X0)
        G, H, A_, B_, Lambdas_ = pred.compute_model_matrices(y, None)
        # y = pred.compute_next_lifted_state(G, H, y, u[:, j, :])
        Yp_list.append(y)
        Xp_list.append(pred.decode(y)[0])
        Lambdas_list.append(Lambdas_)
        B_list.append(B_)
        A_list.append(A_)
        # U_list.append(u[:, j, :])

        # Yp_list = pred.lift_a_trajectory(x=x)
        # Xrecon_list = pred.reconstruct_trajectory(x=x)
        # for j in range(num_pred_steps):
        #     Xp_list.append(x[:, ((j + 1) * STF), :])
        #     Yp_list.append(pred._model.encoder(x[:, ((j + 1) * STF), :]))

    Xp[0, :, :] = torch.stack(Xp_list, dim=1).detach().numpy()[0, :, :]
    # Xp_pred[i, :, :] = torch.stack(Xp_pred_list, dim=1).detach().numpy()[0, :, :]
    Yp[0, :, :] = torch.stack(Yp_list, dim=1).detach().numpy()[0, :, :]
    # Xrecon[i, :, :] = torch.stack(Xrecon_list, dim=1).detach().numpy()[0, :, :]
    # Yp_pred[i, :, :] = torch.stack(Yp_pred_list, dim=1).detach().numpy()[0, :, :]
    Lambdas[0, :, :] = torch.stack(Lambdas_list, dim=1).detach().numpy()[0, :, :]
    # U[i, :, :] = torch.stack(U_list, dim=1).detach().numpy()[0, :, :]
    A[0, :, :] = torch.stack(A_list, dim=1).detach().numpy()[0, :, :, :]
    B[0, :, :] = torch.stack(B_list, dim=1).detach().numpy()[0, :, :, :]

    # Xp = inverse_normalize(Xp, model_info["normalizer_params"]["params_x"])
    # Xp_pred = inverse_normalize(Xp_pred, model_info["normalizer_params"]["params_x"])
    print("Prediction done!")

    """ Plot Results """
    ####################################################################################
    time_pred = np.linspace(0, sample_time_override * num_pred_steps, num_pred_steps)
    # errors_Xrecon = np.abs(Xrecon - Xp)
    # errors_Xp_pred = np.abs(Xp_pred - Xp)
    # errors_Yp_pred = np.abs(Yp_pred - Yp)

    save_to_csv(Xp, Yp, Lambdas, model_dir)

    #### Plot - Errors """
    # plot_errors(Xp, errors_Xp_pred, errors_Xrecon, states_name, model_dir, save=True)
    #### Plot - Eigen functions """
    plot_eigen_functions(Yp, Lambdas, model_dir, save=True)
    #### Plot - B Matrix """
    # plot_A_Matrix(Yp, A, model_dir, save=True)
    #### Plot - B Matrix """
    # plot_B_Matrix(Yp, B, model_dir, save=True)

    # plot_encoder_map(Yp, Xp, model_dir, save=True)
    #### Plot - states phase and time domain """
    # plot_states(
    #     time_pred,
    #     Xp,
    #     Xp_pred,
    #     U,
    #     states_name,
    #     inputs_name,
    #     model_dir,
    #     save=True,
    #     Xobs=None,
    #     # Xrecon=Xrecon,
    # )
    #### Plot - lifted states phase and time domain """
    # plot_lifted_states(
    #     time_pred,
    #     Yp,
    #     Yp_pred,
    #     U,
    #     inputs_name,
    #     model_dir,
    #     save=True,
    #     Yobs=None,
    # )


def plot_states(
    time,
    Xp,
    Xp_pred,
    U,
    states_name,
    inputs_name,
    save_dir,
    save=False,
    Xobs=None,
    Xrecon=None,
):
    print("Plotting states...")
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(
        6,
        2,
        figure=fig,
        height_ratios=[1, 1, 1, 1, 1, 1],
        width_ratios=[2.5, 1],
    )
    axs = []
    axs.append(fig.add_subplot(gs[0:3, 1]))  # ax[0] - small plot at the top right
    axs.append(fig.add_subplot(gs[3:6, 1]))  # ax[1] - small plot middle right
    # axs.append(fig.add_subplot(gs[8:12, 1]))  # ax[2] - small plot buttom right
    axs.append(fig.add_subplot(gs[0:2, 0]))  # ax[3] - wide plot
    axs.append(axs[2].twinx())  # ax[4] - Twin axis for ax[3],
    axs.append(fig.add_subplot(gs[2:4, 0]))  # ax[5] - wide plot
    axs.append(axs[4].twinx())  # ax[6] - Twin axis for ax[5],
    # axs.append(fig.add_subplot(gs[6:9, 0]))  # ax[7] - wide plot
    # axs.append(axs[7].twinx())  # ax[8] - Twin axis for ax[7],
    axs.append(fig.add_subplot(gs[4:6, 0]))  # ax[9] - wide plot

    custom_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    linestyles = ["-", "--", "-", "--", "-", "--"]

    for index, i in enumerate(plot_index_list):

        color = custom_colors[index % len(custom_colors)]

        counter = 0
        for j in range(0, len(states_name), 2):
            axs[counter].plot(
                Xp[i, :, j],
                Xp[i, :, j + 1],
                c=color,
                label="truth" if index == 0 else "",
                linewidth=1,
            )
            axs[counter].plot(
                Xp_pred[i, :, j],
                Xp_pred[i, :, j + 1],
                c=color,
                label="pred" if index == 0 else "",
                linewidth=0.5,
            )
            axs[counter].plot(
                Xp[i, 0, j], Xp[i, 0, j + 1], c=color, marker="o", markersize=2
            )
            axs[counter].plot(
                Xp_pred[i, 0, j],
                Xp_pred[i, 0, j + 1],
                c=color,
                marker="o",
                markersize=2,
            )
            if Xobs is not None:
                axs[counter].plot(
                    Xobs[i, :, j],
                    Xobs[i, :, j + 1],
                    c="red",
                    label="obs" if index == 0 else "",
                    linewidth=1,
                    linestyle=linestyles[1],
                )
                axs[counter].plot(
                    Xobs[i, 0, j], Xobs[i, 0, j + 1], c="red", marker="o", markersize=2
                )
            if Xrecon is not None:
                axs[counter].plot(
                    Xrecon[i, :, j],
                    Xrecon[i, :, j + 1],
                    c="black",
                    label="recon" if index == 0 else "",
                    linewidth=1,
                    linestyle=linestyles[1],
                )
                axs[counter].plot(
                    Xrecon[i, 0, j],
                    Xrecon[i, 0, j + 1],
                    c="black",
                    marker="o",
                    markersize=2,
                )
            counter = counter + 1

        counter = 2
        for j in range(0, len(states_name), 1):
            axs[counter].plot(
                time,
                Xp[i, :, j],
                c=color,
                label=states_name[j] + "$_{truth}$" if index == 0 else "",
                linewidth=1,
                linestyle=linestyles[counter - 3],
            )
            axs[counter].plot(time[0], Xp[i, 0, j], c=color, marker="o", markersize=2)
            axs[counter].plot(
                time,
                Xp_pred[i, :, j],
                c=color,
                label=states_name[j] + "$_{pred}$" if index == 0 else "",
                linewidth=0.5,
                linestyle=linestyles[counter - 3],
            )
            axs[counter].plot(
                time[0], Xp_pred[i, 0, j], c=color, marker="o", markersize=2
            )
            if Xobs is not None:
                axs[counter].plot(
                    time,
                    Xobs[i, :, j],
                    c="red",
                    label=states_name[j] + "$_{obs}$" if index == 0 else "",
                    linewidth=1,
                    linestyle=linestyles[counter - 3],
                )
                axs[counter].plot(
                    time[0], Xobs[i, 0, j], c="red", marker="o", markersize=2
                )
            if Xrecon is not None:
                axs[counter].plot(
                    time,
                    Xrecon[i, :, j],
                    c="black",
                    label=states_name[j] + "$_{obs}$" if index == 0 else "",
                    linewidth=1,
                    linestyle=linestyles[counter - 3],
                )
                axs[counter].plot(
                    time[0], Xrecon[i, 0, j], c="black", marker="o", markersize=2
                )
            counter = counter + 1

        for j in range(0, len(inputs_name), 1):
            axs[6].plot(
                time,
                U[i, :, j],
                c=color,
                label=inputs_name[j] if index == 0 else "",
                linewidth=1,
                linestyle="-",
            )
            axs[6].plot(time[0], U[i, 0, j], c=color, marker="o", markersize=2)

    # Set titles, labels, and other properties for each subplot
    counter = 0
    for j in range(0, len(states_name), 2):
        axs[counter].set_title("State phase portrait")
        axs[counter].set_xlabel(states_name[j] + " $[mm]$")
        axs[counter].set_ylabel(states_name[j + 1] + " $[mm/s]$")
        axs[counter].legend(loc="best")
        axs[counter].minorticks_on()
        axs[counter].grid(
            which="both", color="lightgray", linestyle="--", linewidth=0.5
        )
        # axs[counter].set_aspect("equal", adjustable="box")
        counter = counter + 1

    counter = 2
    for j in range(0, len(states_name), 2):
        axs[counter].set_title("States vs time")
        axs[counter].set_ylabel(states_name[j] + " $[mm]$")
        axs[counter + 1].set_ylabel(states_name[j + 1] + " $[mm/s]$")
        axs[counter].minorticks_on()
        axs[counter].grid(
            which="both", color="lightgray", linestyle="--", linewidth=0.5
        )
        handles1, labels1 = axs[counter].get_legend_handles_labels()
        handles2, labels2 = axs[counter + 1].get_legend_handles_labels()
        axs[counter].legend(handles1 + handles2, labels1 + labels2, loc="best", ncol=2)
        counter = counter + 2
    axs[counter - 2].set_xlabel("Time [s]")

    axs[6].set_title("Inputs vs time")
    axs[6].set_xlabel("Time [s]")
    axs[6].set_ylabel("inputs [mm]")
    axs[6].legend(loc="best")
    axs[6].minorticks_on()
    axs[6].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    plt.tight_layout()  # Adjust layout to make room for the colorbar
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.draw()
    plt.pause(0.1)
    if save:
        plt.savefig(os.path.join(save_dir, "Prediction_States.pdf"), format="pdf")
        plt.savefig(
            os.path.join(save_dir, "PNG/Prediction_States.png"), format="png", dpi=300
        )

    ## 3D figure
    # fig = plt.figure(figsize=(12, 7.5))
    # gs = GridSpec(1, 1, figure=fig, height_ratios=[1], width_ratios=[1])
    # ax = fig.add_subplot(
    #     gs[0, 0], projection="3d"
    # )  # ax - single 3D plot, ensure to add 3D projection
    # axs = []
    # axs.append(ax)
    # # for index, i in enumerate(plot_index_list):
    # #     color = custom_colors[index % len(custom_colors)]
    # #     axs[0].plot(Xp[i, :, 4], Xp[i, :, 2], Xp[i, :, 0], c=color, label="Truth " + str(i), linewidth=1)
    # #     axs[0].plot(Xp_pred[i, :, 4], Xp_pred[i, :, 2], Xp_pred[i, :, 0], c=color, linestyle='--', label="Prediction " + str(i), linewidth=1)
    # # axs[0].plot(0.0, 0.0, 0.0, c='black', label="Base",  marker='o', markersize=2)

    # # Set titles, labels, and other properties for each subplot
    # axs[0].set_title("3D Tip Position")
    # # axs[0].set_xlabel(states_name[4] + " $[mm]$")
    # # axs[0].set_ylabel(states_name[2] + " $[mm]$")
    # # axs[0].set_zlabel(states_name[0] + " $[mm]$")
    # axs[0].minorticks_on()
    # # axs[0].set_aspect("equal", adjustable="box")
    # axs[0].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)
    # if save:
    #     plt.savefig(os.path.join(save_dir, "3D_Tip_Posision.pdf"), format="pdf")


def plot_lifted_states(
    time, Yp, Yp_pred, U, inputs_name, save_dir, save=False, Yobs=None
):
    print("Plotting lifted states...")
    fig = plt.figure(figsize=(12, 7.5))
    gs = GridSpec(
        6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[2.5, 1]
    )
    axs = []
    axs.append(fig.add_subplot(gs[0:3, 1]))  # ax[0] - small plot at the top right
    axs.append(fig.add_subplot(gs[3:6, 1]))  # ax[1] - small plot middle right
    axs.append(fig.add_subplot(gs[0:2, 0]))  # ax[3] - wide plot
    axs.append(axs[2].twinx())  # ax[4] - Twin axis for ax[3],
    axs.append(fig.add_subplot(gs[2:4, 0]))  # ax[5] - wide plot
    axs.append(axs[4].twinx())  # ax[6] - Twin axis for ax[5],
    axs.append(fig.add_subplot(gs[4:6, 0]))  # ax[7] - wide plot

    custom_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    linestyles = ["-", "--", "-", "--", "-", "--"]

    # color = colors[3]
    for index, i in enumerate(plot_index_list):
        color = custom_colors[index % len(custom_colors)]

        counter = 0
        for j in range(0, len(lifted_states_name), 2):
            axs[counter].plot(
                Yp[i, :, j],
                Yp[i, :, j + 1],
                c=color,
                label="truth" if index == 0 else "",
                linewidth=1,
            )
            axs[counter].plot(
                Yp_pred[i, :, j],
                Yp_pred[i, :, j + 1],
                c=color,
                label="pred" if index == 0 else "",
                linewidth=0.5,
            )
            axs[counter].plot(
                Yp[i, 0, j], Yp[i, 0, j + 1], c=color, marker="o", markersize=2
            )
            axs[counter].plot(
                Yp_pred[i, 0, j],
                Yp_pred[i, 0, j + 1],
                c=color,
                marker="o",
                markersize=2,
            )
            if Yobs is not None:
                axs[counter].plot(
                    Yobs[i, :, j],
                    Yobs[i, :, j + 1],
                    c="red",
                    label="obs" if index == 0 else "",
                    linewidth=1,
                    linestyle=linestyles[1],
                )
                axs[counter].plot(
                    Yobs[i, 0, j], Yobs[i, 0, j + 1], c="red", marker="o", markersize=2
                )
            counter = counter + 1

        counter = 2
        for j in range(0, len(lifted_states_name), 1):
            axs[counter].plot(
                time,
                Yp[i, :, j],
                c=color,
                label=lifted_states_name[j] + "$_{truth}$" if index == 0 else "",
                linewidth=0.5,
                linestyle=linestyles[counter - 2],
            )
            axs[counter].plot(time[0], Yp[i, 0, j], c=color, marker="o", markersize=2)
            axs[counter].plot(
                time,
                Yp_pred[i, :, j],
                c=color,
                label=lifted_states_name[j] + "$_{pred}$" if index == 0 else "",
                linewidth=1,
                linestyle=linestyles[counter - 2],
            )
            axs[counter].plot(
                time[0], Yp_pred[i, 0, j], c=color, marker="o", markersize=2
            )
            if Yobs is not None:
                axs[counter].plot(
                    time,
                    Yobs[i, :, j],
                    c="red",
                    label=lifted_states_name[j] + "$_{obs}$" if index == 0 else "",
                    linewidth=1,
                    linestyle=linestyles[counter - 2],
                )
                axs[counter].plot(
                    time[0], Yobs[i, 0, j], c="red", marker="o", markersize=2
                )
            counter = counter + 1

        for j in range(0, len(inputs_name), 1):
            axs[6].plot(
                time,
                U[i, :, j],
                c=color,
                label=inputs_name[j] if index == 0 else "",
                linewidth=1,
                linestyle=linestyles[j],
            )
            axs[6].plot(time[0], U[i, 0, j], c=color, marker="o", markersize=2)

    # Set titles, labels, and other properties for each subplot
    counter = 0
    for j in range(0, len(lifted_states_name), 2):
        axs[counter].set_title("State phase portrait")
        axs[counter].set_xlabel(lifted_states_name[j] + " $[mm]$")
        axs[counter].set_ylabel(lifted_states_name[j + 1] + " $[mm/s]$")
        axs[counter].legend(loc="best")
        axs[counter].minorticks_on()
        axs[counter].grid(
            which="both", color="lightgray", linestyle="--", linewidth=0.5
        )
        counter = counter + 1
        # axs[counter].set_aspect("equal", adjustable="box")

    counter = 2
    for j in range(0, len(lifted_states_name), 2):
        axs[counter].set_title("States vs time")
        axs[counter].set_ylabel(lifted_states_name[j] + " $[mm]$")
        axs[counter + 1].set_ylabel(lifted_states_name[j + 1] + " $[mm/s]$")
        axs[counter].minorticks_on()
        axs[counter].grid(
            which="both", color="lightgray", linestyle="--", linewidth=0.5
        )
        handles2, labels2 = axs[counter].get_legend_handles_labels()
        handles3, labels3 = axs[counter + 1].get_legend_handles_labels()
        axs[counter].legend(handles2 + handles3, labels2 + labels3, loc="best", ncol=2)

        counter = counter + 2
    axs[counter - 2].set_xlabel("Time [s]")

    axs[6].set_title("Inputs vs time")
    axs[6].set_xlabel("Time [s]")
    axs[6].set_ylabel(inputs_name[0] + " $[mm]$")
    axs[6].legend(loc="best")
    axs[6].minorticks_on()
    axs[6].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    plt.tight_layout()  # Adjust layout to make room for the colorbar
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    plt.draw()
    plt.pause(0.1)
    if save:
        plt.savefig(
            os.path.join(save_dir, "Prediction_Lifted_States.pdf"), format="pdf"
        )
        plt.savefig(
            os.path.join(save_dir, "PNG/Prediction_Lifted_States.png"),
            format="png",
            dpi=600,
        )


def plot_errors(Xp, X_errors, ED_errors, states_name, save_dir, save=False):
    print("Plotting errors...")
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))

    # Group 1: Dimensions 0, 2, and 4
    indices = list(range(0, X_errors.shape[2], 2))
    mag_error = np.sqrt(sum(X_errors[:, :, idx] ** 2 for idx in indices))
    errors = [X_errors[:, :, i].flatten() for i in indices]
    # errors = []
    errors.append(mag_error.flatten())

    mag_measure = np.sqrt(sum(Xp[:, :, idx] ** 2 for idx in indices))
    measures = [Xp[:, :, i].flatten() for i in indices]
    measures.append(mag_measure.flatten())

    # epsilon = 1e-10
    # measures = [m + epsilon for m in measures]
    # divisions = [e / m for e, m in zip(errors, measures)]
    # errors = divisions

    bp1 = axs[0, 0].boxplot(errors)
    axs[0, 0].set_title("Position prediction error")
    x_labels = [states_name[i] for i in indices] + ["RMS"]
    axs[0, 0].set_xticklabels(x_labels)
    axs[0, 0].set_ylabel("Error [mm]")
    axs[0, 0].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Annotate with mean and sd at the bottom
    for i, line in enumerate(bp1["caps"]):
        if i % 2 == 0:  # Check if it's the lower cap of the whisker
            x, y = line.get_xydata()[1]  # bottom of lower whisker
            axs[0, 0].text(
                x - 0.5,
                y - 0.0,
                f"{np.mean(errors[i//2]):.2f} ± {np.std(errors[i//2]):.2f}",
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=7,
                rotation=90,
            )

    # Group 2: Dimensions 1, 3, and 5
    indices = list(range(+1, X_errors.shape[2] + 1, 2))
    mag_error = np.sqrt(sum(X_errors[:, :, idx] ** 2 for idx in indices))
    errors = [X_errors[:, :, i].flatten() for i in indices]
    # errors = []
    errors.append(mag_error.flatten())

    mag_measure = np.sqrt(sum(Xp[:, :, idx] ** 2 for idx in indices))
    measures = [Xp[:, :, i].flatten() for i in indices]
    measures.append(mag_measure.flatten())

    # epsilon = 1e-10
    # measures = [m + epsilon for m in measures]
    # divisions = [e / m for e, m in zip(errors, measures)]
    # errors = divisions

    bp2 = axs[0, 1].boxplot(errors)
    axs[0, 1].set_title("Velocity prediction error")
    x_labels = [states_name[i] for i in indices] + ["RMS"]
    axs[0, 0].set_xticklabels(x_labels)
    axs[0, 1].set_ylabel("Error [mm/s]")
    axs[0, 1].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Annotate with mean and sd at the bottom
    for i, line in enumerate(bp2["caps"]):
        if i % 2 == 0:  # Check if it's the lower cap of the whisker
            x, y = line.get_xydata()[1]  # bottom of lower whisker
            axs[0, 1].text(
                x - 0.5,
                y - 0.3,
                f"{np.mean(errors[i//2]):.2f} ± {np.std(errors[i//2]):.2f}",
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=7,
                rotation=90,
            )

    # Group 2: Dimensions 0, 2, and 4
    indices = list(range(0, ED_errors.shape[2], 2))
    mag_error = np.sqrt(sum(ED_errors[:, :, idx] ** 2 for idx in indices))
    errors = [ED_errors[:, :, i].flatten() for i in indices]
    # errors = []
    errors.append(mag_error.flatten())

    mag_measure = np.sqrt(sum(Xp[:, :, idx] ** 2 for idx in indices))
    measures = [Xp[:, :, i].flatten() for i in indices]
    measures.append(mag_measure.flatten())

    # epsilon = 1e-10
    # measures = [m + epsilon for m in measures]
    # divisions = [e / m for e, m in zip(errors, measures)]
    # errors = divisions

    bp1 = axs[1, 0].boxplot(errors)
    axs[1, 0].set_title("Position reconstruction error")
    x_labels = [states_name[i] for i in indices] + ["RMS"]
    axs[0, 0].set_xticklabels(x_labels)
    axs[1, 0].set_ylabel("Error [mm]")
    axs[1, 0].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Annotate with mean and sd at the bottom
    for i, line in enumerate(bp1["caps"]):
        if i % 2 == 0:  # Check if it's the lower cap of the whisker
            x, y = line.get_xydata()[1]  # bottom of lower whisker
            axs[1, 0].text(
                x - 0.5,
                y - 0.0,
                f"{np.mean(errors[i//2]):.2f} ± {np.std(errors[i//2]):.2f}",
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=7,
                rotation=90,
            )

    # Group 2: Dimensions 1, 3, and 5
    indices = list(range(+1, ED_errors.shape[2] + 1, 2))
    mag_error = np.sqrt(sum(ED_errors[:, :, idx] ** 2 for idx in indices))
    errors = [ED_errors[:, :, i].flatten() for i in indices]
    # errors = []
    errors.append(mag_error.flatten())

    mag_measure = np.sqrt(sum(Xp[:, :, idx] ** 2 for idx in indices))
    measures = [Xp[:, :, i].flatten() for i in indices]
    measures.append(mag_measure.flatten())

    # epsilon = 1e-10
    # measures = [m + epsilon for m in measures]
    # divisions = [e / m for e, m in zip(errors, measures)]
    # errors = divisions

    bp2 = axs[1, 1].boxplot(errors)
    axs[1, 1].set_title("Velocity reconstruction error")
    x_labels = [states_name[i] for i in indices] + ["RMS"]
    axs[0, 0].set_xticklabels(x_labels)
    axs[1, 1].set_ylabel("Error [mm/s]")
    axs[1, 1].grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Annotate with mean and sd at the bottom
    for i, line in enumerate(bp2["caps"]):
        if i % 2 == 0:  # Check if it's the lower cap of the whisker
            x, y = line.get_xydata()[1]  # bottom of lower whisker
            axs[1, 1].text(
                x - 0.5,
                y - 0.3,
                f"{np.mean(errors[i//2]):.2f} ± {np.std(errors[i//2]):.2f}",
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.draw()
    plt.pause(0.1)
    if save:
        plt.savefig(os.path.join(save_dir, "Prediction_Errors.pdf"), format="pdf")


def plot_eigen_functions(Yp, Lambdas, save_dir, save=False):
    print("Plotting eigen functions..")

    lifted_states_name = [
        "$y_1$",
        "$y_2$",
        "$y_3$",
        "$y_4$",
        "$y_5$",
        "$y_6$",
    ]  # Example state names

    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    sc1 = axs[0].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=Lambdas[:, :, 0].flatten(),
        cmap="viridis",
        s=2,
    )
    sc2 = axs[1].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=Lambdas[:, :, 1].flatten(),
        cmap="viridis",
        s=2,
    )
    # sc3 = axs[0,1].scatter(Yp[:, :, 2].flatten(), Yp[:, :, 3].flatten(), c=Lambdas[:, :, 2].flatten(), cmap='viridis', s=2)
    # sc4 = axs[1,1].scatter(Yp[:, :, 2].flatten(), Yp[:, :, 3].flatten(), c=Lambdas[:, :, 3].flatten(), cmap='viridis', s=2)

    # Set the grid to appear behind plot elements
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Add color bars for each scatter plot
    cbar_kwargs = {"fraction": 0.046, "pad": 0.04}  # Adjust these values as needed
    fig.colorbar(sc1, ax=axs[0], **cbar_kwargs)
    fig.colorbar(sc2, ax=axs[1], **cbar_kwargs)
    # fig.colorbar(sc3, ax=axs[0,1], **cbar_kwargs)
    # fig.colorbar(sc4, ax=axs[1,1], **cbar_kwargs)

    axs[0].set_title("$\mu_1$")
    axs[0].set_xlabel(lifted_states_name[0])
    axs[0].set_ylabel(lifted_states_name[1])
    axs[0].minorticks_on()
    axs[0].set_aspect("equal", adjustable="box")

    axs[1].set_title("$\omega_1$")
    axs[1].set_xlabel(lifted_states_name[0])
    axs[1].set_ylabel(lifted_states_name[1])
    axs[1].minorticks_on()
    axs[1].set_aspect("equal", adjustable="box")

    # axs[0, 1].set_title("$\mu_2$")
    # axs[0, 1].set_xlabel(lifted_states_name[2])
    # axs[0, 1].set_ylabel(lifted_states_name[3])
    # axs[0, 1].minorticks_on()
    # axs[0, 1].set_aspect("equal", adjustable="box")

    # axs[1, 1].set_title("$\omega_2$")
    # axs[1, 1].set_xlabel(lifted_states_name[2])
    # axs[1, 1].set_ylabel(lifted_states_name[3])
    # axs[1, 1].minorticks_on()
    # axs[1, 1].set_aspect("equal", adjustable="box")

    plt.tight_layout()  # Adjust layout to make room for the colorbar
    plt.draw()
    plt.pause(0.1)
    if save:
        plt.savefig(
            os.path.join(save_dir, "Prediction_Eigen function.pdf"), format="pdf"
        )
        plt.savefig(
            os.path.join(save_dir, "PNG/Prediction_Eigen.png"), format="png", dpi=300
        )


def plot_A_Matrix(Yp, A, save_dir, save=False):
    print("Plotting G Matrix ...")

    lifted_states_name = [
        "$y_1$",
        "$y_2$",
        "$y_3$",
        "$y_4$",
        "$y_5$",
        "$y_6$",
    ]  # Example state names

    fig, axs = plt.subplots(2, 2, figsize=(7, 8))

    sc1 = axs[0, 0].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=A[:, :, 0, 0].flatten(),
        cmap="viridis",
        s=2,
    )
    sc2 = axs[0, 1].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=A[:, :, 0, 1].flatten(),
        cmap="viridis",
        s=2,
    )
    sc3 = axs[1, 0].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=A[:, :, 1, 0].flatten(),
        cmap="viridis",
        s=2,
    )
    sc4 = axs[1, 1].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=A[:, :, 1, 1].flatten(),
        cmap="viridis",
        s=2,
    )

    # Set the grid to appear behind plot elements
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Add color bars for each scatter plot
    cbar_kwargs = {"fraction": 0.046, "pad": 0.04}  # Adjust these values as needed
    fig.colorbar(sc1, ax=axs[0, 0], **cbar_kwargs)
    fig.colorbar(sc2, ax=axs[0, 1], **cbar_kwargs)
    fig.colorbar(sc3, ax=axs[1, 0], **cbar_kwargs)
    fig.colorbar(sc4, ax=axs[1, 1], **cbar_kwargs)

    axs[0, 0].set_title("$G_{1,1}$")
    axs[0, 0].set_xlabel(lifted_states_name[0])
    axs[0, 0].set_ylabel(lifted_states_name[1])
    axs[0, 0].minorticks_on()
    # axs[0, 0].set_aspect("equal", adjustable="box")

    axs[0, 1].set_title("$G_{1,2}$")
    axs[0, 1].set_xlabel(lifted_states_name[0])
    axs[0, 1].set_ylabel(lifted_states_name[1])
    axs[0, 1].minorticks_on()
    # axs[0, 1].set_aspect("equal", adjustable="box")

    axs[1, 0].set_title("$G_{2,1}$")
    axs[1, 0].set_xlabel(lifted_states_name[0])
    axs[1, 0].set_ylabel(lifted_states_name[1])
    axs[1, 0].minorticks_on()
    # axs[1, 0].set_aspect("equal", adjustable="box")

    axs[1, 1].set_title("$G_{2,2}$")
    axs[1, 1].set_xlabel(lifted_states_name[0])
    axs[1, 1].set_ylabel(lifted_states_name[1])
    axs[1, 1].minorticks_on()
    # axs[1, 1].set_aspect("equal", adjustable="box")

    plt.tight_layout()  # Adjust layout to make room for the colorbar
    plt.draw()
    plt.pause(0.1)
    if save:
        plt.savefig(os.path.join(save_dir, "Prediction_G_Matrix.pdf"), format="pdf")
        plt.savefig(
            os.path.join(save_dir, "PNG/Prediction_G_Matrix.png"), format="png", dpi=300
        )


def plot_B_Matrix(Yp, B, save_dir, save=False):
    print("Plotting B Matrix ...")

    lifted_states_name = [
        "$y_1$",
        "$y_2$",
        "$y_3$",
        "$y_4$",
        "$y_5$",
        "$y_6$",
    ]  # Example state names

    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    sc1 = axs[0].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=B[:, :, 0].flatten(),
        cmap="viridis",
        s=2,
    )
    sc2 = axs[1].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=B[:, :, 1].flatten(),
        cmap="viridis",
        s=2,
    )
    # sc3 = axs[0, 1].scatter(
    #     Yp[:, :, 0].flatten(),
    #     Yp[:, :, 1].flatten(),
    #     c=B[:, :, 2].flatten(),
    #     cmap="viridis",
    #     s=2,
    # )
    # sc4 = axs[1, 1].scatter(
    #     Yp[:, :, 0].flatten(),
    #     Yp[:, :, 1].flatten(),
    #     c=B[:, :, 3].flatten(),
    #     cmap="viridis",
    #     s=2,
    # )

    # Set the grid to appear behind plot elements
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Add color bars for each scatter plot
    cbar_kwargs = {"fraction": 0.046, "pad": 0.04}  # Adjust these values as needed
    fig.colorbar(sc1, ax=axs[0], **cbar_kwargs)
    fig.colorbar(sc2, ax=axs[1], **cbar_kwargs)
    # fig.colorbar(sc3, ax=axs[0, 1], **cbar_kwargs)
    # fig.colorbar(sc4, ax=axs[1, 1], **cbar_kwarg
    axs[0].set_title("$H_{1,1}$")
    axs[0].set_xlabel(lifted_states_name[0])
    axs[0].set_ylabel(lifted_states_name[1])
    axs[0].minorticks_on()
    # axs[0].set_aspect("equal", adjustable="box")

    axs[1].set_title("$H_{2,1}$")
    axs[1].set_xlabel(lifted_states_name[0])
    axs[1].set_ylabel(lifted_states_name[1])
    axs[1].minorticks_on()
    # axs[1].set_aspect("equal", adjustable="box")

    # axs[0, 1].set_title("$B_{3,1}$")
    # axs[0, 1].set_xlabel(lifted_states_name[0])
    # axs[0, 1].set_ylabel(lifted_states_name[1])
    # axs[0, 1].minorticks_on()
    # axs[0, 1].set_aspect("equal", adjustable="box")

    # axs[1, 1].set_title("$B_{4,1}$")
    # axs[1, 1].set_xlabel(lifted_states_name[0])
    # axs[1, 1].set_ylabel(lifted_states_name[1])
    # axs[1, 1].minorticks_on()
    # axs[1, 1].set_aspect("equal", adjustable="box")

    plt.tight_layout()  # Adjust layout to make room for the colorbar
    plt.draw()
    plt.pause(0.1)
    if save:
        plt.savefig(os.path.join(save_dir, "Prediction_H_Matrix.pdf"), format="pdf")
        plt.savefig(
            os.path.join(save_dir, "PNG/Prediction_H_Matrix.png"), format="png", dpi=300
        )


def plot_encoder_map(Yp, Xp, save_dir, save=False):
    print("Plotting decoder map ...")

    lifted_states_name = [
        "$y_1$",
        "$y_2$",
        "$y_3$",
        "$y_4$",
        "$y_5$",
        "$y_6$",
    ]  # Example state names

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    sc1 = axs[0, 0].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=Xp[:, :, 0].flatten(),
        cmap="viridis",
        s=2,
    )
    sc2 = axs[1, 0].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=Xp[:, :, 1].flatten(),
        cmap="viridis",
        s=2,
    )
    sc3 = axs[0, 1].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=Xp[:, :, 2].flatten(),
        cmap="viridis",
        s=2,
    )
    sc4 = axs[1, 1].scatter(
        Yp[:, :, 0].flatten(),
        Yp[:, :, 1].flatten(),
        c=Xp[:, :, 3].flatten(),
        cmap="viridis",
        s=2,
    )

    # Set the grid to appear behind plot elements
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

    # Add color bars for each scatter plot
    cbar_kwargs = {"fraction": 0.046, "pad": 0.04}  # Adjust these values as needed
    fig.colorbar(sc1, ax=axs[0, 0], **cbar_kwargs)
    fig.colorbar(sc2, ax=axs[1, 0], **cbar_kwargs)
    fig.colorbar(sc3, ax=axs[0, 1], **cbar_kwargs)
    fig.colorbar(sc4, ax=axs[1, 1], **cbar_kwargs)

    axs[0, 0].set_title("$X_1$")
    axs[0, 0].set_xlabel(lifted_states_name[0])
    axs[0, 0].set_ylabel(lifted_states_name[1])
    axs[0, 0].minorticks_on()
    # axs[0, 0].set_aspect("equal", adjustable="box")

    axs[1, 0].set_title("$X_2$")
    axs[1, 0].set_xlabel(lifted_states_name[0])
    axs[1, 0].set_ylabel(lifted_states_name[1])
    axs[1, 0].minorticks_on()
    # axs[1, 0].set_aspect("equal", adjustable="box")

    axs[0, 1].set_title("$X_3$")
    axs[0, 1].set_xlabel(lifted_states_name[0])
    axs[0, 1].set_ylabel(lifted_states_name[1])
    axs[0, 1].minorticks_on()
    # axs[0, 1].set_aspect("equal", adjustable="box")

    axs[1, 1].set_title("$X_4$")
    axs[1, 1].set_xlabel(lifted_states_name[0])
    axs[1, 1].set_ylabel(lifted_states_name[1])
    axs[1, 1].minorticks_on()
    # axs[1, 1].set_aspect("equal", adjustable="box")

    plt.tight_layout()  # Adjust layout to make room for the colorbar
    plt.draw()
    plt.pause(0.1)
    if save:
        plt.savefig(os.path.join(save_dir, "Decoder_map.pdf"), format="pdf")
        plt.savefig(
            os.path.join(save_dir, "PNG/Decoder_map.png"), format="png", dpi=300
        )


def save_to_csv(Xp, Yp, Lambdas, save_dir):
    
    # Get the first dimension size to determine the number of files
    N = Yp.shape[0]

    column_names = ['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'l_mu', 'l_omega']  # Adjust the number of columns as needed
    
    for i in range(N):
        data_slice = np.concatenate((Xp[i, :, :], Yp[i, :, :], Lambdas[i, :, :]), axis = 1)
        df = pd.DataFrame(data_slice, columns=column_names)
        
        # Save DataFrame to CSV
        filename = f'circle_result_{i+1}.csv'
        csv_dir = os.path.join(save_dir, 'csv', 'circle')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        df.to_csv(os.path.join(csv_dir, filename), index=False, float_format='%.6f')
    print('Saved in CSV folder')

def random_point_in_circle(radius):
    """
    Generates a random point inside a circle with a given radius.

    Parameters:
    radius (float): The radius of the circle.

    Returns:
    tuple: (x, y) coordinates of the random point.
    """
    # Generate random angle between 0 and 2*pi
    angle = np.random.uniform(0, 2 * np.pi)
    # Generate random radius r, 0 <= r <= radius, uniformly distributed in the area (r^2)
    r = radius * np.sqrt(np.random.uniform(0, 1))
    # Convert polar to Cartesian coordinates
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y

if __name__ == "__main__":
    main()
