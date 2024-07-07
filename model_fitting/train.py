import os, sys, json
print(os.getcwd())
sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trainer import Trainer
from model_fitting.utils import (
    plot_error,
    plot_trend,
    plot_states,
    PlotLearning,
    MLPDataset,
    Weighted_MSE_Loss,
    MAE_Metrix,
    init_tn_weights,
    init_dl_weights,
    init_he_weights,
    inverse_normalize,
    DeepKoopmanExplicitLoss,
    DeepKoopmanExplicitMetric,
)
from model_fitting.models import (
    MLP,
    DoubleMLP,
    States_Auxilary,
    Input_Auxilary,
    LinearKoopman,
    LinearKoopmanWithInput,
    TrainableKoopmanDynamics,
    CombinedDeepKoopman,
    CombinedKoopman_withAux_withoutInput,
    CombinedKoopman_withAux_withInput,
)

# ###### Check for availabel processors ###### #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


"""Set Paramters"""
####################################################################################
model_name = "catheter_1T_unforced_3.1"
model_details = {
    "model_name": model_name,
    "desc": "one tendon with negative - unforced - mlp encoder - adjust encoder size",
}
### dataset related parameters
dataset_options = {
    "dataset_name": "catheter_1T_unforced_3.1",
    # "dataset_name": "catheter_1T_unforced_2.1",
    "add_sequence_dimension": True,  # True for TCN and LSTM
    "fit_nomalization": 'X',  #'X', 'U', 'All',   'X' for unforced system - None for forced system to use the unforced params
    "prenomalization": True,
    "sample_step": 1,
    "sample_time": None,
}
### training options
training_options = {
    "transfer_from": None,
    # "transfer_from": "catheter_1T_forced_1.2.3",
    "transfer_input_aux": True,
    "n_epochs": 3000,
    "n_recon_epoch": 40,  # number of first epocs only with reconstruction loss
    "learning_rate": 1e-3,
    "decay_rate": 0.9988,
    "batch_size": 256,
    "lossfunc_weights": ([1e-1, 1e-7, 1e-13]),
    "num_pred_steps": 10,
    "train_encoder": True,
    "train_decoder": True,
    "train_states_auxilary": True,
    "train_inputs_auxilary": False,
    "cuda": True,
}
### model params for Koopman operator
koopman_params = {
    "num_complexeigens_pairs": 1,
    "num_realeigens": 0,
    "sample_time": None,  # loads from the dataste options
    "structure": "Jordan",  # "Jordan" or "Controlable"
}
### model params for encoder
encoder_params = {
    "architecture": "MLP",  # DoubleMLP, MLP
    "state_size": 4,
    "hidden_sizes": ([128, 128, 128, 128, 128]),
    # "hidden_sizes": ([160, 160, 160, 160, 160]),
    # "hidden_sizes": ([80, 80, 80, 80]),
    "lifted_state_size": koopman_params["num_realeigens"]
    + 2 * koopman_params["num_complexeigens_pairs"],
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
    "sample_time_new": None,
}
### changes in decoder paramters
decoder_params = encoder_params.copy()
decoder_params["hidden_sizes"] = [128, 128, 128, 128, 128]
decoder_params["architecture"] = "MLP"
### model params for the auxilary network to estimate the eigen values of the A matrix
states_auxilary_params = {
    "architecture": "MLP",
    "hidden_sizes": ([30, 30, 30]),
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
}
### model params for the auxilary network to estimate the B matrix
inputs_auxilary_params = {
    "architecture": "MLP",
    "system_input_size": 1,
    "hidden_sizes": ([30, 30, 30]),
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
}

# driven paramters
states_auxilary_params["num_complexeigens_pairs"] = koopman_params[
    "num_complexeigens_pairs"
]
states_auxilary_params["num_realeigens"] = koopman_params["num_realeigens"]
states_auxilary_params["state_size"] = encoder_params["lifted_state_size"]
states_auxilary_params["output_size"] = encoder_params["lifted_state_size"]
# driven paramters
inputs_auxilary_params["lifted_state_size"] = encoder_params["lifted_state_size"]
inputs_auxilary_params["output_shape"] = [
    encoder_params["lifted_state_size"],
    # encoder_params["state_size"],
    inputs_auxilary_params["system_input_size"],
]

#### create the model saving folder if not exists
model_dir = os.path.join(os.getcwd(), "trained_models", model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


"""Load Pretrained Model"""
####################################################################################
normalizer_params = None
if training_options["transfer_from"] is not None:
    pretrained_model_dir = os.path.join(
        os.getcwd(), "trained_models", training_options["transfer_from"]
    )
    pretrained_model_info_path = os.path.join(pretrained_model_dir, "model_info.json")
    if os.path.isfile(pretrained_model_info_path):
        with open(pretrained_model_info_path, "r") as fp:
            model_info = json.load(fp)
    else:
        error_message = f"[{pretrained_model_info_path}] does not exist!"
        raise FileNotFoundError(error_message)
    ### load noarmalizer params if exists in the pretrained model
    normalizer_params = model_info["normalizer_params"]


"""Load Dataset"""
####################################################################################
#### load dataset_info json file
datasets_dir = os.path.join(os.getcwd(), "datasets", dataset_options["dataset_name"])
dataset_info_path = os.path.join(datasets_dir, "dataset_info.json")
if os.path.isfile(dataset_info_path):
    with open(dataset_info_path, "r") as fp:
        dataset_info = json.load(fp)
        dataset_options["seq_length"] = dataset_info["sequence_length"]
        dataset_options["test_seq_length"] = dataset_info["test_sequence_length"]
        dataset_options["sample_time_new"] = (
            dataset_info["sample_time"] * dataset_options["sample_step"]
        )
        koopman_params["sample_time"] = dataset_options["sample_time_new"]
        encoder_params["sample_time"] = dataset_options["sample_time_new"]
else:
    error_message = f"[{dataset_info_path}] does not exist!"
    raise FileNotFoundError(error_message)

#### Load train datasets
print("Loading datasets...")
dataset_train = MLPDataset(
    datasets_dir=datasets_dir,
    filename="train.txt",
    seq_length=dataset_options["seq_length"],
    add_dim=dataset_options["add_sequence_dimension"],
    fit_normalize=dataset_options["fit_nomalization"],
    prenomalize=dataset_options["prenomalization"],
    scaler_params=normalizer_params,
    sample_step=dataset_options["sample_step"],
)
print("Loading train datasets finished!")

#### get normalizer params for normalize val and test datasets with the same paramters
normalizer_params = dataset_train.get_normalizer_params()

#### Load validation datasets
dataset_val = MLPDataset(
    datasets_dir=datasets_dir,
    filename="val.txt",
    seq_length=dataset_options["seq_length"],
    add_dim=dataset_options["add_sequence_dimension"],
    fit_normalize=None,
    prenomalize=dataset_options["prenomalization"],
    scaler_params=normalizer_params,
    sample_step=dataset_options["sample_step"],
)
print("Loading val datasets finished!")

#### Load test datasets
dataset_test = MLPDataset(
    datasets_dir=datasets_dir,
    filename="test.txt",
    seq_length=dataset_options["test_seq_length"],
    add_dim=dataset_options["add_sequence_dimension"],
    fit_normalize=None,
    prenomalize=dataset_options["prenomalization"],
    scaler_params=normalizer_params,
    sample_step=dataset_options["sample_step"],
)
print("Loading test datasets finished!")

####  Initilize dataloaders
dataloader_train = DataLoader(
    dataset=dataset_train, batch_size=training_options["batch_size"], shuffle=False
)
dataloader_valid = DataLoader(
    dataset=dataset_val, batch_size=training_options["batch_size"], shuffle=False
)
dataloader_test = DataLoader(dataset=dataset_test, shuffle=False)


"""Initialize the Model"""
####################################################################################
if encoder_params["architecture"] == "MLP":
    encoder = MLP(
        input_size=encoder_params["state_size"],
        hidden_sizes=encoder_params["hidden_sizes"],
        output_size=encoder_params["lifted_state_size"],
        activation=encoder_params["activation"],
        input_normalizer_params=None,
        device=device,
    ).to(device)
elif encoder_params["architecture"] == "DualMLP":
    encoder = DoubleMLP(
        input_size=encoder_params["state_size"],
        hidden_sizes=encoder_params["hidden_sizes"],
        output_size=encoder_params["lifted_state_size"],
        activation=encoder_params["activation"],
        input_normalizer_params=None,
        device=device,
    ).to(device)
decoder = MLP(
    input_size=decoder_params["lifted_state_size"],
    hidden_sizes=decoder_params["hidden_sizes"],
    output_size=decoder_params["state_size"],
    activation=decoder_params["activation"],
    output_normalizer_params=None,
    device=device,
).to(device)
auxilary = States_Auxilary(
    num_complexeigens_pairs=states_auxilary_params["num_complexeigens_pairs"],
    num_realeigens=states_auxilary_params["num_realeigens"],
    hidden_sizes=states_auxilary_params["hidden_sizes"],
    activation=states_auxilary_params["activation"],
).to(device)
input_auxilary = Input_Auxilary(
    input_size=inputs_auxilary_params["lifted_state_size"],
    hidden_sizes=inputs_auxilary_params["hidden_sizes"],
    output_shape=inputs_auxilary_params["output_shape"],
    activation=inputs_auxilary_params["activation"],
).to(device)
koopmanOperator = TrainableKoopmanDynamics(
    num_complexeigens_pairs=koopman_params["num_complexeigens_pairs"],
    num_realeigens=koopman_params["num_realeigens"],
    sample_time=koopman_params["sample_time"],
    structure=koopman_params["structure"],  # "Jordan", "Controlable"
    # device = device
).to(device)

# koopmanOperator = LinearKoopman(dimension=encoder_params["lifted_state_size"]).to(device)
# koopmanOperator = LinearKoopmanWithInput(lifted_states_dimension=encoder_params["lifted_state_size"], inpus_dimension=encoder_params["input_size"]).to(device)
# model = CombinedDeepKoopman(encoder, koopmanOperator, decoder)
# model = CombinedKoopman_withAux_withoutInput(encoder, auxilary, koopmanOperator, decoder)
model = CombinedKoopman_withAux_withInput(
    encoder, auxilary, koopmanOperator, decoder, input_auxilary
)

### initialze weights and biases -  random of loaded from pretrained
if training_options["transfer_from"] is None:
    # ### initilize weights
    # model.apply(init_weights)
    # model.apply(init_dl_weights)
    model.apply(init_tn_weights)
# load pretrained network
else:
    pretrained_model_dir = os.path.join(
        os.getcwd(), "trained_models", training_options["transfer_from"]
    )
    encoder_dict = torch.load(
        os.path.join(pretrained_model_dir, "trained_model", "TorchTrained_Encoder.pt"),
        map_location=torch.device("cpu"),
    )
    decoder_dict = torch.load(
        os.path.join(pretrained_model_dir, "trained_model", "TorchTrained_Decoder.pt"),
        map_location=torch.device("cpu"),
    )
    states_auxilary_dict = torch.load(
        os.path.join(
            pretrained_model_dir, "trained_model", "TorchTrained_States_Auxilary.pt"
        ),
        map_location=torch.device("cpu"),
    )
    koopman_dict = torch.load(
        os.path.join(pretrained_model_dir, "trained_model", "TorchTrained_Koopman.pt"),
        map_location=torch.device("cpu"),
    )

    model.encoder.load_state_dict(encoder_dict["state_dict"])
    model.decoder.load_state_dict(decoder_dict["state_dict"])
    model.states_auxilary.load_state_dict(states_auxilary_dict["state_dict"])
    model.koopman.load_state_dict(koopman_dict["state_dict"])

    #### check for pretrained inputs aux model
    file_path = os.path.join(
        pretrained_model_dir, "trained_model", "TorchTrained_Inputs_Auxilary.pt"
    )
    if os.path.exists(file_path) and training_options["transfer_input_aux"]:
        inputs_auxilary_dict = torch.load(file_path, map_location=torch.device("cpu"))
        model.inputs_auxilary.load_state_dict(inputs_auxilary_dict["state_dict"])
    else:
        print("File does not exist:", file_path)
        print("Inputs_Auxilary initiated with random parameters")
        model.inputs_auxilary.apply(init_tn_weights)


"""Loss Function, Optimizer, and Learning Rate Scheduler"""
####################################################################################
# loss_function = nn.MSELoss()
# weight = torch.tensor(training_options["mseloss_weights"])
# loss_function = Weighted_MSE_Loss(weight=weight)
loss_function = DeepKoopmanExplicitLoss(
    alpha1=training_options["lossfunc_weights"][0],
    alpha2=training_options["lossfunc_weights"][1],
    alpha_reg=training_options["lossfunc_weights"][2],
)
loss_type = "MSE"
metric_function = DeepKoopmanExplicitMetric()
metric_type = "MAE"
optimizer = torch.optim.Adam(model.parameters(), lr=training_options["learning_rate"])
lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=training_options["decay_rate"])


"""Save Model Info Before Training"""
####################################################################################
model_parameters = {
    "model_details": model_details.copy(),
    "encoder_parameters": encoder_params.copy(),
    "decoder_parameters": decoder_params.copy(),
    "states_auxilary_parameters": states_auxilary_params.copy(),
    "koopman_parameters": koopman_params.copy(),
    "inputs_auxilary_parameters": inputs_auxilary_params.copy(),
}
info = {
    "encoder_params": encoder_params.copy(),
    "decoder_params": decoder_params.copy(),
    "states_auxilary_params": states_auxilary_params.copy(),
    "koopman_params": koopman_params.copy(),
    "inputs_auxilary_params": inputs_auxilary_params.copy(),
    "normalizer_params": normalizer_params.copy(),
    "training_options": training_options.copy(),
    "dataset_options": dataset_options.copy(),
    "dataset_info": dataset_info.copy(),
}
with open(os.path.join(model_dir, "model_info.json"), "w") as fp:
    json.dump(info, fp)
    fp.close()


""" Train """
####################################################################################
trainer = Trainer(
    model=model,
    crit=loss_function,
    metric=metric_function,
    optim=optimizer,
    lr_scheduler=lr_scheduler,
    train_dl=dataloader_train,
    val_dev_dl=dataloader_valid,
    cuda=training_options["cuda"],
    early_stopping_patience=None,
    save_dir=model_dir,
    model_parameters=model_parameters,
    training_parameters=training_options,
    normalilzer_scale=(
        normalizer_params["params_y"]["max"][0]
        - normalizer_params["params_y"]["min"][0]
    ),
    sequence_length=dataset_options["seq_length"],
)
#### Train
loss_train, loss_validation, metric_validation = trainer.fit(
    epochs=training_options["n_epochs"], plot_learning=True
)
print("Training finished!")

#### Plot and save train history figures and .csv
plt.savefig(os.path.join(model_dir, "Training_Loss.pdf"), format="pdf")
train_history = pd.DataFrame(
    {
        "epoch": np.arange(1, training_options["n_epochs"] + 1),
        "train_" + loss_type: loss_train,
        "val_" + loss_type: loss_validation,
        "val_" + metric_type + "_recon": metric_validation[:, 0],
        "val_" + metric_type + "_pred": metric_validation[:, 1],
        "val_" + metric_type + "_lin": metric_validation[:, 2],
    }
)
train_history.to_csv(os.path.join(model_dir, "Train_Results.csv"), sep=",", index=False)

#### Save the trained model parameters
trainer.save_model()


"""Predict the Test Data"""
####################################################################################
### initialize required variables
Yp = np.empty(
    (
        len(dataloader_test),
        dataset_options["seq_length"] - 1,
        encoder_params["lifted_state_size"],
    )
)
Yp_pred = np.empty(
    (
        len(dataloader_test),
        dataset_options["seq_length"] - 1,
        encoder_params["lifted_state_size"],
    )
)
Xp = np.empty(
    (
        len(dataloader_test),
        training_options["num_pred_steps"],
        encoder_params["state_size"],
    )
)
Xp_pred = np.empty(
    (
        len(dataloader_test),
        training_options["num_pred_steps"],
        encoder_params["state_size"],
    )
)

model = trainer._model.cpu()
for i, (x, u, _) in enumerate(dataloader_test, 0):
    Yp_list = []  ## truth value of y_plus from advanced encoded x
    Yp_pred_list = []  ## predicted value of y_plus from iterating the Koopman model
    Xp_list = []  ## truth value of x_plus from advanced x
    Xp_pred_list = (
        []
    )  ## predicted value of x_plus from decoding the iterated Koopman model
    x0 = x[:, 0, :]
    y0 = model.encoder(x0)
    for j in range(dataset_options["seq_length"] - 1):
        y_adv, omega = model.koopman(y0, u[:, j, :])
        Yp_pred_list.append(y_adv)
        Yp_list.append(model.encoder(x[:, j + 1, :]))
        if j < training_options["num_pred_steps"]:
            Xp_pred_list.append(model.decoder(y_adv))
            Xp_list.append(x[:, j + 1, :])
        y0 = y_adv
    Xp[i, :, :] = torch.stack(Xp_list, dim=1).detach().numpy()[0, :, :]
    Xp_pred[i, :, :] = torch.stack(Xp_pred_list, dim=1).detach().numpy()[0, :, :]
    Yp[i, :, :] = torch.stack(Yp_list, dim=1).detach().numpy()[0, :, :]
    Yp_pred[i, :, :] = torch.stack(Yp_pred_list, dim=1).detach().numpy()[0, :, :]

# if dataset_options["prenomalization"]:
#     x_plus_truth = inverse_normalize(x_truth, normalizer_params["params_y"])
# # pred_y = inverse_normalize(
#     np.delete(pred_y, 0, 0), normalizer_params["params_y"]
# )  # delete the first row which is nan
# error = np.abs(pred_yp - truth_xp)

error_X = np.sqrt(np.mean(np.square(Xp_pred - Xp), axis=2))
error_Y = np.sqrt(np.mean(np.square(Yp_pred - Yp), axis=2))

avg_error_X = np.mean(error_X)
avg_error_Y = np.mean(error_Y)
std_deviation_X = np.std(error_X)
std_deviation_Y = np.std(error_Y)
print(
    f"Average Error X: {avg_error_X:.8f}, Standard Deviation X: {std_deviation_X:.8f}"
)
print(
    f"Average Error Y: {avg_error_Y:.8f}, Standard Deviation Y: {std_deviation_Y:.8f}"
)

# ########## Save prediction results in .csv
# test_prediction = pd.DataFrame(
#     {
#         "v_dot_truth": truth[:, 0],
#         "theta_dot_truth": truth[:, 1],
#         "v_dot_truth": y_plus_pred[:, 0],
#         "theta_dot_truth": y_plus_pred[:, 1],
#         "v_dot_error": error[:, 0],
#         "theta_dot_error": error[:, 1],
#     }
# )
# test_prediction.to_csv(
#     os.path.join(model_dir, "Test_Prediction.csv"), sep=",", index=False
# )

(train_errors) = {
    "loss_train": loss_train[-1],
    "loss_valid": loss_validation[-1],
    "metric_valid_recon": metric_validation[-1, 0],
    "metric_valid_pred": metric_validation[-1, 1],
    "metric_valid_lin": metric_validation[-1, 2],
    "error_test_pred": avg_error_X.tolist(),
    "error_test_linear": avg_error_Y.tolist(),
}

with open(os.path.join(model_dir, "model_info.json"), "r+") as fp:
    info = json.load(fp)
    info["train_history"] = train_errors
    fp.seek(0)
    json.dump(info, fp)
    fp.close()


"""Plot and Save Prediction Results Figures"""
####################################################################################
error_list = [error_X, error_Y]
fig_1 = plot_error(error_list, labels=["x", "y"])
plt.savefig(os.path.join(model_dir, "Prediction_Error.pdf"), format="pdf")
fig_2 = plot_states(Xp, Xp_pred, Yp, Yp_pred)
plt.savefig(os.path.join(model_dir, "Prediction_Trend.pdf"), format="pdf")

print("Done!")
