import os, sys, json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(os.getcwd())
sys.path.append(os.getcwd())
current_directory = os.path.dirname(__file__)

from trainer import Trainer
from model_fitting_nature.utils import (
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
from model_fitting_nature.models import (
    MLP,
    Auxilary,
    LinearKoopman,
    LinearKoopmanWithInput,
    TrainableKoopmanDynamics,
    CombinedDeepKoopman,
    CombinedKoopman_withAux,
)

# ###### Check for availabel processors ###### #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

#################################
#        Adjust parameters      #
#################################

model_name = "Catheter_3"
koopman_params = {  ### model params for encoder, decoder is driven accordingly
    "num_complexeigens_pairs": 2,  
    "num_realeigens": 1,  
    "sample_time": None,     # loads from the dataste options
}
encoder_params = {  ### model params for encoder, decoder is driven accordingly
    "name": model_name,
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
dataset_options = {
    "name": "catheter",
    "add_sequence_dimension": True,  # True for TCN and LSTM
    "do_nomalization": True,
}
training_options = {
    "n_epochs": 2000,
    "n_recon_epoch": 10,        # number of first ecpocs only with reconstruction loss
    "learning_rate": 2e-3,
    "decay_rate": 0.9988,
    "batch_size": 128,
    "lossfunc_weights": ([1e-1, 1e-7, 1e-13]),
    "num_pred_steps": 10,
    "cuda": False,
}

# ##### some preparations ##### #
# create the model saving folder if not exists
model_dir = os.path.join(os.getcwd(), "trained_models", model_name)     
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# load dataset_info json file
datasets_dir = os.path.join(os.getcwd(), "datasets", dataset_options["name"])
dataset_info_path = os.path.join(datasets_dir, "dataset_info.json")
if os.path.isfile(dataset_info_path):
    with open(dataset_info_path, "r") as fp:
        dataset_info = json.load(fp)
        encoder_params["sample_time"] = dataset_info["sample_time"]
        dataset_options["seq_length"] = dataset_info["sequence_length"]
        koopman_params["sample_time"] = dataset_info["sample_time"]
else:
    error_message = f"[{dataset_info_path}] does not exist!"
    raise FileNotFoundError(error_message)

#################################
#       Load test datasets      #
#################################

# ########## Load train datasets
print("Loading datasets...")
dataset_train = MLPDataset(
    datasets_dir=datasets_dir,
    filename="train.txt",
    sample_time=encoder_params["sample_time"],
    seq_length=dataset_options["seq_length"],
    add_dim=dataset_options["add_sequence_dimension"],
    normalizer=dataset_options["do_nomalization"],
    scaler_params=None,
)
print("Loading train datasets finished!")

# ### get normalizer object
if dataset_options["do_nomalization"]:
    normalizer_params = dataset_train.get_normalizer_params()
else:
    normalizer_params = None

# ########## Load validation datasets
dataset_val = MLPDataset(
    datasets_dir=datasets_dir,
    filename="val.txt",
    sample_time=encoder_params["sample_time"],
    seq_length=dataset_options["seq_length"],
    add_dim=dataset_options["add_sequence_dimension"],
    normalizer=dataset_options["do_nomalization"],
    scaler_params=normalizer_params,
)
print("Loading val datasets finished!")

# ########## Load test datasets
dataset_test = MLPDataset(
    datasets_dir=datasets_dir,
    filename="test.txt",
    sample_time=encoder_params["sample_time"],
    seq_length=dataset_options["seq_length"],
    add_dim=dataset_options["add_sequence_dimension"],
    normalizer=dataset_options["do_nomalization"],
    scaler_params=normalizer_params,
)
print("Loading test datasets finished!")

####  Initilize dataloaders  ####
dataloader_train = DataLoader(
    dataset=dataset_train, batch_size=training_options["batch_size"], shuffle=False
)
dataloader_valid = DataLoader(
    dataset=dataset_val, batch_size=training_options["batch_size"], shuffle=False
)
dataloader_test = DataLoader(dataset=dataset_test, shuffle=False)

################################
#     Initialize the model     #
################################

encoder = MLP(
    input_size=encoder_params["state_size"],
    hidden_sizes=encoder_params["hidden_sizes"],
    output_size=encoder_params["lifted_state_size"],
    activation=encoder_params["activation"],
).to(device)
decoder = MLP(
    input_size=encoder_params["lifted_state_size"],
    hidden_sizes=encoder_params["hidden_sizes"],
    output_size=encoder_params["state_size"],
    activation=encoder_params["activation"],
).to(device)
auxilary = Auxilary(
    input_size=auxilary_params["state_size"],
    params=auxilary_params,
).to(device)
koopmanOperator = TrainableKoopmanDynamics(koopman_params["num_complexeigens_pairs"], koopman_params["num_realeigens"], koopman_params["sample_time"]).to(device)
# koopmanOperator = LinearKoopman(dimension=encoder_params["lifted_state_size"]).to(device)
# koopmanOperator = LinearKoopmanWithInput(lifted_states_dimension=encoder_params["lifted_state_size"], inpus_dimension=encoder_params["input_size"]).to(device)

# model = CombinedDeepKoopman(encoder, koopmanOperator, decoder)
model = CombinedKoopman_withAux(encoder, auxilary, koopmanOperator, decoder)

# ### initilize weights
# model.apply(init_weights)
# model.apply(init_dl_weights)
model.apply(init_tn_weights)

# ########## Define the loss function, optimizer, and learning rate scheduler
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

model_parameters = {
    "encoder_parameters": encoder_params.copy(),
    "auxilary_parameters": auxilary_params.copy(),
    "koopman_parameters": koopman_params.copy(),
}

# ########## Initilize Trainer object
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

info = {
    "encoder_parameters": encoder_params,
    "auxilary_parameters": auxilary_params,
    "koopman_parameters": koopman_params,
    "normalizer_params": normalizer_params,
    "training_options": training_options,
    "dataset_options": dataset_options,
    "dataset_info": dataset_info,
}
with open(os.path.join(model_dir, "model_info.json"), "w") as fp:
    json.dump(info, fp)
    fp.close()

###############################
#            Train            #
###############################

loss_train, loss_validation, metric_validation = trainer.fit(
    epochs=training_options["n_epochs"], plot_learning=True
)
print("Training finished!")

# ########## Plot and save train history figures and .csv
plt.savefig(os.path.join(model_dir, "Training_Loss.pdf"), format="pdf")
train_history = pd.DataFrame(
    {
        "epoch": np.arange(1, training_options["n_epochs"] + 1),
        "train_" + loss_type: loss_train,
        "val_" + loss_type: loss_validation,
        "val_" + metric_type + "_recon": metric_validation[:,0],
        "val_" + metric_type + "_pred": metric_validation[:,1],
        "val_" + metric_type + "_lin": metric_validation[:,2],
    }
)
train_history.to_csv(os.path.join(model_dir, "Train_Results.csv"), sep=",", index=False)

# ########## Save the model
trainer.save_model()

###############################
#   Predict on the test data  #
###############################

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
    Xp_pred_list = []  ## predicted value of x_plus from decoding the iterated Koopman model
    x0 = x[:, 0, :]
    y0 = model.encoder(x0)
    for j in range(dataset_options["seq_length"] - 1):
        y_adv, omega = model.koopman(y0, u[:,j,:])
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

# if dataset_options["do_nomalization"]:
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
    "metric_valid_recon": metric_validation[-1,0],
    "metric_valid_pred": metric_validation[-1,1],
    "metric_valid_lin": metric_validation[-1,2],
    "error_test_pred": avg_error_X.tolist(),
    "error_test_linear": avg_error_Y.tolist(),
}

with open(os.path.join(model_dir, "model_info.json"), "r+") as fp:
    info = json.load(fp)
    info["train_history"] = train_errors
    fp.seek(0)
    json.dump(info, fp)
    fp.close()

# ########## Plot and save prediction results figures
error_list = [error_X, error_Y]
fig_1 = plot_error(error_list, labels=["x", "y"])
plt.savefig(os.path.join(model_dir, "Prediction_Error.pdf"), format="pdf")
fig_2 = plot_states(Xp, Xp_pred, Yp, Yp_pred)
plt.savefig(os.path.join(model_dir, "Prediction_Trend.pdf"), format="pdf")

print("Done!")
