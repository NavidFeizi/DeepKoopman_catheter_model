import os, sys, json
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

sys.path.append(os.getcwd())
from trainer import Trainer
from model_fitting.utils import (
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
    States_Auxiliary,
    Input_Auxiliary,
    LinearKoopman,
    LinearKoopmanWithInput,
    TrainableKoopmanDynamics,
    CombinedDeepKoopman,
    CombinedKoopman_withAux_withoutInput,
    CombinedKoopman_withAux_withInput,
)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(" Running on %s", DEVICE)


# Paramters
MODEL_DETAILS = {
    "model_name": "catheter_1T_forced_3.1.3",
    "desc": "one tendon with negative - unforced - mlp encoder - adjust encoder size",
}

# Dataset related parameters
dataset_options = {
    "dataset_name": "catheter_1T_forced_3.1",
    # "dataset_name": "catheter_1T_unforced_2.1",
    "add_sequence_dimension": True,  # True for TCN and LSTM
    "fit_normalization": None,  #'X', 'U', 'All',   'X' for unforced system - None for forced system to use the unforced params
    "prenormalization": True,
    "sample_step": 1,
    "sample_time": None,
}

# Training options
training_options = {
    # "transfer_from": None,
    "transfer_from": "catheter_1T_unforced_3.1",
    "transfer_input_aux": False,
    "transfer_states_aux": True,
    "transfer_auto_encoder": True,
    "n_epochs": 30,
    "n_recon_epoch": 00,  # number of first epocs only with reconstruction loss
    "learning_rate": 4e-4,
    "decay_rate": 0.9988,
    "batch_size": 450,
    "lossfunc_weights": ([1e-1, 1e-7, 1e-13]),
    "num_pred_steps": 60,
    "train_encoder": True,
    "train_decoder": True,
    "train_states_auxiliary": False,
    "train_inputs_auxiliary": True,
    "cuda": True,
}

# Koopman operator parameters
koopman_params = {
    "num_complexeigens_pairs": 1,
    "num_realeigens": 0,
    "sample_time": None,  # loads from the dataste options
    "structure": "Jordan",  # "Jordan" or "Controlable"
}

# Encoder parameters
encoder_params = {
    "architecture": "MLP",  # DoubleMLP, MLP
    "state_size": 4,
    "hidden_sizes": ([128, 128, 128, 128, 128]),
    # "hidden_sizes": ([64, 64, 64, 64, 64]),
    # "hidden_sizes": ([80, 80, 80, 80]),
    "lifted_state_size": koopman_params["num_realeigens"]
    + 2 * koopman_params["num_complexeigens_pairs"],
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
    "sample_time_new": None,
}

# Decoder paramters
decoder_params = encoder_params.copy()
decoder_params["hidden_sizes"] = [128, 128, 128, 128, 128]
decoder_params["architecture"] = "MLP"  # MLP or DualMLP

# model params for the auxiliary network to estimate the eigen values of the A matrix
states_auxiliary_params = {
    "architecture": "MLP",
    "hidden_sizes": ([30, 30, 30]),
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
}

# model params for the auxiliary network to estimate the B matrix
inputs_auxiliary_params = {
    "architecture": "MLP",
    "system_input_size": 1,
    "hidden_sizes": ([128, 128, 128, 128]),
    "activation": "ReLU",  # ReLU, Tanh, Sigmoid, LeakyReLU ...
}

# driven paramters
states_auxiliary_params["num_complexeigens_pairs"] = koopman_params[
    "num_complexeigens_pairs"
]
states_auxiliary_params["num_realeigens"] = koopman_params["num_realeigens"]
states_auxiliary_params["state_size"] = encoder_params["lifted_state_size"]
states_auxiliary_params["output_size"] = encoder_params["lifted_state_size"]
inputs_auxiliary_params["lifted_state_size"] = encoder_params["lifted_state_size"]
inputs_auxiliary_params["output_shape"] = [
    encoder_params["lifted_state_size"],
    # encoder_params["state_size"],         # this is for when you want to train B instead of B_phi. Other parts of the code must be corrected accordingly
    inputs_auxiliary_params["system_input_size"],
]

# Create the model saving folder
model_dir = os.path.join(os.getcwd(), "trained_models", MODEL_DETAILS["model_name"])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


####################################################################################
def main():
    datasets_dir = os.path.join(
        os.getcwd(), "datasets", dataset_options["dataset_name"]
    )
    dataset_info_path = os.path.join(datasets_dir, "dataset_info.json")

    # Load dataset info JSON file
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
        logger.error(" dataset_info.json does not exist in the expected address")
        raise FileNotFoundError(f"{dataset_info_path} ")

    # Load normalizer_params from the pretrained Model
    pretrained_model_name = training_options["transfer_from"]
    normalizer_params = load_normalizer_params(pretrained_model_name)

    # Load Dataset
    dataset_train, dataset_val, dataset_test = load_datasets(
        datasets_dir, dataset_options, normalizer_params
    )
    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=training_options["batch_size"], shuffle=False
    )
    dataloader_valid = DataLoader(
        dataset=dataset_val, batch_size=training_options["batch_size"], shuffle=False
    )
    dataloader_test = DataLoader(dataset=dataset_test, shuffle=False)

    # Create the model structure
    model = create_model(
        encoder_params,
        decoder_params,
        states_auxiliary_params,
        inputs_auxiliary_params,
        koopman_params,
    )

    # Initialize model weights
    model = initialize_weights(model, training_options)

    # Loss Function, Optimizer, and Learning Rate Scheduler
    loss_function = DeepKoopmanExplicitLoss(
        alpha1=training_options["lossfunc_weights"][0],
        alpha2=training_options["lossfunc_weights"][1],
        alpha_reg=training_options["lossfunc_weights"][2],
    )
    loss_type = "MSE"
    metric_function = DeepKoopmanExplicitMetric()
    metric_type = "MAE"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_options["learning_rate"]
    )
    lr_scheduler = ExponentialLR(
        optimizer=optimizer, gamma=training_options["decay_rate"]
    )

    # Save model info before training
    model_parameters = {
        "model_details": MODEL_DETAILS.copy(),
        "encoder_parameters": encoder_params.copy(),
        "decoder_parameters": decoder_params.copy(),
        "states_auxiliary_parameters": states_auxiliary_params.copy(),
        "koopman_parameters": koopman_params.copy(),
        "inputs_auxiliary_parameters": inputs_auxiliary_params.copy(),
    }
    save_model_info(
        model_dir,
        model_parameters,
        normalizer_params,
        training_options,
        dataset_options,
        dataset_info,
    )

    # setup trainer object
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

    # Train the model
    loss_train, loss_validation, metric_validation = trainer.fit(
        epochs=training_options["n_epochs"], plot_learning=True
    )
    logger.info(" Training finished!")

    # Save the trained model parameters
    trainer.save_model()

    save_train_history(model_dir, loss_type, metric_type, loss_train, loss_validation, metric_validation)


def load_normalizer_params(model_name):
    """
    Load the pretrained model information and normalizer parameters.
    """
    logger = logging.getLogger(f"{__name__}.load_normalizer_params")
    normalizer_params = None
    if model_name is not None:
        pretrained_model_dir = os.path.join(os.getcwd(), "trained_models", model_name)
        pretrained_model_info_path = os.path.join(
            pretrained_model_dir, "model_info.json"
        )
        if os.path.isfile(pretrained_model_info_path):
            with open(pretrained_model_info_path, "r") as fp:
                model_info = json.load(fp)
            if "normalizer_params" not in model_info:
                raise KeyError("'normalizer_params' not found in 'model_info'")
            else:
                normalizer_params = model_info["normalizer_params"]
        else:
            logger.error(
                " model_info.json does not exist in the expected address to load normalize params"
            )
            raise FileNotFoundError(f"{pretrained_model_info_path} does not exist!")
    return normalizer_params


def load_datasets(datasets_dir, dataset_options, normalizer_params):
    """
    Load datasets for training, validation, and testing.

    Parameters:
    - datasets_dir: Directory where the datasets are stored.
    - dataset_options: Options for the dataset.
    - normalizer_params: Parameters for normalizing the data.

    Returns:
    - dataset_train: Training dataset.
    - dataset_val: Validation dataset.
    - dataset_test: Testing dataset.
    """
    logger = logging.getLogger(f"{__name__}.load_datasets")

    logger.info(" Loading datasets ...")
    dataset_train = MLPDataset(
        datasets_dir=datasets_dir,
        filename="train.txt",
        seq_length=dataset_options["seq_length"],
        add_dim=dataset_options["add_sequence_dimension"],
        fit_normalize=dataset_options["fit_normalization"],
        prenomalize=dataset_options["prenormalization"],
        scaler_params=normalizer_params,
        sample_step=dataset_options["sample_step"],
    )
    logger.info(" Loading train datasets finished")

    normalizer_params = dataset_train.get_normalizer_params()

    dataset_val = MLPDataset(
        datasets_dir=datasets_dir,
        filename="val.txt",
        seq_length=dataset_options["seq_length"],
        add_dim=dataset_options["add_sequence_dimension"],
        fit_normalize=None,
        prenomalize=dataset_options["prenormalization"],
        scaler_params=normalizer_params,
        sample_step=dataset_options["sample_step"],
    )
    logger.info(" Loading validation datasets finished")

    dataset_test = MLPDataset(
        datasets_dir=datasets_dir,
        filename="test.txt",
        seq_length=dataset_options["test_seq_length"],
        add_dim=dataset_options["add_sequence_dimension"],
        fit_normalize=None,
        prenomalize=dataset_options["prenormalization"],
        scaler_params=normalizer_params,
        sample_step=dataset_options["sample_step"],
    )
    logger.info(" Loading test datasets finished")

    return dataset_train, dataset_val, dataset_test


def create_model(
    encoder_params,
    decoder_params,
    states_auxiliary_params,
    inputs_auxiliary_params,
    koopman_params,
):
    """
    Defining the architecture and instantiating the components of the model
    """
    logger = logging.getLogger(f"{__name__}.create_model")

    # Initialize encoder
    if encoder_params["architecture"] == "MLP":
        encoder = MLP(
            input_size=encoder_params["state_size"],
            hidden_sizes=encoder_params["hidden_sizes"],
            output_size=encoder_params["lifted_state_size"],
            activation=encoder_params["activation"],
            input_normalizer_params=None,
        ).to(DEVICE)
    elif encoder_params["architecture"] == "DualMLP":
        encoder = DoubleMLP(
            input_size=encoder_params["state_size"],
            hidden_sizes=encoder_params["hidden_sizes"],
            output_size=encoder_params["lifted_state_size"],
            activation=encoder_params["activation"],
            input_normalizer_params=None,
        ).to(DEVICE)
    else:
        logger.error(
            "Unsupported encoder architecture: %s", encoder_params["architecture"]
        )
        raise ValueError(
            f"Unsupported encoder architecture: {encoder_params['architecture']}"
        )

    # Initialize decoder
    decoder = MLP(
        input_size=decoder_params["lifted_state_size"],
        hidden_sizes=decoder_params["hidden_sizes"],
        output_size=decoder_params["state_size"],
        activation=decoder_params["activation"],
        output_normalizer_params=None,
    ).to(DEVICE)

    # Initialize states auxiliary network
    states_auxiliary = States_Auxiliary(
        num_complexeigens_pairs=states_auxiliary_params["num_complexeigens_pairs"],
        num_realeigens=states_auxiliary_params["num_realeigens"],
        hidden_sizes=states_auxiliary_params["hidden_sizes"],
        activation=states_auxiliary_params["activation"],
    ).to(DEVICE)

    # Initialize inputs auxiliary network
    inputs_auxiliary = Input_Auxiliary(
        input_size=inputs_auxiliary_params["lifted_state_size"],
        hidden_sizes=inputs_auxiliary_params["hidden_sizes"],
        output_shape=inputs_auxiliary_params["output_shape"],
        activation=inputs_auxiliary_params["activation"],
    ).to(DEVICE)

    # Initialize Koopman operator
    koopman_operator = TrainableKoopmanDynamics(
        num_complexeigens_pairs=koopman_params["num_complexeigens_pairs"],
        num_realeigens=koopman_params["num_realeigens"],
        sample_time=koopman_params["sample_time"],
        structure=koopman_params["structure"],  # "Jordan", "Controlable"
    ).to(DEVICE)

    # Combine all components into the final model
    model = CombinedKoopman_withAux_withInput(
        encoder, states_auxiliary, koopman_operator, decoder, inputs_auxiliary
    )

    logger.info(" Model created successfully")
    return model


def initialize_weights(model, training_options):
    """
    Initialize weights and biases of the model, either randomly or from pretrained models.

    Parameters:
    - model: The model to initialize.
    - training_options: Options for training the model.

    Returns:
    - model: The model with initialized weights and biases.
    """
    logger = logging.getLogger(f"{__name__}.initialize_weights")

    if training_options["transfer_from"] is None:
        model.apply(init_tn_weights)

    # load pretrained network wegihts and biases
    else:
        pretrained_model_dir = os.path.join(
            os.getcwd(), "trained_models", training_options["transfer_from"]
        )
        koopman_dict = torch.load(
            os.path.join(
                pretrained_model_dir, "trained_model", "TorchTrained_Koopman.pt"
            ),
            map_location=torch.device("cpu"),
        )
        model.koopman.load_state_dict(koopman_dict["state_dict"])

        ## auto encoder model weights
        encoder_file_path = os.path.join(
            pretrained_model_dir, "trained_model", "TorchTrained_Encoder.pt"
        )
        decoder_file_path = os.path.join(
            pretrained_model_dir, "trained_model", "TorchTrained_Decoder.pt"
        )
        if (
            os.path.exists(encoder_file_path)
            and os.path.exists(decoder_file_path)
            and training_options["transfer_auto_encoder"]
        ):
            encoder_dict = torch.load(
                encoder_file_path, map_location=torch.device("cpu")
            )
            decoder_dict = torch.load(
                decoder_file_path, map_location=torch.device("cpu")
            )
            model.encoder.load_state_dict(encoder_dict["state_dict"])
            model.decoder.load_state_dict(decoder_dict["state_dict"])
            logger.info(
                " Auto encoder initiated from '%s'", training_options["transfer_from"]
            )
        else:
            model.encoder.apply(init_tn_weights)
            model.decoder.apply(init_tn_weights)
            logger.info(" Auto encoder initiated with random weights")

        ## states aux model weights
        file_path = os.path.join(
            pretrained_model_dir, "trained_model", "TorchTrained_states_auxiliary.pt"
        )
        if os.path.exists(file_path) and training_options["transfer_states_aux"]:
            states_auxiliary_dict = torch.load(
                file_path, map_location=torch.device("cpu")
            )
            model.states_auxiliary.load_state_dict(states_auxiliary_dict["state_dict"])
            logger.info(
                " states_auxiliary initiated from '%s'",
                training_options["transfer_from"],
            )
        else:
            model.inputs_auxiliary.apply(init_tn_weights)
            logger.info(" states_auxiliary initiated with random weights")

        ## check for pretrained inputs aux model weights
        file_path = os.path.join(
            pretrained_model_dir, "trained_model", "TorchTrained_Inputs_auxiliary.pt"
        )
        if os.path.exists(file_path) and training_options["transfer_input_aux"]:
            inputs_auxiliary_dict = torch.load(
                file_path, map_location=torch.device("cpu")
            )
            model.inputs_auxiliary.load_state_dict(inputs_auxiliary_dict["state_dict"])
            logger.info(
                " Inputs_auxiliary initiated from '%s'",
                training_options["transfer_from"],
            )
        else:
            logger.info(" Inputs_auxiliary initiated with random parameters")
            model.inputs_auxiliary.apply(init_tn_weights)

    return model


def save_model_info(
    model_dir,
    model_parameters,
    normalizer_params,
    training_options,
    dataset_options,
    dataset_info,
):
    """
    Save model information to a JSON file.
    """
    info = {
        "encoder_params": model_parameters["encoder_parameters"].copy(),
        "decoder_params": model_parameters["decoder_parameters"].copy(),
        "states_auxiliary_params": model_parameters["states_auxiliary_parameters"].copy(),
        "koopman_params": model_parameters["koopman_parameters"].copy(),
        "inputs_auxiliary_params": model_parameters["inputs_auxiliary_parameters"].copy(),
        "normalizer_params": normalizer_params.copy(),
        "training_options": training_options.copy(),
        "dataset_options": dataset_options.copy(),
        "dataset_info": dataset_info.copy(),
    }
    with open(os.path.join(model_dir, "model_info.json"), "w") as fp:
        json.dump(info, fp)


def save_train_history(model_dir, loss_type, metric_type, loss_train, loss_validation, metric_validation):
        # Plot and save train history figures and CSV
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
        train_history.to_csv(
            os.path.join(model_dir, "Train_Results.csv"), sep=",", index=False
        )

if __name__ == "__main__":
    main()
