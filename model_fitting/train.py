import os, sys, json
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.getcwd())
from trainer import Trainer
from model_fitting.utils import (
    MLPDataset,
    init_tn_weights,
    init_dl_weights,
    init_he_weights,
    inverse_normalize,
)

from model_fitting.loss_functions import (
    DeepKoopmanExplicitLoss,
    DeepKoopmanExplicitMetric,
    WeightedMseLoss,
    MaeMetric,
)

from model_fitting.models import (
    MLP,
    DoubleMLP,
    DelayedMLP,
    States_Auxiliary,
    Input_Auxiliary,
    TrainableA, 
    TrainableB,
    LinearKoopmanWithInput,
    VariableKoopman,
    CombinedDeepKoopman,
    CombinedKoopman_withAux_withoutInput,
    CombinedKoopman_withAux_withInput,
    CombinedDeepKoopman_without_decoder,
)

from config import (
    MODEL_DETAILS,
    dataset_options,
    training_options,
    koopman_params,
    encoder_params,
    decoder_params,
    inputs_params,
    DEVICE,
    model_dir,
)

logger.info(" Running on %s", DEVICE)


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
        dataset_options["sample_time_new"] = (dataset_info["sample_time"] * dataset_options["sample_step"])
        koopman_params["sample_time"] = dataset_options["sample_time_new"]
        encoder_params["sample_time"] = dataset_options["sample_time_new"]
    else:
        logger.error(" dataset_info.json does not exist in the expected address")
        raise FileNotFoundError(f"{dataset_info_path} ")

    # Load normalizer_params from the pretrained Model
    pretrained_model_name = training_options["transfer_from"]
    normalizer_params = load_normalizer_params(pretrained_model_name)

    # Load Dataset
    normalizer_params, dataset_train, dataset_val, dataset_test = load_datasets(
        datasets_dir, dataset_options, normalizer_params
    )
    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=training_options["batch_size"], shuffle=False
    )
    dataloader_valid = DataLoader(
        dataset=dataset_val, batch_size=training_options["batch_size"], shuffle=False
    )
    # dataloader_test = DataLoader(dataset=dataset_test, shuffle=False)

    # Create the model structure
    model = create_model(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        koopman_params=koopman_params,
        inputs_params=inputs_params,
    )

    # Initialize model weights
    model = initialize_weights(model, training_options)

    # Loss Function, Optimizer, and Learning Rate Scheduler
    loss_function = DeepKoopmanExplicitLoss(
        alpha_1=training_options["lossfunc_weights"][0],
        alpha_2=training_options["lossfunc_weights"][1],
        alpha_reg=training_options["lossfunc_weights"][2],
        weights_x=torch.tensor(training_options["weights_x"]).to(DEVICE),
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
        "decoder_parameters": decoder_params,
        "koopman_parameters": koopman_params.copy(),
        "inputs_parameters": inputs_params.copy(),
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
        normalizer_scale=(
            normalizer_params["params_y"]["max"][0]
            - normalizer_params["params_y"]["min"][0]
        ),
        sequence_length=dataset_options["seq_length"],
    )

    # Train the model
    loss_train, loss_validation, metric_validation = trainer.fit(
        epochs=training_options["n_epochs"], plot_learning_history=True
    )
    logger.info(" Training finished!")

    # Save the trained model parameters
    trainer.save_model()

    save_train_history(
        model_dir,
        loss_type,
        metric_type,
        loss_train,
        loss_validation,
        metric_validation,
    )


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

    return normalizer_params, dataset_train, dataset_val, dataset_test


def create_model(
    encoder_params,
    decoder_params,
    koopman_params,
    inputs_params,
):
    """
    Defining the architecture and instantiating the components of the model
    """
    logger = logging.getLogger(f"{__name__}.create_model")

    # Initialize encoder
    if encoder_params["architecture"] == "MLP":
        encoder = MLP(
            input_size=encoder_params["state_size"],
            hidden_size=encoder_params["hidden_size"],
            num_layers=encoder_params["num_layers"],
            output_size=encoder_params["output_size"],
            extend_states=encoder_params["extend_lifted_state"],
            activation=encoder_params["activation"],
            input_normalizer_params=None,
        ).to(DEVICE)
    elif encoder_params["architecture"] == "DualMLP":
        encoder = DoubleMLP(
            input_size=encoder_params["state_size"],
            hidden_size=encoder_params["hidden_size"],
            num_layers=encoder_params["num_layers"],
            output_size=encoder_params["output_size"],
            activation=encoder_params["activation"],
            input_normalizer_params=None,
        ).to(DEVICE)
    elif encoder_params["architecture"] == "DelayedMLP":
        encoder = DelayedMLP(
            input_size=encoder_params["state_size"],
            hidden_size=encoder_params["hidden_size"],
            num_layers=encoder_params["num_layers"],
            len_history=encoder_params["len_history"],
            output_size=encoder_params["output_size"],
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
    if decoder_params is not None:
        if decoder_params["architecture"] == "MLP":
            decoder = MLP(
                input_size=decoder_params["lifted_state_size"],
                hidden_size=decoder_params["hidden_size"],
                num_layers=decoder_params["num_layers"],
                output_size=decoder_params["state_size"],
                extend_states=encoder_params["extend_lifted_state"],
                activation=decoder_params["activation"],
                output_normalizer_params=None,
            ).to(DEVICE)
        elif decoder_params["architecture"] == "DelayedMLP":
            decoder = DelayedMLP(
                input_size=decoder_params["lifted_state_size"],
                hidden_size=decoder_params["hidden_size"],
                num_layers=decoder_params["num_layers"],
                len_history=decoder_params["len_history"],
                output_size=decoder_params["state_size"],
                activation=decoder_params["activation"],
                input_normalizer_params=None,
            ).to(DEVICE)
        else:
            logger.error(
                "Unsupported decoder architecture: %s", decoder_params["architecture"]
            )
            raise ValueError(
                f"Unsupported decoder architecture: {decoder_params['architecture']}"
            )

    # Initialize Koopman operator
    if koopman_params["space_varying"]:
        koopman_operator = VariableKoopman(
            num_complexeigens_pairs=koopman_params["num_complexeigens_pairs"],
            num_realeigens=koopman_params["num_realeigens"],
            sample_time=koopman_params["sample_time"],
            structure=koopman_params["structure"],  # "Jordan", "Controlable"
        ).to(DEVICE)
        states_auxiliary = States_Auxiliary(
            num_complexeigens_pairs=koopman_params["auxiliary_params"]["num_complexeigens_pairs"],
            num_realeigens=koopman_params["auxiliary_params"]["num_realeigens"],
            hidden_sizes=koopman_params["auxiliary_params"]["hidden_sizes"],
            activation=koopman_params["auxiliary_params"]["activation"],
        ).to(DEVICE)
        inputs_auxiliary = Input_Auxiliary(
            input_size=inputs_params["inputs_auxiliary_params"]["lifted_state_size"],
            hidden_sizes=inputs_params["inputs_auxiliary_params"]["hidden_sizes"],
            output_shape=inputs_params["inputs_auxiliary_params"]["output_shape"],
            activation=inputs_params["inputs_auxiliary_params"]["activation"],
        ).to(DEVICE)
    else:
        states_matrix = TrainableA(
            states_dim=koopman_params["lifted_state_size"],
        ).to(DEVICE)
        actuation_matrix = TrainableB(
            states_dim=koopman_params["lifted_state_size"],
            inputs_dim=inputs_params["system_input_size"],
        ).to(DEVICE)


    # Combine all components into the final model
    if koopman_params["space_varying"]:
        model = CombinedKoopman_withAux_withInput(
            encoder, states_auxiliary, koopman_operator, decoder, inputs_auxiliary
        )
    else:
        if decoder_params is None:
            model = CombinedDeepKoopman_without_decoder(
                encoder=encoder, states_matrix=states_matrix, actuation_matrix=actuation_matrix
            )
        else:
            model = CombinedDeepKoopman(
                encoder=encoder, states_matrix=states_matrix, actuation_matrix=actuation_matrix, decoder=decoder
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
        logger.info(" All model components initiated with random weights")

    # load pretrained network wegihts and biases
    else:
        pretrained_model_dir = os.path.join(
            os.getcwd(), "trained_models", training_options["transfer_from"]
        )
        koopman_dict = torch.load(
            os.path.join(pretrained_model_dir, "trained_model", "koopman.pt"),
            map_location=torch.device("cpu"),
        )
        model.koopman.load_state_dict(koopman_dict["state_dict"])

        ## auto encoder model weights
        encoder_file_path = os.path.join(
            pretrained_model_dir, "trained_model", "encoder.pt"
        )
        decoder_file_path = os.path.join(
            pretrained_model_dir, "trained_model", "decoder.pt"
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
            pretrained_model_dir, "trained_model", "states_auxiliary.pt"
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
            pretrained_model_dir, "trained_model", "inputs_auxiliary.pt"
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
        "decoder_params": model_parameters["decoder_parameters"],
        "koopman_params": model_parameters["koopman_parameters"].copy(),
        "inputs_params": model_parameters["inputs_parameters"].copy(),
        "normalizer_params": normalizer_params.copy(),
        "training_options": training_options.copy(),
        "dataset_options": dataset_options.copy(),
        "dataset_info": dataset_info.copy(),
    }
    with open(os.path.join(model_dir, "model_info.json"), "w") as fp:
        json.dump(info, fp)


def save_train_history(
    model_dir, loss_type, metric_type, loss_train, loss_validation, metric_validation
):
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
        os.path.join(model_dir, "training_history.csv"), sep=",", index=False
    )


if __name__ == "__main__":
    main()
