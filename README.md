<div align="center">

# Deep Koopman Cathter Model

</div>

## Introduction
A deeplearning based Koopman approach for modeling the nonlinear dynamics of tendon driven continuum robots.
This repository contains the codes for training the model networks usign Pytroch.
This project aims to train a dynamical model of tendon-driven continuum robots using the deep Koopman approach. The code is organized into several modules to handle different aspects of the model, including data loading, model architecture, training, and prediction.

![Diagram](./figures/Diagram.png)

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Training the Model](#training-the-model)
4. [Making Predictions](#making-predictions)
5. [Project Structure](#project-structure)
6. [References](#references)


## Installation

To set up the environment and install the required dependencies, follow these steps:

1. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Configuration parameters are stored in the `config.py` file. This file includes details such as model parameters, dataset options, training options, and more.


## Training the Model

To train the model, run the `train.py` script. This script handles loading the dataset, initializing the model, and training it using the configurations specified in `config.py`.

```bash
python train.py
```

### Key Components in `train.py`:
- **Data Loading**: Loads and preprocesses the dataset.
- **Model Creation**: Initializes the model architecture.
- **Training**: Trains the model using specified loss functions, optimizers, and schedulers.

## Making Predictions

After training the model, you can make predictions using the `predict.py` script. This script loads the trained model and performs predictions on new data.

```bash
python predict.py
```

### Key Components in `predict.py`:
- **Model Loading**: Loads the trained model and its parameters.
- **Prediction**: Uses the model to make predictions on new input data.

## Project Structure

- `config.py`: Contains configuration parameters for the model, training, and dataset.
- `train.py`: Script to train the model.
- `predict.py`: Script to make predictions using the trained model.
- `trainer.py`: Contains the `Trainer` class which handles the training loop and model saving.
- `models.py`: Defines the model architectures used in the project.
- `utils.py`: Utility functions for data processing and normalization.
- `loss_functions.py`: Custom loss functions used for training the model.
- `predictor.py`: Defines the `Predictor` class for making predictions using the trained model.


## Results


![Robot state prediction](./figures/Prediction_States.png)

![Lifted state prediction](./figures/Prediction_Lifted_States.png)

![Eigen values](./figures/Prediction_Eigen.png)

![Input matrix map](./figures/Prediction_B_Matrix.png)

![Decoder map](./figures/Decoder_map.png)


## References

- Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). Deep learning for universal linear embeddings of nonlinear dynamics. *Nature communications*, 9(1), 4950.

