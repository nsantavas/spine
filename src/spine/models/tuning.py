import random
import string
from typing import Dict

import mlflow
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spine.data import datasets, utils
from spine.models import models, train


class Tuner:
    def __init__(
        self,
        directory: str,
        input_size: int,
        output_size: int,
        model_type: str = "BaselineModel",
        epochs: int = 10,
        n_trials: int = 10,
    ):
        """
        Initialize the tuner.

        Args:
            directory (str): Path to the dataset directory.
            input_size (int): Size of the input data.
            output_size (int): Size of the output data.
            model_type (str, optional): Type of the model to be used. Defaults to "BaselineModel".
            epochs (int, optional): Number of epochs for training. Defaults to 10.
            n_trials (int, optional): Number of tuning trials. Defaults to 10.
        """
        self.directory = directory
        self.input_size = input_size
        self.output_size = output_size
        self.model_type = model_type
        self.epochs = epochs
        self.n_trials = n_trials
        self.loss_function = nn.MSELoss()
        self.best_params = None

    def prepare_dataset(self, trial: optuna.Trial) -> Dataset:
        """
        Prepare dataset based on the model type.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            Dataset: The prepared dataset.
        """

        if self.model_type in ["BaselineModel", "MLP"]:
            return datasets.StaticSkeletonDataset(self.directory)

        elif self.model_type in ["RNNModel", "LSTMModel", "GRUModel"]:
            sequence_length = trial.suggest_categorical("sequence_length", [5, 6, 7, 10])
            return datasets.SequentialSkeletonDataset(
                self.directory, sequence_length=sequence_length
            )
        else:
            raise ValueError(f"Model type {self.model_type} not supported.")

    def instantiate_model(self, trial: optuna.Trial) -> nn.Module:
        """
        Create a model instance based on the model type.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            Module: The instantiated model.
        """
        if self.model_type == "BaselineModel":
            return models.BaselineModel(self.input_size, self.output_size)

        elif self.model_type == "MLP":
            hidden_size = trial.suggest_int("hidden_size", 16, 256)
            return models.MLP(self.input_size, hidden_size, self.output_size)

        elif self.model_type == "RNNModel":
            num_layers = trial.suggest_int("num_layers", 1, 3)
            hidden_size = trial.suggest_int("hidden_size", 16, 256)
            return models.RNNModel(
                self.input_size, hidden_size, self.output_size, num_layers=num_layers
            )

        elif self.model_type == "LSTMModel":
            num_layers = trial.suggest_int("num_layers", 1, 3)
            hidden_size = trial.suggest_int("hidden_size", 16, 256)
            return models.LSTMModel(
                self.input_size, hidden_size, self.output_size, num_layers=num_layers
            )

        else:
            raise ValueError(f"Model type {self.model_type} not supported.")

    def define_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Define hyperparameters for the training.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            Dict: A dictionary containing the hyperparameters.
        """

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        optimizer_selection = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

        optimizer = torch.optim.Adam if optimizer_selection == "Adam" else torch.optim.SGD

        hyperparameters = {
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": optimizer,
        }

        return hyperparameters

    def objective(self, trial: optuna.Trial) -> float:
        """
        Define the objective function for optimization.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            float: The value of the objective function.
        """
        random_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

        mlruns_dir = f"file:///{'/'.join(self.directory.split('/')[:-3])}/mlruns"
        mlflow.set_tracking_uri(mlruns_dir)

        with mlflow.start_run(run_name=f"TuningTrial_{self.model_type}_{random_string}"):
            mlflow.log_param("model_type", self.model_type)
            hyperparameters = self.define_hyperparameters(trial)
            dataset = self.prepare_dataset(trial)
            model = self.instantiate_model(trial)

            train_set, val_set, test_set = utils.split_dataset(
                dataset, train_ratio=0.7, val_ratio=0.2
            )

            train_dataloader = DataLoader(
                train_set, batch_size=hyperparameters["batch_size"], shuffle=True
            )
            val_dataloader = DataLoader(val_set, batch_size=hyperparameters["batch_size"])
            test_dataloader = DataLoader(test_set, batch_size=hyperparameters["batch_size"])
            optimizer = hyperparameters["optimizer"](model.parameters(), lr=hyperparameters["lr"])

            trainer = train.Trainer(
                model, self.loss_function, optimizer, metric_fn=nn.L1Loss(), device="cpu"
            )

            for key, value in trial.params.items():
                mlflow.log_param(key, value)

            train_val_losses = trainer.train_model(
                train_dataloader, val_dataloader, epochs=self.epochs
            )
            test_losses = trainer.test_model(test_dataloader)

            self._log_metrics(train_val_losses)
            self._log_metrics(test_losses)

            return train_val_losses["val_losses"][-1]

    def _log_metrics(self, data: Dict) -> None:
        """Log metrics

        Args:
            data: The data to log.
        """
        for key, value in data.items():
            if isinstance(value, float):
                mlflow.log_metric(key, value)
            else:
                [mlflow.log_metric(key, val) for val in value]

    def tune(self) -> None:
        """
        Perform the hyperparameter tuning.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        self.best_params = study.best_params  # type: ignore

        print("Best trial:")
        trial = study.best_trial
        print("Value: ", trial.value)
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    def get_best_params(self) -> dict:
        """
        Get the best parameters found during the tuning.

        Returns:
            dict: A dictionary containing the best parameters.
        """
        return self.best_params  # type: ignore
