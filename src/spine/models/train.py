from typing import Callable, List, Optional, Tuple

import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: _Loss,
        optimizer: Optimizer,
        device: str = "cuda",
        metric_fn: Optional[Callable] = None,
    ):
        """
        Initialize the Trainer class.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            loss_function (_Loss): The loss function used for training.
            optimizer (Optimizer): The optimization algorithm.
            device (str): The device to which model and data should be moved before computation.
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

        self.metric_fn = metric_fn
        self.train_metric_values: List[float] = []
        self.val_metric_values: List[float] = []

    def train_model(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model for a given number of epochs.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs to train for.

        Returns:
            tuple: A tuple containing lists of training and validation losses for each epoch.
        """
        train_losses = []
        val_losses = []
        self.model.to(self.device)

        progress_bar = tqdm(range(epochs), position=0, leave=True)
        for epoch in progress_bar:
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._evaluate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_metric = self._compute_metric(train_loader) if self.metric_fn else None
            val_metric = self._compute_metric(val_loader) if self.metric_fn else None

            if train_metric and val_metric:
                self.train_metric_values.append(train_metric)
                self.val_metric_values.append(val_metric)

            # Update tqdm progress bar
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"  # noqa: E501
            )

        return train_losses, val_losses

    def test_model(self, test_loader: DataLoader) -> float:
        """
        Evaluate the model on the test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            float: The test loss.
        """
        test_loss = self._evaluate(test_loader)
        tqdm.write(f"Test Loss: {test_loss:.4f}")
        return test_loss

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            loader (DataLoader): DataLoader for the epoch's data.

        Returns:
            float: The training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = self._move_to_device(inputs, targets)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader) -> float:
        """
        Evaluate the model on a given dataset.

        Args:
            loader (DataLoader): DataLoader for the data to be evaluated.

        Returns:
            float: The evaluation loss.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = self._move_to_device(inputs, targets)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(loader)

    def _move_to_device(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Move the inputs and targets to the specified device.

        Args:
            inputs (torch.Tensor): Model input tensors.
            targets (torch.Tensor): Target tensors.

        Returns:
            tuple: A tuple containing the input and target tensors moved to the desired device.
        """
        return inputs.to(self.device), targets.to(self.device)

    def _compute_metric(self, loader: DataLoader) -> float:
        """
        Compute the metric for a given dataset.

        Args:
            loader (DataLoader): DataLoader for the data to be evaluated.

        Returns:
            float: The computed metric.
        """
        if not self.metric_fn:
            raise ValueError("No metric function specified.")

        self.model.eval()
        total_metric = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = self._move_to_device(inputs, targets)
                outputs = self.model(inputs)
                metric_val = self.metric_fn(outputs, targets)
                total_metric += metric_val

        return total_metric / len(loader)
