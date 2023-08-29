import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """
        Baseline model with a single fully connected layer.

        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
        """
        super(BaselineModel, self).__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the ReLU activation function.
        """

        x = F.relu(self.fc(x))
        return x


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Multi-Layer Perceptron (MLP) model with multiple fully connected layers.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output.
        """
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """
        Recurrent Neural Network (RNN) model with an RNN layer and a fully connected layer.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output.
            num_layers (int): The number of RNN layers. Default is 1.
        """
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out, _ = self.rnn(x)
        x = self.fc(out[:, -1, :])
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """
        LSTM model with an LSTM layer and a fully connected layer.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output.
            num_layers (int): The number of LSTM layers. Default is 1.
        """
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out, (h, c) = self.lstm(x)
        x = self.fc(out[:, -1, :])
        return x
