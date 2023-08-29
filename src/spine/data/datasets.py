import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, directory: str):
        """
        Initialize the StaticSkeletonDataset.

        Args:
            directory (str): The path to the directory containing the data files.
        """
        self.input_features = [
            "Head",
            "ShoulderRight",
            "ShoulderLeft",
            "ElbowRight",
            "ElbowLeft",
            "HipLeft",
            "HipRight",
        ]
        self.target_features = ["Spine", "Spine1", "Spine2", "Spine3"]
        self.head_joints = ["EyeLeft", "EyeRight", "EarLeft", "EarRight", "Nose"]
        self._directory = directory

        # Calculating sorted inputs and outputs once during initialization
        self.sorted_inputs = self._get_sorted_joints(self.input_features)
        self.sorted_outputs = self._get_sorted_joints(self.target_features)

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single CSV file and return its dataframe.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataframe from the file.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _min_max_scale_tensor(
        tensor: torch.Tensor, tensor_min: torch.Tensor, tensor_max: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Min-Max scaling to a tensor using provided min and max values.

        Args:
            tensor (torch.Tensor): Tensor to be scaled.
            tensor_min (torch.Tensor): Minimum value for scaling.
            tensor_max (torch.Tensor): Maximum value for scaling.

        Returns:
            torch.Tensor: Scaled tensor.
        """
        scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return scaled_tensor

    def _get_sorted_joints(self, features: List[str]) -> List[str]:
        """
        Generate sorted joint columns based on given feature list.

        Args:
            features (List[str]): List of features to be sorted.

        Returns:
            List[str]: Sorted list of joint columns.
        """
        joints_grouped_by_limb = {
            "Head": ["Head"] if "Head" in features else [],
            "Upper Limb": [
                j
                for j in ["ShoulderRight", "ElbowRight", "ShoulderLeft", "ElbowLeft"]
                if j in features
            ],
            "Lower Limb": [j for j in ["HipRight", "HipLeft"] if j in features],
            "Spine": [j for j in ["Spine", "Spine1", "Spine2", "Spine3"] if j in features],
        }

        # Generate the sorted column list
        sorted_columns = []
        for limb, joints in joints_grouped_by_limb.items():
            for joint in joints:
                for axis in ["_x", "_y", "_z"]:
                    sorted_columns.append(joint + axis)
        return sorted_columns

    def _average_head_joints(self, data: pd.DataFrame) -> None:
        """
        Average the head joints and add the results to the dataframe.

        Args:
            data (pd.DataFrame): Dataframe to which averaged head joints will be added.
        """
        for axis in ["_x", "_y", "_z"]:
            joints = [joint + axis for joint in self.head_joints]

            # Check if all joints are present in the dataframe
            if set(joints).issubset(data.columns):
                data["Head" + axis] = data[joints].mean(axis=1)
            else:
                print(f"Warning: Missing some head joints in axis {axis}. Cannot compute average.")


class StaticSkeletonDataset(BaseDataset):
    """
    A dataset representation for static skeleton data.

    This class provides functionality to load, preprocess, and access skeleton data stored in CSV
    files within a specified directory. The data represents 3D coordinates for different skeletal
    joints. The input features are a set of predefined skeletal joints, and the target features
    represent the spine joints. All joints have X, Y, and Z coordinates.

    Attributes:
        input_features (List[str]): List of features used as input.
        target_features (List[str]): List of features used as targets.
        head_joints (List[str]): List of joints in the head.
        sorted_inputs (List[str]): Sorted list of input joint columns.
        sorted_outputs (List[str]): Sorted list of output joint columns.
        inputs_tensor (torch.Tensor): Tensor representation of input features.
        outputs_tensor (torch.Tensor): Tensor representation of target features.
        minimum (torch.Tensor): Minimum value across all data.
        maximum (torch.Tensor): Maximum value across all data.
    """

    def __init__(self, directory: str):
        """
        Initialize the StaticSkeletonDataset.

        Args:
            directory (str): The path to the directory containing the data files.
        """
        super().__init__(directory)
        # Load and process data
        self._df = self._load_and_process_data()

        # Convert dataframe to tensors
        self.inputs_tensor = torch.tensor(self._df[self.sorted_inputs].values, dtype=torch.float32)
        self.outputs_tensor = torch.tensor(
            self._df[self.sorted_outputs].values, dtype=torch.float32
        )

        self.minimum, self.maximum = torch.aminmax(
            torch.cat([self.inputs_tensor.flatten(), self.outputs_tensor.flatten()])
        )
        self.inputs_tensor = self._min_max_scale_tensor(
            self.inputs_tensor, self.minimum, self.maximum
        )
        self.outputs_tensor = self._min_max_scale_tensor(
            self.outputs_tensor, self.minimum, self.maximum
        )

    def _load_and_process_data(self) -> pd.DataFrame:
        """
        Load, process, and concatenate all dataframes in the directory.

        Returns:
            pd.DataFrame: Concatenated and processed dataframe.
        """
        # Load all CSV files in the directory
        dataframes = [
            self._load_file(os.path.join(self._directory, filename))
            for filename in os.listdir(self._directory)
            if filename.endswith(".csv")
        ]

        # Concatenate all loaded dataframes
        concatenated_data = pd.concat(dataframes, ignore_index=True)

        # Average the head joints
        self._average_head_joints(concatenated_data)

        # Keep only the required columns
        keep_columns = (
            [n + "_x" for n in self.input_features + self.target_features]
            + [n + "_y" for n in self.input_features + self.target_features]
            + [n + "_z" for n in self.input_features + self.target_features]
        )

        return concatenated_data[keep_columns]

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors for the given index.
        """
        return self.inputs_tensor[idx], self.outputs_tensor[idx]

    def __repr__(self) -> str:
        """
        String representation of the StaticSkeletonDataset.

        Returns:
            str: String representation.
        """
        return (
            f"StaticSkeletonDataset(\n"
            f"  Directory: {self._directory}\n"
            f"  Number of samples: {len(self._df)}\n"
            f"  Input Features: {self.input_features}\n"
            f"  Target Features: {self.target_features}\n"
            f"  Sorted Inputs: {self.sorted_inputs}\n"
            f"  Sorted Outputs: {self.sorted_outputs}\n"
            f"  Minimum value: {self.minimum}\n"
            f"  Maximum value: {self.maximum}\n"
            f")"
        )


class SequentialSkeletonDataset(BaseDataset):
    def __init__(self, directory: str, sequence_length: int = 5):
        """
        Initialize the StaticSkeletonDataset.

        Args:
            directory (str): The path to the directory containing the data files.
        """
        super().__init__(directory)

        self.sequence_length = sequence_length

        # Calculating sorted inputs and outputs once during initialization
        self.sorted_inputs = self._get_sorted_joints(self.input_features)
        self.sorted_outputs = self._get_sorted_joints(self.target_features)

        # Load and process data
        self._df = self._load_and_process_data()
        self.minimum = pd.concat(self._df, ignore_index=True).to_numpy().min()
        self.maximum = pd.concat(self._df, ignore_index=True).to_numpy().max()

        self._sequences = self.create_dataframe_sequences(self._df, self.sequence_length)

    def _load_and_process_data(self) -> pd.DataFrame:
        """
        Load, process, and concatenate all dataframes in the directory.

        Returns:
            pd.DataFrame: Concatenated and processed dataframe.
        """
        # Load all CSV files in the directory
        dataframes = [
            self._load_file(os.path.join(self._directory, filename))
            for filename in os.listdir(self._directory)
            if filename.endswith(".csv")
        ]

        keep_columns = (
            [n + "_x" for n in self.input_features + self.target_features]
            + [n + "_y" for n in self.input_features + self.target_features]
            + [n + "_z" for n in self.input_features + self.target_features]
        )

        for idx in range(len(dataframes)):
            self._average_head_joints(dataframes[idx])
            dataframes[idx] = dataframes[idx][keep_columns]

        return dataframes

    def create_dataframe_sequences(
        self, dataframes: List[pd.DataFrame], sequence_length: int = 5
    ) -> List[pd.DataFrame]:
        all_sequences = []

        for df in dataframes:
            for idx in range(len(df) - sequence_length):
                sequence_df = df.iloc[idx : idx + sequence_length].reset_index(drop=True)  # noqa
                all_sequences.append(sequence_df)

        return all_sequences

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors for the given index.
        """
        inputs = torch.tensor(
            self._sequences[idx][self.sorted_inputs].iloc[: self.sequence_length].values
        )
        outputs = torch.tensor(self._sequences[idx][self.sorted_outputs].iloc[-1])

        inputs = self._min_max_scale_tensor(inputs, self.minimum, self.maximum).to(torch.float32)
        outputs = self._min_max_scale_tensor(outputs, self.minimum, self.maximum).to(torch.float32)

        return inputs, outputs

    def __repr__(self) -> str:
        return "SequentialSkeletonDataset(\n"
        f"  Directory: {self._directory}\n"
        f"  Number of samples: {len(self._sequences)}\n"
        f"  Input Features: {self.input_features}\n"
        f"  Target Features: {self.target_features}\n"
        f"  Sorted Inputs: {self.sorted_inputs}\n"
        f"  Sorted Outputs: {self.sorted_outputs}\n"
        f"  Minimum value: {self.minimum}\n"
        f"  Maximum value: {self.maximum}\n"
        ")"
