import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(data):
    """
    Prepare the data for analysis by splitting the combined columns into separate columns.
    """
    for column in data.columns[1:]:
        data[[f"{column}_x", f"{column}_y", f"{column}_z"]] = (
            data[column].str.split(";", expand=True).astype(float)
        )

    # Drop the original combined columns
    data = data.drop(columns=np.concatenate([data.columns[1:35], data.columns[-3:]]))

    return data


def extract_coordinates(coord_string: str) -> Tuple[float, ...]:
    """
    Extract x, y, z coordinates from a semicolon-separated string.

    Args:
        coord_string (str): The string containing x, y, z coordinates separated by semicolons.

    Returns:
        Tuple[float, float, float]: The extracted x, y, z coordinates as floats.
    """
    return tuple(map(float, coord_string.split(";")))


class AnimationFileProcessor:
    """
    Processes animation files to split and recenter coordinates.

    This class is designed to iterate through a source directory of CSV files,
    split each joint's coordinates into x, y, and z, recenter the coordinates based
    on the `PlayerPosition`, and save the processed data to a target directory.

    Attributes:
        source_directory (str): The directory containing the source CSV files.
        target_directory (str): The directory where processed CSV files will be saved.
    """

    def __init__(self, source_directory: str, target_directory: str):
        """
        Initializes the AnimationFileProcessor with source and target directories.

        Args:
            source_directory (str): The directory containing the source CSV files.
            target_directory (str): The directory where processed CSV files will be saved.
        """
        self.source_directory = source_directory
        self.target_directory = target_directory
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

    def process_files(self) -> None:
        """
        Processes all CSV files in the source directory and saves them to the target directory.

        Iterates through each CSV file in the source directory, splits and recenters
        the joint's coordinates, and saves the processed data to the target directory.
        """
        for file_name in os.listdir(self.source_directory):
            if file_name.endswith(".csv"):
                file_path = os.path.join(self.source_directory, file_name)
                df = pd.read_csv(file_path)

                df = self._split_coordinates(df)
                df = self._recenter_coordinates(df)

                # Save the processed dataframe to the target directory
                output_path = os.path.join(self.target_directory, f"clean_{file_name}")
                df.to_csv(output_path, index=False)
                print(f"Processed and saved to {output_path}")

    def _split_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the coordinates in the dataframe into separate x, y, z columns.

        Args:
            df (pd.DataFrame): The dataframe containing semicolon-separated coordinates.

        Returns:
            pd.DataFrame: The dataframe with separate columns for x, y, z coordinates.
        """
        for col in df.columns:
            # Ensure that the data type is string before checking for ';'
            if isinstance(df[col].iloc[0], str):
                df[col + "_x"] = df[col].apply(lambda coord: float(coord.split(";")[0]))
                df[col + "_y"] = df[col].apply(lambda coord: float(coord.split(";")[1]))
                df[col + "_z"] = df[col].apply(lambda coord: float(coord.split(";")[2]))
        return df

    def _recenter_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recenters the coordinates in the dataframe based on the `PlayerPosition`.

        Args:
            df (pd.DataFrame): The dataframe containing the split x, y, z coordinates.

        Returns:
            pd.DataFrame: The dataframe with recentered coordinates.
        """
        player_x = df["PlayerPosition_x"]
        player_y = df["PlayerPosition_y"]
        player_z = df["PlayerPosition_z"]

        for col in df.columns:
            if col.endswith("_x"):
                df[col] -= player_x
            elif col.endswith("_y"):
                df[col] -= player_y
            elif col.endswith("_z"):
                df[col] -= player_z

        df = df.drop(columns=["PlayerPosition_x", "PlayerPosition_y", "PlayerPosition_z"])
        return df


class DataFrameChecker:
    """
    A class to validate the integrity of dataframes stored in a directory.

    Attributes:
        directory (str): Path to the directory containing the dataframes to be validated.
        reference_columns (List[str]): The longest set of columns discovered from the dataframes
        in the directory.

    Methods:
        read_dataframe(path: str) -> pd.DataFrame:
            Reads a dataframe from a specified file path.
        discover_dataframe_columns() -> List[str]:
            Discovers and returns the longest column set from dataframes in the directory.
        check_dataframe_integrity(data: pd.DataFrame) -> None:
            Checks the integrity of a given dataframe.
        check_dataframe_columns(data: pd.DataFrame) -> None:
            Validates that dataframe columns match the reference columns.
        check_all_dataframes() -> None:
            Iterates over all dataframes in the directory and validates their integrity.
    """

    def __init__(self, directory: str):
        """
        Initialize the DataFrameChecker with a specified directory.

        Args:
            directory (str): Path to the directory containing dataframes.
        """
        self.directory = directory
        self.reference_columns = self.discover_dataframe_columns()

    def read_dataframe(self, path: str) -> pd.DataFrame:
        """
        Read a dataframe from a file.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        return pd.read_csv(path)

    def discover_dataframe_columns(self) -> List[str]:
        """
        Discover and return the longest column set from dataframes in the directory.

        Returns:
            List[str]: List of columns.
        """
        reference_columns: List[str] = []
        for file in os.listdir(self.directory):
            if file.endswith(".csv"):
                path = os.path.join(self.directory, file)
                data = self.read_dataframe(path)
                if len(data.columns) > len(reference_columns):
                    reference_columns = data.columns
        return reference_columns

    def check_dataframe_integrity(self, data: pd.DataFrame) -> None:
        """
        Check the integrity of the dataframe.

        Args:
            data (pd.DataFrame): The dataframe to be checked.

        Raises:
            ValueError: If there is a mismatch in number of joints.
        """
        joint_counts = data.iloc[:, 1:-1].applymap(lambda x: x.count(";")).sum(axis=1)
        if not all(joint_counts == joint_counts.iloc[0]):
            raise ValueError("Not all frames have the same number of joints.")

        self._check_equal_number_of_values(data)

    def _check_equal_number_of_values(self, data: pd.DataFrame) -> None:
        """
        Check that all cells have the same number of values.

        Args:
            data (pd.DataFrame): The dataframe to be checked.

        Raises:
            ValueError: If not all cells have the same number of values.
        """

        def get_cell_length(cell):
            return len(cell.split(";")) if isinstance(cell, str) else 1

        cell_lengths = data.applymap(get_cell_length)
        equal_lengths = (cell_lengths == cell_lengths.iloc[0]).all().all()
        if not equal_lengths:
            raise ValueError("Not all cells have the same number of values.")

    def check_dataframe_columns(self, data: pd.DataFrame) -> None:
        """
        Check that dataframe columns match the reference columns.

        Args:
            data (pd.DataFrame): The dataframe to be checked.

        Raises:
            ValueError: If columns don't match the reference columns.
        """
        if not all(data.columns == self.reference_columns):
            raise ValueError("Not all frames have the same number of joints.")

    def check_all_dataframes(self) -> None:
        """
        Iterate over all dataframes in a directory and check their integrity.
        """
        for file in os.listdir(self.directory):
            if file.endswith(".csv"):
                path = os.path.join(self.directory, file)
                data = self.read_dataframe(path)
                try:
                    logger.info(f"Checking file {file}")
                    self.check_dataframe_integrity(data)
                    self.check_dataframe_columns(data)
                except ValueError as e:
                    logger.error(f"Error in file {file}: {e}")


def split_dataset(dataset, train_ratio, val_ratio):
    """
    Split a dataset into training, validation, and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Proportion of data to go to training set.
        val_ratio (float): Proportion of data to go to validation set.

    Returns:
        train_set, val_set, test_set: The split datasets.
    """
    assert (
        0 < train_ratio < 1 and 0 < val_ratio < 1 and (train_ratio + val_ratio) < 1
    ), "Invalid ratios"

    total_samples = len(dataset)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    return train_set, val_set, test_set
