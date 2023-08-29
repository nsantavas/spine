import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


class AnimationFile:
    """
    Represents an individual animation file, and contains functionality to read the file and
    compute correlations.

    Attributes:
        file_path (str): Path to the CSV file.
        spine_columns (List[str]): List of column names corresponding to spine joints.
        data (pd.DataFrame): Data read from the CSV file.
        correlations_X (pd.DataFrame): Correlations for the X coordinate.
        correlations_Y (pd.DataFrame): Correlations for the Y coordinate.
        correlations_Z (pd.DataFrame): Correlations for the Z coordinate.
    """

    def __init__(self, file_path: str, spine_columns: List[str]):
        """
        Initialize an AnimationFile object.

        Args:
            file_path (str): Path to the CSV file.
            spine_columns (List[str]): List of column names corresponding to spine joints.
        """
        self.file_path = file_path
        self.spine_columns = spine_columns
        self.data = pd.read_csv(self.file_path)
        self.data = self._recenter_coordinates()
        (
            self.correlations_X,
            self.correlations_Y,
            self.correlations_Z,
        ) = self._compute_correlations()

    def _compute_correlations(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute correlations for X, Y, Z coordinates.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing correlations for
            X, Y, Z coordinates.
        """
        all_joints = self.data.drop(columns=["Frame", "PlayerPosition"])
        all_joints_X = all_joints.applymap(lambda coord: float(coord.split(";")[0]))
        all_joints_Y = all_joints.applymap(lambda coord: float(coord.split(";")[1]))
        all_joints_Z = all_joints.applymap(lambda coord: float(coord.split(";")[2]))

        correlations_X = all_joints_X.corr()[self.spine_columns]
        correlations_Y = all_joints_Y.corr()[self.spine_columns]
        correlations_Z = all_joints_Z.corr()[self.spine_columns]

        return correlations_X, correlations_Y, correlations_Z

    def _recenter_coordinates(self) -> pd.DataFrame:
        """
        Recenter coordinates of joints relative to the PlayerPosition.

        Returns:
            pd.DataFrame: Updated dataframe with recentered coordinates.
        """
        data = self.data.copy()

        # Splitting PlayerPosition into X, Y, Z
        player_positions = data["PlayerPosition"].str.split(";", expand=True).astype(float)
        player_positions.columns = ["PlayerPosition_X", "PlayerPosition_Y", "PlayerPosition_Z"]

        # Iterate over all joint columns
        for column in data.columns:
            if column not in ["Frame", "PlayerPosition"]:
                joint_positions = data[column].str.split(";", expand=True).astype(float)
                joint_positions.columns = [f"{column}_X", f"{column}_Y", f"{column}_Z"]

                # Subtract PlayerPosition from joint position
                joint_positions[f"{column}_X"] -= player_positions["PlayerPosition_X"]
                joint_positions[f"{column}_Y"] -= player_positions["PlayerPosition_Y"]
                joint_positions[f"{column}_Z"] -= player_positions["PlayerPosition_Z"]

                # Concatenate back into the original format
                data[column] = (
                    joint_positions[f"{column}_X"].astype(str)
                    + ";"
                    + joint_positions[f"{column}_Y"].astype(str)
                    + ";"
                    + joint_positions[f"{column}_Z"].astype(str)
                )

        return data


class AnimationDirectory:
    """
    Represents a directory containing multiple animation files, and provides methods to aggregate
    statistics over all files.

    Attributes:
        directory_path (str): Path to the directory containing animation CSV files.
        spine_columns (List[str]): List of column names corresponding to spine joints.
        animation_files (List[AnimationFile]): List of AnimationFile objects extracted from
        the directory.
        avg_correlations_X (pd.DataFrame): Average correlations for the X coordinate over
        all files.
        avg_correlations_Y (pd.DataFrame): Average correlations for the Y coordinate over
        all files.
        avg_correlations_Z (pd.DataFrame): Average correlations for the Z coordinate over
        all files.
    """

    def __init__(self, directory_path: str, spine_columns: List[str]):
        """
        Initialize an AnimationDirectory object.

        Args:
            directory_path (str): Path to the directory containing animation CSV files.
            spine_columns (List[str]): List of column names corresponding to spine joints.
        """
        self.directory_path = directory_path
        self.spine_columns = spine_columns
        self.animation_files = self._get_animation_files()
        (
            self.avg_correlations_X,
            self.avg_correlations_Y,
            self.avg_correlations_Z,
        ) = self._aggregate_correlations()

    def _get_animation_files(self) -> List[AnimationFile]:
        """
        Retrieve a list of AnimationFile objects from the directory.

        Returns:
            List[AnimationFile]: List of AnimationFile objects.
        """
        animation_files = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.directory_path, filename)
                animation_files.append(AnimationFile(file_path, self.spine_columns))
        return animation_files

    def _aggregate_correlations(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Aggregate correlations over all animation files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing average correlations
            for X, Y, Z coordinates.
        """
        agg_correlations_X = [file.correlations_X for file in self.animation_files]
        agg_correlations_Y = [file.correlations_Y for file in self.animation_files]
        agg_correlations_Z = [file.correlations_Z for file in self.animation_files]

        avg_correlations_X = pd.concat(agg_correlations_X).groupby(level=0).mean()
        avg_correlations_Y = pd.concat(agg_correlations_Y).groupby(level=0).mean()
        avg_correlations_Z = pd.concat(agg_correlations_Z).groupby(level=0).mean()

        return avg_correlations_X, avg_correlations_Y, avg_correlations_Z

    @property
    def ranked_average_correlations(self) -> pd.DataFrame:
        """
        Rank the average correlations for each spine joint.

        Returns:
            pd.DataFrame: Dataframe containing ranked average correlations for each spine joint.
        """
        avg_correlations = (
            self.avg_correlations_X + self.avg_correlations_Y + self.avg_correlations_Z
        ) / 3
        avg_correlations = avg_correlations.mean(axis=1).sort_values(ascending=False)
        avg_correlations.drop(index=self.spine_columns, inplace=True)
        return avg_correlations


class SpineAutocorrelation:
    """
    Analyzes and visualizes the autocorrelation of spine joint movements from a CSV file.

    This class provides methods to compute and visualize the autocorrelation of Spine's X, Y,
    and Z joint movements.

    Attributes:
        data (pd.DataFrame): Loaded spine data from the provided CSV file.

    """

    def __init__(self, filepath: str) -> None:
        """
        Initializes the SpineAutocorrelation instance.

        Args:
            filepath (str): Path to the CSV file containing various spine joints' X, Y, and Z data.

        Raises:
            ValueError: If the CSV file doesn't contain the required columns.
        """
        self.data = pd.read_csv(filepath)
        self.spine_columns = [
            "Spine_x",
            "Spine_y",
            "Spine_z",
            "Spine1_x",
            "Spine1_y",
            "Spine1_z",
            "Spine2_x",
            "Spine2_y",
            "Spine2_z",
            "Spine3_x",
            "Spine3_y",
            "Spine3_z",
        ]
        self.spine_types = ["Spine", "Spine1", "Spine2", "Spine3"]

        if not all(col in self.data.columns for col in self.spine_columns):
            raise ValueError(
                f"CSV file must contain the following columns: {', '.join(self.spine_columns)}."
            )

    def compute_autocorrelation(self, spine: str, lag: int = None) -> List[float]:
        """
        Computes the autocorrelation for a given spine joint movement data series.

        Args:
            spine (str): The spine joint movement data series to compute autocorrelation for.
            lag (int, optional): The lag for which the autocorrelation is computed. Defaults
            to None.

        Returns:
            List[float]: List of autocorrelation values for different lags.
        """
        assert spine in self.data.columns, f"Spine joint {spine} not found in data."
        if not lag:
            lag = len(self.data[spine])

        return acf(self.data[spine], nlags=lag, fft=True)

    def compute_average_autocorrelation(
        self, axis: str, lags: int, plot: bool = False
    ) -> List[float]:
        """
        Computes the average autocorrelation for a given axis of spine joint movement data series.

        Args:
            axis (str): The axis of spine joint movement data series to compute autocorrelation
            for.
            lags (int): The number of lags for which to compute the average autocorrelation.

        Returns:
            List[float]: List of average autocorrelation values for different lags.
        """
        assert axis in ["x", "y", "z"], "Axis must be one of 'x', 'y', or 'z'"

        axis_columns = [col for col in self.spine_columns if col.endswith(f"_{axis}")]
        autocorrelations = []
        for column in axis_columns:
            autocorrelations.append(self.compute_autocorrelation(column, lags))

        return np.mean(np.array(autocorrelations), axis=0)

    def visualize_average_autocorrelation(self, lags: int, directory: str = None) -> None:
        """
        Computes and visualizes the average autocorrelation for a given axis of spine joint
        movement data series.

        Args:
            axis (str): The axis of spine joint movement data series to compute autocorrelation
            for.
            lags (int): The number of lags for which to compute and visualize autocorrelation.
        """

        plt.figure(figsize=(12, 9))
        colors = ["blue", "red", "green"]

        for i, axis in enumerate(["x", "y", "z"]):
            results = self.compute_average_autocorrelation(axis, lags)
            plt.plot(
                range(lags + 1),
                results,
                label=f"Average Autocorrelation for {axis} Axis",
                color=colors[i],
                marker="x",
            )

        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title("Average Autocorrelation for Spine Joint Movements")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if directory:
            plt.savefig(os.path.join(directory, "single_average_autocorrelation.png"))

        plt.show()

    def visualize_autocorrelation(self, lags: int, directory: str = None) -> None:
        """
        Computes and visualizes the autocorrelation for Spine's X, Y, and Z joint movements.

        Args:
            lags (int): The number of lags for which to compute and visualize autocorrelation.
        """
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 10))
        colors = ["blue", "red", "green", "purple"]

        for i, spine in enumerate(self.spine_columns):
            spine_type = spine.split("_")[0]
            plot_acf(
                self.data[spine],
                ax=axes[self.spine_columns.index(spine) // 2, self.spine_columns.index(spine) % 2],
                title=f"Autocorrelation for {spine}",
                lags=lags,
                color=colors[self.spine_types.index(spine_type)],
            )

        plt.tight_layout()

        if directory:
            fig.savefig(os.path.join(directory, "autocorrelation.png"))

        plt.show()


class DirectorySpineAnalyzer:
    """This class is used to analyze the autocorrelation of spine joint movements for every CSV
    file in a directory.

    Attributes:
        directory_path (str): Path to the directory containing CSV files

    """

    def __init__(self, directory_path: str):
        """This method initializes the DirectorySpineAnalyzer instance.

        Args:
            directory_path: Path to the directory containing CSV files
        """
        self.directory_path = directory_path

    def analyze_all(self, lags: int):
        """Analyze all CSV files in the directory

        Args:
            lags: Number of lags

        Returns:
            List[float]: List of autocorrelation values for different lags
        """

        def pad_to_length(arr, length, pad_value=0):
            return np.pad(arr, (0, length - len(arr)), "constant", constant_values=pad_value)

        x_autocorrelation, y_autocorrelation, z_autocorrelation = [], [], []
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.directory_path, filename)
                analyzer = SpineAutocorrelation(filepath=file_path)
                x_autocorrelation.append(analyzer.compute_average_autocorrelation("x", lags))
                y_autocorrelation.append(analyzer.compute_average_autocorrelation("y", lags))
                z_autocorrelation.append(analyzer.compute_average_autocorrelation("z", lags))

        x_padded = [pad_to_length(arr, lags + 1) for arr in x_autocorrelation]
        y_padded = [pad_to_length(arr, lags + 1) for arr in y_autocorrelation]
        z_padded = [pad_to_length(arr, lags + 1) for arr in z_autocorrelation]
        x_autocorrelation = np.mean(np.array(x_padded), axis=0)
        y_autocorrelation = np.mean(np.array(y_padded), axis=0)
        z_autocorrelation = np.mean(np.array(z_padded), axis=0)

        return x_autocorrelation, y_autocorrelation, z_autocorrelation

    def visualize_autocorrelation(self, lags: int, directory: str = None) -> None:
        """Visualize the autocorrelation

        Args:
            lags: Number of lags
        """
        x_autocorrelation, y_autocorrelation, z_autocorrelation = self.analyze_all(lags)

        avg_autocorrelation = {
            "x": x_autocorrelation,
            "y": y_autocorrelation,
            "z": z_autocorrelation,
        }
        colors = ["blue", "red", "green"]

        plt.figure(figsize=(12, 9))
        for i, (axis, autocorrelation) in enumerate(avg_autocorrelation.items()):
            plt.plot(
                range(lags + 1), autocorrelation, color=colors[i], marker="x", label=f"{axis} Axis"
            )
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")

        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        if directory:
            plt.savefig(os.path.join(directory, "global_autocorrelation.png"))

        plt.show()


class TemporalCorrelationAnalyzer:
    """
    A class to analyze temporal correlations between various joints and spine joints in 3D motion
    data.

    This class provides tools to study the time-delayed relationship between movements of different
    joints and the spine. It can calculate individual and average correlations for specified lags,
    visualize these correlations, and identify lags where the average correlation drops below a
    given threshold.

    Attributes:
        data (pd.DataFrame): Input data containing 3D joint positions.
        axes (str): Axis of interest (e.g., 'x', 'y', or 'z').
        max_lag (int): Maximum number of frames (or lags) for correlation analysis.
        spine_joints (list): List of spine joints based on the specified axis.
        joint_prefixes (list, optional): List of joint prefixes to focus the analysis on.
        all_joints (list): List of all joints to be analyzed.
        correlation_data (dict): Stores computed correlations for each joint with each spine joint.
        average_curve (list): Stores average correlation values across all joints for each lag.

    Methods:
        get_selected_joints(): Returns the list of joints to be analyzed.
        compute_avg_correlations(joint, spine_joint): Calculates average temporal correlation for
        a joint with a spine joint.
        analyze_correlations(): Computes the temporal correlations for all selected joints with
        each spine joint.
        compute_avg_curve(): Calculates and stores the average correlation across all selected
        joints for each lag.
        plot_correlations(plot_avg_curve): Visualizes the temporal correlations.
        frames_below_threshold(threshold): Returns the lag where average correlation first drops
        below the threshold.
    """

    def __init__(
        self, data: pd.DataFrame, axes: str = "y", max_lag: int = 20, joint_prefixes: list = None
    ):
        """
        Initialize the TemporalCorrelationAnalyzer.

        Args:
            data (pd.DataFrame): The input dataframe containing joint positions.
            axes (str, optional): The axis of interest (x, y, or z). Defaults to 'y'.
            max_lag (int, optional): The maximum number of frames (lags) to consider for temporal
            correlation. Defaults to 20.
            joint_prefixes (list, optional): List of joint prefixes to analyze. If not provided,
            all joints excluding spine joints are considered.
        """
        self.data = data.copy(deep=True)
        self.axes = axes
        self.max_lag = max_lag
        self.spine_joints = [f"Spine_{axes}", f"Spine1_{axes}", f"Spine2_{axes}", f"Spine3_{axes}"]
        self.joint_prefixes = joint_prefixes
        self._prepare_data()
        self.all_joints = self.get_selected_joints()
        self.correlation_data: Dict = {}
        self.average_curve: List = []

    def _prepare_data(self) -> None:
        """
        Prepare the data for analysis by splitting the combined columns into separate columns.
        """
        for column in self.data.columns[1:]:
            self.data[[f"{column}_x", f"{column}_y", f"{column}_z"]] = (
                self.data[column].str.split(";", expand=True).astype(float)
            )

        # Drop the original combined columns
        self.data = self.data.drop(
            columns=np.concatenate([self.data.columns[1:36], self.data.columns[-3:]])
        )

    def get_selected_joints(self) -> list:
        """
        Get the list of joints to be analyzed based on joint_prefixes and axis of interest.

        Returns:
            list: List of joint names to be analyzed.
        """
        if self.joint_prefixes:
            return [
                col
                for col in self.data.columns
                if f"_{self.axes}" in col
                and any(col.startswith(prefix) for prefix in self.joint_prefixes)
            ]
        else:
            return [
                col
                for col in self.data.columns
                if f"_{self.axes}" in col and col not in self.spine_joints
            ]

    def compute_avg_correlations(self, joint: str, spine_joint: str) -> list:
        """
        Compute the average temporal correlation between a given joint and a spine joint for
        each lag.

        Args:
            joint (str): The name of the joint to be analyzed.
            spine_joint (str): The name of the spine joint to be analyzed.

        Returns:
            list: List of correlation values for each lag.
        """
        avg_correlations = []
        for t in range(1, self.max_lag + 1):
            shifted_data = self.data[spine_joint].shift(t).dropna()
            current_correlation = shifted_data.corr(self.data[joint].iloc[t:])
            avg_correlations.append(current_correlation)
        return avg_correlations

    def analyze_correlations(self) -> None:
        """
        Analyze and compute the temporal correlations for all selected joints with each spine
        joint.

        Populates the correlation_data attribute with computed correlations.
        """
        for joint in self.all_joints:
            joint_correlations = {}
            for spine_joint in self.spine_joints:
                joint_correlations[spine_joint] = self.compute_avg_correlations(joint, spine_joint)
            self.correlation_data[joint] = joint_correlations

    def compute_avg_curve(self) -> None:
        """
        Compute and store the average correlation across all selected joints for each lag.

        Populates the average_curve attribute with average correlations.
        """
        total_correlations = np.zeros(self.max_lag)
        total_count = 0
        for joint, joint_correlations in self.correlation_data.items():
            for spine_joint, correlations in joint_correlations.items():
                total_correlations += correlations
                total_count += 1
        self.average_curve = (total_correlations / total_count).tolist()

    def plot_correlations(self, plot_avg_curve: bool = False, directory: str = None) -> None:
        """
        Visualize the temporal correlations.

        Args:
            plot_avg_curve (bool, optional): If True, plots only the average correlation curve.
            If False, plots individual correlations of each joint with each spine joint. Defaults
            to False.
        """

        marker_styles = {
            "Head": "o",
            "Hip": "s",
            "Nose": "^",
            "Ear": "v",
            "Eye": "+",
            "Shoulder": "x",
            "Elbow": "D",
        }

        plt.figure(figsize=(12, 9))
        if plot_avg_curve:
            plt.plot(
                range(1, self.max_lag + 1),
                self.average_curve,
                label="Average Correlation",
                linewidth=2,
                color="black",
            )
        else:
            for joint, joint_correlations in self.correlation_data.items():
                _joint_prefix = (
                    joint.replace("Right", "")
                    .replace("Left", "")
                    .replace("_y", "")
                    .replace("_z", "")
                    .replace("_x", "")
                )
                for spine_joint, correlations in joint_correlations.items():
                    plt.plot(
                        range(1, self.max_lag + 1),
                        correlations,
                        marker=marker_styles[_joint_prefix]
                        if _joint_prefix in marker_styles
                        else "*",
                        label=f"{joint} vs {spine_joint}",
                    )

        plt.xlabel("Time Lag (frames)")
        plt.ylabel("Correlation")
        plt.title(f"Temporal Correlations of Joints with Spine Joints ({self.axes}-axis)")
        plt.xticks(np.arange(1, self.max_lag + 1))
        plt.legend(
            loc="upper center", bbox_to_anchor=(1.15, 1.15)
        )  # Adjust the bbox_to_anchor values as needed
        plt.grid(True)
        plt.tight_layout()
        if directory:
            plt.savefig(
                os.path.join(directory, f"temporal_correlations_{self.axes}_axis.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.show()

    def frames_below_threshold(self, threshold: float) -> int:
        """
        Get the number of frames for which the average correlation is below the given threshold.

        Args:
            threshold (float): Correlation threshold value.

        Returns:
            int: The lag (or number of frames) at which the average correlation falls below the
            threshold. Returns max_lag if correlation never goes below threshold.
        """
        for lag, correlation in enumerate(self.average_curve, start=1):
            if correlation < threshold:
                return lag
        return self.max_lag


class TemporalAnalysisDirectory:
    """This class is used to analyze temporal correlations for all files in a directory."""

    def __init__(self, directory: str, threshold: float, joint_prefixes: List[str]):
        """Initialize the TemporalAnalysisDirectory.

        Args:
            directory: The directory containing the files to be analyzed.
            threshold: The threshold value for average correlation.
            joint_prefixes: The list of joint prefixes to be analyzed.
        """
        self.directory = directory
        self.threshold = threshold
        self.joint_prefixes = joint_prefixes

    def analyze_all(self):
        """Analyze all files in the directory and return the average lag above the threshold.

        Returns:
            float: The average lag above the threshold.
        """
        lags = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.directory, filename)
                data = pd.read_csv(file_path)
                for axis in ["x", "y", "z"]:
                    analyzer = TemporalCorrelationAnalyzer(
                        data, axis, joint_prefixes=self.joint_prefixes
                    )
                    analyzer.analyze_correlations()
                    analyzer.compute_avg_curve()
                    lag = analyzer.frames_below_threshold(self.threshold)
                    lags.append(lag)

        return np.mean(lags)
