import os
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from spine.data import constants, utils


class SkeletonFrame:
    """
    Represents a single frame of a skeleton, facilitating operations such as averaging and
    normalizing joint coordinates.

    Attributes:
        frame (pd.Series): Data for a specific frame containing joint coordinates.

    Methods:
        average_coordinates(joint_list: List[str]) -> str:
            Compute the average coordinates for a given list of joints.

        average_joints() -> None:
            Average the coordinates of certain groups of joints and update them in the frame.

        normalize_by_player_position() -> None:
            Normalize all joint positions by the PlayerPosition within the frame.

        get_frame() -> pd.Series:
            Retrieve the processed frame data after averaging and normalization.
    """

    def __init__(self, frame: pd.Series):
        """This method initializes the class.

        Args:
            frame: A single frame of skeleton data.
        """
        self.frame = frame.copy()

    def average_coordinates(self, joint_list: List[str]) -> str:
        """This method averages the coordinates of a list of joints.

        Args:
            joint_list: A list of joints to average.

        Returns:
            A string containing the averaged coordinates.
        """
        x_values, y_values, z_values = [], [], []
        for joint in joint_list:
            x, y, z = utils.extract_coordinates(self.frame[joint])
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)
        avg_x = sum(x_values) / len(x_values)
        avg_y = sum(y_values) / len(y_values)
        avg_z = sum(z_values) / len(z_values)
        return f"{avg_x};{avg_y};{avg_z}"

    def average_joints(self) -> None:
        """This method averages the coordinates of particular joints of the skeleton, to get
        a more compact representation of the skeleton. The averaged joints are the joints of
        the head the hands and the feet.
        """
        self.frame["Head"] = self.average_coordinates(
            ["Nose", "EyeLeft", "EyeRight", "EarLeft", "EarRight"]
        )
        self.frame["HandLeft"] = self.average_coordinates(
            ["WristLeft", "ThumbLeft", "LeftIndexFinger", "PinkyLeft"]
        )
        self.frame["HandRight"] = self.average_coordinates(
            ["WristRight", "ThumbRight", "RightIndexFinger", "PinkyRight"]
        )
        self.frame["FootLeft"] = self.average_coordinates(
            ["AnkleLeft", "ToesLeft", "BigToeLeft", "SmallToeLeft"]
        )
        self.frame["FootRight"] = self.average_coordinates(
            ["AnkleRight", "ToesRight", "BigToeRight", "SmallToeRight"]
        )

    def normalize_by_player_position(self) -> None:
        """This method normalizes the coordinates of the skeleton by the player position."""

        player_x, player_y, player_z = utils.extract_coordinates(self.frame["PlayerPosition"])
        for joint in self.frame.index[1:-1]:  # Excluding timestamp and PlayerPosition
            x, y, z = utils.extract_coordinates(self.frame[joint])
            self.frame[joint] = f"{x - player_x};{y - player_y};{z - player_z}"

    def get_frame(self) -> pd.Series:
        """This method returns the frame.

        Returns:
            The frame.
        """
        return self.frame


class SkeletonAnimator:
    """
    Responsible for plotting and animating the skeleton frames.

    Attributes:
        data (pd.DataFrame): Data containing the joint coordinates of the skeleton for multiple
        frames.

    Methods:
        plot_skeleton(ax: plt.Axes, frame_index: int) -> None:
            Plot the skeleton for a specific frame on the provided axes.

        animate(save_path: str = None) -> Union[None, HTML]:
            Generate an animation of the skeleton across all frames.
    """

    def __init__(self, data: pd.DataFrame):
        """This method initializes the class.

        Args:
            data: The data containing the skeleton information.
        """
        self.data = data

    def plot_skeleton(self, ax: plt.Axes, frame_index: int) -> None:
        """This method plots a skeleton for a given frame on provided axes.

        Args:
            ax: The axes on which to plot the skeleton.
            frame_index: The index of the frame to plot.
        """
        frame_data = SkeletonFrame(self.data.iloc[frame_index])
        frame_data.normalize_by_player_position()
        frame_data.average_joints()

        frame = frame_data.get_frame()

        for joint in frame.index[1:]:
            x, y, z = utils.extract_coordinates(frame[joint])
            if joint in constants.SPINE_JOINTS:
                ax.scatter(x, z, y, color="red", s=50, label=joint)
            elif joint == "Head":
                ax.scatter(x, z, y, color="green", s=50, label=joint)
            elif joint in ["HandLeft", "HandRight", "FootLeft", "FootRight"]:
                ax.scatter(x, z, y, color="blue", s=50, label=joint)

        # Connect joints
        for start, end in constants.CONNECTIONS:
            x1, y1, z1 = utils.extract_coordinates(frame[start])
            x2, y2, z2 = utils.extract_coordinates(frame[end])
            linewidth = (
                2.5 if start in constants.SPINE_JOINTS or end in constants.SPINE_JOINTS else 1
            )
            ax.plot([x1, x2], [z1, z2], [y1, y2], "k-", linewidth=linewidth)

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])
        ax.set_title(f"Simplified Skeleton - Frame {frame_index}")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.view_init(elev=20, azim=30)

    def plot_frame(self, frame_index: int, directory: str = None) -> None:
        """This method plots a specific frame with a distinct spine.

        Args:
            frame_index: The index of the frame to plot.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        self.plot_skeleton(ax, frame_index)
        if directory:
            plt.savefig(
                os.path.join(directory, f"frame_{frame_index}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.tight_layout()
        plt.show()

    def animate(self, save_path: str = None) -> Union[None, HTML]:
        """
        Animate the skeleton frames.

        Args:
            data (pd.DataFrame): The data containing the skeleton information.
            save_path (str, optional): Path to save the animation. If provided, the animation
            will be saved as a .mp4 file. If ffmpeg is not installed, it will fall back to Pillow.
            In case of Pillow, the animation needs to be saved as a .gif file, so the save_path
            must end with .gif. Defaults to None.

        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Initialization function for the animation
        def init():
            ax.clear()
            return (ax,)

        # Update function for the animation
        def update(frame_index):
            ax.clear()
            self.plot_skeleton(ax, frame_index)
            return (ax,)

        ani = FuncAnimation(
            fig, update, frames=len(self.data), init_func=init, blit=False, repeat=False
        )

        # If save_path is provided, save the animation
        if save_path:
            ani.save(save_path, writer="ffmpeg", fps=24, bitrate=1800)
            return None
        else:
            # This is for displaying the animation in Jupyter notebooks
            plt.close(fig)
            return HTML(ani.to_jshtml())


class BarVisualization:
    """
    Helper class for visualizing results, particularly the joint rankings based on correlations.

    Methods:
        display_ranked_joints(X, Y, Z, save_filename=None): Displays a bar plot of joints ranked
        by their combined correlations and optionally saves the figure.
    """

    @staticmethod
    def display_ranked_joints(
        X: pd.DataFrame, Y: pd.DataFrame, Z: pd.DataFrame, save_filename=None
    ) -> None:
        """
        Display a bar plot of joints ranked by their combined correlations.

        Args:
            X (pd.DataFrame): Correlations for X coordinate.
            Y (pd.DataFrame): Correlations for Y coordinate.
            Z (pd.DataFrame): Correlations for Z coordinate.
            save_filename (str or None): Optional filename to save the figure.
        """
        combined_correlation = (abs(X) + abs(Y) + abs(Z)) / 3
        ranked_joints_combined = combined_correlation.mean(axis=1).sort_values(ascending=False)
        ranked_joints_combined.drop(index=["Spine", "Spine1", "Spine2", "Spine3"], inplace=True)

        plt.figure(figsize=(15, 10))
        sns.barplot(
            y=ranked_joints_combined.index, x=ranked_joints_combined.values, palette="viridis"
        )
        plt.title("Joint Rankings Based on Combined Correlations Across All Axes")
        plt.xlabel("Combined Correlation Score")
        plt.ylabel("Joints")

        if save_filename:
            plt.savefig(save_filename, bbox_inches="tight")
            print(f"Figure saved as {save_filename}")
        else:
            plt.show()


def plot_curves(train_curve: List[float], val_curve: List[float], title: str = "Loss Curves"):
    """
    Plots training and validation curves.

    Args:
        train_curve (List[float]): List of training losses/metrics.
        val_curve (List[float]): List of validation losses/metrics.
        title (str): Title for the plot.
    """
    epochs = range(1, len(train_curve) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_curve, label="Training", marker="o")
    plt.plot(epochs, val_curve, label="Validation", marker="x")

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
