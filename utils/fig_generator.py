import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib import cm
import seaborn as sns


def plot_values_over_rounds(data_path, confidence=True, file_name='plot.pdf'):
    """
    Generates a figure for values of different methods and keeps the found maximum until every step.
    The function plots using the data in the csv files under data_path.
    """
    # List to store data for each method
    method_data = {}

    # Loop through all files in the data_path
    for filename in os.listdir(data_path):
        if filename.endswith("_results.csv"):
            function_name = filename.replace("_results.csv", "")
            file_path = os.path.join(data_path, filename)

            # Read CSV file
            df = pd.read_csv(file_path)

            # Extract 'i', 'Mean', and 'StdDev'
            rounds = df['i']
            means = df['Mean']
            stds = df['StdDev']

            # Store the data for plotting
            method_data[function_name] = (rounds, means, stds)

    # Sorting the dictionary by the function_name (i.e., the key)
    method_data = {k: method_data[k] for k in sorted(method_data)}

    # Set the plotting style
    sns.set(style='whitegrid', context='paper', font_scale=1.5)
    plt.figure(figsize=(12, 8))

    # Define a color palette
    palette = sns.color_palette("tab10", len(method_data))

    # Plotting
    for idx, (method, (rounds, means, stds)) in enumerate(method_data.items()):
        line_label = 'AdaLIPO+' if method == 'AdaLIPO_P' else method
        plt.plot(rounds, means, label=line_label, color=palette[idx], lw=2)

        # Add confidence interval
        if confidence:
            plt.fill_between(rounds, means - stds / 5, means + stds / 5,
                             color=palette[idx], alpha=0.2)

    # Enhancing the plot with labels, title, grid, and legend
    plt.xlabel('Evaluations', fontsize=25)
    plt.ylabel('Maximum Value', fontsize=25)
    # plt.title('Maximum Values Over Rounds for Different Methods', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a legend only if labels exist
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(loc='best', fontsize=22, frameon=True, fancybox=True, shadow=False)

    # Save the figure
    plt.tight_layout()
    path_ = f"figures/functions/ablations_"
    plt.savefig(os.path.join(path_, file_name), bbox_inches="tight", dpi=300)

    # Show plot
    # plt.show()
    # plt.close()


class FigGenerator:
    def __init__(self, f):
        """
        Class to generate figures for visualizing the optimization process
        f: function optimized (Function)
        """

        self.f = f

    def gen_figure(
        self, eval_points: np.array, eval_values: np.array, path: str = None
    ):
        """
        Generates a figure
        eval_points: points where the function was evaluated (np.array)
        eval_values: values of the function at the evaluation points (np.array)
        path: path to save the figure (str) (optional)
        """

        dim = eval_points.shape[1]
        if dim == 1:
            self.gen_1D(eval_points, eval_values)
        elif dim == 2:
            self.gen_2D(eval_points, eval_values)
        else:
            raise ValueError(
                f"Cannot generate a figure for {dim}-dimensional functions"
            )

        if path is not None:
            plt.savefig(path, bbox_inches="tight")
        else:
            plt.show()
        plt.clf()
        plt.close()

    def gen_1D(self, eval_points, eval_values):
        """
        Generates a figure for 1D functions
        """

        x = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 1000)
        y = self.f(x)

        plt.plot(x, y)
        plt.scatter(
            eval_points,
            eval_values,
            c=eval_values,
            label="evaluations",
            cmap="viridis",
            zorder=2,
        )
        plt.colorbar(fraction=0.046, pad=0.04)
        # plt.plot(eval_points, eval_values, linewidth=0.5, color="black")
        plt.xlabel("$X$")
        plt.ylabel("$f(x)$")
        # plt.legend()

    def gen_2D(self, eval_points, eval_values):
        """
        Generates a figure for 2D functions
        """

        x = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 10000)
        y = np.linspace(self.f.bounds[1][0], self.f.bounds[1][1], 10000)
        x, y = np.meshgrid(x, y)
        z = self.f([x, y])

        fig = plt.figure(figsize=(15, 15))
        ax = plt.axes(projection="3d", computed_zorder=False)

        ax.plot_surface(
            x, y, z, cmap="RdYlBu", linewidth=0, antialiased=True, zorder=4.4
        )

        cb = ax.scatter(
            eval_points[:, 0],
            eval_points[:, 1],
            eval_values,
            c=eval_values,
            cmap="viridis",
            zorder=4.5,
        )

        # plt.colorbar(cb, fraction=0.046, pad=0.04)

        ax.set_xlabel("$X$", fontsize=22)
        ax.set_ylabel("$Y$", fontsize=22)

    def gen_empty_2D(self, path):
        """
        Generates a figure for 2D functions
        """

        x = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 1000)
        y = np.linspace(self.f.bounds[1][0], self.f.bounds[1][1], 1000)
        x, y = np.meshgrid(x, y)
        z = self.f([x, y])

        fig = plt.figure(figsize=(15, 15))
        ax = plt.axes(projection="3d", computed_zorder=False)

        ax.plot_surface(
            x, y, z, cmap="RdYlBu", linewidth=0, antialiased=True, zorder=4.4
        )

        ax.set_xlabel("$X$", fontsize=22)
        ax.set_ylabel("$Y$", fontsize=22)

        if path is not None:
            plt.savefig(path, bbox_inches="tight")
        else:
            plt.show()
        plt.clf()
        plt.close()


def plot_epsilon(average1, average2, average3, std1=None, std2=None, std3=None, title=""):
    # Function to parse input data
    if std1 is None:
        std1 = []

    def parse_input(input_data):
        # If input is a string, split by commas and convert to float
        if isinstance(input_data, str):
            return np.array([float(x.strip()) for x in input_data.split(',')]).astype(float)
        elif isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data).astype(float)
        else:
            raise ValueError(f"Invalid input type: {type(input_data)}. Expected str, list, or np.ndarray.")

    # Create x-axis values based on the length of average1
    x = np.arange(len(average1))

    # Plotting averages
    plt.plot(average1, '-', color='orange', label='AdaLIPO')
    plt.fill_between(x, average1 - std1, average1 + std1,
                     alpha=0.3, color="gold")  # Lighter std color

    plt.plot(average2, '-', color='orangered', label='AdaLIPO_P')
    plt.fill_between(x, average2 - std2, average2 + std2,
                     alpha=0.3, color="orangered")  # Lighter std color

    plt.plot(average3[1:], '-', color='navy', label='ECP')
    plt.fill_between(x, average3 - std3[1:], average3 + std3[1:],
                     alpha=0.3, color="lightblue")  # Lighter std color

    plt.title(f"{title}")
    plt.xlabel('Method Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/epsilon", bbox_inches="tight")
    plt.show()
