import matplotlib.pyplot as plt
import numpy as np


# Function to plot 10 graphs on the same axis
def plot_multiple_graphs(ax, x, y_values, name):
    for y in y_values:
        ax.plot(x, y)
    ax.set_title(name)


def plot_graphs(X, Y):
    # Create the main figure with 4 subplots, each having 10 graphs
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    # Generate 10 sets of example data for the graphs
    plot_multiple_graphs(axs[0], X, Y[0], "Accuracy")
    plot_multiple_graphs(axs[1], X, Y[1], "Loss")
    plot_multiple_graphs(axs[2], X, Y[2], "Data Loss")
    plot_multiple_graphs(axs[3], X, Y[3], "Regularization Loss")

    # Adjust the layout of the subplots to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()
