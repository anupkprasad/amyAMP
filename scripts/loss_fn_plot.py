import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append(os.path.expanduser("~/workspace/amyAMP/"))
from scripts import util, plotStyle
plotStyle.setPlotStyle()

path_model = os.path.expanduser("~/workspace/amyAMP/model_saved/")
run_num = 2

for r in [1]:
    collection = torch.load(path_model + 'collectedseqs_loss_epochinfo_r' + str(r + 1) + '.json')
    loss_all = collection['loss_all']
    
    loss_all = [x[:2000] for x in loss_all]  # Truncate to first 2000 epochs for better visualization
    # Extract the four lists from loss_all
    loss_1, loss_2, loss_3, loss_4 = loss_all

    # Create separate plots for each loss list
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    # Plot Loss 1
    axs[0, 0].plot(loss_1, color="blue", linewidth=1.5)
    axs[0, 0].set_title("Loss 1", fontsize=10, fontweight="bold")
    axs[0, 0].set_xlabel("Epochs", fontsize=8)
    axs[0, 0].set_ylabel("Loss Value", fontsize=8)
    axs[0, 0].grid(alpha=0.3, linestyle="--")   

    # Plot Loss 2
    axs[0, 1].plot(loss_2, color="green", linewidth=1.5)
    axs[0, 1].set_title("Loss 2", fontsize=10, fontweight="bold")
    axs[0, 1].set_xlabel("Epochs", fontsize=8)
    axs[0, 1].set_ylabel("Loss Value", fontsize=8)
    axs[0, 1].grid(alpha=0.3, linestyle="--")

    # Plot Loss 3
    axs[1, 0].plot(loss_3, color="orange", linewidth=1.5)
    axs[1, 0].set_title("Loss 3", fontsize=10, fontweight="bold")
    axs[1, 0].set_xlabel("Epochs", fontsize=8)
    axs[1, 0].set_ylabel("Loss Value", fontsize=8)
    axs[1, 0].grid(alpha=0.3, linestyle="--")

    # Plot Loss 4
    axs[1, 1].plot(loss_4, color="red", linewidth=1.5)
    axs[1, 1].set_title("Loss 4", fontsize=10, fontweight="bold")
    axs[1, 1].set_xlabel("Epochs", fontsize=8)
    axs[1, 1].set_ylabel("Loss Value", fontsize=8)
    axs[1, 1].grid(alpha=0.3, linestyle="--")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(path_model + f"loss_curves_separate_run_{r + 1}.png", dpi=300, bbox_inches="tight")
    plt.show()