import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.calibration import gather_foreground, reliability_curve


def plot_reliability(probs, true_labels, title, output_path, n_bins=15):
    confidences, correct, _, _ = gather_foreground(probs, true_labels)
    centers, accuracies, _, counts = reliability_curve(confidences, correct, n_bins)
    valid = counts > 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.bar(centers[valid], accuracies[valid], width=1.0 / n_bins, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_uncertainty_map(image, prediction, reference, uncertainty, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(image[..., 0], cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(reference, cmap="viridis")
    axes[1].set_title("Reference")
    axes[2].imshow(prediction, cmap="viridis")
    axes[2].set_title("Prediction")
    heat = axes[3].imshow(uncertainty, cmap="hot")
    axes[3].set_title("Uncertainty")
    fig.colorbar(heat, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(history_list, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for fold, history in enumerate(history_list):
        ax.plot(history["loss"], label=f"train fold {fold}", alpha=0.5)
        ax.plot(history["val_loss"], linestyle="--", label=f"val fold {fold}", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=6)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
