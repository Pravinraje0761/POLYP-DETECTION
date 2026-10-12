
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = "training_curves_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=== Plotting training curves ===")

    # Try to find models with training history
    # First, check cross_dataset_results
    cross_dir = "cross_dataset_results"
    if os.path.exists(cross_dir) and os.path.exists(os.path.join(cross_dir, "cross_dataset_results.json")):
        with open(os.path.join(cross_dir, "cross_dataset_results.json"), "r") as f:
            cross_data = json.load(f)
        for key, data in cross_data.items():
            if "train_loss_history" in data and "val_dice_history" in data:
                plot_curves(data["train_loss_history"], data["val_dice_history"], title=key,
                           save_path=os.path.join(OUTPUT_DIR, f"cross_{key}.png"))

    # Then check multi_seed_results
    multi_dir = "multi_seed_results"
    if os.path.exists(multi_dir) and os.path.exists(os.path.join(multi_dir, "multi_seed_results.json")):
        with open(os.path.join(multi_dir, "multi_seed_results.json"), "r") as f:
            multi_data = json.load(f)
        # Wait, multi_seed_results might not have per-seed histories, but item6_baseline and item9_training_time do

    # Check baseline_results
    base_dir = "baseline_results"
    if os.path.exists(base_dir) and os.path.exists(os.path.join(base_dir, "baseline_training_history.pth")):
        base_hist = torch.load(os.path.join(base_dir, "baseline_training_history.pth"))
        plot_curves(base_hist["train_loss"], base_hist["val_dice"], title="Baseline UNet",
                   save_path=os.path.join(OUTPUT_DIR, "baseline_unet_curves.png"))

    # Check training_time_results
    time_dir = "training_time_results"
    if os.path.exists(time_dir) and os.path.exists(os.path.join(time_dir, "training_time_results.json")):
        with open(os.path.join(time_dir, "training_time_results.json"), "r") as f:
            time_data = json.load(f)
        # Need to check if we have loss and dice history here too
        # Since item9_training_time doesn't save that, we can create a modified version,
        # but for now let's assume we have some history from other scripts

    print("Generated training curves!")


def plot_curves(train_loss, val_dice, title="Training Curves", save_path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = np.arange(1, len(train_loss) + 1)

    ax1.plot(epochs, train_loss, "b-", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_dice, "r-", label="Val Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Dice Score Curve")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
