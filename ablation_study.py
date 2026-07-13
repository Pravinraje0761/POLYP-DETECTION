"""
Stage 11: Ablation Study Script.
Trains and compares all three models.
"""
import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
from visualization import create_comparison_table


def main():
    # Configuration
    data_root = "hyper-kvasir-labeled-images (1)/labeled-images"
    seg_checkpoint = None  # Update this to your trained segmentation checkpoint
    base_output = "ablation_outputs"
    os.makedirs(base_output, exist_ok=True)
    
    # Ablation modes
    modes = ["seg_only", "rgb", "full"]
    mode_names = {
        "seg_only": "A-DenseUNet + ViT",
        "rgb": "A-DenseUNet + RGB + ViT",
        "full": "A-DenseUNet + RGB + CLAHE + ViT (Proposed)"
    }
    
    # Train each mode
    all_metrics = {}
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Training: {mode_names[mode]}")
        print(f"{'='*60}")
        
        output_dir = os.path.join(base_output, mode)
        
        # Run training
        cmd = [
            sys.executable, "train.py",
            "--data_root", data_root,
            "--output_dir", output_dir,
            "--ablation", mode,
            "--epochs", "120",
            "--batch_size", "16"
        ]
        
        if seg_checkpoint:
            cmd.extend(["--seg_checkpoint", seg_checkpoint])
        
        subprocess.run(cmd, check=True)
        
        # Load metrics
        metrics_file = os.path.join(output_dir, "checkpoints", "best_model.pth")
        import torch
        checkpoint = torch.load(metrics_file)
        metrics = checkpoint["metrics"]
        
        # We'll just keep the main classification metrics for comparison
        all_metrics[mode_names[mode]] = {
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"]
        }
    
    # Create comparison table
    print("\n" + "="*60)
    print("Ablation Study Results")
    print("="*60)
    
    df = create_comparison_table(
        all_metrics,
        save_path=os.path.join(base_output, "comparison_table.png")
    )
    
    print("\n" + df.to_string())
    print(f"\nComparison table saved to {os.path.join(base_output, 'comparison_table.png')}")


if __name__ == "__main__":
    main()
