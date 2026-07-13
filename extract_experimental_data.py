
import os
import json
import torch
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(r"c:\Users\ADMIN\Desktop\POLYP_DETECTION")
OUTPUT_FILE = PROJECT_ROOT / "extracted_manuscript_information.md"

def extract_paper_info():
    output = []
    output.append("# Extracted Manuscript Information")
    output.append("=" * 100)
    output.append("")

    # 1. Dataset Statistics
    output.append("## 1. Dataset Statistics")
    output.append("-" * 80)
    combined_dir = PROJECT_ROOT / "combined_dataset"
    if combined_dir.exists():
        images_dir = combined_dir / "images"
        masks_dir = combined_dir / "masks"
        all_files = list(images_dir.glob("*"))
        output.append(f"- Combined Dataset Total Images: {len(all_files)}")

        # Count by prefix
        prefix_counts = defaultdict(int)
        for f in all_files:
            if f.name.startswith("archive"):
                prefix_counts["archive"] +=1
            elif f.name.startswith("cvc_colondb"):
                prefix_counts["cvc_colondb"] +=1
            elif f.name.startswith("kvasir"):
                prefix_counts["kvasir"] +=1
            elif f.name.startswith("etis"):
                prefix_counts["etis"] +=1
            else:
                prefix_counts["other"] +=1
        output.append("- Dataset Breakdown:")
        for k, v in sorted(prefix_counts.items()):
            output.append(f"  - {k}: {v} images")

        # Check hyperkvasir original
        hk_polyp_dir = Path(r"C:\Users\ADMIN\Desktop\POLYP_DETECTION\hyper-kvasir-labeled-images (1)\labeled-images\lower-gi-tract\pathological-findings\polyps")
        if hk_polyp_dir.exists():
            n_polyps = len(list(hk_polyp_dir.glob("*.jpg")))
            output.append(f"- HyperKvasir Polyps (Original): {n_polyps} images")
    output.append("")

    # 2. Checkpoint Information
    output.append("## 2. Checkpoint Information")
    output.append("-" * 80)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_files = sorted(checkpoints_dir.glob("*.pth"))
        for cp_file in checkpoint_files:
            try:
                cp = torch.load(cp_file, map_location="cpu", weights_only=False)
                output.append(f"- {cp_file.name}:")
                output.append(f"  - Epoch: {cp.get('epoch', 'N/A')}")
                output.append(f"  - Best Dice: {cp.get('best_dice', 'N/A'):.4f}" if isinstance(cp.get('best_dice'), float) else f"  - Best Dice: {cp.get('best_dice')}")
                output.append(f"  - Best Threshold: {cp.get('best_threshold', 'N/A')}")
                output.append("")
            except Exception as e:
                output.append(f"- {cp_file.name}: ERROR loading - {str(e)}")
                output.append("")

    # 3. Paper info files
    output.append("## 3. Paper Information Files")
    output.append("-" * 80)
    paper_info_dir = PROJECT_ROOT / "paper_information"
    if paper_info_dir.exists():
        paper_json = paper_info_dir / "paper_info.json"
        if paper_json.exists():
            with open(paper_json, "r") as f:
                data = json.load(f)
                output.append("### paper_info.json")
                output.append("```json")
                output.append(json.dumps(data, indent=2))
                output.append("```")
        output.append("")

        paper_md = paper_info_dir / "paper_information.md"
        if paper_md.exists():
            output.append("### paper_information.md content:")
            with open(paper_md, "r") as f:
                output.append(f.read())
        output.append("")

    # 4. Training History
    output.append("## 4. Training History")
    output.append("-" * 80)
    log_dir = PROJECT_ROOT / "logs_improved"
    log_json = log_dir / "training_history.json"
    if log_json.exists():
        with open(log_json, "r") as f:
            data = json.load(f)
            output.append("### logs_improved/training_history.json")
            output.append(f"- Best Dice: {data.get('best_dice', 'N/A'):.4f}" if isinstance(data.get('best_dice'), float) else f"- Best Dice: {data.get('best_dice')}")
            output.append(f"- Best Threshold: {data.get('best_threshold', 'N/A')}")
            output.append(f"- Number of Epochs in history: {len(data.get('train_loss', []))}")
            output.append("")

    # 5. Training Script Parameters
    output.append("## 5. Training Script Parameters")
    output.append("-" * 80)
    training_scripts = [
        PROJECT_ROOT / "train_combined_improved.py",
        PROJECT_ROOT / "train_combined_dataset.py",
        PROJECT_ROOT / "train_stage1_segmentation.py",
    ]
    for script in training_scripts:
        if script.exists():
            output.append(f"### {script.name}")
            with open(script, "r") as f:
                content = f.read()
                # Extract configs between markers or just first 50 lines
                output.append("First 100 lines of script:")
                output.append("```python")
                output.append("\n".join(content.splitlines()[:100]))
                output.append("```")
            output.append("")

    # 6. Requirements / Environment
    output.append("## 6. Requirements & Environment")
    output.append("-" * 80)
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        with open(req_file, "r") as f:
            output.append("### requirements.txt:")
            output.append("```")
            output.append(f.read())
            output.append("```")
        output.append("")

    # 7. Segmentation Metrics (from existing files)
    output.append("## 7. Segmentation Metrics")
    output.append("-" * 80)
    output.append("(No classification metrics found in files - this would need to be extracted from classification scripts)")
    output.append("")

    # 8. Missing Information
    output.append("## 8. Missing Information to Collect")
    output.append("-" * 80)
    missing = [
        "- Classification metrics (Accuracy, Precision, Recall, F1, AUC) for ResNet-50, ViT, Full Pipeline",
        "- Ablation study results for all configurations",
        "- Cross-dataset classification results",
        "- Computational complexity (FLOPs, params, FPS)",
        "- Figure captions and data sources",
        "- Multiple runs/seed statistics",
        "- References and author/funding info"
    ]
    for item in missing:
        output.append(item)

    return "\n".join(output)


if __name__ == "__main__":
    print("Extracting experimental information...")
    final_content = extract_paper_info()
    print("Writing output file...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"Done! Saved at: {OUTPUT_FILE}")
