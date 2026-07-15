
import os
import json


def get_source(filename):
    if filename.startswith("cvc_clinic") or filename.startswith("archive"):
        return "CVC-ClinicDB"
    elif filename.startswith("kvasir"):
        return "Kvasir-SEG"
    elif filename.startswith("etis"):
        return "ETIS-LaribPolypDB"
    elif filename.startswith("cvc_colon"):
        return "CVC-ColonDB"
    elif filename.startswith("hyperkvasir"):
        return "HyperKvasir"
    elif filename.startswith("cvc"):
        return "CVC-ClinicDB"  # Assume others?
    elif filename.startswith("1") or filename.startswith("2") or filename.startswith("3") or filename.startswith("4") or filename.startswith("5") or filename.startswith("6") or filename.startswith("7") or filename.startswith("8") or filename.startswith("9") or filename.startswith("0"):
        return "ETIS-LaribPolypDB"
    else:
        return "Unknown"


def main():
    split_path = "checkpoints/ultimate_split_indices.json"
    data_root = "combined_dataset"
    images_dir = os.path.join(data_root, "images")

    with open(split_path, "r") as f:
        split = json.load(f)

    image_files = sorted(os.listdir(images_dir))
    unknown_files = [f for f in image_files if get_source(f) == "Unknown"]
    print("=== UNKNOWN FILES (first 20) ===")
    for f in unknown_files[:20]:
        print(f)

    all_sources = [get_source(f) for f in image_files]
    total_counts = {s: 0 for s in set(all_sources)}
    for src in all_sources:
        total_counts[src] += 1

    split_counts = {
        "train": {s: 0 for s in set(all_sources)},
        "val": {s: 0 for s in set(all_sources)},
        "test": {s: 0 for s in set(all_sources)},
    }

    for split_name, indices in split.items():
        for idx in indices:
            src = get_source(image_files[idx])
            split_counts[split_name][src] += 1

    print("=== TOTAL COUNTS PER SOURCE ===")
    for src, cnt in total_counts.items():
        print(f"{src}: {cnt}")

    print("\n=== TRAIN/VAL/TEST COUNTS PER SOURCE ===")
    for split_name in ["train", "val", "test"]:
        print(f"\n{split_name.upper()}:")
        for src, cnt in split_counts[split_name].items():
            print(f"  {src}: {cnt}")
        print(f"  Total {split_name}: {sum(split_counts[split_name].values())}")

    results = {
        "total_counts": total_counts,
        "split_counts": split_counts,
    }
    with open("split_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved split_analysis_results.json!")


if __name__ == "__main__":
    main()
