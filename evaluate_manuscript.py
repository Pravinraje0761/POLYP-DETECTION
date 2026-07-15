
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from adenseunet_boosted import ADenseUNet_Boosted


DEVICE = torch.device("cpu")
DATA_ROOT = "combined_dataset"
SPLIT_PATH = "checkpoints/ultimate_split_indices.json"


class PolypDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.transform = transform
        images_dir = os.path.join(data_root, "images")
        masks_dir = os.path.join(data_root, "masks")
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.sources = [self._get_source(f) for f in self.image_files]

    @staticmethod
    def _get_source(filename):
        if filename.startswith("cvc_clinic") or filename.startswith("archive") or filename.startswith("cvc_300"):
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
            return "CVC-ClinicDB"
        elif filename.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            return "ETIS-LaribPolypDB"
        else:
            return "Unknown"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image, mask, self.sources[idx], idx


def get_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def compute_metrics(y_true, y_pred, y_score):
    """
    Compute all required metrics
    y_true: H,W binary
    y_pred: H,W binary
    y_score: H,W float (probability)
    Returns: dict with all metrics
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    eps = 1e-8
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def compute_average_metrics(metrics_list):
    avg = {}
    keys = ["dice", "iou", "precision", "recall", "specificity", "accuracy", "f1"]
    for key in keys:
        avg[key] = np.mean([m[key] for m in metrics_list])
    return avg


def predict_tta(model, img_tensor):
    """
    Test-time augmentation: average predictions over original, horizontal flip, vertical flip
    """
    with torch.no_grad():
        preds = [torch.sigmoid(model(img_tensor))]
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=(3,)))), dims=(3,)))  # hflip
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(img_tensor, dims=(2,)))), dims=(2,)))  # vflip
        return torch.stack(preds, 0).mean(dim=0)


def evaluate_model(model, test_loader, use_tta=False, model_name="model"):
    """
    Evaluate model on test_loader
    Returns:
      all_metrics: list of per-image metrics
      per_source_metrics: dict of source -> list of per-image metrics
    """
    model.eval()
    all_metrics = []
    per_source_metrics = {}

    for batch in tqdm(test_loader, desc=f"Evaluating {model_name} (tta={use_tta})"):
        imgs, masks, sources, idxs = batch
        imgs = imgs.to(DEVICE)

        with torch.no_grad():
            if use_tta:
                probs = predict_tta(model, imgs)
            else:
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)

        probs_np = probs.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(probs_np.shape[0]):
            src = sources[i]
            prob = probs_np[i, 0]  # (H, W)
            mask = masks_np[i]     # (H, W) because our dataset returns (1, H, W)
            if len(mask.shape) == 3:
                mask = mask[0]
            pred = (prob > 0.5).astype(np.float32)
            metrics = compute_metrics(mask, pred, prob)
            all_metrics.append(metrics)
            if src not in per_source_metrics:
                per_source_metrics[src] = []
            per_source_metrics[src].append(metrics)

    avg_all = compute_average_metrics(all_metrics)
    avg_per_source = {}
    for src, metrics_list in per_source_metrics.items():
        avg_per_source[src] = compute_average_metrics(metrics_list)

    return {
        "all": avg_all,
        "per_source": avg_per_source
    }


def count_parameters(model):
    encoder_params = 0
    decoder_params = 0
    # For ADenseUNet_Boosted:
    if hasattr(model, "init_conv"):  # ADenseUNet_Boosted
        encoder = [model.init_conv, model.dense1, model.trans1, model.dense2, model.trans2, model.dense3, model.trans3, model.bottleneck]
        decoder = [model.up3, model.att3, model.dec3, model.up2, model.att2, model.dec2, model.up1, model.att1, model.dec1, model.up_final, model.final_conv]
        for mod in encoder:
            encoder_params += sum(p.numel() for p in mod.parameters())
        for mod in decoder:
            decoder_params += sum(p.numel() for p in mod.parameters())
    elif hasattr(model, "encoder") and hasattr(model, "decoder"):  # SMP models
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        if hasattr(model, "segmentation_head"):
            decoder_params += sum(p.numel() for p in model.segmentation_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "encoder": encoder_params,
        "decoder": decoder_params,
        "total": total_params
    }


def load_adenseunet(checkpoint_path):
    model = ADenseUNet_Boosted(in_channels=3, out_channels=1).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # Load state dict, handle key differences (up_final vs final_up)
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("final_up."):
            new_state_dict[k.replace("final_up.", "up_final.")] = v
        elif k.startswith("up_final."):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    return model, ckpt


def load_smp_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = smp.UnetPlusPlus(
        encoder_name=ckpt["encoder"], encoder_weights=None, in_channels=3, classes=1, deep_supervision=False
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def main():
    print("=" * 80)
    print("STARTING MANUSCRIPT EVALUATION")
    print("=" * 80)

    print("\n--- Step 1: Load dataset and split ---")
    base_ds = PolypDataset(DATA_ROOT)
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)

    results = {
        "item1": {}, "item2": {}, "item5": {}, "item7": {}
    }

    # ----------------------------------------------------------------------
    # Evaluate Pretrained Unet++ (uses 320px and this split)
    # ----------------------------------------------------------------------
    print("\n--- Step 2: Evaluate Pretrained Unet++ (320px) ---")
    model_pp, ckpt_pp = load_smp_model("checkpoints/best_pretrained_unetpp.pth")
    val_transforms_pp = get_transforms(320)
    test_ds_pp = Subset(PolypDataset(DATA_ROOT, val_transforms_pp), split["test"])
    test_loader_pp = DataLoader(test_ds_pp, batch_size=2, shuffle=False, num_workers=0)
    results_no_tta_pp = evaluate_model(model_pp, test_loader_pp, use_tta=False, model_name="Pretrained_UnetPP")
    results_tta_pp = evaluate_model(model_pp, test_loader_pp, use_tta=True, model_name="Pretrained_UnetPP")
    params_pp = count_parameters(model_pp)
    print("Pretrained Unet++ params:", params_pp)

    results["item1"]["Pretrained_UnetPP"] = results_no_tta_pp["all"]
    results["item2"]["Pretrained_UnetPP"] = results_no_tta_pp["per_source"]
    results["item5"]["Pretrained_UnetPP"] = {
        "without_tta": results_no_tta_pp["all"],
        "with_tta": results_tta_pp["all"]
    }
    results["item7"]["Pretrained_UnetPP"] = params_pp

    # ----------------------------------------------------------------------
    # Evaluate ADenseUNet_combined_honest (uses 320px and this split)
    # ----------------------------------------------------------------------
    print("\n--- Step 3: Evaluate ADenseUNet_combined_honest (320px) ---")
    model_ad, ckpt_ad = load_adenseunet("checkpoints/best_adenseunet_combined_honest.pth")
    val_transforms_ad = get_transforms(320)
    test_ds_ad = Subset(PolypDataset(DATA_ROOT, val_transforms_ad), split["test"])
    test_loader_ad = DataLoader(test_ds_ad, batch_size=2, shuffle=False, num_workers=0)
    results_no_tta_ad = evaluate_model(model_ad, test_loader_ad, use_tta=False, model_name="ADenseUNet_combined_honest")
    results_tta_ad = evaluate_model(model_ad, test_loader_ad, use_tta=True, model_name="ADenseUNet_combined_honest")
    params_ad = count_parameters(model_ad)
    print("ADenseUNet_combined_honest params:", params_ad)

    results["item1"]["ADenseUNet_combined_honest"] = results_no_tta_ad["all"]
    results["item2"]["ADenseUNet_combined_honest"] = results_no_tta_ad["per_source"]
    results["item5"]["ADenseUNet_combined_honest"] = {
        "without_tta": results_no_tta_ad["all"],
        "with_tta": results_tta_ad["all"]
    }
    results["item7"]["ADenseUNet_combined_honest"] = params_ad

    with open("manuscript_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("Evaluation complete! Saved manuscript_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
