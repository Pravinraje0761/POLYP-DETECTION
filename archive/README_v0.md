# Colorectal Polyp Detection Framework

A complete end-to-end framework for colorectal polyp detection using Hyper-Kvasir dataset.

## Pipeline Overview

1. **Stage 1 - Segmentation**: A-DenseUNet for polyp segmentation
2. **Stage 2 - ROI Extraction**: Soft ROI extraction using probability mask (no hard thresholding)
3. **Stage 3 - RGB Enhancement**: Selective RGB enhancement (contrast stretching, gamma correction, white balance, channel normalization) applied only inside ROI
4. **Stage 4 - LAB + CLAHE**: Convert to LAB color space and apply CLAHE only on L channel
5. **Stage 5 - Classification**: Vision Transformer (pretrained ViT-B/16) for polyp classification
6. **Stage 6 - Training**: 70/15/15 train/val/test split with extensive augmentations, mixed precision, cosine annealing, early stopping
7. **Stage 7 - Evaluation**: Accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix
8. **Stage 8 - Grad-CAM**: Visual explanations with 6-panel figures
9. **Stage 9 - Attention Rollout**: Transformer attention visualization
10. **Stage 10 - Explainability Metrics**: Attention Concentration Ratio (ACR), Grad-CAM IoU, Pointing Game
11. **Stage 11 - Ablation Study**: Compare three model variants

## Project Structure

```
POLYP_DETECTION/
├── adenseunet.py          # A-DenseUNet model
├── ablation_study.py      # Ablation study script
├── attention_rollout.py   # Attention rollout implementation
├── clahe.py               # LAB + CLAHE enhancement
├── dataset.py             # Dataset and dataloaders
├── evaluate.py            # Evaluation and explainability
├── gradcam.py             # Grad-CAM implementation
├── metrics.py             # All metrics (classification, segmentation, explainability)
├── preprocessing.py       # Complete preprocessing pipeline
├── rgb_enhancement.py     # Selective RGB enhancement
├── roi_extraction.py      # Soft ROI extraction
├── train.py               # Training script
├── visualization.py       # Visualization functions
├── vit_classifier.py      # ViT classifier with Focal Loss
├── requirements.txt       # Dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train A-DenseUNet (Segmentation)
First, train the segmentation model using the original `train_segmentation.py` (from earlier work) or implement it using `adenseunet.py`.

### 2. Train Polyp Classifier
```bash
python train.py --data_root "hyper-kvasir-labeled-images (1)/labeled-images" --seg_checkpoint "path/to/seg_checkpoint.pth" --ablation "full"
```

### 3. Evaluate
```bash
python evaluate.py --data_root "hyper-kvasir-labeled-images (1)/labeled-images" --seg_checkpoint "path/to/seg_checkpoint.pth" --cls_checkpoint "path/to/cls_checkpoint.pth"
```

### 4. Run Ablation Study
```bash
python ablation_study.py
```

## Modules

### `adenseunet.py`
- A-DenseUNet model with dense blocks, atrous convolutions, attention gates
- Combined loss: Dice + BCE

### `roi_extraction.py`
- Soft ROI extraction: ROI = mask × image
- Preserves context around polyp

### `rgb_enhancement.py`
- Contrast stretching
- Gamma correction
- White balance
- RGB channel normalization

### `clahe.py`
- RGB → LAB conversion
- CLAHE (clipLimit=2.0, tileGridSize=(8,8)) on L channel
- LAB → RGB conversion

### `vit_classifier.py`
- Pretrained ViT-B/16
- Focal Loss for handling class imbalance
- Attention weight extraction hooks

### `metrics.py`
- Classification: accuracy, precision, recall, specificity, F1, ROC-AUC
- Segmentation: Dice, IoU
- Explainability: ACR, Grad-CAM IoU, Pointing Game
- Visualization: ROC curve, confusion matrix, training curves

### `gradcam.py`
- Grad-CAM implementation for ViT
- Verifies gradients and activations
- No `torch.no_grad()` during generation

### `attention_rollout.py`
- Attention rollout for ViT
- Head fusion (mean/max)
- Residual connections

### `visualization.py`
- Functions for visualizing all stages
- Publication-quality figures

## Citation
If you use this code, please consider citing the relevant papers:
- A-DenseUNet paper
- ViT paper
- Hyper-Kvasir dataset paper
```
