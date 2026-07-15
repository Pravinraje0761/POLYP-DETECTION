
# Manuscript Preparation Summary

## Key Files Generated
- `split_analysis_results.json`: Per-source train/val/test split counts
- `manuscript_results.json`: All quantitative metrics from evaluation
- `manuscript_summary.md`: This summary


## Evaluation Results (Item 1, 2, 5, 7)

### Item 1: Pooled Test Set Metrics
Two models evaluated using **ultimate_split_indices.json** (fixed held-out test set, n=309):

#### Pretrained Unet++ (EfficientNet-B4, 320px)
| Metric | Value |
|--------|-------|
| Dice | 0.8774 |
| IoU | 0.8103 |
| Precision | 0.8848 |
| Recall | 0.8930 |
| Specificity | 0.9798 |
| Accuracy | 0.9723 |
| F1 | 0.8774 |

#### ADenseUNet_combined_honest (320px)
| Metric | Value |
|--------|-------|
| Dice | 0.7886 |
| IoU | 0.7007 |
| Precision | 0.8311 |
| Recall | 0.8059 |
| Specificity | 0.9827 |
| Accuracy | 0.9516 |
| F1 | 0.7886 |


### Item 2: Per-Source Breakdown
#### Pretrained Unet++
| Source | Dice | IoU | Precision | Recall | Specificity | Accuracy | F1 |
|--------|------|-----|-----------|--------|-------------|----------|----|
| CVC-ColonDB | 0.8426 | 0.7615 | 0.8501 | 0.8554 | 0.9872 | 0.9848 | 0.8426 |
| Kvasir-SEG | 0.8797 | 0.8165 | 0.8867 | 0.9025 | 0.9699 | 0.9609 | 0.8797 |
| CVC-ClinicDB | 0.8935 | 0.8281 | 0.9017 | 0.8989 | 0.9918 | 0.9839 | 0.8935 |

#### ADenseUNet_combined_honest
| Source | Dice | IoU | Precision | Recall | Specificity | Accuracy | F1 |
|--------|------|-----|-----------|--------|-------------|----------|----|
| CVC-ColonDB | 0.6793 | 0.5851 | 0.6996 | 0.7212 | 0.9845 | 0.9636 | 0.6793 |
| Kvasir-SEG | 0.8088 | 0.7212 | 0.8561 | 0.8268 | 0.9773 | 0.9375 | 0.8088 |
| CVC-ClinicDB | 0.8178 | 0.7333 | 0.8654 | 0.8200 | 0.9906 | 0.9677 | 0.8178 |


### Item 5: Test-Time Augmentation Comparison
#### Pretrained Unet++
| Metric | Without TTA | With TTA |
|--------|-------------|----------|
| Dice | 0.8774 | 0.8838 |
| IoU | 0.8103 | 0.8201 |
| Precision | 0.8848 | 0.8963 |
| Recall | 0.8930 | 0.8926 |
| Specificity | 0.9798 | 0.9834 |
| Accuracy | 0.9723 | 0.9750 |
| F1 | 0.8774 | 0.8838 |

#### ADenseUNet_combined_honest
| Metric | Without TTA | With TTA |
|--------|-------------|----------|
| Dice | 0.7886 | 0.7985 |
| IoU | 0.7007 | 0.7127 |
| Precision | 0.8311 | 0.8476 |
| Recall | 0.8059 | 0.8082 |
| Specificity | 0.9827 | 0.9847 |
| Accuracy | 0.9516 | 0.9538 |
| F1 | 0.7886 | 0.7985 |


### Item 7: Parameter Count
| Model | Encoder | Decoder | Total |
|-------|---------|---------|-------|
| Pretrained Unet++ | 17,548,616 | 3,264,497 | 20,813,113 |
| ADenseUNet_combined_honest | 3,442,720 | 3,142,150 | 6,584,870 |


### Item 12: Final Per-Source and Split Counts
#### Total Per Source
| Source | Count |
|--------|-------|
| CVC-ColonDB | 380 |
| CVC-ClinicDB | 672 |
| Kvasir-SEG | 1000 |
| **Total** | 2052 |

#### Train/Val/Test Per Source
| Source | Train | Val | Test |
|--------|-------|-----|------|
| CVC-ColonDB | 265 | 60 |55 |
| CVC-ClinicDB |469 |107 |96 |
| Kvasir-SEG |702 |140 |158 |
| **Total** |1436 |307 |309 |


### Item 13: Split Confirmation
- All evaluations used the fixed `checkpoints/ultimate_split_indices.json`, generated once and reused for all experiments on this combined dataset.


## Items Marked as "Requires Additional Training/Setup"
These items require running new training experiments:
- **Item 3**: Cross-dataset generalization (train on one source only, test on another fully disjoint source)
- **Item 4**: Multi-seed variance (2-3 different random seed runs with same split proportions)
- **Item 6**: Baseline comparison (plain UNet or DoubleUNet trained on same split)
- **Item 8**: FLOPs/inference latency measurement
- **Item 9**: Training time per epoch and total time (requires running full training and logging time)
- **Item 10**: Qualitative results (sample images with original, GT, predicted, overlay)
- **Item 11**: Training curves (loss/Dice vs epoch for both train and val)


## Items Marked as "Blocked (Requires Normal Mucosa Dataset)"
These require a labeled normal mucosa cohort, which we do not currently have:
- **Item 14**: Normal vs polyp classification metrics
- **Item 15**: Grad-CAM visualizations
- **Item 16**: Five-config ablation study (baseline ViT to full pipeline)
