# Reproducing the paper

Every number in the paper is produced by a standalone script. This page maps each table and
figure to that script and says whether the script is in this repository yet.

**Status key**
- ✅ in repo and matches the protocol described in the paper
- ⚠️ in repo, but an earlier version that differs from the paper's final protocol (difference noted)
- ⏳ cited in the paper, not yet uploaded; will be added before publication

Run every command from the repository root after `pip install -e .` and preparing the data
as described in [DATA.md](DATA.md). Scripts read `combined_dataset/` and
`checkpoints/ultimate_split_indices.json` relative to the current directory.

## Segmentation

| Paper item | What it reports | Script | Status |
|---|---|---|---|
| Table 1 | Dataset composition and fixed split | `experiments/segmentation/prepare_combined_dataset.py`, `experiments/segmentation/analyze_split.py` | ⚠️ `analyze_split.py` counts CVC-300 under CVC-ClinicDB (672 images, as in `results/manuscript_summary.md`); the paper's Table 1 lists them separately |
| Section 4.1, Table 5 row 3 | A-DenseUNet training (Focal Tversky + boundary loss, 320 px, AdamW) | `experiments/segmentation/train_adenseunet.py` | ⚠️ architecture (6,584,870 params, checked in `tests/`) and loss weights match; this version trains at 224 px on its own `random_split(seed=42)` rather than at 320 px on the fixed split file |
| Table 4 | Test-time augmentation, UNet++ and A-DenseUNet | `experiments/segmentation/evaluate_test_set.py` → `results/manuscript_results.json` | ✅ reproduces Table 4 exactly |
| Table 3, Supp. Table S1 | UNet++ pooled and per-source metrics | `experiments/segmentation/evaluate_test_set.py` | ⚠️ this version gives UNet++ Dice 0.8774; Table 3 reports 0.8799 from a later re-evaluation, with CVC-300 reported separately |
| Table 5 row 1 | Plain U-Net from scratch, shared protocol | `experiments/segmentation/train_baseline_unet.py` | ✅ BCE, AdamW, cosine warm restarts, 320 px, 100 epochs / patience 15 |
| Table 5 row 2 | U-Net with ImageNet-pretrained ResNet-34 | — | ⏳ the repo's baseline script trains only the from-scratch row (`encoder_weights=None`) |
| Table 5 rows 5–6, Section 6.5 | Same-split PraNet and ResUNet++ retrains; Res2Net weight transplant | `train_samesplit_sota.py` | ⏳ |
| Section 6.3 | Multi-seed variance (seeds 42 / 123 / 2024, split resampled per seed) | `experiments/segmentation/multi_seed.py` | ⚠️ this version uses seeds 42 / 1337 / 9999 and keeps the fixed split |
| Section 6.6 | Paired bootstrap, UNet++ vs. A-DenseUNet | — | ⏳ |
| Table 14, Figure 10 | Cross-dataset transfer, 6 source pairs | `experiments/segmentation/cross_dataset.py` | ⚠️ partitions by source, so train and test are disjoint by construction; the explicit filename-overlap assertion described in the paper is not in this version |
| Table 15 | Parameters, FLOPs, latency | `experiments/segmentation/compute_cost.py` | ⚠️ covers A-DenseUNet and UNet++; the ViT row is not in this version |
| — | Training time per epoch | `experiments/segmentation/training_time.py` | ✅ supporting detail, not a paper table |
| Figure 3 | Qualitative segmentation examples | `generate_qualitative_figure.py` | ⏳ (earlier version: `experiments/segmentation/qualitative_examples.py`) |
| — | Training curves | `experiments/segmentation/training_curves.py` | ✅ supporting detail |

## Classification and the failure mechanism

The classification-stage code in `src/polypgate/preprocessing/` and
`src/polypgate/models/vit_classifier.py` is an **earlier version** than the one in the paper
(it uses mask dilation and blur, conditional gamma, and a torchvision LoRA ViT, where the paper
uses `S = P**1.5`, fixed gamma 0.9 and a timm ViT-B/16 with a single-logit head). The final
classification code is among the scripts below.

| Paper item | What it reports | Script | Status |
|---|---|---|---|
| Tables 7–8, Figure 4 | Five-configuration ablation on Raw / Corrected / v2 | `polyp_pipeline_completion/phase1_essential/04_ablation_study*.py` | ⏳ (an earlier three-mode version is in `archive/ablation_study.py`) |
| Table 9, Figures 6–8 | Crop vs. white-balance mechanism ablation | `check_distribution_shift_mechanism.py`, `generate_mechanism_figure.py` | ⏳ |
| Figure 5 | Probability-collapse examples | `generate_collapse_figure.py` | ⏳ |
| Table 10 | Same check on UNet++ | `check_unetpp_mechanism.py` | ⏳ |
| Section 6.12 | Colour-augmented A-DenseUNet retrain (strong and weak) | `train_adenseunet_augfix.py`, `train_adenseunet_augfix_weak.py`, `check_augfix_mechanism.py`, `check_augfix_mechanism_weak.py` | ⏳ |
| Tables 11–12, Figure 9 | Classification recovery with the fixed segmenter | `04_ablation_study_augfix.py`, `04_ablation_study_augfix_v2.py`, `generate_extra_figures.py` | ⏳ |
| Section 6.13, Table 13 | Near-duplicate leakage check and sensitivity sweep | `check_classification_leakage.py`, `leakage_sensitivity_analysis.py` | ⏳ |
| Section 6.16 | ViT classifier training | — | ⏳ (earlier version: `archive/train.py`, which does not run as is) |
| Table 16, Figure 11 | Grad-CAM / Grad-CAM++ metrics | — | ⏳ (Grad-CAM and attention rollout implementations: `polypgate.explain`) |

## Split files

| File | Used by | Status |
|---|---|---|
| `checkpoints/ultimate_split_indices.json` | every segmentation experiment | ⏳ |
| `checkpoints/vit_cls_split.json`, `vit_cls_split_corrected.json`, `vit_cls_split_v2.json` | classification experiments | ⏳ |

## Software and hardware used in the paper

Python 3.10.11, PyTorch 2.6.0 (CUDA 12.4), torchvision 0.21.0, timm 1.0.28,
albumentations 2.0.8, one NVIDIA RTX 3060 (12 GB). `environment.yml` pins these.

Seeds: segmentation primary split 42; multi-seed 42 / 123 / 2024; classification and
mechanism-ablation sampling 42.
