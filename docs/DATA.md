# Data

All data used in the paper is public. Nothing in this repository contains patient data,
and no dataset files are committed: download them yourself and place them as described below.

## Segmentation (Sections 3, 6.1–6.6, 6.14)

Four sources, pooled into one dataset of **2,052** images with pixel-level masks:

| Source | Images | Reference | Where to get it |
|---|---:|---|---|
| CVC-ClinicDB | 612 | Bernal et al., 2015 | <https://polyp.grand-challenge.org/CVCClinicDB/> |
| CVC-300 | 60 | Bernal et al., 2015 | Distributed with the PraNet / EndoScene test sets |
| CVC-ColonDB | 380 | Bernal et al., 2012 | Distributed with the PraNet test sets |
| Kvasir-SEG | 1,000 | Jha et al., 2020 | <https://datasets.simula.no/kvasir-seg/> |

`experiments/segmentation/prepare_combined_dataset.py` builds the pooled folder:

```bash
python experiments/segmentation/prepare_combined_dataset.py \
    --clinicdb-dir path/to/CVC-ClinicDB/PNG \
    --kvasir-dir   path/to/Kvasir-SEG \
    --test-dir     path/to/TestDataset          # contains CVC-300/ and CVC-ColonDB/
```

```
combined_dataset/
├── images/   # <source-prefix>_<original-name>.png
└── masks/    # same file names, binary masks
```

The per-image source is recovered from the filename prefix the script writes:
`archive_*` (CVC-ClinicDB), `kvasir_*` (Kvasir-SEG), `cvc_300_*` (CVC-300) and
`cvc_colondb_*` (CVC-ColonDB). Every per-source table is computed this way.

### Fixed split

A single 70 / 15 / 15 split (1,436 / 307 / 309 images) was generated once with seed 42 and
reused for **every** segmentation experiment (all six models in Table 5). Zero overlap between
train/val/test was verified programmatically. The split file is
`checkpoints/ultimate_split_indices.json`.

| Source | Total | Train | Val | Test |
|---|---:|---:|---:|---:|
| CVC-ClinicDB | 612 | 427 | 97 | 88 |
| CVC-300 | 60 | 42 | 10 | 8 |
| CVC-ColonDB | 380 | 265 | 60 | 55 |
| Kvasir-SEG | 1,000 | 702 | 140 | 158 |
| **Total** | **2,052** | **1,436** | **307** | **309** |

## Classification (Sections 3.1, 6.7–6.13, 6.16)

From **HyperKvasir** (Borgli et al., 2020, <https://datasets.simula.no/hyper-kvasir/>,
labeled-images release). Three dataset variants are used:

| Variant | Positive class | Negative class | Images | Split (train/val/test) | Split file |
|---|---|---|---:|---|---|
| Raw | 1,028 polyp frames | 4,104 normal-mucosa landmark frames | 5,132 | 3,594 / 769 / 769 | `checkpoints/vit_cls_split.json` |
| Corrected | same | same, after 60% center-crop + gray-world white balance + per-channel normalisation | 5,132 | 3,594 / 769 / 769 | `checkpoints/vit_cls_split_corrected.json` |
| v2 (pathology-matched) | same, corrected | 857 ulcerative-colitis grade 0–3 and hemorrhoid frames, corrected | 1,885 | 1,321 / 282 / 282 | `checkpoints/vit_cls_split_v2.json` |

Why three variants: the Raw variant contains two shortcuts the authors identified themselves.
A linear probe on the four corner pixels alone reaches 97.75% accuracy (burned-in on-screen
overlays and a colour difference between capture sessions), and an 8×8 thumbnail probe still
reaches 92.67% after that fix (close-up vs. wide-angle shot types). The Corrected and v2
variants remove these in turn. See Section 3.1 of the paper.

## Known limitations of the data

- HyperKvasir's public release has **no patient or procedure identifiers**, so classification
  splits are image-level. Section 6.13 quantifies the resulting near-duplicate risk
  (1.30% of test images) and shows it does not inflate accuracy.
- None of the segmentation sources publish patient identifiers or demographics, so splits are
  image-level and no subgroup analysis is possible.
- CVC-300 contributes only 8 test images; its per-source numbers are not meaningful on their own.

> **Status:** the split JSON files are not yet in the repository. They will be added together
> with the remaining experiment scripts (see [REPRODUCING.md](REPRODUCING.md)).
