
# Manuscript Item Scripts

## Item 3: Cross-Dataset Generalization
- **File**: `item3_cross_dataset.py`
- **Run**: `python item3_cross_dataset.py`
- **Output**: `cross_dataset_results/` directory with trained models and `cross_dataset_results.json`
- **Description**: Trains on each source dataset and tests on all other disjoint sources

## Item 4: Multi-Seed Variance
- **File**: `item4_multi_seed.py`
- **Run**: `python item4_multi_seed.py`
- **Output**: `multi_seed_results/` directory with per-seed models and `multi_seed_results.json` (mean/std test dice)
- **Seeds Used**: [42, 1337, 9999]

## Item 6: Baseline Comparison
- **File**: `item6_baseline.py`
- **Run**: `python item6_baseline.py`
- **Output**: `baseline_results/` directory with trained baseline UNet and `baseline_results.json`
- **Baseline Model**: Plain UNet (no pretrained encoder)

## Item 8: FLOPs & Latency
- **File**: `item8_flops_latency.py`
- **Run**: `python item8_flops_latency.py`
- **Output**: `flops_latency_results.json`
- **Requires**: `thop` (install with `pip install thop`)

## Item 9: Training Time Tracking
- **File**: `item9_training_time.py`
- **Run**: `python item9_training_time.py`
- **Output**: `training_time_results/` directory and `training_time_results.json`
- **Tracks**: Total time, avg epoch time, per-epoch time

## Item 10: Qualitative Results
- **File**: `item10_qualitative.py`
- **Run**: `python item10_qualitative.py`
- **Output**: `qualitative_results/` directory with PNG images of examples (original, GT, prediction, overlay)
- **Requires**: `matplotlib` (install with `pip install matplotlib`)

## Item 11: Training Curves
- **File**: `item11_training_curves.py`
- **Run**: `python item11_training_curves.py`
- **Output**: `training_curves_results/` directory with PNG plots
- **Requires**: `matplotlib`
- **Depends On**: Output from items 3, 6, or 9 being present
