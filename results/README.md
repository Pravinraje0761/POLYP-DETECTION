# Results

Metric files written by the scripts in `experiments/`, committed so the reported numbers can
be checked without re-running anything.

| File | Produced by | Contents |
|---|---|---|
| `manuscript_results.json` | `experiments/segmentation/evaluate_test_set.py` | Fixed-split test metrics for UNet++ and A-DenseUNet, per-source breakdown, with/without TTA, parameter counts (paper Table 4) |
| `split_analysis_results.json` | `experiments/segmentation/analyze_split.py` | Per-source train/val/test counts of the fixed split |
| `manuscript_summary.md` | written during manuscript preparation | Human-readable summary of the two files above |

These files predate some final paper numbers; [docs/REPRODUCING.md](../docs/REPRODUCING.md)
notes where a table in the paper comes from a later run.
