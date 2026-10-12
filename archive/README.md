# Archive: development history

This folder keeps the exploratory and intermediate code written while the project
developed: earlier model variants, training runs tuned against a Dice target, checkpoint
inspection utilities, one-off demos, and the first version of the classification pipeline.

Nothing here is needed to reproduce the paper. It is kept, rather than deleted, so the
history of how the final experiments were reached stays inspectable. The paper's
experiments are in [`../experiments/`](../experiments) and are mapped table by table in
[`../docs/REPRODUCING.md`](../docs/REPRODUCING.md).

**Status:** not maintained. Imports have been updated so every file still compiles and
resolves against the `polypgate` package, but scripts have not been re-run since they
were archived. Several contain hard-coded local paths.

Run a script from the repository root (after `pip install -e .`) with:

```bash
python archive/<script>.py
```

Sibling modules in this folder (such as `adenseunet_simple.py`) are found automatically
because Python puts the script's own folder on the import path.

## What is here

| Group | Files | Notes |
|---|---|---|
| Earlier A-DenseUNet variants | `adenseunet.py`, `adenseunet_fixed.py`, `adenseunet_simple*.py`, `adenseunet_checkpoint_matching.py`, `adenseunet_improved_v2.py`, `ultimate_adenseunet.py`, `src/` | Superseded by `polypgate.models.ADenseUNet`. `adenseunet.py` and `adenseunet_fixed.py` raise a shape error at every input size tested (224 to 512 px) and are kept for reference only. |
| Earlier segmentation training runs | `train_*.py`, `new_model_train_model.py` | Exploratory runs. File names such as `train_to_93.py` record the Dice target being explored at the time; the paper reports the final, fixed-split results instead. |
| First classification pipeline | `train.py`, `evaluate.py`, `ablation_study.py`, `run_all_stages_*.py`, `end_to_end_model.py`, `hyper_kvasir_11_stages.py` | Three-mode ablation that predates the paper's five-configuration design. Depends on `adenseunet.py` above, so it does not run as is. |
| Evaluation and inspection utilities | `evaluate_*.py`, `check_*.py`, `inspect_*.py`, `print_checkpoint_stats.py`, `fix_checkpoint*.py`, `test_*.py` | Ad hoc checkpoint and data checks. |
| Demos and batch processing | `demo_*.py`, `process_*.py`, `inference*.py`, `visualize_*.py`, `find_polyp.py`, `sample_archive_dataset_demo.py` | Stage-by-stage visual demos. |
| Notes | `README_v0.md`, `ITEM_README.md`, `collect_all_paper_information.py`, `extract_experimental_data.py` | Original README and manuscript-preparation notes. |
