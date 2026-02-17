# SilentWear

Code release for **SilentWear: an Ultra-Low Power Wearable Interface for EMG-based Silent Speech Recognition**

This repository is organized as a **reproducibility artifact**:
1. *(Optional)* convert raw BioGUI recordings to `.h5`
2. Generate **EMG windows + features**
3. Run **Global** and **Inter-Session** experiments
4. generate **paper tables/figures** from saved run summaries

> Dataset: released separately on Hugging Face. The code can either use the released `wins_and_features/` directly (fast path),
> or regenerate windows/features for new window sizes (ablation path).

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data layout

Set `DATA_DIR` to the dataset root folder. The code expects the Hugging Face release layout:

```text
DATA_DIR/
├── data_raw_and_filt/
└── wins_and_features/
```

If you want to collect your own data using your own recordings from, see **Optional: raw data preprocessing** below.

---

## Reproduce paper results (recommended)

Run the end-to-end script:

```bash
export DATA_DIR=/path/to/dataset_root
bash scripts/reproduce_paper.sh
```

Outputs are written to:

```text
artifacts/
├── models/     # per-run folders, each with cv_summary.csv + run_cfg.json
├── tables/     # CSVs used for paper numbers
└── figures/    # SVG confusion matrices and plots
```

### Regenerate windows/features for ablations

The dataset ships one example windowing configuration. To rerun ablations with a different window size:

```bash
export DATA_DIR=/path/to/dataset_root
export REGEN_WINDOWS=1
export WINDOW_S=1.4
bash scripts/reproduce_paper.sh
```

---

## Optional: raw data preprocessing (only if you collected new data)

If you recorded new data using BioGUI, convert `.bio` recordings to `.h5` using:

```text
utils/I_data_preparation/data_preparation.py
```

Then run windowing/feature extraction as above.

---

## Configs

Open-release config templates (no internal paths):
- `config/open_release_create_windows.yaml`
- `config/open_release_base_models_config.yaml`

Model definitions live in:
- `config/models_configs/`

---

## Citation

See the dataset card / paper for citation details.
