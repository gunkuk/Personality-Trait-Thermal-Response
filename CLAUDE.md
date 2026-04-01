# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
# Standard run
python run_pipeline.py dataset_version=v1

# Skip stages
python run_pipeline.py dataset_version=v1 skip_build=true
python run_pipeline.py dataset_version=v1 skip_posthoc=true

# Select feature structure
python run_pipeline.py dataset_version=v1 data_structure=no_tlx

# Multi-run sweep (Hydra multirun)
python run_pipeline.py -m dataset_version=v1 data_structure=full,no_tlx,PHY,PSY

# View MLflow results
mlflow ui  # then open http://localhost:5000
```

## Pipeline Architecture

Three sequential stages, each as a separate subprocess:

1. **`00_data/modules/5. Input Matrix/build_dataset.py`** — reads `00_data/00_raw/rawdata_wrong corrected_0319.xlsx`, selects feature blocks per `data_structure`, standardizes, writes `00_data/02_processed/X_subject_{version}_{structure}.csv`

2. **`01_clustering/6. Clustering (K-meas, Ward, GMM)/run_clustering.py`** — reads the processed CSV via `configs/default.yaml` (overridden by a temp config), runs KMeans/Ward/GMM for k=2~6, writes results to `03_outputs/<timestamp>__<experiment_name>/`

3. **`02_notebooks/PER_posthoc.py`** — reads cluster assignments + personality xlsx, runs FFM/MBTI continuous/MBTI binary analyses, writes into `<clustering_run_dir>/post_hoc/`

`run_pipeline.py` is the orchestrator. It uses **Hydra** (`configs/pipeline.yaml`) for config and starts an **MLflow** parent run that wraps all three stages.

## Config Files

- `configs/pipeline.yaml` — Hydra config for `run_pipeline.py` (dataset_version, data_structure, tracking_uri, skip_* flags, posthoc_analysis_modes)
- `configs/default.yaml` — clustering config for `run_clustering.py` (k_values, blocks, analysis params)

## Feature Blocks

Defined identically in `build_dataset.py` and `default.yaml`:
- **PHY**: HR, EDA tonic, pulse amplitude, SKT
- **PSY**: TSV, thermal comfort (m2)
- **BHR**: SET-point delta (p7-m7)
- **TLX**: NASA-TLX cognitive load
- **EUP**: euphoria items (eup5, eup16)

`data_structure` can be `full`, `no_tlx`, or any comma-separated subset of block names (e.g., `PHY,PSY`).

## MLflow Tracking

`tracking_uri` defaults to `sqlite:///mlflow.db` (local file, no server needed). To use the MLflow UI, run `mlflow ui` separately and change `tracking_uri` to `http://127.0.0.1:5000` in `configs/pipeline.yaml`.

## Working with Claude Code

**Test runs and commits are handled by the user, not Claude.**
- When debugging is needed, the user runs commands and pastes output back to Claude.
- Claude should describe what to run and what output to look for, then wait.
- Claude should not execute `run_pipeline.py` or long-running processes itself.

## Output Structure

```
03_outputs/<timestamp>__<experiment_name>/
  summary.xlsx           # best clustering solution per method
  xlsx/                  # per-method label files
  figures/               # plots
  post_hoc/              # PER_posthoc.py results (ffm, mbti_cont, mbti_bin)
  run_metadata.json
  used_config.yaml
```
