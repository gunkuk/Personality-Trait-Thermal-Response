# 핵심: 기존 3개 전처리 스크립트를 순차 실행한 뒤, versioned clustering input dataset과 manifest를 생성한다.
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class BuildDatasetConfig:
    dataset_version: str = "v1"
    data_structure: str = "full"
    experiment_name: str = "subject_level_dataset_build"
    tracking_uri: str = "mlruns"
    run_name: str | None = None
    use_existing_interim: bool = False
    register_latest_alias: bool = True


def parse_args() -> BuildDatasetConfig:
    parser = argparse.ArgumentParser(description="Build versioned clustering dataset from raw/interim files.")
    parser.add_argument("--dataset_version", type=str, required=True, help="Dataset version tag. e.g. v1, v2")
    parser.add_argument("--data_structure", type=str, default="full", help="Feature structure preset. e.g. full, no_tlx, no_eup, phy_psy_only")
    parser.add_argument("--experiment_name", type=str, default="subject_level_dataset_build")
    parser.add_argument("--tracking_uri", type=str, default="mlruns")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_existing_interim", action="store_true", help="Skip raw->interim generation and reuse existing interim workbook.")
    parser.add_argument("--no_latest_alias", action="store_true", help="Do not overwrite current/latest alias files.")
    args = parser.parse_args()
    return BuildDatasetConfig(
        dataset_version=args.dataset_version,
        data_structure=args.data_structure,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name,
        use_existing_interim=args.use_existing_interim,
        register_latest_alias=not args.no_latest_alias,
    )


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent.parent    # MBTI 루트
STANDARDIZATION_DIR = PROJECT_ROOT / "00_data" / "modules" / "3. standarization"
PROCESSED_DIR = PROJECT_ROOT / "00_data" / "02_processed"
INTERIM_DIR = PROJECT_ROOT / "00_data" / "01_interim"
ARTIFACT_DIR = PROJECT_ROOT / "01_model" / "OUT" / "dataset_build"

FEATURE_SCRIPT = STANDARDIZATION_DIR / "build_input_matrix_subject_level.py"
IMPUTE_SCRIPT = STANDARDIZATION_DIR / "missing_data.py"
BLOCK_STD_SCRIPT = STANDARDIZATION_DIR / "block-wise_standardization.py"

INPUT_MATRIX_XLSX = INTERIM_DIR / "input_matrix_subject_level.xlsx"
IMPUTED_XLSX = INTERIM_DIR / "input_matrix_subject_level_imputed.xlsx"
STD_SCRIPT_INPUT_XLSX = STANDARDIZATION_DIR / "input_matrix_subject_level.xlsx"
STD_SCRIPT_OUTPUT_XLSX = STANDARDIZATION_DIR / "input_matrix_standardized.xlsx"


BLOCKS: Dict[str, List[str]] = {
    "PHY": [
        "hr_dtst_env_sd",
        "tonic_dtst_env_sd",
        "pulse_amplitude_dtst_env_sd",
        "skt_dtst_env_sd",
    ],
    "PSY": [
        "tsv_20minus10_env_sd",
        "m2_20minus10_env_sd",
    ],
    "BHR": [
        "p7_minus_m7_overall_mean",
        "p7_minus_m7_env_sd",
    ],
    "TLX": [
        "tlx1_dtst_env_sd",
    ],
    "EUP": [
        "eup5",
        "eup16",
    ],
}

DATA_STRUCTURE_PRESETS: Dict[str, List[str]] = {
    "full": ["PHY", "PSY", "BHR", "TLX", "EUP"],
    "no_tlx": ["PHY", "PSY", "BHR", "EUP"],
    "no_eup": ["PHY", "PSY", "BHR", "TLX"],
    "phy_psy_only": ["PHY", "PSY"],
    "reactivity_only": ["PHY", "PSY", "TLX"],
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)



def run_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    print(f"[RUN] {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")



def prepare_standardization_input() -> None:
    if not IMPUTED_XLSX.exists():
        raise FileNotFoundError(f"Imputed workbook not found: {IMPUTED_XLSX}")
    shutil.copy2(IMPUTED_XLSX, STD_SCRIPT_INPUT_XLSX)



def select_blocks(data_structure: str) -> Dict[str, List[str]]:
    if data_structure in DATA_STRUCTURE_PRESETS:
        block_names = DATA_STRUCTURE_PRESETS[data_structure]
    else:
        block_names = [token.strip().upper() for token in data_structure.split(",") if token.strip()]
        if not block_names:
            raise ValueError(f"Invalid data_structure: {data_structure}")
        unknown = [b for b in block_names if b not in BLOCKS]
        if unknown:
            raise ValueError(f"Unknown blocks in data_structure: {unknown}")
    return {b: BLOCKS[b] for b in block_names}



def impute_and_block_standardize(df: pd.DataFrame, blocks: Dict[str, List[str]]) -> pd.DataFrame:
    if "id" not in df.columns:
        raise KeyError("Expected 'id' column in subject_features_only sheet.")

    ids = df["id"].copy()
    X_scaled_blocks: List[pd.DataFrame] = []

    for block_name, cols in blocks.items():
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Block {block_name} columns missing from dataframe: {missing_cols}")

        block_data = df[cols].copy()
        for col in cols:
            block_data[col] = pd.to_numeric(block_data[col], errors="coerce")
            mean_val = block_data[col].mean()
            block_data[col] = block_data[col].fillna(mean_val)

        scaler = StandardScaler()
        Z = scaler.fit_transform(block_data)
        Z = Z / np.sqrt(len(cols))
        X_scaled_blocks.append(pd.DataFrame(Z, columns=cols, index=df.index))

    X_final = pd.concat(X_scaled_blocks, axis=1)
    X_final.insert(0, "id", ids)
    return X_final



def build_dataset(cfg: BuildDatasetConfig) -> tuple[pd.DataFrame, Dict[str, object]]:
    if not cfg.use_existing_interim:
        run_script(FEATURE_SCRIPT)
        run_script(IMPUTE_SCRIPT)
    else:
        print("[SKIP] Reusing existing interim workbook.")

    if not INPUT_MATRIX_XLSX.exists():
        raise FileNotFoundError(f"Input matrix workbook not found: {INPUT_MATRIX_XLSX}")

    # 원 스크립트 호환을 위해 경로 맞춰둔다.
    prepare_standardization_input()
    if BLOCK_STD_SCRIPT.exists():
        try:
            run_script(BLOCK_STD_SCRIPT)
        except Exception as exc:
            print(f"[WARN] block-wise_standardization.py failed; internal standardization will be used instead. reason={exc}")

    subject_df = pd.read_excel(INPUT_MATRIX_XLSX, sheet_name="subject_features_only")
    selected_blocks = select_blocks(cfg.data_structure)
    final_df = impute_and_block_standardize(subject_df, selected_blocks)

    feature_cols = [c for c in final_df.columns if c != "id"]
    build_info = {
        "dataset_version": cfg.dataset_version,
        "data_structure": cfg.data_structure,
        "selected_blocks": list(selected_blocks.keys()),
        "n_subjects": int(final_df.shape[0]),
        "n_features": int(len(feature_cols)),
        "feature_names": feature_cols,
        "source_input_workbook": str(INPUT_MATRIX_XLSX),
        "source_imputed_workbook": str(IMPUTED_XLSX),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return final_df, build_info



def save_outputs(df: pd.DataFrame, build_info: Dict[str, object]) -> Dict[str, Path]:
    stem = f"X_subject_{build_info['dataset_version']}_{build_info['data_structure']}"
    xlsx_path = PROCESSED_DIR / f"{stem}.xlsx"
    csv_path = PROCESSED_DIR / f"{stem}.csv"
    json_path = PROCESSED_DIR / f"{stem}.json"
    manifest_path = PROCESSED_DIR / "dataset_manifest.csv"

    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(build_info, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_row = pd.DataFrame([
        {
            "dataset_version": build_info["dataset_version"],
            "data_structure": build_info["data_structure"],
            "created_at": build_info["generated_at"],
            "n_subjects": build_info["n_subjects"],
            "n_features": build_info["n_features"],
            "feature_names": ", ".join(build_info["feature_names"]),
            "selected_blocks": ", ".join(build_info["selected_blocks"]),
            "xlsx_path": str(xlsx_path),
            "csv_path": str(csv_path),
            "config_json": str(json_path),
        }
    ])
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        manifest_row = pd.concat([old, manifest_row], ignore_index=True)
    manifest_row.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    outputs = {
        "xlsx": xlsx_path,
        "csv": csv_path,
        "json": json_path,
        "manifest": manifest_path,
    }

    return outputs



def write_latest_alias(outputs: Dict[str, Path], build_info: Dict[str, object]) -> None:
    latest_csv = PROCESSED_DIR / "X_subject_current.csv"
    latest_xlsx = PROCESSED_DIR / "X_subject_current.xlsx"
    latest_json = PROCESSED_DIR / "X_subject_current.json"

    shutil.copy2(outputs["csv"], latest_csv)
    shutil.copy2(outputs["xlsx"], latest_xlsx)
    latest_json.write_text(json.dumps(build_info, ensure_ascii=False, indent=2), encoding="utf-8")



def log_mlflow(cfg: BuildDatasetConfig, build_info: Dict[str, object], outputs: Dict[str, Path]) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    run_name = cfg.run_name or f"build_{cfg.dataset_version}_{cfg.data_structure}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dataset_version": cfg.dataset_version,
            "data_structure": cfg.data_structure,
            "selected_blocks": ",".join(build_info["selected_blocks"]),
            "use_existing_interim": cfg.use_existing_interim,
            "register_latest_alias": cfg.register_latest_alias,
        })
        mlflow.log_metrics({
            "n_subjects": build_info["n_subjects"],
            "n_features": build_info["n_features"],
        })

        config_snapshot = ARTIFACT_DIR / f"build_config_{cfg.dataset_version}_{cfg.data_structure}.json"
        config_snapshot.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(config_snapshot), artifact_path="config")
        for name, path in outputs.items():
            mlflow.log_artifact(str(path), artifact_path="dataset_outputs")



def main() -> None:
    cfg = parse_args()
    ensure_dirs()
    df, build_info = build_dataset(cfg)
    outputs = save_outputs(df, build_info)
    if cfg.register_latest_alias:
        write_latest_alias(outputs, build_info)
    log_mlflow(cfg, build_info, outputs)

    print("[DONE] dataset build finished")
    print(f"  version      : {cfg.dataset_version}")
    print(f"  structure    : {cfg.data_structure}")
    print(f"  n_subjects   : {build_info['n_subjects']}")
    print(f"  n_features   : {build_info['n_features']}")
    print(f"  saved csv    : {outputs['csv']}")
    print(f"  saved xlsx   : {outputs['xlsx']}")
    print(f"  manifest csv : {outputs['manifest']}")


if __name__ == "__main__":
    main()
