# 핵심: raw data에서 최종 subject-level clustering 입력 dataset 버전을 생성한다.
import argparse
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parents[1]           # 00_data/
RAW_PATH = DATA_ROOT / "00_raw" / "rawdata_wrong corrected_0319.xlsx"
PROCESSED_DIR = DATA_ROOT / "02_processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── block definitions ────────────────────────────────────────────────────────
ALL_BLOCKS = {
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
    "TLX": ["tlx1_dtst_env_sd"],
    "EUP": ["eup5", "eup16"],
}

SPECIAL_STRUCTURES = {
    "full":   list(ALL_BLOCKS.keys()),
    "no_tlx": ["PHY", "PSY", "BHR", "EUP"],
}


# ─── utils ────────────────────────────────────────────────────────────────────
def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return unicodedata.normalize("NFKC", str(x)).strip()

def norm_col(x) -> str:
    x = norm_text(x).lower()
    x = re.sub(r"[^\w\s]", "_", x)
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x

def std_common_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ren = {}
    for c in out.columns:
        nc = norm_col(c)
        if nc in {"no", "no_", "id"}:
            ren[c] = "id"
        elif nc == "name":
            ren[c] = "name"
        elif nc == "env":
            ren[c] = "env"
        elif nc in {"time_min", "time"}:
            ren[c] = "time_min"
    return out.rename(columns=ren)

def load_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)
    df = std_common_cols(df)
    if "id" not in df.columns:
        raise ValueError(f"{sheet_name}: id column not found")
    df = df[df["id"].notna()].copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)
    if "name" in df.columns:
        df["name"] = df["name"].map(norm_text)
    if "env" in df.columns:
        df["env"] = df["env"].map(norm_text)
    return df

def find_col(df: pd.DataFrame, target: str) -> str:
    target_n = norm_col(target)
    for c in df.columns:
        if norm_col(c) == target_n:
            return c
    raise KeyError(f"Column not found: {target}")

def find_minute_cols(df: pd.DataFrame, prefixes: Iterable[str]) -> List[str]:
    prefix_ns = [norm_col(p) for p in prefixes]
    out = []
    for c in df.columns:
        nc = norm_col(c)
        for p in prefix_ns:
            if re.fullmatch(rf"{re.escape(p)}_(\d+)", nc):
                out.append(c)
                break
    return sorted(out, key=lambda c: int(re.search(r"_(\d+)$", norm_col(c)).group(1)))

def safe_sd(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.std(ddof=1)) if len(s) >= 2 else np.nan

def env_letter_to_condition(env_val) -> str:
    env = norm_text(env_val).upper()
    mapping = {
        "A": "AB", "B": "AB", "C": "CD", "D": "CD",
        "E": "EF", "F": "EF", "G": "GH", "H": "GH",
        "AB": "AB", "CD": "CD", "EF": "EF", "GH": "GH",
    }
    return mapping.get(env, np.nan)

def phase_from_env(env_val) -> str:
    env = norm_text(env_val).upper()
    if env in {"A", "C", "E", "G"}:
        return "static"
    if env in {"B", "D", "F", "H"}:
        return "dynamic"
    return np.nan

def add_condition_phase(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["condition"] = out["env"].map(env_letter_to_condition)
    out["phase"] = out["env"].map(phase_from_env)
    return out


# ─── feature builders ─────────────────────────────────────────────────────────
def make_phy_feature(df: pd.DataFrame, minute_prefixes: List[str], output_col: str) -> pd.DataFrame:
    minute_cols = find_minute_cols(df, minute_prefixes)
    work = df[["id", "env"] + minute_cols].copy()
    for c in minute_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    static_cols  = [c for c in minute_cols if 1  <= int(re.search(r"_(\d+)$", norm_col(c)).group(1)) <= 10]
    dynamic_cols = [c for c in minute_cols if 11 <= int(re.search(r"_(\d+)$", norm_col(c)).group(1)) <= 20]
    work["static_mean"]  = work[static_cols].mean(axis=1, skipna=True)
    work["dynamic_mean"] = work[dynamic_cols].mean(axis=1, skipna=True)
    work["delta_dt_st"]  = work["dynamic_mean"] - work["static_mean"]
    cond_mean = work.groupby(["id", "env"], as_index=False)["delta_dt_st"].mean()
    return cond_mean.groupby("id")["delta_dt_st"].apply(safe_sd).reset_index(name=output_col)

def make_psy_features(df: pd.DataFrame, vars_: List[str]) -> pd.DataFrame:
    work = add_condition_phase(df)
    work["time_min"] = pd.to_numeric(work["time_min"], errors="coerce")
    out = None
    for var in vars_:
        col = find_col(work, var)
        work[col] = pd.to_numeric(work[col], errors="coerce")
        sub = (
            work[work["time_min"].isin([10, 20])]
            .groupby(["id", "condition", "time_min"], as_index=False)[col].mean()
            .pivot_table(index=["id", "condition"], columns="time_min", values=col)
            .reset_index()
        )
        sub["delta_20_10"] = sub[20] - sub[10]
        feat = sub.groupby("id")["delta_20_10"].apply(safe_sd).reset_index(
            name=f"{norm_col(var)}_20minus10_env_sd"
        )
        out = feat if out is None else out.merge(feat, on="id", how="outer")
    return out

def make_bhr_features(df: pd.DataFrame, p_var: str = "P7", m_var: str = "M7") -> pd.DataFrame:
    work = add_condition_phase(df)
    p_col = find_col(work, p_var)
    m_col = find_col(work, m_var)
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work[m_col] = pd.to_numeric(work[m_col], errors="coerce")
    work["p7_minus_m7"] = work[p_col] - work[m_col]
    overall = (
        work.groupby("id", as_index=False)["p7_minus_m7"]
        .mean()
        .rename(columns={"p7_minus_m7": "p7_minus_m7_overall_mean"})
    )
    env_mean = work.groupby(["id", "condition"], as_index=False)["p7_minus_m7"].mean()
    env_sd = env_mean.groupby("id")["p7_minus_m7"].apply(safe_sd).reset_index(name="p7_minus_m7_env_sd")
    return overall.merge(env_sd, on="id", how="outer")

def make_tlx_features(df: pd.DataFrame, vars_: List[str]) -> pd.DataFrame:
    work = add_condition_phase(df)
    out = None
    for var in vars_:
        col = find_col(work, var)
        work[col] = pd.to_numeric(work[col], errors="coerce")
        pivot = (
            work.groupby(["id", "condition", "phase"], as_index=False)[col].mean()
            .pivot_table(index=["id", "condition"], columns="phase", values=col)
            .reset_index()
        )
        pivot["delta_dt_st"] = pivot["dynamic"] - pivot["static"]
        feat = pivot.groupby("id")["delta_dt_st"].apply(safe_sd).reset_index(
            name=f"{norm_col(var)}_dtst_env_sd"
        )
        out = feat if out is None else out.merge(feat, on="id", how="outer")
    return out

def make_raw_features(df: pd.DataFrame, vars_: List[str]) -> pd.DataFrame:
    keep = ["id"]
    ren = {}
    for v in vars_:
        c = find_col(df, v)
        keep.append(c)
        ren[c] = norm_col(v)
    out = df[keep].copy().rename(columns=ren)
    agg = {"id": "first"}
    for c in out.columns:
        if c != "id":
            agg[c] = lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan
    return out.groupby("id", as_index=False).agg(agg)


# ─── main build ───────────────────────────────────────────────────────────────
def build_subject_dataset(data_structure: str = "full", use_existing_interim: bool = False) -> pd.DataFrame:
    # 1. raw load
    if use_existing_interim:
        interim_path = DATA_ROOT / "01_interim" / "input_matrix_subject_level.xlsx"
        raw_df = pd.read_excel(interim_path, sheet_name="subject_features_only")
    else:
        sheets = {s: load_sheet(RAW_PATH, s) for s in ["ECG", "EDA", "PPG", "ST", "SUR", "TLX", "EUP"]}

        # 2. input matrix 생성
        ecg = make_phy_feature(sheets["ECG"], ["HR"], "hr_dtst_env_sd")
        eda = make_phy_feature(sheets["EDA"], ["tonic"], "tonic_dtst_env_sd")
        ppg = make_phy_feature(sheets["PPG"], ["pulse_amplitude_mean", "pulse_amplitude"], "pulse_amplitude_dtst_env_sd")
        st  = make_phy_feature(sheets["ST"],  ["st_mean", "skt_mean", "skt", "st"], "skt_dtst_env_sd")
        psy = make_psy_features(sheets["SUR"], ["TSV", "M2"])
        bhr = make_bhr_features(sheets["SUR"])
        tlx = make_tlx_features(sheets["TLX"], ["TLX1"])
        eup = make_raw_features(sheets["EUP"], ["EUP5", "EUP16"])

        raw_df = ecg
        for part in [eda, ppg, st, psy, bhr, tlx, eup]:
            raw_df = raw_df.merge(part, on="id", how="outer")
        raw_df = raw_df.sort_values("id").reset_index(drop=True)

    # 5. feature selection
    if data_structure in SPECIAL_STRUCTURES:
        selected_blocks = {k: ALL_BLOCKS[k] for k in SPECIAL_STRUCTURES[data_structure]}
    else:
        requested = [b.strip().upper() for b in data_structure.split(",")]
        selected_blocks = {k: ALL_BLOCKS[k] for k in requested if k in ALL_BLOCKS}
        if not selected_blocks:
            raise ValueError(f"No valid blocks in data_structure='{data_structure}'. Valid: {list(ALL_BLOCKS)}")

    feature_cols = [col for cols in selected_blocks.values() for col in cols]
    ids = raw_df["id"].reset_index(drop=True)
    X = raw_df[feature_cols].copy().reset_index(drop=True)
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 3. missing 처리: column mean imputation
    for col in feature_cols:
        X[col] = X[col].fillna(X[col].mean())

    # 4. scaling: block-wise z-score + divide by sqrt(block_size)
    scaled_parts = []
    for block_name, cols in selected_blocks.items():
        Z = StandardScaler().fit_transform(X[cols])
        Z = Z / np.sqrt(len(cols))
        scaled_parts.append(pd.DataFrame(Z, columns=cols))

    # 6. 최종 X 반환
    X_final = pd.concat(scaled_parts, axis=1)
    X_final.insert(0, "id", ids)
    return X_final


def save_manifest(df: pd.DataFrame, save_path: Path, dataset_version: str, data_structure: str) -> None:
    feature_cols = [c for c in df.columns if c != "id"]
    manifest = pd.DataFrame([{
        "dataset_version": dataset_version,
        "data_structure": data_structure,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": RAW_PATH.name,
        "n_subjects": df.shape[0],
        "n_features": len(feature_cols),
        "feature_names": ", ".join(feature_cols),
        "cleaning_version": "missing_data_v1",
        "scaling_version": "blockwise_standardization_v1",
        "selection_version": "fixed_feature_set_v1",
    }])
    manifest_path = save_path.with_name("dataset_manifest.csv")
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        manifest = pd.concat([old, manifest], ignore_index=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_version",    type=str, default="v1")
    parser.add_argument("--data_structure",     type=str, default="full")
    parser.add_argument("--experiment_name",    type=str, default="build")
    parser.add_argument("--tracking_uri",       type=str, default="mlruns")
    parser.add_argument("--use_existing_interim", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    df = build_subject_dataset(
        data_structure=args.data_structure,
        use_existing_interim=args.use_existing_interim,
    )
    save_path = PROCESSED_DIR / f"X_subject_{args.dataset_version}_{args.data_structure}.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    save_manifest(df, save_path, args.dataset_version, args.data_structure)
    print(f"[DONE] Saved dataset: {save_path}")
    print(f"  shape    : {df.shape}")
    print(f"  features : {[c for c in df.columns if c != 'id']}")

    try:
        import os
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", args.experiment_name))
        parent_run_id = os.environ.get("MLFLOW_RUN_ID")
        ctx = (
            mlflow.start_run(run_id=parent_run_id, nested=True)
            if parent_run_id
            else mlflow.start_run()
        )
        with ctx:
            mlflow.log_params({
                "dataset_version": args.dataset_version,
                "data_structure": args.data_structure,
                "n_subjects": df.shape[0],
                "n_features": df.shape[1] - 1,
                "source_file": RAW_PATH.name,
            })
            mlflow.log_artifact(str(save_path))
    except Exception as e:
        print(f"[MLflow] logging skipped: {e}")


if __name__ == "__main__":
    main()
