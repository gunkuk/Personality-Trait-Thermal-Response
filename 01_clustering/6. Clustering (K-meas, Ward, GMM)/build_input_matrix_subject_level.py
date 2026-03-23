
# 요약: rawdata_wrong corrected_0319.xlsx에서 지정한 12개 subject-level feature를 생성하고,
# 첫 3행에 메타데이터(data source / representing value / target data)를 넣은 input matrix를 저장한다.

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Iterable, List

import numpy as np
import pandas as pd


# =========================
# 1) 경로 설정
# =========================
# 실사용 시 아래 두 경로를 그대로 사용하면 된다.
INPUT_XLSX = Path(r"C:\Users\rjs11\Desktop\1\PAPER\00_SCI\MBTI\model\1. data cleaning\domain knowledge\rawdata_wrong corrected_0319.xlsx")
OUTPUT_XLSX = Path(r"C:\Users\rjs11\Desktop\1\PAPER\00_SCI\MBTI\model\3. feature selection & standarization\input_matrix_subject_level.xlsx")

# 컨테이너/테스트용 fallback
FALLBACK_INPUT = Path("/mnt/data/rawdata_wrong corrected_0319.xlsx")
FALLBACK_OUTPUT = Path("/mnt/data/input_matrix_subject_level.xlsx")


# =========================
# 2) 타깃 변수
# =========================
PSY_VARS = ["TSV", "M2"]
TLX_VARS = ["TLX1"]
RAW_BI_VARS = ["sex"]
RAW_EUP_VARS = ["EUP5", "EUP16"]


# =========================
# 3) 유틸
# =========================
def pick_input_path() -> Path:
    return INPUT_XLSX if INPUT_XLSX.exists() else FALLBACK_INPUT

def pick_output_path() -> Path:
    return OUTPUT_XLSX if OUTPUT_XLSX.parent.exists() else FALLBACK_OUTPUT

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
    out = out.rename(columns=ren)
    return out

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
    out = sorted(out, key=lambda c: int(re.search(r"_(\d+)$", norm_col(c)).group(1)))
    return out

def safe_sd(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=1))

def env_letter_to_condition(env_val) -> str:
    env = norm_text(env_val).upper()
    mapping = {
        "A": "AB", "B": "AB",
        "C": "CD", "D": "CD",
        "E": "EF", "F": "EF",
        "G": "GH", "H": "GH",
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


# =========================
# 4) PHY [dt-st, SD]
# =========================
def make_phy_feature(df: pd.DataFrame, minute_prefixes: List[str], output_col: str) -> pd.DataFrame:
    minute_cols = find_minute_cols(df, minute_prefixes)
    if len(minute_cols) < 20:
        raise ValueError(f"{output_col}: expected >=20 minute columns, got {len(minute_cols)}")

    work = df[["id", "env"] + minute_cols].copy()
    for c in minute_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    static_cols = [c for c in minute_cols if 1 <= int(re.search(r"_(\d+)$", norm_col(c)).group(1)) <= 10]
    dynamic_cols = [c for c in minute_cols if 11 <= int(re.search(r"_(\d+)$", norm_col(c)).group(1)) <= 20]

    work["static_mean"] = work[static_cols].mean(axis=1, skipna=True)
    work["dynamic_mean"] = work[dynamic_cols].mean(axis=1, skipna=True)
    work["delta_dt_st"] = work["dynamic_mean"] - work["static_mean"]

    cond_mean = work.groupby(["id", "env"], as_index=False)["delta_dt_st"].mean()
    feat = cond_mean.groupby("id")["delta_dt_st"].apply(safe_sd).reset_index(name=output_col)
    return feat


# =========================
# 5) TLX [dt-st, SD]
# =========================
def make_tlx_features(df: pd.DataFrame, vars_: List[str]) -> pd.DataFrame:
    work = add_condition_phase(df)
    out = None

    for var in vars_:
        col = find_col(work, var)
        work[col] = pd.to_numeric(work[col], errors="coerce")

        pivot = (
            work.groupby(["id", "condition", "phase"], as_index=False)[col]
            .mean()
            .pivot_table(index=["id", "condition"], columns="phase", values=col)
            .reset_index()
        )
        pivot["delta_dt_st"] = pivot["dynamic"] - pivot["static"]
        feat = pivot.groupby("id")["delta_dt_st"].apply(safe_sd).reset_index(name=f"{norm_col(var)}_dtst_env_sd")
        out = feat if out is None else out.merge(feat, on="id", how="outer")

    return out


# =========================
# 6) SUR: PSY [20-10, SD] / BHR [RAW mean, env SD]
# =========================
def make_psy_features(df: pd.DataFrame, vars_: List[str]) -> pd.DataFrame:
    work = add_condition_phase(df)
    work["time_min"] = pd.to_numeric(work["time_min"], errors="coerce")

    out = None
    for var in vars_:
        col = find_col(work, var)
        work[col] = pd.to_numeric(work[col], errors="coerce")

        sub = (
            work[work["time_min"].isin([10, 20])]
            .groupby(["id", "condition", "time_min"], as_index=False)[col]
            .mean()
            .pivot_table(index=["id", "condition"], columns="time_min", values=col)
            .reset_index()
        )
        sub["delta_20_10"] = sub[20] - sub[10]
        feat = sub.groupby("id")["delta_20_10"].apply(safe_sd).reset_index(name=f"{norm_col(var)}_20minus10_env_sd")
        out = feat if out is None else out.merge(feat, on="id", how="outer")

    return out

def make_bhr_features(df: pd.DataFrame, p_var: str = "P7", m_var: str = "M7") -> pd.DataFrame:
    work = add_condition_phase(df)
    p_col = find_col(work, p_var)
    m_col = find_col(work, m_var)
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work[m_col] = pd.to_numeric(work[m_col], errors="coerce")
    work["p7_minus_m7"] = work[p_col] - work[m_col]

    overall = work.groupby("id", as_index=False)["p7_minus_m7"].mean().rename(columns={"p7_minus_m7": "p7_minus_m7_overall_mean"})
    env_mean = work.groupby(["id", "condition"], as_index=False)["p7_minus_m7"].mean()
    env_sd = env_mean.groupby("id")["p7_minus_m7"].apply(safe_sd).reset_index(name="p7_minus_m7_env_sd")
    return overall.merge(env_sd, on="id", how="outer")


# =========================
# 7) RAW
# =========================
def make_raw_features(df: pd.DataFrame, vars_: List[str]) -> pd.DataFrame:
    keep = ["id"]
    ren = {}
    for v in vars_:
        c = find_col(df, v)
        keep.append(c)
        ren[c] = norm_col(v)
    out = df[keep].copy().rename(columns=ren)

    # 같은 id가 여러 번 나타나는 경우, 각 변수별 첫 non-null 값을 대표값으로 사용
    agg = {"id": "first"}
    for c in out.columns:
        if c == "id":
            continue
        agg[c] = lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan
    out = out.groupby("id", as_index=False).agg(agg)
    return out


# =========================
# 8) 메타데이터
# =========================
def build_metadata(feature_cols: List[str]) -> pd.DataFrame:
    mapping = {
        "hr_dtst_env_sd": ("PHY", "dt-st, SD across 4 ENV", "HR"),
        "tonic_dtst_env_sd": ("PHY", "dt-st, SD across 4 ENV", "tonic"),
        "pulse_amplitude_dtst_env_sd": ("PHY", "dt-st, SD across 4 ENV", "pulse amplitude"),
        "skt_dtst_env_sd": ("PHY", "dt-st, SD across 4 ENV", "skt"),
        "tsv_20minus10_env_sd": ("PSY", "20-10, SD across 4 ENV", "TSV"),
        "m2_20minus10_env_sd": ("PSY", "20-10, SD across 4 ENV", "M2"),
        "p7_minus_m7_overall_mean": ("BHR", "RAW overall mean", "P7-M7"),
        "p7_minus_m7_env_sd": ("BHR", "SD of ENV means", "P7-M7"),
        "tlx1_dtst_env_sd": ("TLX", "dt-st, SD across 4 ENV", "TLX1"),
        "sex": ("SEX", "RAW", "SEX"),
        "eup5": ("EUP", "RAW", "EUP5"),
        "eup16": ("EUP", "RAW", "EUP16"),
    }
    row1 = {"id": "data_source"}
    row2 = {"id": "representing_value"}
    row3 = {"id": "target_data"}
    for c in feature_cols:
        row1[c], row2[c], row3[c] = mapping[c]
    return pd.DataFrame([row1, row2, row3])


# =========================
# 9) 메인 실행
# =========================
def main() -> None:
    input_path = pick_input_path()
    output_path = pick_output_path()

    sheets = {s: load_sheet(input_path, s) for s in ["ECG", "EDA", "PPG", "ST", "SUR", "TLX", "BI", "EUP"]}

    ecg = make_phy_feature(sheets["ECG"], ["HR"], "hr_dtst_env_sd")
    eda = make_phy_feature(sheets["EDA"], ["tonic"], "tonic_dtst_env_sd")
    ppg = make_phy_feature(sheets["PPG"], ["pulse_amplitude_mean", "pulse_amplitude"], "pulse_amplitude_dtst_env_sd")
    st = make_phy_feature(sheets["ST"], ["st_mean", "skt_mean", "skt", "st"], "skt_dtst_env_sd")

    psy = make_psy_features(sheets["SUR"], PSY_VARS)
    bhr = make_bhr_features(sheets["SUR"], "P7", "M7")
    tlx = make_tlx_features(sheets["TLX"], TLX_VARS)
    sex = make_raw_features(sheets["BI"], RAW_BI_VARS)
    eup = make_raw_features(sheets["EUP"], RAW_EUP_VARS)

    final_df = ecg
    for part in [eda, ppg, st, psy, bhr, tlx, sex, eup]:
        final_df = final_df.merge(part, on="id", how="outer")

    ordered_cols = [
        "id",
        "hr_dtst_env_sd",
        "tonic_dtst_env_sd",
        "pulse_amplitude_dtst_env_sd",
        "skt_dtst_env_sd",
        "tsv_20minus10_env_sd",
        "m2_20minus10_env_sd",
        "p7_minus_m7_overall_mean",
        "p7_minus_m7_env_sd",
        "tlx1_dtst_env_sd",
        "sex",
        "eup5",
        "eup16",
    ]
    final_df = final_df.reindex(columns=ordered_cols).sort_values("id").reset_index(drop=True)

    meta = build_metadata(ordered_cols[1:]).reindex(columns=ordered_cols)
    export_df = pd.concat([meta, final_df], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="input_matrix", index=False)
        final_df.to_excel(writer, sheet_name="subject_features_only", index=False)

    print(f"[DONE] input={input_path}")
    print(f"[DONE] output={output_path}")
    print(f"[INFO] n_subjects={final_df['id'].nunique()}, n_features={len(ordered_cols) - 1}")
    print("[INFO] missing_by_feature")
    print(final_df.isna().sum())

if __name__ == "__main__":
    main()
