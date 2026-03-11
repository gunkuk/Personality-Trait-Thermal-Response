# ============================================================
# 핵심 요약
# - PHY.xlsx의 ST / EDA / ECG 시트를 읽음
# - 각 시나리오(사람 × 환경)에서 baseline correction 수행

# - 환경값이 AB, CD처럼 2개 시나리오가 붙어 있는 경우
#   -> 앞 글자는 1~10분, 뒤 글자는 11~20분으로 분리

# - 기존 processed 시트는 유지
# - 추가로 A~H 시나리오별 시트를 생성
# ============================================================

import os
import pandas as pd
import numpy as np

INPUT_PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\data\PHY.xlsx"
OUTPUT_PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT\PHY_processed.xlsx"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

SCENARIOS = list("ABCDEFGH")
DYNAMIC_SCENARIOS = {"B", "D", "F", "H"}


# =========================
# UTIL
# =========================
def normalize_env_label(x):
    return str(x).strip().upper()


def minute_map(df, prefix, min_minute=1, max_minute=20):
    out = {}
    for m in range(min_minute, max_minute + 1):
        c = f"{prefix}_{m}"
        if c in df.columns:
            out[m] = c
    return out


def get_phase_minutes_from_existing(minute_dict, phase_start, phase_end):
    return [m for m in sorted(minute_dict.keys()) if phase_start <= m <= phase_end]


def _row_slope(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = times[mask]
    y = values[mask]
    if np.unique(x).size < 2:
        return np.nan
    return np.polyfit(x, y, 1)[0]


def _row_auc(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = times[mask]
    y = values[mask]
    return np.trapz(y, x)


def normalize_by_window(df, prefix, baseline_minutes, suffix="_bc", baseline_label=None):
    out = df.copy()
    m_map = minute_map(out, prefix)

    if len(m_map) == 0:
        print(f"[WARN] {prefix}: no columns found")
        return out, None

    baseline_minutes_exist = [m for m in baseline_minutes if m in m_map]
    if len(baseline_minutes_exist) == 0:
        print(f"[WARN] {prefix}: no baseline cols found for {baseline_minutes}")
        return out, None

    baseline_cols = [m_map[m] for m in baseline_minutes_exist]
    baseline_mean = out[baseline_cols].mean(axis=1)

    if baseline_label is None:
        mean_col = f"{prefix}_baseline_mean"
        mins_col = f"{prefix}_baseline_minutes"
    else:
        mean_col = f"{prefix}_baseline_mean_{baseline_label}"
        mins_col = f"{prefix}_baseline_minutes_{baseline_label}"

    out[mean_col] = baseline_mean
    out[mins_col] = ",".join(map(str, baseline_minutes_exist))

    for m, c in m_map.items():
        out[f"{c}{suffix}"] = out[c] - baseline_mean

    return out, baseline_minutes_exist


def make_phase_features_auto(df, prefix, use_normalized=True, suffix="_bc"):
    out = df.copy()
    m_map = minute_map(out, prefix)

    if len(m_map) == 0:
        print(f"[WARN] {prefix}: no columns found for phase features")
        return out

    static_minutes = get_phase_minutes_from_existing(m_map, 1, 10)
    dynamic_minutes = get_phase_minutes_from_existing(m_map, 11, 20)

    if use_normalized:
        static_cols = [f"{m_map[m]}{suffix}" for m in static_minutes if f"{m_map[m]}{suffix}" in out.columns]
        dynamic_cols = [f"{m_map[m]}{suffix}" for m in dynamic_minutes if f"{m_map[m]}{suffix}" in out.columns]
    else:
        static_cols = [m_map[m] for m in static_minutes]
        dynamic_cols = [m_map[m] for m in dynamic_minutes]

    if len(static_cols) == 0 or len(dynamic_cols) == 0:
        print(f"[WARN] {prefix}: insufficient static/dynamic cols")
        return out

    out[f"{prefix}_static_mean"] = out[static_cols].mean(axis=1)
    out[f"{prefix}_dynamic_mean"] = out[dynamic_cols].mean(axis=1)
    out[f"{prefix}_delta_dynamic_minus_static"] = (
        out[f"{prefix}_dynamic_mean"] - out[f"{prefix}_static_mean"]
    )

    out[f"{prefix}_static_minutes"] = ",".join(map(str, static_minutes))
    out[f"{prefix}_dynamic_minutes"] = ",".join(map(str, dynamic_minutes))

    return out


def add_dynamic_features_auto(
    df,
    prefix,
    use_normalized=True,
    suffix="_bc",
    pre_minutes=(8, 9, 10),
    start_minutes=(11, 12),
    end_minutes=(19, 20),
):
    out = df.copy()
    m_map = minute_map(out, prefix)

    if len(m_map) == 0:
        print(f"[WARN] {prefix}: no columns found for dynamic features")
        return out

    dynamic_minutes = get_phase_minutes_from_existing(m_map, 11, 20)
    if len(dynamic_minutes) < 2:
        print(f"[WARN] {prefix}: insufficient dynamic minutes")
        return out

    if use_normalized:
        dyn_cols = [f"{m_map[m]}{suffix}" for m in dynamic_minutes if f"{m_map[m]}{suffix}" in out.columns]
        pre_cols = [f"{m_map[m]}{suffix}" for m in pre_minutes if m in m_map and f"{m_map[m]}{suffix}" in out.columns]
        start_cols = [f"{m_map[m]}{suffix}" for m in start_minutes if m in m_map and f"{m_map[m]}{suffix}" in out.columns]
        end_cols = [f"{m_map[m]}{suffix}" for m in end_minutes if m in m_map and f"{m_map[m]}{suffix}" in out.columns]
    else:
        dyn_cols = [m_map[m] for m in dynamic_minutes if m in m_map]
        pre_cols = [m_map[m] for m in pre_minutes if m in m_map]
        start_cols = [m_map[m] for m in start_minutes if m in m_map]
        end_cols = [m_map[m] for m in end_minutes if m in m_map]

    if len(dyn_cols) < 2:
        print(f"[WARN] {prefix}: insufficient dynamic cols")
        return out

    out[f"{prefix}_dynamic_slope"] = out[dyn_cols].apply(
        lambda row: _row_slope(row.values, dynamic_minutes),
        axis=1
    )

    out[f"{prefix}_dynamic_auc"] = out[dyn_cols].apply(
        lambda row: _row_auc(row.values, dynamic_minutes),
        axis=1
    )

    out[f"{prefix}_dynamic_std"] = out[dyn_cols].std(axis=1)

    if len(start_cols) > 0 and len(end_cols) > 0:
        start_mean = out[start_cols].mean(axis=1)
        end_mean = out[end_cols].mean(axis=1)
        out[f"{prefix}_dynamic_end_minus_start"] = end_mean - start_mean

    if len(pre_cols) > 0 and len(start_cols) > 0:
        pre_mean = out[pre_cols].mean(axis=1)
        start_mean = out[start_cols].mean(axis=1)
        out[f"{prefix}_transition_jump"] = start_mean - pre_mean

    return out


def add_transition_baseline_features_auto(
    df,
    prefix,
    suffix="_bc_8to10",
    start_minutes=(11, 12),
    dynamic_minutes=tuple(range(11, 21)),
    end_minutes=(19, 20),
):
    out = df.copy()
    m_map = minute_map(out, prefix)

    if len(m_map) == 0:
        print(f"[WARN] {prefix}: no columns found for transition-baseline features")
        return out

    dyn_cols = [f"{m_map[m]}{suffix}" for m in dynamic_minutes if m in m_map and f"{m_map[m]}{suffix}" in out.columns]
    start_cols = [f"{m_map[m]}{suffix}" for m in start_minutes if m in m_map and f"{m_map[m]}{suffix}" in out.columns]
    end_cols = [f"{m_map[m]}{suffix}" for m in end_minutes if m in m_map and f"{m_map[m]}{suffix}" in out.columns]

    if len(dyn_cols) < 2:
        print(f"[WARN] {prefix}: insufficient dynamic cols for transition-baseline features")
        return out

    dyn_times = [m for m in dynamic_minutes if m in m_map and f"{m_map[m]}{suffix}" in out.columns]

    out[f"{prefix}_dynamic_mean_from_8to10"] = out[dyn_cols].mean(axis=1)

    out[f"{prefix}_dynamic_slope_from_8to10"] = out[dyn_cols].apply(
        lambda row: _row_slope(row.values, dyn_times),
        axis=1
    )

    out[f"{prefix}_dynamic_auc_from_8to10"] = out[dyn_cols].apply(
        lambda row: _row_auc(row.values, dyn_times),
        axis=1
    )

    out[f"{prefix}_dynamic_std_from_8to10"] = out[dyn_cols].std(axis=1)

    if len(start_cols) > 0 and len(end_cols) > 0:
        start_mean = out[start_cols].mean(axis=1)
        end_mean = out[end_cols].mean(axis=1)
        out[f"{prefix}_dynamic_end_minus_start_from_8to10"] = end_mean - start_mean

    if len(start_cols) > 0:
        start_mean = out[start_cols].mean(axis=1)
        out[f"{prefix}_transition_jump_11to12_minus_8to10"] = start_mean

    return out


def safe_log1p(df, col):
    if col in df.columns:
        df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))
    return df


def is_processed_summary_feature(col: str) -> bool:
    keep_suffixes = [
        "_baseline_mean",
        "_static_mean",
        "_dynamic_mean",
        "_delta_dynamic_minus_static",
        "_delta_11to20_minus_1to10",
        "_1to10",
        "_11to20",
    ]

    if col.startswith("log1p_"):
        return True

    for suf in keep_suffixes:
        if col.endswith(suf):
            return True

    return False


def is_scenario_summary_feature(col: str) -> bool:
    extra_suffixes = [
        "_dynamic_slope",
        "_dynamic_end_minus_start",
        "_dynamic_auc",
        "_dynamic_std",
        "_transition_jump",
        "_baseline_mean_8to10",
        "_dynamic_mean_from_8to10",
        "_dynamic_slope_from_8to10",
        "_dynamic_end_minus_start_from_8to10",
        "_dynamic_auc_from_8to10",
        "_dynamic_std_from_8to10",
        "_transition_jump_11to12_minus_8to10",
    ]

    if is_processed_summary_feature(col):
        return True

    for suf in extra_suffixes:
        if col.endswith(suf):
            return True

    return False


def slim_processed_df(df):
    meta_cols = [c for c in ["연번", "이름", "성별", "MBTI", "환경"] if c in df.columns]
    feature_cols = [c for c in df.columns if is_processed_summary_feature(c)]

    keep_cols = meta_cols + [c for c in feature_cols if c not in meta_cols]
    out = df[keep_cols].copy()

    if "환경" in out.columns:
        out["환경"] = out["환경"].map(normalize_env_label)

    if "연번" in out.columns and "환경" in out.columns:
        out = out.sort_values(["연번", "환경"]).reset_index(drop=True)

    return out


def slim_scenario_df(df):
    meta_cols = [c for c in ["연번", "이름", "성별", "MBTI", "환경"] if c in df.columns]
    feature_cols = [c for c in df.columns if is_scenario_summary_feature(c)]

    keep_cols = meta_cols + [c for c in feature_cols if c not in meta_cols]
    out = df[keep_cols].copy()

    if "환경" in out.columns:
        out["환경"] = out["환경"].map(normalize_env_label)

    if "연번" in out.columns and "환경" in out.columns:
        out = out.sort_values(["연번", "환경"]).reset_index(drop=True)

    return out


# ============================================================
# NEW: AB / CD 같은 결합 시나리오를 시나리오별 row로 분리
# ============================================================
def split_combined_env_rows_for_sheet(df):
    """
    환경이 AB처럼 붙어 있을 때:
    - A: 1~10분 row
    - B: 11~20분 row
    로 분해해서 반환

    시나리오 시트용이므로,
    앞/뒤 구간에 해당하는 feature만 남기고 이름도 일부 통일한다.
    """
    df = slim_scenario_df(df).copy()

    meta_cols = [c for c in ["연번", "이름", "성별", "MBTI", "환경"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    rows = []

    for _, row in df.iterrows():
        env = normalize_env_label(row.get("환경", ""))

        # 단일 시나리오인 경우 그대로 사용
        if len(env) == 1 and env in SCENARIOS:
            new_row = row.to_dict()
            new_row["원본환경"] = env
            new_row["분할구간"] = "single"
            rows.append(new_row)
            continue

        # AB / CD / EF / GH 형태 처리
        if len(env) >= 2 and env[0] in SCENARIOS and env[1] in SCENARIOS:
            front_env = env[0]   # 1~10분
            back_env = env[1]    # 11~20분

            front_row = {c: row[c] for c in meta_cols if c in row.index}
            back_row = {c: row[c] for c in meta_cols if c in row.index}

            front_row["환경"] = front_env
            back_row["환경"] = back_env

            front_row["원본환경"] = env
            back_row["원본환경"] = env

            front_row["분할구간"] = "1to10"
            back_row["분할구간"] = "11to20"

            for col in feature_cols:
                val = row[col]

                # ---------- 앞 시나리오(1~10분) ----------
                if col.endswith("_static_mean"):
                    front_row[col.replace("_static_mean", "_scenario_mean")] = val
                elif col.endswith("_1to10"):
                    front_row[col.replace("_1to10", "_scenario")] = val
                elif col.endswith("_baseline_mean"):
                    front_row[col] = val
                # 나머지 dynamic 관련 지표는 front에 넣지 않음

                # ---------- 뒤 시나리오(11~20분) ----------
                if col.endswith("_dynamic_mean"):
                    back_row[col.replace("_dynamic_mean", "_scenario_mean")] = val
                elif col.endswith("_11to20"):
                    back_row[col.replace("_11to20", "_scenario")] = val
                elif (
                    col.endswith("_dynamic_slope")
                    or col.endswith("_dynamic_end_minus_start")
                    or col.endswith("_dynamic_auc")
                    or col.endswith("_dynamic_std")
                    or col.endswith("_transition_jump")
                    or col.endswith("_baseline_mean_8to10")
                    or col.endswith("_dynamic_mean_from_8to10")
                    or col.endswith("_dynamic_slope_from_8to10")
                    or col.endswith("_dynamic_end_minus_start_from_8to10")
                    or col.endswith("_dynamic_auc_from_8to10")
                    or col.endswith("_dynamic_std_from_8to10")
                    or col.endswith("_transition_jump_11to12_minus_8to10")
                ):
                    back_row[col] = val

            rows.append(front_row)
            rows.append(back_row)
            continue

        # 예상치 못한 라벨이면 일단 그대로 보존
        new_row = row.to_dict()
        new_row["원본환경"] = env
        new_row["분할구간"] = "unknown"
        rows.append(new_row)

    out = pd.DataFrame(rows)

    front_cols = [c for c in ["연번", "이름", "성별", "MBTI", "환경", "원본환경", "분할구간"] if c in out.columns]
    other_cols = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + other_cols]

    if "연번" in out.columns and "환경" in out.columns:
        out = out.sort_values(["환경", "연번"]).reset_index(drop=True)

    return out


def build_scenario_sheet(env_label, st_df, eda_df, ecg_df):
    """
    A~H 시나리오별로 ST/EDA/ECG를 하나의 시트로 병합
    - 환경값이 AB/CD처럼 붙어 있어도 split 후 병합
    """
    env_label = normalize_env_label(env_label)

    def _filter_env(df):
        tmp = split_combined_env_rows_for_sheet(df).copy()
        if "환경" not in tmp.columns:
            return tmp.iloc[0:0].copy()
        return tmp.loc[tmp["환경"] == env_label].copy()

    st_part = _filter_env(st_df)
    eda_part = _filter_env(eda_df)
    ecg_part = _filter_env(ecg_df)

    parts = [x for x in [st_part, eda_part, ecg_part] if len(x) > 0]

    if len(parts) == 0:
        return pd.DataFrame(columns=["연번", "이름", "성별", "MBTI", "환경", "원본환경", "분할구간", "시나리오유형"])

    merged = parts[0].copy()
    for part in parts[1:]:
        keys = [
            c for c in ["연번", "이름", "성별", "MBTI", "환경", "원본환경", "분할구간"]
            if c in merged.columns and c in part.columns
        ]
        merged = merged.merge(part, on=keys, how="outer")

    merged["시나리오유형"] = "dynamic" if env_label in DYNAMIC_SCENARIOS else "static"

    front_cols = [c for c in ["연번", "이름", "성별", "MBTI", "환경", "원본환경", "분할구간", "시나리오유형"] if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + other_cols]

    if "연번" in merged.columns:
        merged = merged.sort_values("연번").reset_index(drop=True)

    return merged


# =========================
# LOAD
# =========================
df_st = pd.read_excel(INPUT_PATH, sheet_name="ST")
df_eda = pd.read_excel(INPUT_PATH, sheet_name="EDA")
df_ecg = pd.read_excel(INPUT_PATH, sheet_name="ECG")

rename_map = {
    "subject": "연번",
    "name": "이름",
    "condition": "환경"
}
df_ecg = df_ecg.rename(columns={k: v for k, v in rename_map.items() if k in df_ecg.columns})

for _df in [df_st, df_eda, df_ecg]:
    if "환경" in _df.columns:
        _df["환경"] = _df["환경"].map(normalize_env_label)


# =========================
# ST SHEET
# =========================
df_st_proc = df_st.copy()

for prefix in ["st_mean", "at_mean"]:
    df_st_proc, _ = normalize_by_window(
        df_st_proc, prefix,
        baseline_minutes=[1, 2, 3],
        suffix="_bc",
        baseline_label=None
    )

    df_st_proc, _ = normalize_by_window(
        df_st_proc, prefix,
        baseline_minutes=[8, 9, 10],
        suffix="_bc_8to10",
        baseline_label="8to10"
    )

    df_st_proc = make_phase_features_auto(df_st_proc, prefix, use_normalized=True, suffix="_bc")
    df_st_proc = add_dynamic_features_auto(df_st_proc, prefix, use_normalized=True, suffix="_bc")
    df_st_proc = add_transition_baseline_features_auto(df_st_proc, prefix, suffix="_bc_8to10")

df_st_proc_save = slim_processed_df(df_st_proc)


# =========================
# EDA SHEET
# =========================
df_eda_proc = df_eda.copy()

df_eda_proc, _ = normalize_by_window(
    df_eda_proc, "tonic",
    baseline_minutes=[1, 2, 3],
    suffix="_bc",
    baseline_label=None
)
df_eda_proc, _ = normalize_by_window(
    df_eda_proc, "tonic",
    baseline_minutes=[8, 9, 10],
    suffix="_bc_8to10",
    baseline_label="8to10"
)

df_eda_proc = make_phase_features_auto(df_eda_proc, "tonic", use_normalized=True, suffix="_bc")
df_eda_proc = add_dynamic_features_auto(df_eda_proc, "tonic", use_normalized=True, suffix="_bc")
df_eda_proc = add_transition_baseline_features_auto(df_eda_proc, "tonic", suffix="_bc_8to10")

df_eda_proc = make_phase_features_auto(df_eda_proc, "phasic", use_normalized=False)
df_eda_proc = add_dynamic_features_auto(df_eda_proc, "phasic", use_normalized=False)

df_eda_proc, _ = normalize_by_window(
    df_eda_proc, "phasic",
    baseline_minutes=[8, 9, 10],
    suffix="_bc_8to10",
    baseline_label="8to10"
)
df_eda_proc = add_transition_baseline_features_auto(df_eda_proc, "phasic", suffix="_bc_8to10")

for col in [
    "scr_frequency_1to10", "scr_amplitude_1to10",
    "scr_frequency_11to20", "scr_amplitude_11to20",
    "phasic_mean_1to10", "phasic_mean_11to20",
    "phasic_std_1to10", "phasic_std_11to20",
    "phasic_max_1to10", "phasic_max_11to20",
    "eda_variability_1to10", "eda_variability_11to20"
]:
    df_eda_proc = safe_log1p(df_eda_proc, col)

for base in [
    "scr_frequency", "scr_amplitude", "tonic_trend",
    "phasic_mean", "phasic_std", "phasic_max", "eda_variability"
]:
    c1 = f"{base}_1to10"
    c2 = f"{base}_11to20"
    if c1 in df_eda_proc.columns and c2 in df_eda_proc.columns:
        df_eda_proc[f"{base}_delta_11to20_minus_1to10"] = df_eda_proc[c2] - df_eda_proc[c1]

df_eda_proc_save = slim_processed_df(df_eda_proc)


# =========================
# ECG SHEET
# =========================
df_ecg_proc = df_ecg.copy()

for prefix in ["HR", "MeanRR", "SDNN", "RMSSD", "LF", "HF", "LF_HF"]:
    df_ecg_proc, _ = normalize_by_window(
        df_ecg_proc, prefix,
        baseline_minutes=[1, 2, 3],
        suffix="_bc",
        baseline_label=None
    )

    df_ecg_proc, _ = normalize_by_window(
        df_ecg_proc, prefix,
        baseline_minutes=[8, 9, 10],
        suffix="_bc_8to10",
        baseline_label="8to10"
    )

    df_ecg_proc = make_phase_features_auto(df_ecg_proc, prefix, use_normalized=True, suffix="_bc")
    df_ecg_proc = add_dynamic_features_auto(df_ecg_proc, prefix, use_normalized=True, suffix="_bc")
    df_ecg_proc = add_transition_baseline_features_auto(df_ecg_proc, prefix, suffix="_bc_8to10")

df_ecg_proc_save = slim_processed_df(df_ecg_proc)


# =========================
# SCENARIO SHEETS A~H
# =========================
scenario_sheets = {}
for scn in SCENARIOS:
    scenario_sheets[scn] = build_scenario_sheet(
        env_label=scn,
        st_df=df_st_proc,
        eda_df=df_eda_proc,
        ecg_df=df_ecg_proc,
    )


# =========================
# SAVE
# =========================
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    df_st_proc_save.to_excel(writer, sheet_name="ST_processed", index=False)
    df_eda_proc_save.to_excel(writer, sheet_name="EDA_processed", index=False)
    df_ecg_proc_save.to_excel(writer, sheet_name="ECG_processed", index=False)

    for scn in SCENARIOS:
        scenario_sheets[scn].to_excel(writer, sheet_name=scn, index=False)

print(f"Saved: {OUTPUT_PATH}")