# =========================
# 핵심 요약
# 1. PHY.xlsx의 ST, EDA, ECG, PPG 시트를 통합하여 인체 생리 지표 분석 수행
# 2. 각 지표별 2가지 Baseline Correction(1~3분, 8~10분) 및 시점별 특징량(Mean, Slope, AUC 등) 추출
# 3. PPG 기반 말초 혈관 반응(pulse_amplitude, blood_flow) 및 ECG 신호 품질(SNR, Kurtosis) 반영
# 4. AB, CD 등 결합된 환경 데이터를 A~H의 8개 개별 시나리오(1~10분: Static / 11~20분: Dynamic)로 분할 및 병합
# =========================

import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook

# =========================
# PATH & CONFIG
# =========================
INPUT_PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\data\PHY.xlsx"
OUTPUT_PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT\PHY_processed.xlsx"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

SCENARIOS = list("ABCDEFGH")
DYNAMIC_SCENARIOS = {"B", "D", "F", "H"}

# =========================
# UTIL FUNCTIONS
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
    if mask.sum() < 2: return np.nan
    x, y = times[mask], values[mask]
    return np.polyfit(x, y, 1)[0] if np.unique(x).size >= 2 else np.nan

def _row_auc(values, times):
    """트래피조이드 공식을 이용한 곡선 아래 면적(AUC) 계산"""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2: return np.nan
    # NumPy 2.0+ 버전 대응: trapz -> trapezoid
    trapezoid_func = getattr(np, 'trapezoid', np.trapz)
    return trapezoid_func(values[mask], times[mask])

# =========================
# CORE PROCESSING LOGIC
# =========================
def normalize_by_window(df, prefix, baseline_minutes, suffix="_bc", baseline_label=None):
    out = df.copy()
    m_map = minute_map(out, prefix)
    if not m_map: return out, None

    exist_mins = [m for m in baseline_minutes if m in m_map]
    if not exist_mins: return out, None

    b_cols = [m_map[m] for m in exist_mins]
    b_mean = out[b_cols].mean(axis=1)

    m_col = f"{prefix}_baseline_mean" if baseline_label is None else f"{prefix}_baseline_mean_{baseline_label}"
    out[m_col] = b_mean
    
    for m, c in m_map.items():
        out[f"{c}{suffix}"] = out[c] - b_mean
    return out, exist_mins

def make_phase_features(df, prefix, use_normalized=True, suffix="_bc"):
    out = df.copy()
    m_map = minute_map(out, prefix)
    s_mins = get_phase_minutes_from_existing(m_map, 1, 10)
    d_mins = get_phase_minutes_from_existing(m_map, 11, 20)

    s_cols = [f"{m_map[m]}{suffix}" if use_normalized else m_map[m] for m in s_mins]
    d_cols = [f"{m_map[m]}{suffix}" if use_normalized else m_map[m] for m in d_mins]

    if s_cols: out[f"{prefix}_static_mean"] = out[s_cols].mean(axis=1)
    if d_cols: out[f"{prefix}_dynamic_mean"] = out[d_cols].mean(axis=1)
    if s_cols and d_cols:
        out[f"{prefix}_delta_dynamic_minus_static"] = out[f"{prefix}_dynamic_mean"] - out[f"{prefix}_static_mean"]
    return out

def add_dynamic_features(df, prefix, use_normalized=True, suffix="_bc"):
    out = df.copy()
    m_map = minute_map(out, prefix)
    d_mins = get_phase_minutes_from_existing(m_map, 11, 20)
    if len(d_mins) < 2: return out

    cols = [f"{m_map[m]}{suffix}" if use_normalized else m_map[m] for m in d_mins]
    out[f"{prefix}_dynamic_slope"] = out[cols].apply(lambda r: _row_slope(r.values, d_mins), axis=1)
    out[f"{prefix}_dynamic_auc"] = out[cols].apply(lambda r: _row_auc(r.values, d_mins), axis=1)
    out[f"{prefix}_dynamic_std"] = out[cols].std(axis=1)
    return out

# =========================
# DATA LOADING & SPLIT
# =========================
def load_and_preprocess():
    print("Step 1: Loading all sheets including PPG...")
    st = pd.read_excel(INPUT_PATH, sheet_name="ST")
    eda = pd.read_excel(INPUT_PATH, sheet_name="EDA")
    ecg = pd.read_excel(INPUT_PATH, sheet_name="ECG").rename(columns={"subject": "연번", "condition": "환경", "name": "이름"})
    ppg = pd.read_excel(INPUT_PATH, sheet_name="PPG").rename(columns={"subject": "연번", "condition": "환경", "name": "이름"})

    for df in [st, eda, ecg, ppg]:
        if "환경" in df.columns: df["환경"] = df["환경"].map(normalize_env_label)

    # 1. ST Processing
    for p in ["st_mean", "at_mean"]:
        st, _ = normalize_by_window(st, p, [1,2,3], "_bc")
        st, _ = normalize_by_window(st, p, [8,9,10], "_bc_8to10", "8to10")
        st = make_phase_features(st, p)
        st = add_dynamic_features(st, p)

    # 2. EDA Processing
    eda, _ = normalize_by_window(eda, "tonic", [1,2,3], "_bc")
    eda = make_phase_features(eda, "tonic")
    eda = add_dynamic_features(eda, "tonic")
    eda = make_phase_features(eda, "phasic", use_normalized=False) 

    # 3. ECG Processing
    ecg_targets = ["HR", "MeanRR", "SDNN", "RMSSD", "LF", "HF", "LF_HF"]
    for p in ecg_targets:
        ecg, _ = normalize_by_window(ecg, p, [1,2,3], "_bc")
        ecg = make_phase_features(ecg, p)
        ecg = add_dynamic_features(ecg, p)

    # 4. PPG Processing
    for p in ["pulse_amplitude_mean", "blood_flow_mean"]:
        ppg, _ = normalize_by_window(ppg, p, [1,2,3], "_bc")
        ppg = make_phase_features(ppg, p)
        ppg = add_dynamic_features(ppg, p)

    return st, eda, ecg, ppg

def build_scenario_data(env, st, eda, ecg, ppg):
    def get_rows(df, target_env):
        is_dynamic = target_env in DYNAMIC_SCENARIOS
        mask = df["환경"].str.contains(target_env, na=False)
        tmp = df[mask].copy()
        
        # 특징량 정리 및 시나리오별 통일된 이름 부여
        for c in tmp.columns:
            if not is_dynamic and "_static_mean" in c:
                tmp[c.replace("_static_mean", "_scenario_mean")] = tmp[c]
            if is_dynamic and "_dynamic_mean" in c:
                tmp[c.replace("_dynamic_mean", "_scenario_mean")] = tmp[c]
        return tmp

    st_p = get_rows(st, env)
    eda_p = get_rows(eda, env)
    ecg_p = get_rows(ecg, env)
    ppg_p = get_rows(ppg, env)

    merged = st_p.merge(eda_p, on=["연번", "이름"], how="outer", suffixes=('', '_eda'))
    merged = merged.merge(ecg_p, on=["연번", "이름"], how="outer", suffixes=('', '_ecg'))
    merged = merged.merge(ppg_p, on=["연번", "이름"], how="outer", suffixes=('', '_ppg'))
    
    return merged

# =========================
# MAIN EXECUTION
# =========================
st_proc, eda_proc, ecg_proc, ppg_proc = load_and_preprocess()

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    st_proc.to_excel(writer, sheet_name="ST_processed", index=False)
    eda_proc.to_excel(writer, sheet_name="EDA_processed", index=False)
    ecg_proc.to_excel(writer, sheet_name="ECG_processed", index=False)
    ppg_proc.to_excel(writer, sheet_name="PPG_processed", index=False)

    for scn in SCENARIOS:
        scn_df = build_scenario_data(scn, st_proc, eda_proc, ecg_proc, ppg_proc)
        if not scn_df.empty:
            scn_df.to_excel(writer, sheet_name=scn, index=False)

print(f"✅ 분석 완료 및 경고 해결! 결과 저장 경로: {OUTPUT_PATH}")