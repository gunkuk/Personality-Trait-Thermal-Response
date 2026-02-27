import os
import re
import numpy as np
import pandas as pd

# =========================
# CONFIG  (컬럼명은 "소문자 기준")
# =========================
PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\MBTI.xlsx"
SHEETS_TO_RUN = [0, 2]  # 1번째 시트(0), 3번째 시트(2)

ID_COL = "연번"
ENV_COL = "환경"
TIME_COL = "시간"

# MBTI / FFM
MBTI_COLS = ["e", "n", "t", "j", "a"]
FFM_COLS = ["o1", "c1", "e1", "a1", "n1"]

# Sheet0 thermal + behavior
RESP_COLS = ["tsv", "tcv", "ta", "tp", "pt"]
P_COLS = [f"p{i}" for i in range(1, 9)]
M_COLS = [f"m{i}" for i in range(1, 9)]

STATIC_TIMES = [0, 5, 10]
DYNAMIC_TIMES = [10, 15, 20]

MISSING_TOKENS = ["-", "—", "–", "", "#n/a", "#na", "n/a", "na"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_XLSX = os.path.join(HERE, "spearman_sheet0_plus_sheet2.xlsx")


# =========================
# Helpers
# =========================
def norm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    return frame

def keep_existing(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in frame.columns]

def to_numeric_df(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame[cols].copy()
    out = out.replace(MISSING_TOKENS, np.nan)
    # FutureWarning 방지: replace 후 infer_objects
    out = out.infer_objects(copy=False)
    out = out.apply(pd.to_numeric, errors="coerce")
    return out

def lower_triangle_only(corr: pd.DataFrame, keep_diag: bool = False) -> pd.DataFrame:
    """
    keep_diag=False: 하삼각(대각 제외)만 남김 (상삼각+대각 제거)
    keep_diag=True : 하삼각+대각 남김
    """
    n = corr.shape[0]
    k = 1 if keep_diag else 0  # keep_diag=False면 k=0 -> 대각 포함 상삼각 제거
    mask = np.triu(np.ones((n, n), dtype=bool), k=k)
    return corr.mask(mask)

def env_to_set(env: str) -> str:
    e = str(env).strip().upper()
    if e in ["A", "B"]:
        return "AB"
    if e in ["C", "D"]:
        return "CD"
    if e in ["E", "F"]:
        return "EF"
    if e in ["G", "H"]:
        return "GH"
    return np.nan

def set_to_direction(set_code: str) -> int:
    s = str(set_code).strip().upper()
    if s in ["AB", "GH"]:
        return -1  # cooling
    if s in ["CD", "EF"]:
        return +1  # heating
    return np.nan

def make_phase(time_series: pd.Series) -> pd.Series:
    return np.where(
        time_series.isin(STATIC_TIMES), "static",
        np.where(time_series.isin(DYNAMIC_TIMES), "dynamic", pd.NA)
    )

def find_cols_regex(frame: pd.DataFrame, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [c for c in frame.columns if rx.match(c)]


# =========================
# Sheet 0: Thermal(반복측정) RAW + Δ + Δgap
# =========================
def run_sheet0(frame: pd.DataFrame) -> dict:
    frame = norm_columns(frame)

    id_col = ID_COL.lower()
    env_col = ENV_COL.lower()
    time_col = TIME_COL.lower()

    assert id_col in frame.columns, f"Missing {id_col}"
    assert env_col in frame.columns, f"Missing {env_col}"
    assert time_col in frame.columns, f"Missing {time_col}"

    mbti_cols = keep_existing(frame, MBTI_COLS)
    ffm_cols = keep_existing(frame, FFM_COLS)
    personality_cols = mbti_cols + ffm_cols
    if not personality_cols:
        raise ValueError("No MBTI/FFM columns found on sheet 0.")

    resp_cols = keep_existing(frame, RESP_COLS)
    p_cols = keep_existing(frame, P_COLS)
    m_cols = keep_existing(frame, M_COLS)

    # 시나리오 파생
    frame["set"] = frame[env_col].map(env_to_set)
    frame["dir"] = frame["set"].map(set_to_direction)
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")

    # RAW Spearman
    raw_cols = personality_cols + resp_cols + p_cols + m_cols
    raw_num = to_numeric_df(frame, raw_cols)
    raw_corr = raw_num.corr(method="spearman")
    raw_corr_L = lower_triangle_only(raw_corr, keep_diag=False)

    # Δ 테이블: 사람 × 세트
    dff = frame.dropna(subset=["set", "dir", time_col]).copy()
    dff[time_col] = pd.to_numeric(dff[time_col], errors="coerce")
    dff = dff.dropna(subset=[time_col])
    dff[time_col] = dff[time_col].astype(int)

    dff["phase"] = make_phase(dff[time_col])
    dff = dff.dropna(subset=["phase"])

    agg_cols = resp_cols + p_cols + m_cols
    dff_num = dff.copy()
    dff_num[agg_cols] = to_numeric_df(dff_num, agg_cols)

    phase_means = (
        dff_num.groupby([id_col, "set", "dir", "phase"], as_index=False)[agg_cols]
               .mean()
    )

    wide = phase_means.pivot(index=[id_col, "set", "dir"], columns="phase", values=agg_cols)
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    wide = wide.reset_index()

    # Δ(동적-정적)
    for v in agg_cols:
        s = f"{v}_static"
        d = f"{v}_dynamic"
        if s in wide.columns and d in wide.columns:
            wide[f"Δ{v}"] = wide[d] - wide[s]

    # Δgap_i = (M_i - P_i)_dynamic - (M_i - P_i)_static
    for i in range(1, 9):
        ms = f"m{i}_static"
        md = f"m{i}_dynamic"
        ps = f"p{i}_static"
        pd_ = f"p{i}_dynamic"
        if all(c in wide.columns for c in [ms, md, ps, pd_]):
            wide[f"Δgap{i}"] = (wide[md] - wide[pd_]) - (wide[ms] - wide[ps])

    # 성격 병합
    personality_unique = frame[[id_col] + personality_cols].drop_duplicates(subset=[id_col])
    delta_tbl = wide.merge(personality_unique, on=id_col, how="left")

    # Δ 컬럼 수집(ΔTSV..., ΔP..., ΔM..., Δgap...)
    delta_cols = [c for c in delta_tbl.columns if c.startswith("Δ")]

    # Δ Spearman: (성격 + dir) ↔ (Δ...)
    corr_cols = personality_cols + ["dir"] + delta_cols
    corr_df = delta_tbl[corr_cols].copy()
    corr_df = corr_df.replace(MISSING_TOKENS, np.nan).infer_objects(copy=False)
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

    delta_corr = corr_df.corr(method="spearman")
    delta_corr_L = lower_triangle_only(delta_corr, keep_diag=False)

    return {
        "raw_corr_lower": raw_corr_L,
        "delta_corr_lower": delta_corr_L,
        "delta_table": delta_tbl
    }


# =========================
# Sheet 2: 설문(개인 1행) RAW Spearman only
# =========================
def run_sheet2() -> dict:
    """
    Sheet index 2 (3번째 시트):
    - header는 1행 (네 파일 구조 기준)
    - 성격(MBTI+FFM) ↔ EUP / SP / DSP Spearman
    """
    frame = pd.read_excel(PATH, sheet_name=2, header=1)
    frame = norm_columns(frame)

    id_col = ID_COL.lower()
    assert id_col in frame.columns, "Missing '연번' in sheet 2"

    mbti_cols = keep_existing(frame, MBTI_COLS)
    ffm_cols = keep_existing(frame, FFM_COLS)
    personality_cols = mbti_cols + ffm_cols

    print("[Sheet2] MBTI:", mbti_cols)
    print("[Sheet2] FFM :", ffm_cols)

    if not personality_cols:
        raise ValueError("No MBTI/FFM columns found on sheet 2.")

    eup_cols = [c for c in frame.columns if c.startswith("eup")]
    sp_cols = [c for c in ["sp_c", "sp_h", "dsp"] if c in frame.columns]
    target_cols = eup_cols + sp_cols

    if not target_cols:
        raise ValueError("No EUP/SP/DSP columns found on sheet 2.")

    use_cols = personality_cols + target_cols
    num = to_numeric_df(frame, use_cols)

    corr = num.corr(method="spearman")
    corr_L = lower_triangle_only(corr, keep_diag=False)

    personality_vs_targets = corr.loc[personality_cols, target_cols]

    return {
        "raw_corr_lower": corr_L,
        "personality_vs_targets": personality_vs_targets
    }


# =========================
# MAIN
# =========================
all_outputs = {}

for sh in SHEETS_TO_RUN:
    if sh == 0:
        df0 = pd.read_excel(PATH, sheet_name=0)
        out0 = run_sheet0(df0)
        all_outputs["S0_RAW_corr_lower"] = out0["raw_corr_lower"]
        all_outputs["S0_DELTA_corr_lower"] = out0["delta_corr_lower"]
        all_outputs["S0_DELTA_table"] = out0["delta_table"]

    elif sh == 2:
        out2 = run_sheet2()
        all_outputs["S2_RAW_corr_lower"] = out2["raw_corr_lower"]
        all_outputs["S2_personality_vs_EUP_SP"] = out2["personality_vs_targets"]

    else:
        raise ValueError(f"Unhandled sheet index: {sh}")

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    for name, obj in all_outputs.items():
        obj.to_excel(writer, sheet_name=name, index=True)

print("완료.")
print("저장 파일:", OUT_XLSX)
print("생성 시트:", " / ".join(all_outputs.keys()))