import pandas as pd
import numpy as np
import re
import os

# =========================
# CONFIG  (컬럼명은 "소문자 기준")
# =========================
PATH_BI = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\BI+SUR.xlsx"
PATH_PHY = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\PHY.xlsx"
SHEETS_TO_RUN = [0, 2]  # BI+SUR에서 돌릴 시트: 0, 2

ID_COL = "연번"
ENV_COL = "환경"
TIME_COL = "시간"

# BI
SEX_COL_CANDIDATES = ["성별", "sex", "gender"]
AGE_COL_CANDIDATES = ["나이", "age"]
BMI_COL_CANDIDATES = ["bmi"]

# MBTI / FFM
MBTI_COLS = ["e", "n", "t", "j", "a"]
FFM_COLS = ["o1", "c1", "e1", "a1", "n1"]

# SUR
RESP_COLS = ["tsv", "tcv", "ta", "tp", "pt"]
P_COLS = [f"p{i}" for i in range(1, 9)]
M_COLS = [f"m{i}" for i in range(1, 9)]

# EXP time intervals   
STATIC_TIMES = [0, 5, 10]
DYNAMIC_TIMES = [10, 15, 20]

# PHY에서 Δphysio를 만들 때 사용할 분 구간(기본: 1~10 vs 11~20)
PHY_STATIC_MINUTES = list(range(1, 11))
PHY_DYNAMIC_MINUTES = list(range(11, 21))

MISSING_TOKENS = ["-", "—", "–", "", "#n/a", "#na", "n/a", "na"]

HERE = os.getcwd()
OUT_XLSX = os.path.join(HERE, "spearman_BI_SUR_plus_PHY.xlsx")


# =========================
# Helpers
# =========================
def norm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    return frame


def keep_existing(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in frame.columns]


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    cands = [c.lower() for c in candidates]
    for c in cands:
        if c in frame.columns:
            return c
    return None


def to_numeric_df(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame[cols].copy()
    out = out.replace(MISSING_TOKENS, np.nan)
    out = out.infer_objects(copy=False)
    out = out.apply(pd.to_numeric, errors="coerce")
    return out


def lower_triangle_only(corr: pd.DataFrame, keep_diag: bool = False) -> pd.DataFrame:
    n = corr.shape[0]
    k = 1 if keep_diag else 0
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
    # 일부 파일에서 HG로 들어오는 케이스 보정
    if e in ["HG"]:
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
        time_series.isin(STATIC_TIMES),
        "static",
        np.where(time_series.isin(DYNAMIC_TIMES), "dynamic", pd.NA),
    )


def encode_sex(series: pd.Series) -> pd.Series:
    """
    sex_bin: 남/남자/M/male -> 1, 여/여자/F/female -> 0, 그 외 NaN
    """
    s = series.astype(str).str.strip().str.lower()
    male = {"m", "male", "남", "남자", "man", "1"}
    female = {"f", "female", "여", "여자", "woman", "0"}
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s.isin(male)] = 1.0
    out[s.isin(female)] = 0.0
    return out


# =========================
# Sheet 0: Thermal(반복측정) RAW + Δ + Δgap
#  + (기본정보 포함 RAW/Δ 상관용 테이블 생성)
# =========================
def run_sheet0(frame: pd.DataFrame) -> dict:
    frame = norm_columns(frame)

    id_col = ID_COL.lower()
    env_col = ENV_COL.lower()
    time_col = TIME_COL.lower()

    assert id_col in frame.columns, f"Missing {id_col}"
    assert env_col in frame.columns, f"Missing {env_col}"
    assert time_col in frame.columns, f"Missing {time_col}"

    # 기본정보 컬럼 탐색
    sex_col = first_existing(frame, SEX_COL_CANDIDATES)
    age_col = first_existing(frame, AGE_COL_CANDIDATES)
    bmi_col = first_existing(frame, BMI_COL_CANDIDATES)

    demo_cols = []
    if sex_col:
        frame["sex_bin"] = encode_sex(frame[sex_col])
        demo_cols.append("sex_bin")
    if age_col:
        demo_cols.append(age_col)
    if bmi_col:
        demo_cols.append(bmi_col)

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

    # RAW Spearman (성격 + 기본정보 + thermal/behavior)
    raw_cols = demo_cols + personality_cols + resp_cols + p_cols + m_cols
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
        dff_num.groupby([id_col, "set", "dir", "phase"], as_index=False)[agg_cols].mean()
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

    # 성격/기본정보 병합(연번 기준)
    keep_cols = [id_col] + personality_cols
    personality_unique = frame[keep_cols].drop_duplicates(subset=[id_col])

    demo_keep = [id_col]
    if "sex_bin" in frame.columns:
        demo_keep.append("sex_bin")
    if age_col:
        demo_keep.append(age_col)
    if bmi_col:
        demo_keep.append(bmi_col)
    demo_unique = frame[demo_keep].drop_duplicates(subset=[id_col])

    delta_tbl = (
        wide.merge(personality_unique, on=id_col, how="left")
            .merge(demo_unique, on=id_col, how="left")
    )

    delta_cols = [c for c in delta_tbl.columns if c.startswith("Δ")]

    # Δ Spearman: (기본정보 + 성격 + dir) ↔ (Δ...)
    corr_cols = demo_cols + personality_cols + ["dir"] + delta_cols
    corr_df = delta_tbl[corr_cols].copy()
    corr_df = corr_df.replace(MISSING_TOKENS, np.nan).infer_objects(copy=False)
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

    delta_corr = corr_df.corr(method="spearman")
    delta_corr_L = lower_triangle_only(delta_corr, keep_diag=False)

    return {
        "raw_corr_lower": raw_corr_L,
        "delta_corr_lower": delta_corr_L,
        "delta_table": delta_tbl,
        "demo_cols": demo_cols,
        "personality_cols": personality_cols,
        "age_col": age_col,
        "bmi_col": bmi_col,
    }


# =========================
# Sheet 2: 설문(개인 1행) RAW Spearman only
#  + (기본정보 포함)
# =========================
def run_sheet2(demo_cols_from_sheet0: list[str] | None = None) -> dict:
    frame = pd.read_excel(PATH_BI, sheet_name=2, header=1)
    frame = norm_columns(frame)

    id_col = ID_COL.lower()
    assert id_col in frame.columns, "Missing '연번' in sheet 2"

    # 기본정보(가능하면 포함)
    sex_col = first_existing(frame, SEX_COL_CANDIDATES)
    age_col = first_existing(frame, AGE_COL_CANDIDATES)
    bmi_col = first_existing(frame, BMI_COL_CANDIDATES)

    demo_cols = []
    if sex_col:
        frame["sex_bin"] = encode_sex(frame[sex_col])
        demo_cols.append("sex_bin")
    if age_col:
        demo_cols.append(age_col)
    if bmi_col:
        demo_cols.append(bmi_col)

    mbti_cols = keep_existing(frame, MBTI_COLS)
    ffm_cols = keep_existing(frame, FFM_COLS)
    personality_cols = mbti_cols + ffm_cols

    if not personality_cols:
        raise ValueError("No MBTI/FFM columns found on sheet 2.")

    eup_cols = [c for c in frame.columns if c.startswith("eup")]
    sp_cols = [c for c in ["sp_c", "sp_h", "dsp"] if c in frame.columns]
    target_cols = eup_cols + sp_cols

    if not target_cols:
        raise ValueError("No EUP/SP/DSP columns found on sheet 2.")

    use_cols = demo_cols + personality_cols + target_cols
    num = to_numeric_df(frame, use_cols)

    corr = num.corr(method="spearman")
    corr_L = lower_triangle_only(corr, keep_diag=False)

    demo_personality = demo_cols + personality_cols
    dp_vs_targets = corr.loc[demo_personality, target_cols]

    return {
        "raw_corr_lower": corr_L,
        "demo_personality_vs_EUP_SP": dp_vs_targets,
    }


# =========================
# PHY parsing utilities
# =========================
def load_base_from_sheet0_for_phy(df0_norm: pd.DataFrame, personality_cols: list[str], age_col: str | None, bmi_col: str | None) -> tuple[pd.DataFrame, list[str]]:
    """
    PHY와 merge할 base 테이블(연번 단위 1행):
    - sex_bin, age, bmi, MBTI, FFM
    """
    id_col = ID_COL.lower()

    base_cols = [id_col] + personality_cols

    # sex_bin / age / bmi
    sex_col = first_existing(df0_norm, SEX_COL_CANDIDATES)
    demo_cols = []
    tmp = df0_norm.copy()

    if sex_col:
        tmp["sex_bin"] = encode_sex(tmp[sex_col])
        demo_cols.append("sex_bin")
    if age_col:
        demo_cols.append(age_col)
    if bmi_col:
        demo_cols.append(bmi_col)

    base_cols = [id_col] + demo_cols + personality_cols
    base = tmp[base_cols].drop_duplicates(subset=[id_col]).copy()

    predictors = demo_cols + personality_cols
    return base, predictors


def phy_norm_set(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    s = s.replace({"HG": "GH"})
    return s


def load_phy_st_or_ecg(sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(PATH_PHY, sheet_name=sheet_name)
    df = norm_columns(df)

    # 기대 컬럼: subject, condition
    if "subject" not in df.columns or "condition" not in df.columns:
        raise ValueError(f"[PHY:{sheet_name}] expected columns 'subject' and 'condition'")

    df = df.rename(columns={"subject": ID_COL.lower(), "condition": "set"})
    df["set"] = phy_norm_set(df["set"])
    return df


def load_phy_eda() -> pd.DataFrame:
    df = pd.read_excel(PATH_PHY, sheet_name="EDA")
    df = norm_columns(df)

    # 기대 컬럼: 연번, 환경
    if ID_COL.lower() not in df.columns or "환경" not in df.columns:
        # 일부 파일에서 환경 컬럼명이 env/condition일 수도 있어 fallback
        env_like = first_existing(df, ["환경", "condition", "env", "set"])
        if env_like is None:
            raise ValueError("[PHY:EDA] cannot find env/condition column")
        df = df.rename(columns={env_like: "환경"})

    df["환경"] = phy_norm_set(df["환경"])
    df = df.rename(columns={"환경": "set"})

    # 1분 단위가 "1min_tonic" + unnamed 컬럼들로 이어지는 형태로 가정
    # 실제 업로드본: "1min_Tonic"가 있었음 -> lower 처리 후 "1min_tonic"
    tonic_col = "1min_tonic"
    if tonic_col not in df.columns:
        raise ValueError("[PHY:EDA] missing '1min_tonic' column after normalization")

    unnamed = [c for c in df.columns if c.startswith("unnamed:")]
    # unnamed는 순서 보장 위해 숫자 부분으로 정렬
    def _unnamed_key(x: str) -> int:
        m = re.search(r"unnamed:(\d+)", x)
        return int(m.group(1)) if m else 10**9

    unnamed_sorted = sorted(unnamed, key=_unnamed_key)
    minute_cols = [tonic_col] + unnamed_sorted

    rename_map = {minute_cols[i]: f"eda_tonic_{i+1}" for i in range(len(minute_cols))}
    out = df[[ID_COL.lower(), "set"] + minute_cols].rename(columns=rename_map)
    return out


def extract_minute_features_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    ST/ECG 형태: feature_1, feature_2, ... feature_20 같은 wide 컬럼을 feature별로 인식.
    반환:
      - 원 df (그대로)
      - minute feature columns 리스트
    """
    minute_cols = []
    rx = re.compile(r"^[a-z0-9]+_\d+$")  # ex) hr_1, rmssd_10 ...
    for c in df.columns:
        if rx.match(str(c)):
            minute_cols.append(c)
    return df, minute_cols


def compute_phy_delta_means(df: pd.DataFrame, minute_cols: list[str], prefix: str) -> pd.DataFrame:
    """
    minute_cols가 feature_minute 형태일 때
    - static_mean (1~10)
    - dynamic_mean (11~20)
    - Δ = dynamic - static
    feature별로 산출하여 wide 형태로 반환
    index key: 연번, set
    """
    id_col = ID_COL.lower()

    # feature -> {minute -> col}
    fmap: dict[str, dict[int, str]] = {}
    for c in minute_cols:
        m = re.match(r"^([a-z0-9]+)_(\d+)$", c)
        if not m:
            continue
        feat = m.group(1)
        minute = int(m.group(2))
        fmap.setdefault(feat, {})[minute] = c

    rows = df[[id_col, "set"]].copy()

    out = rows.copy()
    out_cols = []
    for feat, mm in fmap.items():
        static_cols = [mm[m] for m in PHY_STATIC_MINUTES if m in mm]
        dynamic_cols = [mm[m] for m in PHY_DYNAMIC_MINUTES if m in mm]
        if not static_cols or not dynamic_cols:
            continue

        tmp = df.copy()
        tmp[static_cols + dynamic_cols] = to_numeric_df(tmp, static_cols + dynamic_cols)

        smean = tmp[static_cols].mean(axis=1)
        dmean = tmp[dynamic_cols].mean(axis=1)
        out[f"{prefix}_{feat}_static_mean"] = smean
        out[f"{prefix}_{feat}_dynamic_mean"] = dmean
        out[f"{prefix}_Δ{feat}"] = dmean - smean
        out_cols += [f"{prefix}_{feat}_static_mean", f"{prefix}_{feat}_dynamic_mean", f"{prefix}_Δ{feat}"]

    return out, out_cols


def compute_eda_delta_means(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    EDA 형태: eda_tonic_1..N
    - RAW minute columns 반환
    - Δ(동-정) 반환 (정: 1~10, 동: 11~20)
    """
    id_col = ID_COL.lower()
    minute_cols = [c for c in df.columns if c.startswith("eda_tonic_")]
    # minute number
    def _mnum(c: str) -> int:
        return int(c.split("_")[-1])

    minute_cols = sorted(minute_cols, key=_mnum)

    tmp = df.copy()
    tmp[minute_cols] = to_numeric_df(tmp, minute_cols)

    static_cols = [c for c in minute_cols if 1 <= _mnum(c) <= 10]
    dynamic_cols = [c for c in minute_cols if 11 <= _mnum(c) <= 20]

    out = tmp[[id_col, "set"]].copy()
    out_cols = []
    if static_cols and dynamic_cols:
        out["eda_static_mean"] = tmp[static_cols].mean(axis=1)
        out["eda_dynamic_mean"] = tmp[dynamic_cols].mean(axis=1)
        out["eda_Δtonic"] = out["eda_dynamic_mean"] - out["eda_static_mean"]
        out_cols = ["eda_static_mean", "eda_dynamic_mean", "eda_Δtonic"]

    return tmp, minute_cols, (out, out_cols)


def spearman_predictors_vs_features(merged: pd.DataFrame, predictors: list[str], features: list[str]) -> pd.DataFrame:
    use_cols = predictors + features
    num = to_numeric_df(merged, use_cols)
    corr = num.corr(method="spearman")
    return corr.loc[predictors, features]


# =========================
# MAIN
# =========================
all_outputs = {}

# 1) BI+SUR Sheet0 + Sheet2
df0_raw = pd.read_excel(PATH_BI, sheet_name=0)
out0 = run_sheet0(df0_raw)

all_outputs["S0_RAW_corr_lower"] = out0["raw_corr_lower"]
all_outputs["S0_DELTA_corr_lower"] = out0["delta_corr_lower"]
all_outputs["S0_DELTA_table"] = out0["delta_table"]

out2 = run_sheet2(demo_cols_from_sheet0=out0["demo_cols"])
all_outputs["S2_RAW_corr_lower"] = out2["raw_corr_lower"]
all_outputs["S2_demo_personality_vs_EUP_SP"] = out2["demo_personality_vs_EUP_SP"]

# base for PHY (sheet0에서 predictors 구성)
df0_norm = norm_columns(df0_raw)
base_for_phy, predictors = load_base_from_sheet0_for_phy(
    df0_norm,
    personality_cols=out0["personality_cols"],
    age_col=out0["age_col"],
    bmi_col=out0["bmi_col"],
)

# 2) PHY: ST
st = load_phy_st_or_ecg("ST")
st, st_min_cols = extract_minute_features_wide(st)
st_m = st.merge(base_for_phy, left_on=ID_COL.lower(), right_on=ID_COL.lower(), how="left")
if st_min_cols:
    all_outputs["PHY_ST_pred_vs_minute"] = spearman_predictors_vs_features(st_m, predictors, st_min_cols)

    st_delta_tbl, st_delta_cols = compute_phy_delta_means(st_m, st_min_cols, prefix="st")
    # Δphysio만 (predictors vs Δ)
    st_delta_m = st_delta_tbl.merge(base_for_phy, on=ID_COL.lower(), how="left")
    st_delta_feats = [c for c in st_delta_cols if c.startswith("st_Δ")]
    if st_delta_feats:
        all_outputs["PHY_ST_pred_vs_Δ"] = spearman_predictors_vs_features(st_delta_m, predictors, st_delta_feats)

# 3) PHY: ECG
ecg = load_phy_st_or_ecg("ECG")
ecg, ecg_min_cols = extract_minute_features_wide(ecg)
ecg_m = ecg.merge(base_for_phy, on=ID_COL.lower(), how="left")
if ecg_min_cols:
    all_outputs["PHY_ECG_pred_vs_minute"] = spearman_predictors_vs_features(ecg_m, predictors, ecg_min_cols)

    ecg_delta_tbl, ecg_delta_cols = compute_phy_delta_means(ecg_m, ecg_min_cols, prefix="ecg")
    ecg_delta_m = ecg_delta_tbl.merge(base_for_phy, on=ID_COL.lower(), how="left")
    ecg_delta_feats = [c for c in ecg_delta_cols if c.startswith("ecg_Δ")]
    if ecg_delta_feats:
        all_outputs["PHY_ECG_pred_vs_Δ"] = spearman_predictors_vs_features(ecg_delta_m, predictors, ecg_delta_feats)

# 4) PHY: EDA
eda_raw = load_phy_eda()
eda_raw_m = eda_raw.merge(base_for_phy, on=ID_COL.lower(), how="left")
eda_raw_min_cols = [c for c in eda_raw.columns if c.startswith("eda_tonic_")]
if eda_raw_min_cols:
    all_outputs["PHY_EDA_pred_vs_minute"] = spearman_predictors_vs_features(eda_raw_m, predictors, eda_raw_min_cols)

    eda_num, eda_min_cols, (eda_delta_tbl, eda_delta_cols) = compute_eda_delta_means(eda_raw)
    if eda_delta_cols:
        eda_delta_m = eda_delta_tbl.merge(base_for_phy, on=ID_COL.lower(), how="left")
        eda_delta_feats = [c for c in eda_delta_cols if c.startswith("eda_Δ")]
        if eda_delta_feats:
            all_outputs["PHY_EDA_pred_vs_Δ"] = spearman_predictors_vs_features(eda_delta_m, predictors, eda_delta_feats)

# 5) (선택) Sheet0 Δ테이블과 PHY Δ를 set 단위로 merge → predictors+dir vs (Δthermal+Δgap + Δphysio)
#    - Δ테이블 key: 연번, set, dir
#    - PHY Δ테이블 key: 연번, set
delta_tbl = out0["delta_table"].copy()
delta_tbl = norm_columns(delta_tbl)

id_col = ID_COL.lower()
if "set" in delta_tbl.columns and "dir" in delta_tbl.columns:
    # ST Δ
    try:
        if st_min_cols:
            st_delta_tbl2, st_delta_cols2 = compute_phy_delta_means(st, st_min_cols, prefix="st")
            st_delta_tbl2 = norm_columns(st_delta_tbl2)
            merged = delta_tbl.merge(st_delta_tbl2, on=[id_col, "set"], how="left")
            delta_cols = [c for c in merged.columns if c.startswith("Δ")]  # thermal/behavior Δ + gap
            phys_delta_cols = [c for c in merged.columns if c.startswith("st_Δ")]
            use_predictors = out0["demo_cols"] + out0["personality_cols"] + ["dir"]
            # numeric prep
            corr_df = merged[use_predictors + delta_cols + phys_delta_cols].copy()
            corr_df = corr_df.replace(MISSING_TOKENS, np.nan).infer_objects(copy=False)
            corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr(method="spearman")
            all_outputs["S0Δ_plus_STΔ_corr_lower"] = lower_triangle_only(corr, keep_diag=False)
    except Exception as e:
        all_outputs["WARN_S0Δ_plus_STΔ"] = pd.DataFrame({"warning": [str(e)]})

    # ECG Δ
    try:
        if ecg_min_cols:
            ecg_delta_tbl2, ecg_delta_cols2 = compute_phy_delta_means(ecg, ecg_min_cols, prefix="ecg")
            ecg_delta_tbl2 = norm_columns(ecg_delta_tbl2)
            merged = delta_tbl.merge(ecg_delta_tbl2, on=[id_col, "set"], how="left")
            delta_cols = [c for c in merged.columns if c.startswith("Δ")]
            phys_delta_cols = [c for c in merged.columns if c.startswith("ecg_Δ")]
            use_predictors = out0["demo_cols"] + out0["personality_cols"] + ["dir"]
            corr_df = merged[use_predictors + delta_cols + phys_delta_cols].copy()
            corr_df = corr_df.replace(MISSING_TOKENS, np.nan).infer_objects(copy=False)
            corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr(method="spearman")
            all_outputs["S0Δ_plus_ECGΔ_corr_lower"] = lower_triangle_only(corr, keep_diag=False)
    except Exception as e:
        all_outputs["WARN_S0Δ_plus_ECGΔ"] = pd.DataFrame({"warning": [str(e)]})

    # EDA Δ
    try:
        _, _, (eda_delta_tbl2, eda_delta_cols2) = compute_eda_delta_means(eda_raw)
        if eda_delta_cols2:
            eda_delta_tbl2 = norm_columns(eda_delta_tbl2)
            merged = delta_tbl.merge(eda_delta_tbl2, on=[id_col, "set"], how="left")
            delta_cols = [c for c in merged.columns if c.startswith("Δ")]
            phys_delta_cols = [c for c in merged.columns if c.startswith("eda_Δ")]
            use_predictors = out0["demo_cols"] + out0["personality_cols"] + ["dir"]
            corr_df = merged[use_predictors + delta_cols + phys_delta_cols].copy()
            corr_df = corr_df.replace(MISSING_TOKENS, np.nan).infer_objects(copy=False)
            corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr(method="spearman")
            all_outputs["S0Δ_plus_EDAΔ_corr_lower"] = lower_triangle_only(corr, keep_diag=False)
    except Exception as e:
        all_outputs["WARN_S0Δ_plus_EDAΔ"] = pd.DataFrame({"warning": [str(e)]})


# =========================
# SAVE
# =========================
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    for name, obj in all_outputs.items():
        # Excel sheet name 제한(31자) 방지
        sheet = name[:31]
        obj.to_excel(writer, sheet_name=sheet, index=True)

print("완료.")
print("저장 파일:", OUT_XLSX)
print("생성 시트:", " / ".join(all_outputs.keys()))