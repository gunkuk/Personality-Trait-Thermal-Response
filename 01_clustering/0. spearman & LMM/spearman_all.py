from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# ✅ UTIL import
from utils import (
    norm_columns, first_existing, keep_existing, ensure_columns,
    to_numeric_df, lower_triangle_only, reorder_square,
    encode_sex, env_to_set, set_to_direction, make_phase,
    select_cols_regex, write_excel
)


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    path_bi: str = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\data\BI+SUR.xlsx"
    path_phy: str = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\data\PHY.xlsx"

    out_dir: str = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT"
    out_name: str = "spearman_all_x.xlsx"

    id_col: str = "연번"
    env_col: str = "환경"
    time_col: str = "시간"

    sex_candidates: Tuple[str, ...] = ("성별", "sex", "gender")
    age_candidates: Tuple[str, ...] = ("나이", "age")
    bmi_candidates: Tuple[str, ...] = ("bmi",)

    mbti_cols: Tuple[str, ...] = ("e", "n", "t", "j", "a")
    ffm_cols: Tuple[str, ...] = ("o1", "c1", "e1", "a1", "n1")

    resp_cols: Tuple[str, ...] = ("tsv", "tcv", "ta", "tp", "pt")
    p_cols: Tuple[str, ...] = tuple(f"p{i}" for i in range(1, 9))
    m_cols: Tuple[str, ...] = tuple(f"m{i}" for i in range(1, 9))

    static_times: Tuple[int, ...] = (0, 5, 10)
    dynamic_times: Tuple[int, ...] = (10, 15, 20)

    phy_static_minutes: Tuple[int, ...] = tuple(range(1, 11))
    phy_dynamic_minutes: Tuple[int, ...] = tuple(range(11, 21))

    missing_tokens: Tuple[str, ...] = ("-", "—", "–", "", "#n/a", "#na", "n/a", "na")

    # "35열 및 행 이후" 삽입 위치 (1-index 의미 -> 내부는 0-index slicing)
    insert_after_n: int = 35


CFG = Config()


# =========================
# BI sheet0
# =========================
@dataclass
class Sheet0Result:
    s0_raw_lower: pd.DataFrame
    s0_raw_full: pd.DataFrame
    s0_raw_order: List[str]

    s0_delta_lower: pd.DataFrame
    s0_delta_table: pd.DataFrame

    personality_cols: List[str]
    demo_order: List[str]
    age_col: Optional[str]
    bmi_col: Optional[str]


def run_sheet0(df0_raw: pd.DataFrame, cfg: Config) -> Sheet0Result:
    df = norm_columns(df0_raw)

    id_col = cfg.id_col.lower()
    env_col = cfg.env_col.lower()
    time_col = cfg.time_col.lower()

    for needed in (id_col, env_col, time_col):
        if needed not in df.columns:
            raise ValueError(f"[Sheet0] missing required column: {needed}")

    # demo cols
    sex_col = first_existing(df, cfg.sex_candidates)
    age_col = first_existing(df, cfg.age_candidates)
    bmi_col = first_existing(df, cfg.bmi_candidates)

    demo_order: List[str] = []
    if sex_col:
        df["sex_bin"] = encode_sex(df[sex_col])
        demo_order.append("sex_bin")
    if age_col:
        demo_order.append(age_col)
    if bmi_col:
        demo_order.append(bmi_col)

    # personality
    mbti = keep_existing(df, cfg.mbti_cols)
    ffm = keep_existing(df, cfg.ffm_cols)
    personality = mbti + ffm
    if not personality:
        raise ValueError("[Sheet0] No MBTI/FFM columns found")

    # responses
    resp_cols = keep_existing(df, cfg.resp_cols)
    p_cols = keep_existing(df, cfg.p_cols)
    m_cols = keep_existing(df, cfg.m_cols)

    # derived scenario
    df["set"] = df[env_col].map(env_to_set)
    df["dir"] = df["set"].map(set_to_direction)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    # RAW corr order: personality -> demo -> rest
    raw_order = personality + demo_order + resp_cols + p_cols + m_cols
    raw_num = to_numeric_df(df, raw_order, cfg.missing_tokens)
    raw_full = reorder_square(raw_num.corr(method="spearman"), raw_order)
    raw_lower = lower_triangle_only(raw_full, keep_diag=False)

    # Δ table
    dff = df.dropna(subset=["set", "dir", time_col]).copy()
    dff[time_col] = pd.to_numeric(dff[time_col], errors="coerce")
    dff = dff.dropna(subset=[time_col])
    dff[time_col] = dff[time_col].astype(int)

    dff["phase"] = make_phase(dff[time_col], cfg.static_times, cfg.dynamic_times)
    dff = dff.dropna(subset=["phase"])

    agg_cols = resp_cols + p_cols + m_cols
    dff_num = dff.copy()
    dff_num[agg_cols] = to_numeric_df(dff_num, agg_cols, cfg.missing_tokens)

    phase_means = (
        dff_num.groupby([id_col, "set", "dir", "phase"], as_index=False)[agg_cols]
        .mean()
    )

    wide = phase_means.pivot(index=[id_col, "set", "dir"], columns="phase", values=agg_cols)
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    wide = wide.reset_index()

    # Δ(dynamic - static)
    for v in agg_cols:
        s = f"{v}_static"
        d = f"{v}_dynamic"
        if s in wide.columns and d in wide.columns:
            wide[f"Δ{v}"] = wide[d] - wide[s]

    # Δgap
    for i in range(1, 9):
        ms = f"m{i}_static"
        md = f"m{i}_dynamic"
        ps = f"p{i}_static"
        pd_ = f"p{i}_dynamic"
        if all(c in wide.columns for c in (ms, md, ps, pd_)):
            wide[f"Δgap{i}"] = (wide[md] - wide[pd_]) - (wide[ms] - wide[ps])

    # merge personality + demo into delta_tbl
    personality_unique = df[[id_col] + personality].drop_duplicates(subset=[id_col])
    demo_unique = df[[id_col] + demo_order].drop_duplicates(subset=[id_col])

    delta_tbl = (
        wide.merge(personality_unique, on=id_col, how="left")
        .merge(demo_unique, on=id_col, how="left")
    )

    delta_cols = [c for c in delta_tbl.columns if c.startswith("Δ")]

    # Δ corr order: personality -> demo -> dir -> Δ...
    delta_order = personality + demo_order + ["dir"] + delta_cols
    corr_df = delta_tbl[delta_order].copy()
    corr_df = corr_df.replace(list(cfg.missing_tokens), np.nan).infer_objects(copy=False)
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

    delta_full = reorder_square(corr_df.corr(method="spearman"), delta_order)
    delta_lower = lower_triangle_only(delta_full, keep_diag=False)

    return Sheet0Result(
        s0_raw_lower=raw_lower,
        s0_raw_full=raw_full,
        s0_raw_order=raw_order,
        s0_delta_lower=delta_lower,
        s0_delta_table=delta_tbl,
        personality_cols=personality,
        demo_order=demo_order,
        age_col=age_col,
        bmi_col=bmi_col,
    )


# =========================
# BI sheet2 targets
# =========================
def load_sheet2_targets_df(cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_excel(cfg.path_bi, sheet_name=2, header=1)
    df = norm_columns(df)

    id_col = cfg.id_col.lower()
    if id_col not in df.columns:
        raise ValueError("[Sheet2] Missing '연번'")

    eup_cols = [c for c in df.columns if c.startswith("eup")]
    sp_cols = [c for c in ["sp_c", "sp_h", "dsp"] if c in df.columns]
    targets = eup_cols + sp_cols
    if not targets:
        raise ValueError("[Sheet2] No eup/sp/dsp columns found")

    out = df[[id_col] + targets].copy()
    out[targets] = to_numeric_df(out, targets, cfg.missing_tokens)
    return out, targets


def compute_s0_raw_corr_lower_extended_from_merged(
    df0_raw: pd.DataFrame,
    s0_raw_order: List[str],
    cfg: Config,
) -> pd.DataFrame:
    """
    ✅ S0 원자료(반복측정 행) + S2 설문(개인 1행)을 연번으로 merge 후 Spearman 계산
    ✅ S0 변수 ↔ EUP/SP/DSP 교차블록이 실제 값으로 채워짐
    ✅ survey targets는 35번째 변수 뒤에 삽입
    """
    id_col = cfg.id_col.lower()
    df0 = norm_columns(df0_raw)

    # sex_bin이 order에 있으면 여기서 생성(없으면 NaN으로라도 생성)
    if "sex_bin" in s0_raw_order and "sex_bin" not in df0.columns:
        sex_col = first_existing(df0, cfg.sex_candidates)
        if sex_col is not None:
            df0["sex_bin"] = encode_sex(df0[sex_col])
        else:
            df0["sex_bin"] = np.nan

    # 실제 존재하는 컬럼만 사용
    use_s0_cols = [c for c in s0_raw_order if c in df0.columns]

    # S0 numeric table + merge key
    s0_num = to_numeric_df(df0, use_s0_cols, cfg.missing_tokens)
    s0_num[id_col] = df0[id_col].values

    # S2 targets
    s2_df, targets = load_sheet2_targets_df(cfg)

    merged = s0_num.merge(s2_df, on=id_col, how="left")

    all_cols = use_s0_cols + targets
    corr = merged[all_cols].corr(method="spearman")

    pos = min(cfg.insert_after_n, len(use_s0_cols))
    new_order = use_s0_cols[:pos] + targets + use_s0_cols[pos:]

    corr = reorder_square(corr, new_order)
    return lower_triangle_only(corr, keep_diag=False)


# =========================
# PHY (ST only) + Δmeans
# =========================
def phy_norm_set(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.replace({"HG": "GH"})


def load_phy_st(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.path_phy, sheet_name="ST")
    df = norm_columns(df)
    if "subject" not in df.columns or "condition" not in df.columns:
        raise ValueError("[PHY:ST] expected columns 'subject' and 'condition'")
    df = df.rename(columns={"subject": cfg.id_col.lower(), "condition": "set"})
    df["set"] = phy_norm_set(df["set"])
    return df


def extract_minute_cols(df: pd.DataFrame) -> List[str]:
    # e.g. hr_1, rmssd_10 ...
    return select_cols_regex(df, r"^[a-z0-9]+_\d+$")


def compute_phy_delta_means(
    df: pd.DataFrame,
    minute_cols: List[str],
    prefix: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    minute_cols가 feature_minute 형태일 때 Δ = mean(11~20)-mean(1~10) feature별 산출
    """
    id_col = cfg.id_col.lower()

    fmap: Dict[str, Dict[int, str]] = {}
    for c in minute_cols:
        m = re.match(r"^([a-z0-9]+)_(\d+)$", c)
        if not m:
            continue
        feat = m.group(1)
        minute = int(m.group(2))
        fmap.setdefault(feat, {})[minute] = c

    out = df[[id_col, "set"]].copy()
    out_cols: List[str] = []

    for feat, mm in fmap.items():
        s_cols = [mm[m] for m in cfg.phy_static_minutes if m in mm]
        d_cols = [mm[m] for m in cfg.phy_dynamic_minutes if m in mm]
        if not s_cols or not d_cols:
            continue

        tmp = df.copy()
        tmp[s_cols + d_cols] = to_numeric_df(tmp, s_cols + d_cols, cfg.missing_tokens)

        smean = tmp[s_cols].mean(axis=1)
        dmean = tmp[d_cols].mean(axis=1)

        out[f"{prefix}_Δ{feat}"] = dmean - smean
        out_cols.append(f"{prefix}_Δ{feat}")

    return out, out_cols


def spearman_predictors_vs_features(df: pd.DataFrame, predictors: List[str], features: List[str], cfg: Config) -> pd.DataFrame:
    predictors = [c for c in predictors if c in df.columns]
    features = [c for c in features if c in df.columns]
    num = to_numeric_df(df, predictors + features, cfg.missing_tokens)
    corr = num.corr(method="spearman")
    return corr.loc[predictors, features]


def build_base_for_phy(df0_raw: pd.DataFrame, s0: Sheet0Result, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """
    base: (id) + predictors(성격 + sex_bin/age/bmi)
    """
    df0 = norm_columns(df0_raw)
    id_col = cfg.id_col.lower()

    sex_col = first_existing(df0, cfg.sex_candidates)
    tmp = df0.copy()

    demo_order: List[str] = []
    if sex_col:
        tmp["sex_bin"] = encode_sex(tmp[sex_col])
        demo_order.append("sex_bin")
    if s0.age_col:
        demo_order.append(s0.age_col)
    if s0.bmi_col:
        demo_order.append(s0.bmi_col)

    predictors = s0.personality_cols + demo_order
    predictors = [c for c in predictors if c in tmp.columns]

    base = tmp[[id_col] + predictors].drop_duplicates(subset=[id_col]).copy()
    return base, predictors


def compute_s0_delta_plus_st_delta(
    s0_delta_tbl: pd.DataFrame,
    st_df: pd.DataFrame,
    st_minute_cols: List[str],
    base_predictors: List[str],
    cfg: Config,
) -> pd.DataFrame:
    """
    Output: S0Δ_plus_STΔ_corr_lower only.
    """
    id_col = cfg.id_col.lower()

    st_delta_tbl, st_delta_cols = compute_phy_delta_means(st_df, st_minute_cols, "st", cfg)
    st_delta_tbl = norm_columns(st_delta_tbl)

    merged = norm_columns(s0_delta_tbl).merge(st_delta_tbl, on=[id_col, "set"], how="left")

    delta_cols = [c for c in merged.columns if c.startswith("Δ")]
    phys_cols = [c for c in merged.columns if c.startswith("st_Δ")]

    use_cols = [c for c in base_predictors if c in merged.columns] + delta_cols + phys_cols

    corr_df = merged[use_cols].copy()
    corr_df = corr_df.replace(list(cfg.missing_tokens), np.nan).infer_objects(copy=False)
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

    corr = reorder_square(corr_df.corr(method="spearman"), use_cols)
    return lower_triangle_only(corr, keep_diag=False)


# =========================
# MAIN
# =========================
def main(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, cfg.out_name)

    outputs: Dict[str, pd.DataFrame] = {}

    df0_raw = pd.read_excel(cfg.path_bi, sheet_name=0)
    s0 = run_sheet0(df0_raw, cfg)

    # S0 RAW (extended with sheet2 cross-block real values)
    outputs["S0_RAW_corr_lower"] = compute_s0_raw_corr_lower_extended_from_merged(
        df0_raw=df0_raw,
        s0_raw_order=s0.s0_raw_order,
        cfg=cfg,
    )

    outputs["S0_DELTA_corr_lower"] = s0.s0_delta_lower
    outputs["S0_DELTA_table"] = s0.s0_delta_table

    # PHY ST
    base_for_phy, predictors = build_base_for_phy(df0_raw, s0, cfg)

    st = load_phy_st(cfg)
    st_minute_cols = extract_minute_cols(st)
    st_m = st.merge(base_for_phy, on=cfg.id_col.lower(), how="left")

    if st_minute_cols:
        outputs["PHY_ST_pred_vs_minute"] = spearman_predictors_vs_features(st_m, predictors, st_minute_cols, cfg)

        st_delta_tbl, st_delta_cols = compute_phy_delta_means(st_m, st_minute_cols, "st", cfg)
        if st_delta_cols:
            outputs["PHY_ST_pred_vs_Δ"] = spearman_predictors_vs_features(
                st_delta_tbl.merge(base_for_phy, on=cfg.id_col.lower(), how="left"),
                predictors,
                st_delta_cols,
                cfg,
            )

    # Keep only S0Δ_plus_STΔ_corr_lower as requested
    base_predictors = s0.personality_cols + s0.demo_order + ["dir"]
    if st_minute_cols:
        outputs["S0Δ_plus_STΔ_corr_lower"] = compute_s0_delta_plus_st_delta(
            s0_delta_tbl=s0.s0_delta_table,
            st_df=st_m,
            st_minute_cols=st_minute_cols,
            base_predictors=base_predictors,
            cfg=cfg,
        )

    write_excel(out_path, outputs, index=True)

    print("완료.")
    print("저장 파일:", out_path)
    print("생성 시트:", " / ".join(outputs.keys()))


if __name__ == "__main__":
    main(CFG)