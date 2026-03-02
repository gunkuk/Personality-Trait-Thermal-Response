from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ✅ UTIL import (dir/phase 관련 유틸 제거)
from utils import (
    norm_columns, first_existing, keep_existing,
    to_numeric_df, to_numeric_series,
    encode_sex, mbti_to_bin,
    env_to_set,  # set만 유지
    zscore_inplace, write_excel
)


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    path_bi: str = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\data\BI+SUR.xlsx"
    out_dir: str = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT"
    out_name: str = "lmm_all_x_cont+bin_y.xlsx"

    sheet0: int = 0
    sheet2: int = 2
    sheet2_header: int = 1

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

    missing_tokens: Tuple[str, ...] = ("-", "—", "–", "", "#n/a", "#na", "n/a", "na")

    mbti_bin_threshold: float = 50.0

    include_survey_as_covariate_X: bool = True
    min_nonnull: int = 30

    zscore_continuous_X: bool = True
    zscore_y: bool = False

    fit_reml: bool = False
    maxiter: int = 300


CFG = Config()


# =========================
# Load datasets
# =========================
@dataclass
class Sheet0Bundle:
    df0: pd.DataFrame
    id_col: str
    time_col: str

    mbti_cont: List[str]
    mbti_bin: List[str]
    ffm_cont: List[str]
    demo_cols: List[str]

    survey_targets: List[str]

    y_raw: List[str]
    y_pattern: List[str]


def load_sheet0_and_build_XY(cfg: Config) -> Sheet0Bundle:
    df0 = pd.read_excel(cfg.path_bi, sheet_name=cfg.sheet0)
    df0 = norm_columns(df0)

    id_col = cfg.id_col.lower()
    env_col = cfg.env_col.lower()
    time_col = cfg.time_col.lower()

    for needed in (id_col, env_col, time_col):
        if needed not in df0.columns:
            raise ValueError(f"[Sheet0] missing required column: {needed}")

    # demo
    sex_col = first_existing(df0, cfg.sex_candidates)
    age_col = first_existing(df0, cfg.age_candidates)
    bmi_col = first_existing(df0, cfg.bmi_candidates)

    demo_cols: List[str] = []
    if sex_col is not None:
        df0["sex_bin"] = encode_sex(df0[sex_col])
        demo_cols.append("sex_bin")
    if age_col is not None:
        demo_cols.append(age_col)
    if bmi_col is not None:
        demo_cols.append(bmi_col)

    # personality
    mbti_cont = keep_existing(df0, cfg.mbti_cols)
    ffm_cont = keep_existing(df0, cfg.ffm_cols)
    if not (mbti_cont or ffm_cont):
        raise ValueError("[Sheet0] No MBTI/FFM columns found")

    # numeric + MBTI bin
    for c in mbti_cont + ffm_cont + demo_cols:
        df0[c] = to_numeric_series(df0[c], cfg.missing_tokens)

    mbti_bin: List[str] = []
    for c in mbti_cont:
        bc = f"{c}_bin"
        df0[bc] = mbti_to_bin(df0[c], threshold=cfg.mbti_bin_threshold)
        mbti_bin.append(bc)

    # derived scenario (set/time만 유지)
    df0["set"] = df0[env_col].map(env_to_set)
    df0[time_col] = pd.to_numeric(df0[time_col], errors="coerce")

    # Y groups
    y_raw = keep_existing(df0, cfg.resp_cols)
    y_pattern = keep_existing(df0, cfg.p_cols) + keep_existing(df0, cfg.m_cols)

    return Sheet0Bundle(
        df0=df0,
        id_col=id_col,
        time_col=time_col,
        mbti_cont=mbti_cont,
        mbti_bin=mbti_bin,
        ffm_cont=ffm_cont,
        demo_cols=demo_cols,
        survey_targets=[],
        y_raw=y_raw,
        y_pattern=y_pattern,
    )


def load_sheet2_targets(cfg: Config) -> tuple[pd.DataFrame, List[str]]:
    df2 = pd.read_excel(cfg.path_bi, sheet_name=cfg.sheet2, header=cfg.sheet2_header)
    df2 = norm_columns(df2)

    id_col = cfg.id_col.lower()
    if id_col not in df2.columns:
        raise ValueError("[Sheet2] Missing '연번'")

    eup_cols = [c for c in df2.columns if c.startswith("eup")]
    sp_cols = [c for c in ["sp_c", "sp_h", "dsp"] if c in df2.columns]
    targets = eup_cols + sp_cols
    if not targets:
        raise ValueError("[Sheet2] No eup/sp/dsp columns found")

    df2[targets] = to_numeric_df(df2, targets, cfg.missing_tokens)
    return df2[[id_col] + targets].copy(), targets


# =========================
# Mismatch (Δgap) builder (dir/phase 제거 버전)
# =========================
def build_mismatch_delta_gap(
    df0: pd.DataFrame, bundle: Sheet0Bundle, cfg: Config
) -> tuple[pd.DataFrame, List[str]]:
    """
    mismatch(Y): Δgap1~Δgap8 (연번×set 1행)
    정의:
      gap_i(t) = M_i(t) - P_i(t)
      Δgap_i = mean_t∈dynamic gap_i(t) - mean_t∈static gap_i(t)

    - dynamic/static 분할은 time 값으로 직접 판정:
        static: time in {0,5,10}
        dynamic: time in {10,15,20}
      (10이 둘 다에 들어가 있는 건 원 코드 그대로 유지)
    """
    id_col = bundle.id_col
    time_col = bundle.time_col

    p_cols = [c for c in df0.columns if re.fullmatch(r"p[1-8]", c)]
    m_cols = [c for c in df0.columns if re.fullmatch(r"m[1-8]", c)]
    if not p_cols or not m_cols:
        return pd.DataFrame(), []

    d = df0.dropna(subset=[id_col, "set", time_col]).copy()
    d[p_cols + m_cols] = to_numeric_df(d, p_cols + m_cols, cfg.missing_tokens)

    # time -> phase (dir/phase 컬럼을 만들지 않고 즉석에서)
    static_mask = d[time_col].isin([0, 5, 10])
    dynamic_mask = d[time_col].isin([10, 15, 20])

    d = d.loc[static_mask | dynamic_mask].copy()
    d["phase"] = np.where(static_mask.loc[d.index], "static", "dynamic")

    # (연번, set, phase) 단위로 P/M 평균
    pm = d.groupby([id_col, "set", "phase"], as_index=False)[p_cols + m_cols].mean()

    wide = pm.pivot(index=[id_col, "set"], columns="phase", values=p_cols + m_cols)
    wide.columns = [f"{v}_{ph}" for v, ph in wide.columns]
    wide = wide.reset_index()

    for i in range(1, 9):
        ms, md = f"m{i}_static", f"m{i}_dynamic"
        ps, pd_ = f"p{i}_static", f"p{i}_dynamic"
        if all(c in wide.columns for c in (ms, md, ps, pd_)):
            wide[f"Δgap{i}"] = (wide[md] - wide[pd_]) - (wide[ms] - wide[ps])

    gap_cols = [c for c in wide.columns if c.startswith("Δgap")]
    keep = [id_col, "set"] + gap_cols
    return wide[keep].copy(), gap_cols


# =========================
# Modeling
# =========================
def build_formula(y: str, x_personality: List[str], x_common: List[str]) -> str:
    """
    MixedLM (dir/phase 제거 버전):
      y ~ personality + demo + (optional survey) + time + C(set)
    """
    terms: List[str] = []
    terms += x_personality

    for c in x_common:
        if c == "set":
            continue
        terms.append(c)

    rhs = " + ".join(terms) if terms else "1"
    if "set" in x_common:
        rhs += " + C(set)"

    return f"{y} ~ {rhs}"


def fit_mixedlm_one(df: pd.DataFrame, y: str, formula: str, group_col: str, cfg: Config) -> pd.DataFrame:
    dd = df.dropna(subset=[y, group_col]).copy().reset_index(drop=True)

    if dd.shape[0] < cfg.min_nonnull:
        return pd.DataFrame({"y": [y], "error": [f"nobs<{cfg.min_nonnull}"], "formula": [formula]})

    try:
        model = smf.mixedlm(formula, dd, groups=dd[group_col], re_formula="1", missing="drop")
        res = model.fit(reml=cfg.fit_reml, method="lbfgs", maxiter=cfg.maxiter, disp=False)
    except Exception as e:
        return pd.DataFrame({"y": [y], "error": [str(e)], "formula": [formula]})

    params = res.params
    bse = res.bse
    pvals = res.pvalues

    return pd.DataFrame({
        "y": y,
        "term": params.index.astype(str),
        "coef": params.values,
        "se": bse.values,
        "p": pvals.values,
        "nobs": int(res.nobs),
        "ngroups": int(len(res.model.group_labels)),
        "converged": getattr(res, "converged", np.nan),
        "llf": getattr(res, "llf", np.nan),
        "aic": getattr(res, "aic", np.nan),
        "bic": getattr(res, "bic", np.nan),
        "formula": formula,
    })


def fit_ols_subject_level(subject_df: pd.DataFrame, y_cols: List[str], x_cols: List[str], cfg: Config) -> pd.DataFrame:
    rows = []
    for y in y_cols:
        cols = [y] + x_cols
        dd = subject_df[cols].dropna()
        if dd.shape[0] < cfg.min_nonnull:
            continue

        formula = f"{y} ~ " + (" + ".join(x_cols) if x_cols else "1")
        try:
            res = smf.ols(formula, dd).fit()
        except Exception as e:
            rows.append(pd.DataFrame({"y": [y], "error": [str(e)], "formula": [formula]}))
            continue

        params = res.params
        bse = res.bse
        pvals = res.pvalues
        out = pd.DataFrame({
            "y": y,
            "term": params.index.astype(str),
            "coef": params.values,
            "se": bse.values,
            "p": pvals.values,
            "nobs": int(res.nobs),
            "r2": res.rsquared,
            "adj_r2": res.rsquared_adj,
            "formula": formula,
        })
        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =========================
# MAIN
# =========================
def main(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, cfg.out_name)

    bundle = load_sheet0_and_build_XY(cfg)
    df0 = bundle.df0

    df2, survey_targets = load_sheet2_targets(cfg)
    bundle.survey_targets = survey_targets

    if cfg.include_survey_as_covariate_X:
        df0 = df0.merge(df2, on=bundle.id_col, how="left")

    # set 타입
    df0["set"] = df0["set"].astype("category")

    # common covariates (dir/phase 제거)
    x_common: List[str] = []
    x_common += bundle.demo_cols
    if cfg.include_survey_as_covariate_X:
        x_common += survey_targets
    for c in [bundle.time_col, "set"]:
        if c in df0.columns:
            x_common.append(c)

    # personality sets
    x_person_cont = bundle.mbti_cont + bundle.ffm_cont
    x_person_bin = bundle.mbti_bin + bundle.ffm_cont

    # standardization
    if cfg.zscore_continuous_X:
        cont_cols: List[str] = []
        cont_cols += bundle.mbti_cont + bundle.ffm_cont
        cont_cols += [c for c in bundle.demo_cols if c != "sex_bin"]
        if cfg.include_survey_as_covariate_X:
            cont_cols += survey_targets
        cont_cols += [bundle.time_col]
        zscore_inplace(df0, cont_cols)

    if cfg.zscore_y:
        zscore_inplace(df0, bundle.y_raw + bundle.y_pattern)

    results: Dict[str, pd.DataFrame] = {}

    # RAW
    rows = []
    for y in bundle.y_raw:
        f = build_formula(y, x_person_cont, x_common)
        rows.append(fit_mixedlm_one(df0, y, f, bundle.id_col, cfg))
    results["LMM_raw_cont"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    rows = []
    for y in bundle.y_raw:
        f = build_formula(y, x_person_bin, x_common)
        rows.append(fit_mixedlm_one(df0, y, f, bundle.id_col, cfg))
    results["LMM_raw_bin"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # PATTERN
    rows = []
    for y in bundle.y_pattern:
        f = build_formula(y, x_person_cont, x_common)
        rows.append(fit_mixedlm_one(df0, y, f, bundle.id_col, cfg))
    results["LMM_pattern_cont"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    rows = []
    for y in bundle.y_pattern:
        f = build_formula(y, x_person_bin, x_common)
        rows.append(fit_mixedlm_one(df0, y, f, bundle.id_col, cfg))
    results["LMM_pattern_bin"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # MISMATCH (Δgap)  (dir/phase 제거 버전)
    mismatch_tbl, gap_cols = build_mismatch_delta_gap(df0, bundle, cfg)
    if not mismatch_tbl.empty and gap_cols:
        base_cols = [bundle.id_col] + bundle.mbti_cont + bundle.mbti_bin + bundle.ffm_cont + bundle.demo_cols
        if cfg.include_survey_as_covariate_X:
            base_cols += survey_targets
        base_cols = [c for c in base_cols if c in df0.columns]
        base = df0[base_cols].drop_duplicates(subset=[bundle.id_col]).copy()

        mm = mismatch_tbl.merge(base, on=bundle.id_col, how="left")
        mm["set"] = mm["set"].astype("category")

        # mismatch는 subject-level(연번×set) 테이블이라 time은 없음
        x_common_mm: List[str] = []
        x_common_mm += bundle.demo_cols
        if cfg.include_survey_as_covariate_X:
            x_common_mm += survey_targets
        if "set" in mm.columns:
            x_common_mm.append("set")

        if cfg.zscore_continuous_X:
            cont_cols_mm: List[str] = []
            cont_cols_mm += bundle.mbti_cont + bundle.ffm_cont
            cont_cols_mm += [c for c in bundle.demo_cols if c != "sex_bin"]
            if cfg.include_survey_as_covariate_X:
                cont_cols_mm += survey_targets
            zscore_inplace(mm, cont_cols_mm)

        rows = []
        for y in gap_cols:
            f = build_formula(y, x_person_cont, x_common_mm)
            rows.append(fit_mixedlm_one(mm, y, f, bundle.id_col, cfg))
        results["LMM_mismatch_cont"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        rows = []
        for y in gap_cols:
            f = build_formula(y, x_person_bin, x_common_mm)
            rows.append(fit_mixedlm_one(mm, y, f, bundle.id_col, cfg))
        results["LMM_mismatch_bin"] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    else:
        results["LMM_mismatch_cont"] = pd.DataFrame()
        results["LMM_mismatch_bin"] = pd.DataFrame()

    # RESPONSIBILITY (EUP/SP/DSP): subject-level OLS
    base_subject_cols = [bundle.id_col] + bundle.mbti_cont + bundle.mbti_bin + bundle.ffm_cont + bundle.demo_cols
    base_subject_cols = [c for c in base_subject_cols if c in df0.columns]
    subject = df0[base_subject_cols].drop_duplicates(subset=[bundle.id_col]).merge(df2, on=bundle.id_col, how="left")

    if cfg.zscore_continuous_X:
        cont_cols_subj: List[str] = []
        cont_cols_subj += bundle.mbti_cont + bundle.ffm_cont
        cont_cols_subj += [c for c in bundle.demo_cols if c != "sex_bin"]
        zscore_inplace(subject, cont_cols_subj)

    x_resp_cont = [c for c in (bundle.mbti_cont + bundle.ffm_cont + bundle.demo_cols) if c in subject.columns]
    x_resp_bin = [c for c in (bundle.mbti_bin + bundle.ffm_cont + bundle.demo_cols) if c in subject.columns]

    results["OLS_responsibility_cont"] = fit_ols_subject_level(subject, survey_targets, x_resp_cont, cfg)
    results["OLS_responsibility_bin"] = fit_ols_subject_level(subject, survey_targets, x_resp_bin, cfg)

    write_excel(out_path, results, index=False)

    print("완료.")
    print("저장 파일:", out_path)
    print("생성 시트:", " / ".join(results.keys()))


if __name__ == "__main__":
    main(CFG)