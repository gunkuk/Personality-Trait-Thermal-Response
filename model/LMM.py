from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Tuple, List, Dict

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

    # ✅ PCA options
    use_pca_inputs: bool = True
    pca_path: str = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT\PCA_PM+EUP.xlsx"

    # ✅ 몇 개 PC까지 X로 넣을지
    n_pcs_eup: int = 1
    n_pcs_m: int = 1
    n_pcs_p: int = 1
    n_pcs_mp: int = 1


CFG = Config()


# =========================
# PCA helpers (loadings -> scores)
# =========================
def _load_pca_loadings(pca_path: str, sheet: str) -> pd.DataFrame:
    """
    loadings 시트 구조:
      첫 컬럼: 변수명 (예: EUP1, M1 ...)
      이후 컬럼: PC1, PC2, ...
    """
    L = pd.read_excel(pca_path, sheet_name=sheet, index_col=0)
    L.index = L.index.astype(str)
    L.columns = [str(c) for c in L.columns]
    return L


def _pc_scores_from_loadings(
    df: pd.DataFrame,
    cols_in_df: List[str],
    loadings: pd.DataFrame,
    n_pcs: int,
    prefix: str,
) -> tuple[pd.DataFrame, List[str]]:
    """
    df의 cols_in_df를 '현재 df 기준'으로 z-score 한 뒤,
    loadings를 곱해 PC 점수 생성.

    ⚠️ 엄밀히 동일한 PCA 점수 재현을 원하면,
       PCA를 만들 때의 mu/sigma가 필요하지만 파일에 없어서,
       여기서는 현재 데이터 기반 mu/sigma로 표준화함.
    """
    if n_pcs <= 0:
        return df, []

    # case-insensitive 매칭
    df_cols_l = {c.lower(): c for c in cols_in_df}
    L = loadings.copy()
    L.index = [i.lower() for i in L.index]

    common = [i for i in L.index if i in df_cols_l]
    if not common:
        return df, []

    use_cols = [df_cols_l[i] for i in common]
    L_use = L.loc[common, :]

    pcs = [c for c in L_use.columns if str(c).upper().startswith("PC")]
    pcs = pcs[:n_pcs]
    if not pcs:
        return df, []

    L_use = L_use[pcs]

    X = df[use_cols].astype(float)

    # 현재 df 기준 표준화
    mu = X.mean(axis=0, skipna=True)
    sd = X.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
    Z = (X - mu) / sd

    scores = Z.values @ L_use.values  # (n, p) @ (p, k)
    pc_cols = [f"{prefix}_{c.lower()}" for c in pcs]

    out = pd.DataFrame(scores, columns=pc_cols, index=df.index)

    # 원행에 결측이 있으면 해당 행 PC는 NaN
    out.loc[X.isna().any(axis=1), pc_cols] = np.nan

    for c in pc_cols:
        df[c] = out[c]

    return df, pc_cols


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

    if not (eup_cols or sp_cols):
        raise ValueError("[Sheet2] No eup/sp/dsp columns found")

    df2[eup_cols + sp_cols] = to_numeric_df(df2, eup_cols + sp_cols, cfg.missing_tokens)

    targets: List[str] = []

    # ✅ EUP -> PCA 점수로 변환해 targets로 사용
    if cfg.use_pca_inputs and eup_cols:
        L_eup = _load_pca_loadings(cfg.pca_path, "loadings_EUP")
        df2, eup_pc_cols = _pc_scores_from_loadings(
            df2,
            cols_in_df=eup_cols,
            loadings=L_eup,
            n_pcs=cfg.n_pcs_eup,
            prefix="eup",
        )
        targets += eup_pc_cols
    else:
        targets += eup_cols

    # SP/DSP는 그대로
    targets += sp_cols

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

    static_mask = d[time_col].isin([0, 5, 10])
    dynamic_mask = d[time_col].isin([10, 15, 20])

    d = d.loc[static_mask | dynamic_mask].copy()
    d["phase"] = np.where(static_mask.loc[d.index], "static", "dynamic")

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
      y ~ personality + demo + (optional survey) + time + C(set) + (optional PCA covariates)
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

    # ---- Sheet2 (EUP -> PC if enabled) ----
    df2, survey_targets = load_sheet2_targets(cfg)
    bundle.survey_targets = survey_targets

    if cfg.include_survey_as_covariate_X:
        df0 = df0.merge(df2, on=bundle.id_col, how="left")

    # set 타입
    df0["set"] = df0["set"].astype("category")

    # ✅ M/P/(M-P) -> PCA covariates (row-level)
    m_pc_cols: List[str] = []
    p_pc_cols: List[str] = []
    mp_pc_cols: List[str] = []

    if cfg.use_pca_inputs:
        # df0의 m/p 컬럼 찾기 (실제 존재하는 것만)
        m_cols = sorted([c for c in df0.columns if re.fullmatch(r"m\d+", c)])
        p_cols = sorted([c for c in df0.columns if re.fullmatch(r"p\d+", c)])

        # M PC
        if m_cols:
            L_m = _load_pca_loadings(cfg.pca_path, "loadings_M")
            df0, m_pc_cols = _pc_scores_from_loadings(df0, m_cols, L_m, cfg.n_pcs_m, prefix="m")

        # P PC
        if p_cols:
            L_p = _load_pca_loadings(cfg.pca_path, "loadings_P")
            df0, p_pc_cols = _pc_scores_from_loadings(df0, p_cols, L_p, cfg.n_pcs_p, prefix="p")

        # (M-P) PC: gap_i = m_i - p_i 만든 뒤 PCA
        mp_cols = []
        if m_cols and p_cols:
            m_idx = sorted([int(re.findall(r"\d+", c)[0]) for c in m_cols])
            p_idx = sorted([int(re.findall(r"\d+", c)[0]) for c in p_cols])
            common_idx = sorted(set(m_idx) & set(p_idx))

            for i in common_idx:
                mc, pc = f"m{i}", f"p{i}"
                gc = f"mp{i}"  # 임시 gap
                df0[gc] = to_numeric_series(df0[mc], cfg.missing_tokens) - to_numeric_series(df0[pc], cfg.missing_tokens)
                mp_cols.append(gc)

            if mp_cols:
                L_mp = _load_pca_loadings(cfg.pca_path, "loadings_M-P")
                df0, mp_pc_cols = _pc_scores_from_loadings(df0, mp_cols, L_mp, cfg.n_pcs_mp, prefix="mp")

    # ---- common covariates ----
    x_common: List[str] = []
    x_common += bundle.demo_cols

    if cfg.include_survey_as_covariate_X:
        x_common += survey_targets  # (여기엔 eup_pc* + sp_* 가 포함됨)

    if cfg.use_pca_inputs:
        x_common += [c for c in (m_pc_cols + p_pc_cols + mp_pc_cols) if c in df0.columns]

    for c in [bundle.time_col, "set"]:
        if c in df0.columns:
            x_common.append(c)

    # personality sets (원래대로 유지)
    x_person_cont = bundle.mbti_cont + bundle.ffm_cont
    x_person_bin = bundle.mbti_bin + bundle.ffm_cont

    # ---- standardization ----
    if cfg.zscore_continuous_X:
        cont_cols: List[str] = []
        cont_cols += bundle.mbti_cont + bundle.ffm_cont
        cont_cols += [c for c in bundle.demo_cols if c != "sex_bin"]
        if cfg.include_survey_as_covariate_X:
            cont_cols += survey_targets
        cont_cols += [bundle.time_col]

        # ✅ PCA covariates도 표준화 대상에 포함 (선택)
        if cfg.use_pca_inputs:
            cont_cols += [c for c in (m_pc_cols + p_pc_cols + mp_pc_cols) if c in df0.columns]

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

    # MISMATCH (Δgap)
    mismatch_tbl, gap_cols = build_mismatch_delta_gap(df0, bundle, cfg)
    if not mismatch_tbl.empty and gap_cols:
        base_cols = [bundle.id_col] + bundle.mbti_cont + bundle.mbti_bin + bundle.ffm_cont + bundle.demo_cols
        if cfg.include_survey_as_covariate_X:
            base_cols += survey_targets
        base_cols = [c for c in base_cols if c in df0.columns]
        base = df0[base_cols].drop_duplicates(subset=[bundle.id_col]).copy()

        mm = mismatch_tbl.merge(base, on=bundle.id_col, how="left")
        mm["set"] = mm["set"].astype("category")

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