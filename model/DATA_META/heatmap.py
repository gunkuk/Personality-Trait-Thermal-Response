import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\MBTI.xlsx"
SHEET0 = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "META_METHOD_HEATMAPS")
os.makedirs(OUT_DIR, exist_ok=True)

# Column names (raw file may have Korean headers; we normalize to lower)
ID_COL = "연번"
ENV_COL = "환경"
TIME_COL = "시간"

# MBTI / FFM (already normalized to lower in your pipeline)
MBTI_COLS = ["e", "n", "t", "j", "a"]
FFM_COLS  = ["o1", "c1", "e1", "a1", "n1"]

# Responses / behavior
RESP_COLS = ["tsv", "tcv", "ta", "tp", "pt"]
P_COLS = [f"p{i}" for i in range(1, 9)]
M_COLS = [f"m{i}" for i in range(1, 9)]

STATIC_TIMES  = [0, 5, 10]
DYNAMIC_TIMES = [10, 15, 20]  # kept for phase labeling if you still need it elsewhere

MISSING_TOKENS = ["-", "—", "–", "", "#n/a", "#na", "n/a", "na"]

# Heatmap display threshold: values below this will be masked (white)
MIN_SCORE_TO_COLOR = 2.0

# =========================
# Helpers
# =========================
def norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df

def keep_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace(MISSING_TOKENS, np.nan), errors="coerce")

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

def set_to_dir01(set_code: str):
    """cooling=0, heating=1 (binary for interpretability)"""
    s = str(set_code).strip().upper()
    if s in ["AB", "GH"]:
        return 0
    if s in ["CD", "EF"]:
        return 1
    return np.nan

def safe_phase(time_series: pd.Series) -> pd.Series:
    """Avoid numpy dtype promotion errors by returning object dtype with pd.NA."""
    out = pd.Series(pd.NA, index=time_series.index, dtype="object")
    out.loc[time_series.isin(STATIC_TIMES)] = "static"
    out.loc[time_series.isin(DYNAMIC_TIMES)] = "dynamic"
    return out

def compute_slope(x: np.ndarray, y: np.ndarray):
    """Return slope of y ~ x using simple least squares; requires >=2 points."""
    ok = np.isfinite(x) & np.isfinite(y)
    x2 = x[ok]
    y2 = y[ok]
    if len(x2) < 2:
        return np.nan
    # slope = cov(x,y)/var(x)
    vx = np.var(x2, ddof=0)
    if vx == 0:
        return np.nan
    return np.cov(x2, y2, ddof=0)[0, 1] / vx

# =========================
# X / Y taxonomy (for method suitability mapping)
# =========================
X_FORMS = [
    ("X1_continuous", "Axis scores (continuous)"),
    ("X2_binary_cut50", "Axis scores (>=50 vs <50)"),
    ("X3_type16", "16-type from e/n/t/j cut50"),
    ("X4_interactions", "Axis×Axis / Axis×dir interactions"),
]

Y_FORMS = [
    ("Y1_raw_level", "Raw repeated measures (level)"),
    ("Y2_reactivity_slope", "Reactivity = slope over time (0/5/10/15/20)"),
    ("Y2_social_modulation", "Reactivity = M-P (social modulation)"),
    ("Y3_mismatch_multi", "Mismatch scores (PT-TSV, comfort-behavior gaps, Δgaps)"),
    ("Y4_pattern_multilevel_pca", "Patterns via multi-level PCA"),
    ("Y5_binary_multi", "Binary events (apathy, social inhibition, etc.)"),
]

# Expanded Z (methods)
Z_MODELS = [
    ("Z1_spearman", "Spearman correlation"),
    ("Z2_partial_corr", "Partial correlation (confound-adjusted)"),
    ("Z3_lmm", "Linear mixed model (continuous Y)"),
    ("Z4_gamm", "GAMM (nonlinear + mixed)"),
    ("Z5_gee", "GEE (marginal)"),
    ("Z6_glmm_logit", "GLMM logistic (binary Y)"),
    ("Z7_firth_rare_logit", "Rare-event logistic / Firth"),
    ("Z8_pca_fa", "PCA / Factor analysis"),
    ("Z9_clustering", "Clustering (GMM/kmeans/hierarchical)"),
    ("Z10_supervised_ml", "Supervised ML (RF/XGB/SVR etc.)"),
    ("Z11_multivariate_anova", "MANOVA / multivariate tests"),
    ("Z12_permutation_test", "Permutation / randomization tests"),
]

# =========================
# Diagnostics derived from sheet0 (used for suitability penalties)
# =========================
def build_diagnostics_sheet0():
    df = pd.read_excel(PATH, sheet_name=SHEET0)
    df = norm_columns(df)

    id_col = ID_COL.strip().lower()
    env_col = ENV_COL.strip().lower()
    time_col = TIME_COL.strip().lower()

    for req in [id_col, env_col, time_col]:
        if req not in df.columns:
            raise ValueError(f"Missing required column on sheet0: {req}")

    mbti = keep_existing(df, MBTI_COLS)
    ffm  = keep_existing(df, FFM_COLS)
    personality = mbti + ffm

    resp = keep_existing(df, RESP_COLS)
    pcols = keep_existing(df, P_COLS)
    mcols = keep_existing(df, M_COLS)

    # derive set/dir/time numeric
    df["set"] = df[env_col].map(env_to_set)
    df["dir01"] = df["set"].map(set_to_dir01)
    df[time_col] = to_numeric(df[time_col])

    # type16 availability + rough group sizes (if possible)
    # We assume e/n/t/j are continuous in 0~100 range; cut50
    type16_min_group = np.nan
    if all(c in df.columns for c in ["e", "n", "t", "j"]):
        tmp = df[[id_col, "e", "n", "t", "j"]].drop_duplicates(subset=[id_col]).copy()
        for c in ["e", "n", "t", "j"]:
            tmp[c] = to_numeric(tmp[c])
        tmp = tmp.dropna()
        if len(tmp) > 0:
            def b(v): return "1" if v >= 50 else "0"
            tmp["type16"] = tmp.apply(lambda r: b(r["e"])+b(r["n"])+b(r["t"])+b(r["j"]), axis=1)
            counts = tmp["type16"].value_counts()
            type16_min_group = int(counts.min()) if len(counts) else np.nan

    # event rate example: strict apathy proxy (if columns exist)
    # strict apathy: TCV<0 and mean(P,M)<=0
    apathy_rate = np.nan
    if "tcv" in df.columns:
        tcv = to_numeric(df["tcv"])
        Pm = None
        if pcols:
            Pm = to_numeric(df[pcols].mean(axis=1))
        Mm = None
        if mcols:
            Mm = to_numeric(df[mcols].mean(axis=1))
        if Pm is not None and Mm is not None:
            ap = (tcv < 0) & (Pm <= 0) & (Mm <= 0)
            apathy_rate = float(np.nanmean(ap.astype(float))) if len(ap) else np.nan

    # available time points count per subject-set (for slope feasibility)
    slope_feasible_ratio = np.nan
    if resp:
        sub = df.dropna(subset=["set", time_col]).copy()
        # count unique time points per (id,set)
        ct = sub.groupby([id_col, "set"])[time_col].nunique()
        slope_feasible_ratio = float(np.mean(ct >= 2)) if len(ct) else np.nan

    return {
        "personality_cols": personality,
        "has_repeated_design": True,
        "has_time_series": True,
        "slope_feasible_ratio": slope_feasible_ratio,
        "type16_min_group": type16_min_group,
        "apathy_rate": apathy_rate,
    }

# =========================
# Suitability scoring (0~4) unified across Z
# =========================
def score_cell(z_key: str, x_key: str, y_key: str, dx: dict) -> float:
    """
    Rule-based suitability score:
      total = design_fit + goal_fit + robustness + interpretability
    each in [0,1], sum in [0,4]
    """
    design = 0.0
    goal = 0.0
    robust = 0.0
    interp = 0.0

    repeated = dx.get("has_repeated_design", False)
    slope_ok = dx.get("slope_feasible_ratio", np.nan)
    min_group = dx.get("type16_min_group", np.nan)
    ap_rate = dx.get("apathy_rate", np.nan)

    # ---- Design fit ----
    if z_key in ["Z3_lmm", "Z4_gamm", "Z5_gee", "Z6_glmm_logit"]:
        design = 1.0 if repeated else 0.4
    elif z_key in ["Z1_spearman", "Z2_partial_corr", "Z11_multivariate_anova"]:
        design = 0.5 if repeated else 0.7
    elif z_key in ["Z7_firth_rare_logit"]:
        design = 0.7 if repeated else 0.7  # depends on implementation; still OK for binary
    elif z_key in ["Z8_pca_fa", "Z9_clustering", "Z10_supervised_ml", "Z12_permutation_test"]:
        design = 0.7  # generally adaptable
    else:
        design = 0.6

    # ---- Goal fit ----
    if y_key == "Y1_raw_level":
        if z_key in ["Z3_lmm", "Z4_gamm", "Z5_gee"]:
            goal = 1.0
        elif z_key in ["Z1_spearman", "Z2_partial_corr"]:
            goal = 0.6
        else:
            goal = 0.7

    elif y_key == "Y2_reactivity_slope":
        if z_key in ["Z3_lmm", "Z4_gamm"]:
            goal = 1.0
        elif z_key in ["Z5_gee"]:
            goal = 0.8
        elif z_key in ["Z1_spearman", "Z2_partial_corr"]:
            goal = 0.5
        else:
            goal = 0.7

        # slope feasibility penalty
        if np.isfinite(slope_ok) and slope_ok < 0.7:
            goal -= 0.2

    elif y_key == "Y2_social_modulation":
        if z_key in ["Z3_lmm", "Z5_gee", "Z10_supervised_ml"]:
            goal = 0.9
        elif z_key in ["Z1_spearman", "Z2_partial_corr"]:
            goal = 0.6
        else:
            goal = 0.7

    elif y_key == "Y3_mismatch_multi":
        if z_key in ["Z3_lmm", "Z5_gee", "Z10_supervised_ml", "Z12_permutation_test"]:
            goal = 0.9
        elif z_key in ["Z1_spearman", "Z2_partial_corr"]:
            goal = 0.7
        else:
            goal = 0.7

    elif y_key == "Y4_pattern_multilevel_pca":
        if z_key in ["Z8_pca_fa", "Z9_clustering"]:
            goal = 1.0
        elif z_key in ["Z10_supervised_ml"]:
            goal = 0.8
        else:
            goal = 0.6

    elif y_key == "Y5_binary_multi":
        if z_key in ["Z6_glmm_logit", "Z5_gee", "Z7_firth_rare_logit"]:
            goal = 1.0
        elif z_key in ["Z12_permutation_test"]:
            goal = 0.8
        else:
            goal = 0.4

    goal = float(np.clip(goal, 0.0, 1.0))

    # ---- Robustness ----
    if z_key in ["Z12_permutation_test"]:
        robust = 1.0
    elif z_key in ["Z5_gee"]:
        robust = 0.9
    elif z_key in ["Z3_lmm", "Z4_gamm"]:
        robust = 0.8
    elif z_key in ["Z1_spearman"]:
        robust = 0.7
    elif z_key in ["Z2_partial_corr", "Z11_multivariate_anova"]:
        robust = 0.6
    elif z_key in ["Z10_supervised_ml"]:
        robust = 0.7
    elif z_key in ["Z9_clustering", "Z8_pca_fa"]:
        robust = 0.6
    elif z_key in ["Z6_glmm_logit"]:
        robust = 0.6
    elif z_key in ["Z7_firth_rare_logit"]:
        robust = 0.9
    else:
        robust = 0.6

    # penalties: rare binary events hurt plain GLMM/GEE a bit, help Firth
    if y_key == "Y5_binary_multi" and np.isfinite(ap_rate):
        if ap_rate < 0.03:
            if z_key in ["Z6_glmm_logit", "Z5_gee"]:
                robust -= 0.2
            if z_key in ["Z7_firth_rare_logit"]:
                robust += 0.1

    # penalties: type16 very small groups hurt category-heavy inference models
    if x_key == "X3_type16" and np.isfinite(min_group):
        if min_group < 5:
            if z_key in ["Z11_multivariate_anova", "Z6_glmm_logit", "Z3_lmm", "Z5_gee"]:
                robust -= 0.2
            if z_key in ["Z12_permutation_test"]:
                robust += 0.05

    robust = float(np.clip(robust, 0.0, 1.0))

    # ---- Interpretability ----
    if z_key in ["Z3_lmm", "Z5_gee", "Z6_glmm_logit", "Z7_firth_rare_logit", "Z11_multivariate_anova"]:
        interp = 0.9
    elif z_key in ["Z1_spearman", "Z2_partial_corr"]:
        interp = 0.7
    elif z_key in ["Z4_gamm"]:
        interp = 0.6
    elif z_key in ["Z8_pca_fa", "Z9_clustering"]:
        interp = 0.6
    elif z_key in ["Z10_supervised_ml"]:
        interp = 0.4
    elif z_key in ["Z12_permutation_test"]:
        interp = 0.6
    else:
        interp = 0.6

    # interactions increase complexity penalty for interpretability
    if x_key == "X4_interactions" and z_key in ["Z10_supervised_ml", "Z4_gamm"]:
        interp -= 0.05

    interp = float(np.clip(interp, 0.0, 1.0))

    return design + goal + robust + interp

# =========================
# Build score tables and plot heatmaps
# =========================
def build_score_tables(dx: dict):
    tables = {}
    for z_key, z_name in Z_MODELS:
        mat = pd.DataFrame(index=[x[0] for x in X_FORMS], columns=[y[0] for y in Y_FORMS], dtype=float)
        for x_key, _ in X_FORMS:
            for y_key, _ in Y_FORMS:
                mat.loc[x_key, y_key] = score_cell(z_key, x_key, y_key, dx)
        tables[z_key] = mat
    return tables

def plot_heatmap(mat: pd.DataFrame, title: str, out_png: str, min_score: float = 2.0):
    # mask: values below min_score -> NaN -> white
    plot_df = mat.copy()
    plot_df = plot_df.where(plot_df >= min_score, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(plot_df.values, aspect="auto")

    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_xticklabels(plot_df.columns, rotation=30, ha="right")
    ax.set_yticklabels(plot_df.index)

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Suitability (masked < {min_score})")

    # annotate cells (show full score even if masked? -> show only if not NaN)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            v = plot_df.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    dx = build_diagnostics_sheet0()

    # Save diagnostics
    diag = pd.DataFrame([dx])
    diag_path = os.path.join(OUT_DIR, "diagnostics_sheet0.xlsx")

    tables = build_score_tables(dx)

    # Save all tables to one xlsx
    out_xlsx = os.path.join(OUT_DIR, "meta_method_heatmaps_tables.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        diag.to_excel(w, sheet_name="diagnostics", index=False)
        # also store labels for readability
        pd.DataFrame(X_FORMS, columns=["X_key", "X_desc"]).to_excel(w, sheet_name="X_forms", index=False)
        pd.DataFrame(Y_FORMS, columns=["Y_key", "Y_desc"]).to_excel(w, sheet_name="Y_forms", index=False)
        pd.DataFrame(Z_MODELS, columns=["Z_key", "Z_desc"]).to_excel(w, sheet_name="Z_models", index=False)

        for z_key, mat in tables.items():
            mat.to_excel(w, sheet_name=z_key[:31])  # excel sheet name <=31 chars

    # Plot per-Z heatmap
    for z_key, z_name in Z_MODELS:
        mat = tables[z_key]
        png = os.path.join(OUT_DIR, f"{z_key}_suitability_heatmap.png")
        plot_heatmap(mat, f"{z_key}: {z_name}", png, min_score=MIN_SCORE_TO_COLOR)

    print("DONE.")
    print("Output dir:", OUT_DIR)
    print("Tables:", out_xlsx)
    print("PNGs: one per Z model")

if __name__ == "__main__":
    main()