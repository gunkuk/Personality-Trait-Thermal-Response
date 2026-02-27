import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests

# =========================================================
# CONFIG
# =========================================================
PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\MBTI.xlsx"
SHEET0 = 0

ID_COL = "연번"
ENV_COL = "환경"
TIME_COL = "시간"

# Personality (scores)
MBTI_COLS = ["e", "n", "t", "j", "a"]          # MBTI axes scores (0~100)
FFM_COLS  = ["o1", "c1", "e1", "a1", "n1"]     # FFM scores (0~120 scaled? you already normalized)
X_CONT_COLS = MBTI_COLS + FFM_COLS

# Outcomes (raw, repeated)
RESP_COLS = ["tsv", "tcv", "ta", "tp", "pt"]
P_COLS = [f"p{i}" for i in range(1, 9)]
M_COLS = [f"m{i}" for i in range(1, 9)]

STATIC_TIMES  = [0, 5, 10]
DYNAMIC_TIMES = [10, 15, 20]

MISSING_TOKENS = ["-", "—", "–", "", "#n/a", "#na", "n/a", "na", None]

# Output
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "META_METHOD_HEATMAPS")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_XLSX = os.path.join(OUT_DIR, "meta_method_heatmaps_tables.xlsx")

# X forms (columns in heatmap)
X_FORMS = [
    "X1_continuous",      # scores
    "X2_binary_cut50",    # >=50 vs <50 (per axis)
    "X3_type16",          # 16 type from axis cut50
    "X4_interactions",    # pairwise interactions (e.g., n*j, e*dir, etc.)
]

# Y forms (rows in heatmap) + expanded variants
# We'll create multiple sub-variants for Y3/Y4/Y5 but map them into these main Y classes.
Y_FORMS = [
    "Y1_raw_level",
    "Y2_reactivity_delta",
    "Y3_mismatch_multi",
    "Y4_pattern_multilevel_pca",
    "Y5_binary_multi",
]

# Z models (one heatmap per model)
Z_MODELS = [
    "Z1_Spearman",
    "Z2_PartialCorr",
    "Z3_LMM",
    "Z4_GAMM",
    "Z5_GEE",
    "Z6_GLMM_logit",
    "Z7_RareEvent_Logit_Firth",
    "Z8_PCA_FA",
    "Z9_Clustering",
    "Z10_Supervised_ML",
]

# Unified scoring: 0~4 (Design fit + Goal fit + Robustness + Interpretability)
# We'll compute per-cell score using dataset diagnostics + rulebase.

# =========================================================
# UTILS
# =========================================================
def norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df

def to_numeric_inplace(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(MISSING_TOKENS, np.nan)
            df[c] = pd.to_numeric(df[c], errors="coerce")

def env_to_set(x):
    x = str(x).strip().upper()
    if x in ["A","B"]: return "AB"
    if x in ["C","D"]: return "CD"
    if x in ["E","F"]: return "EF"
    if x in ["G","H"]: return "GH"
    return np.nan

def set_to_dir01(s):
    s = str(s).strip().upper()
    if s in ["AB","GH"]: return 0  # cooling
    if s in ["CD","EF"]: return 1  # heating
    return np.nan

def make_phase(df, time_col):
    df["phase"] = pd.NA
    df.loc[df[time_col].isin(STATIC_TIMES), "static"] = 1
    df.loc[df[time_col].isin(DYNAMIC_TIMES), "dynamic"] = 1
    df["phase"] = pd.NA
    df.loc[df[time_col].isin(STATIC_TIMES), "phase"] = "static"
    df.loc[df[time_col].isin(DYNAMIC_TIMES), "phase"] = "dynamic"
    return df

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s - mu
    return (s - mu) / sd

def safe_mean(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None
    return df[cols].mean(axis=1)

def pca_svd(X: np.ndarray, n_pc: int = 3) -> np.ndarray:
    # X assumed no NaN
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :n_pc] * S[:n_pc]

# =========================================================
# LOAD + BASE TABLE
# =========================================================
df = pd.read_excel(PATH, sheet_name=SHEET0)
df = norm_columns(df)

id_col   = ID_COL.lower()
env_col  = ENV_COL.lower()
time_col = TIME_COL.lower()

for c in [id_col, env_col, time_col]:
    if c not in df.columns:
        raise ValueError(f"Missing essential col: {c}")

df["set"] = df[env_col].map(env_to_set)
df["dir01"] = df["set"].map(set_to_dir01)

df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
df = df.dropna(subset=[id_col, "set", "dir01", time_col])
df[time_col] = df[time_col].astype(int)

df = make_phase(df, time_col)
df = df.dropna(subset=["phase"])

# keep existing columns
X_CONT_COLS = [c for c in X_CONT_COLS if c in df.columns]
RESP_COLS   = [c for c in RESP_COLS if c in df.columns]
P_COLS      = [c for c in P_COLS if c in df.columns]
M_COLS      = [c for c in M_COLS if c in df.columns]

Y_RAW_COLS = RESP_COLS + P_COLS + M_COLS
if not X_CONT_COLS:
    raise ValueError("No personality columns found.")
if not Y_RAW_COLS:
    raise ValueError("No outcome columns found (TSV/TCV/P/M etc).")

to_numeric_inplace(df, X_CONT_COLS + Y_RAW_COLS + ["dir01"])

# z-score X continuous
for x in X_CONT_COLS:
    df[x + "_z"] = zscore(df[x])

# =========================================================
# X FORMS: create data structures
# =========================================================
def build_X_forms(df: pd.DataFrame):
    """
    Returns dict of X-form metadata (not modeling yet)
    """
    out = {}

    # X1 continuous: use *_z
    out["X1_continuous"] = {
        "type": "continuous",
        "cols": [x + "_z" for x in X_CONT_COLS],
        "notes": "standardized continuous scores"
    }

    # X2 binary cut50: per axis -> {x}_bin
    df2 = df.copy()
    bin_cols = []
    for x in X_CONT_COLS:
        if x in df2.columns:
            c = x + "_bin50"
            df2[c] = np.where(df2[x] >= 50, 1, 0)  # 0/1
            bin_cols.append(c)

    out["X2_binary_cut50"] = {
        "type": "binary",
        "cols": bin_cols,
        "notes": "axis-wise binary cut at 50"
    }

    # X3 type16: 16-type from 4 axes e/n, n/s implied? you have e,n,t,j scores
    # We'll form MBTI 16 only if e,n,t,j exist.
    df3 = df2.copy()
    if all(ax in df3.columns for ax in ["e", "n", "t", "j"]):
        def letter(ax, hi, lo):
            return hi if ax >= 50 else lo
        df3["type16"] = df3.apply(lambda r: (
            letter(r["e"], "E", "I") +
            letter(r["n"], "N", "S") +
            letter(r["t"], "T", "F") +
            letter(r["j"], "J", "P")
        ), axis=1)
        out["X3_type16"] = {
            "type": "categorical",
            "cols": ["type16"],
            "notes": "16-type categorical from (e,n,t,j) cut50"
        }
    else:
        out["X3_type16"] = {
            "type": "categorical",
            "cols": [],
            "notes": "missing e/n/t/j to build type16"
        }

    # X4 interactions: pairwise products among MBTI axes + dir interaction
    df4 = df3.copy()
    inter_cols = []
    base = [c for c in ["e_z","n_z","t_z","j_z","a_z"] if c in df4.columns]
    for i in range(len(base)):
        for j in range(i+1, len(base)):
            c = f"{base[i]}__x__{base[j]}"
            df4[c] = df4[base[i]] * df4[base[j]]
            inter_cols.append(c)

    # add interactions with dir01
    for b in base:
        c = f"{b}__x__dir"
        df4[c] = df4[b] * df4["dir01"]
        inter_cols.append(c)

    out["X4_interactions"] = {
        "type": "interaction",
        "cols": inter_cols,
        "notes": "pairwise products + dir interactions (standardized)"
    }

    return out, df4

X_META, dfX = build_X_forms(df)

# =========================================================
# Y FORMS: create derived outcomes (expanded)
# =========================================================
def build_delta_table(df: pd.DataFrame, cols):
    g = df[[id_col, "set", "dir01", "phase"] + cols].groupby(
        [id_col, "set", "dir01", "phase"], as_index=False
    )[cols].mean()

    wide = g.pivot(index=[id_col, "set", "dir01"], columns="phase", values=cols)
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    wide = wide.reset_index()

    dcols = []
    for v in cols:
        s = f"{v}_static"
        d = f"{v}_dynamic"
        if s in wide.columns and d in wide.columns:
            wide[f"d_{v}"] = wide[d] - wide[s]
            dcols.append(f"d_{v}")
    return wide, dcols

def build_Y_forms(df: pd.DataFrame):
    """
    Returns:
      Y_META: dict of y-form -> list of sub-variants columns
      DataFrames for per-form modeling convenience (raw = df, delta = wide, pattern tables etc.)
    """
    Y_META = {}

    # --- Y1 raw level ---
    Y_META["Y1_raw_level"] = {
        "variants": {
            "raw_all": Y_RAW_COLS,
            "raw_survey": RESP_COLS,
            "raw_behavior_P": P_COLS,
            "raw_behavior_M": M_COLS,
        }
    }

    # --- Y2 reactivity delta (dynamic - static) ---
    wide, dcols_all = build_delta_table(df, Y_RAW_COLS)
    Y_META["Y2_reactivity_delta"] = {
        "variants": {
            "delta_all": dcols_all,
            "delta_survey": [c for c in dcols_all if c.startswith("d_") and c[2:] in RESP_COLS],
            "delta_P": [c for c in dcols_all if c.startswith("d_") and c[2:] in P_COLS],
            "delta_M": [c for c in dcols_all if c.startswith("d_") and c[2:] in M_COLS],
        }
    }

    # --- Y3 mismatch multi (continuous + engineered linear combos) ---
    # Create row-level and delta-level mismatch features
    drow = df.copy()

    # Means and social gap
    if P_COLS:
        drow["p_mean"] = safe_mean(drow, P_COLS)
    if M_COLS:
        drow["m_mean"] = safe_mean(drow, M_COLS)
    if P_COLS and M_COLS:
        drow["m_minus_p"] = drow["m_mean"] - drow["p_mean"]  # social difference (multi - solo)
        drow["pm_mean"] = (drow["m_mean"] + drow["p_mean"]) / 2.0

    # (1) discomfort * inaction (continuous)
    if "tcv" in drow.columns:
        discomfort = np.clip(-drow["tcv"], 0, None)
        if "pm_mean" in drow.columns:
            inaction = np.clip(-drow["pm_mean"], 0, None)     # 행동이 0 이하일수록 큼
            drow["mis_apathy_score"] = discomfort * inaction
        else:
            drow["mis_apathy_score"] = discomfort

    # (2) discomfort * social_inhibition (TCV<0인데 M-P가 작거나 음수) -> 사회적 억제 가설
    if "tcv" in drow.columns and "m_minus_p" in drow.columns:
        social_inhib = np.clip(-drow["m_minus_p"], 0, None)   # 다수재실에서 오히려 덜 행동
        drow["mis_social_inhib_score"] = np.clip(-drow["tcv"], 0, None) * social_inhib

    # (3) perception mismatch: |TSV - PT| (너 정의에 맞게 PT를 perceived temp accuracy로 사용)
    # 주의: 네 데이터에서 PT가 "perceived temperature"면, TSV와 scale이 다를 수 있음.
    if "tsv" in drow.columns and "pt" in drow.columns:
        drow["mis_temp_perception_abs"] = (drow["tsv"] - drow["pt"]).abs()

    # (4) multi vs solo behavior distance: L1 norm between M[1..8] and P[1..8]
    # 행동 패턴 차이가 큰 사람(다중재실자 개입의사 변화)
    if P_COLS and M_COLS and len(P_COLS) == len(M_COLS):
        diffs = []
        for i in range(1, 9):
            p = f"p{i}"
            m = f"m{i}"
            if p in drow.columns and m in drow.columns:
                diffs.append((drow[m] - drow[p]).abs())
        if diffs:
            drow["mis_MP_L1"] = np.sum(diffs, axis=0)

    # Delta-level mismatch: Δgap_i = (M_i - P_i)_dynamic - (M_i - P_i)_static
    wide2 = wide.copy()
    gap_dcols = []
    for i in range(1, 9):
        ms, md = f"m{i}_static", f"m{i}_dynamic"
        ps, pd = f"p{i}_static", f"p{i}_dynamic"
        if all(c in wide2.columns for c in [ms, md, ps, pd]):
            col = f"d_gap{i}"
            wide2[col] = (wide2[md] - wide2[pd]) - (wide2[ms] - wide2[ps])
            gap_dcols.append(col)

    # Also delta of mean social gap
    if all(c in wide2.columns for c in ["m_mean_static","m_mean_dynamic","p_mean_static","p_mean_dynamic"]):
        wide2["d_m_minus_p"] = (wide2["m_mean_dynamic"] - wide2["p_mean_dynamic"]) - (wide2["m_mean_static"] - wide2["p_mean_static"])
        gap_dcols.append("d_m_minus_p")

    Y_META["Y3_mismatch_multi"] = {
        "variants": {
            "mis_row_scores": [c for c in [
                "mis_apathy_score",
                "mis_social_inhib_score",
                "mis_temp_perception_abs",
                "mis_MP_L1",
                "m_minus_p",
            ] if c in drow.columns],
            "mis_delta_gaps": gap_dcols,
        }
    }

    # --- Y4 pattern multilevel PCA ---
    # 여러 레벨로 PCA:
    #  L1: P only
    #  L2: M only
    #  L3: P+M (behavior full)
    #  L4: survey + behavior
    #  L5: delta behavior (from wide)
    # each returns PC1..PCk columns
    def pca_table_rowlevel(df_in, cols, prefix, n_pc=3):
        cols = [c for c in cols if c in df_in.columns]
        if len(cols) < 3:
            return None, []
        d = df_in[[id_col, "set", "dir01", "phase"] + cols].dropna().copy()
        X = d[cols].to_numpy(dtype=float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-9)
        scores = pca_svd(X, n_pc=n_pc)
        out_cols = []
        for k in range(n_pc):
            c = f"{prefix}_pc{k+1}"
            d[c] = scores[:, k]
            out_cols.append(c)
        keep = [id_col, "set", "dir01", "phase"] + out_cols
        return d[keep], out_cols

    def pca_table_deltalevel(wide_in, cols, prefix, n_pc=3):
        cols = [c for c in cols if c in wide_in.columns]
        if len(cols) < 3:
            return None, []
        d = wide_in[[id_col, "set", "dir01"] + cols].dropna().copy()
        X = d[cols].to_numpy(dtype=float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-9)
        scores = pca_svd(X, n_pc=n_pc)
        out_cols = []
        for k in range(n_pc):
            c = f"{prefix}_pc{k+1}"
            d[c] = scores[:, k]
            out_cols.append(c)
        keep = [id_col, "set", "dir01"] + out_cols
        return d[keep], out_cols

    pca_tables = {}
    pca_cols_map = {}

    t, cols = pca_table_rowlevel(drow, P_COLS, "P", n_pc=3)
    if t is not None:
        pca_tables["P_only"] = t
        pca_cols_map["P_only"] = cols

    t, cols = pca_table_rowlevel(drow, M_COLS, "M", n_pc=3)
    if t is not None:
        pca_tables["M_only"] = t
        pca_cols_map["M_only"] = cols

    t, cols = pca_table_rowlevel(drow, P_COLS + M_COLS, "PM", n_pc=3)
    if t is not None:
        pca_tables["PM"] = t
        pca_cols_map["PM"] = cols

    t, cols = pca_table_rowlevel(drow, RESP_COLS + P_COLS + M_COLS, "SPM", n_pc=3)
    if t is not None:
        pca_tables["Survey+Behavior"] = t
        pca_cols_map["Survey+Behavior"] = cols

    # delta PCA on delta behaviors only (d_p*, d_m*)
    delta_beh = [c for c in wide.columns if c.startswith("d_") and (c[2:] in P_COLS or c[2:] in M_COLS)]
    t, cols = pca_table_deltalevel(wide, delta_beh, "dPM", n_pc=3)
    if t is not None:
        pca_tables["Delta_Behavior"] = t
        pca_cols_map["Delta_Behavior"] = cols

    # delta PCA on delta gaps (social difference change)
    t, cols = pca_table_deltalevel(wide2, gap_dcols, "dGAP", n_pc=min(3, len(gap_dcols)))
    if t is not None:
        pca_tables["Delta_Gaps"] = t
        pca_cols_map["Delta_Gaps"] = cols

    Y_META["Y4_pattern_multilevel_pca"] = {
        "variants": {k: v for k, v in pca_cols_map.items()}
    }

    # --- Y5 binary multi: multiple event definitions (row-level + delta-level) ---
    # strict/relaxed apathy, social inhibition binary, misperception high binary
    bin_df = drow.copy()
    binaries = {}

    # helper: inaction strict means all items <=0
    if "tcv" in bin_df.columns and P_COLS and M_COLS:
        p_inaction_strict = bin_df[P_COLS].le(0).all(axis=1)
        m_inaction_strict = bin_df[M_COLS].le(0).all(axis=1)
        binaries["apathy_strict"] = ((bin_df["tcv"] < 0) & p_inaction_strict & m_inaction_strict).astype(int)

        # relaxed: mean<=0
        if "pm_mean" in bin_df.columns:
            binaries["apathy_relaxed"] = ((bin_df["tcv"] < 0) & (bin_df["pm_mean"] <= 0)).astype(int)

        # social inhibition: discomfort + (M-P <= 0)
        if "m_minus_p" in bin_df.columns:
            binaries["social_inhib"] = ((bin_df["tcv"] < 0) & (bin_df["m_minus_p"] <= 0)).astype(int)

    # misperception high: top 20% of |TSV-PT|
    if "mis_temp_perception_abs" in bin_df.columns:
        thr = bin_df["mis_temp_perception_abs"].quantile(0.80)
        binaries["misperception_top20"] = (bin_df["mis_temp_perception_abs"] >= thr).astype(int)

    # “multi-occupant shift” event: |M-P| L1 is top 20%
    if "mis_MP_L1" in bin_df.columns:
        thr = bin_df["mis_MP_L1"].quantile(0.80)
        binaries["MP_shift_top20"] = (bin_df["mis_MP_L1"] >= thr).astype(int)

    for k, s in binaries.items():
        bin_df[k] = s

    Y_META["Y5_binary_multi"] = {
        "variants": {
            "binary_events": list(binaries.keys())
        }
    }

    return Y_META, drow, wide2, pca_tables, bin_df

Y_META, dfY_row, dfY_delta, PCA_TABLES, dfY_bin = build_Y_forms(df)

# =========================================================
# DIAGNOSTICS for suitability scoring
# =========================================================
def diag_base(df_row: pd.DataFrame):
    n_rows = len(df_row)
    n_subj = df_row[id_col].nunique()
    sets = df_row["set"].nunique()
    phases = df_row["phase"].nunique()
    return {
        "n_rows": n_rows,
        "n_subj": n_subj,
        "n_sets": sets,
        "n_phases": phases
    }

BASE_DIAG = diag_base(df)

def diag_xform(df_any: pd.DataFrame, xform_key: str):
    meta = X_META[xform_key]
    typ = meta["type"]
    cols = meta["cols"]

    if not cols:
        return {"ok": False, "reason": "no columns"}

    d = df_any.copy()
    if typ == "categorical":
        # one column (type16)
        c = cols[0]
        n_levels = d[c].nunique(dropna=True)
        counts = d[c].value_counts(dropna=True)
        min_group = int(counts.min()) if len(counts) else 0
        return {
            "ok": True, "type": typ, "n_levels": int(n_levels),
            "min_group": min_group
        }

    # numeric/binary/interaction
    miss = d[cols].isna().mean().mean()
    var0 = float((d[cols].std(ddof=0) == 0).mean())
    return {
        "ok": True, "type": typ,
        "n_cols": len(cols),
        "missing_rate": float(miss),
        "zero_var_frac": var0
    }

def diag_yform(df_row: pd.DataFrame, yform_key: str):
    """
    Return properties that affect method suitability.
    """
    if yform_key == "Y1_raw_level":
        # continuous-ish, repeated
        return {
            "ok": True, "kind": "continuous_repeated", "rare_event": False
        }

    if yform_key == "Y2_reactivity_delta":
        # delta: subject x set (not per-time repeated)
        return {
            "ok": True, "kind": "continuous_delta", "rare_event": False
        }

    if yform_key == "Y3_mismatch_multi":
        # mixture: continuous engineered + delta gaps
        return {
            "ok": True, "kind": "continuous_engineered", "rare_event": False
        }

    if yform_key == "Y4_pattern_multilevel_pca":
        return {
            "ok": True, "kind": "pattern_latent", "rare_event": False
        }

    if yform_key == "Y5_binary_multi":
        # need event rate check
        # take union rate across available binaries
        ev_cols = Y_META["Y5_binary_multi"]["variants"]["binary_events"]
        if not ev_cols:
            return {"ok": False, "reason": "no binary events"}
        rates = {}
        for c in ev_cols:
            if c in dfY_bin.columns:
                rates[c] = float(dfY_bin[c].mean())
        min_rate = min(rates.values()) if rates else 0.0
        max_rate = max(rates.values()) if rates else 0.0
        return {
            "ok": True, "kind": "binary_repeated", "rare_event": True,
            "min_event_rate": float(min_rate), "max_event_rate": float(max_rate),
        }

    return {"ok": False, "reason": "unknown yform"}

# =========================================================
# RULE-BASED SUITABILITY SCORING (0~4) - unified across Z
# =========================================================
def score_cell(z: str, xform: str, yform: str, dx: dict, dy: dict, base: dict) -> float:
    """
    Score: 0..4
      1) Design fit
      2) Goal fit
      3) Robustness
      4) Interpretability (paper-friendly)
    """
    if not dx.get("ok", False) or not dy.get("ok", False):
        return 0.0

    # base flags
    repeated = (yform in ["Y1_raw_level", "Y3_mismatch_multi", "Y4_pattern_multilevel_pca", "Y5_binary_multi"])
    delta = (yform == "Y2_reactivity_delta")
    rare = dy.get("rare_event", False)
    kind = dy.get("kind", "")

    x_is_cat = (dx.get("type") == "categorical")
    x_is_bin = (dx.get("type") == "binary")
    x_is_cont = (dx.get("type") == "continuous")
    x_is_inter = (dx.get("type") == "interaction")

    # diagnostics
    n_subj = base["n_subj"]
    n_rows = base["n_rows"]

    # if type16 with tiny min_group => penalize models that need stable group sizes
    min_group = dx.get("min_group", None)

    # event rate for binary
    min_ev = dy.get("min_event_rate", None)
    max_ev = dy.get("max_event_rate", None)

    # -------------------------
    # Initialize sub-scores
    # -------------------------
    design = 0.0
    goal = 0.0
    robust = 0.0
    interp = 0.0

    # -------------------------
    # Z-specific rules
    # -------------------------
    if z == "Z1_Spearman":
        # design: weak for repeated structure
        design = 0.3 if repeated else 0.6
        # goal: quick screening for continuous, poor for pattern/binary
        goal = 0.8 if yform in ["Y1_raw_level", "Y2_reactivity_delta", "Y3_mismatch_multi"] else 0.3
        # robust: ok-ish, but confounding ignored
        robust = 0.6
        # interpretability: ok for early, not final
        interp = 0.5
        # categorical X(type16) not natural for spearman
        if x_is_cat:
            goal -= 0.3
        if yform == "Y5_binary_multi":
            goal -= 0.3

    elif z == "Z2_PartialCorr":
        design = 0.5 if repeated else 0.7
        goal = 0.9 if yform in ["Y1_raw_level", "Y2_reactivity_delta", "Y3_mismatch_multi"] else 0.3
        robust = 0.6
        interp = 0.6
        if x_is_cat:
            goal -= 0.4

    elif z == "Z3_LMM":
        # best for continuous repeated/delta, decent for engineered mismatch, ok for PCA PCs
        design = 1.0 if repeated or delta else 0.8
        goal = 1.0 if kind in ["continuous_repeated","continuous_delta","continuous_engineered","pattern_latent"] else 0.2
        robust = 0.8
        interp = 0.9
        if yform == "Y5_binary_multi":
            goal = 0.1  # not for binary

        # Type16 OK but needs enough group sizes
        if x_is_cat and min_group is not None and min_group < 20:
            robust -= 0.3

    elif z == "Z4_GAMM":
        # non-linear ramp/time curves: strong for raw repeated; less for delta
        design = 1.0 if repeated and yform == "Y1_raw_level" else 0.6
        goal = 1.0 if yform == "Y1_raw_level" else 0.5
        robust = 0.7
        interp = 0.7
        if yform == "Y5_binary_multi":
            goal = 0.2

    elif z == "Z5_GEE":
        # marginal robust for repeated; good for binary too
        design = 0.9 if repeated else 0.6
        goal = 0.8 if repeated else 0.5
        robust = 0.9
        interp = 0.7
        # for binary rare: can suffer separation-ish but more stable than GLMM sometimes
        if yform == "Y5_binary_multi":
            goal = 0.9
            if min_ev is not None and min_ev < 0.03:
                robust -= 0.2

    elif z == "Z6_GLMM_logit":
        # for binary with random effects; can be unstable with rare events
        design = 1.0 if yform == "Y5_binary_multi" else 0.4
        goal = 1.0 if yform == "Y5_binary_multi" else 0.2
        robust = 0.6
        interp = 0.8
        if yform == "Y5_binary_multi" and min_ev is not None and min_ev < 0.03:
            robust -= 0.3

    elif z == "Z7_RareEvent_Logit_Firth":
        # best when binary rare; usually needs aggregation or careful handling
        design = 0.8 if yform == "Y5_binary_multi" else 0.3
        goal = 1.0 if yform == "Y5_binary_multi" else 0.2
        robust = 1.0 if yform == "Y5_binary_multi" else 0.4
        interp = 0.8
        if yform == "Y5_binary_multi" and min_ev is not None and min_ev >= 0.05:
            # not rare; still OK but less necessary
            robust -= 0.2

    elif z == "Z8_PCA_FA":
        # pattern discovery best
        design = 0.8 if yform == "Y4_pattern_multilevel_pca" else 0.4
        goal = 1.0 if yform == "Y4_pattern_multilevel_pca" else 0.3
        robust = 0.8
        interp = 0.7

    elif z == "Z9_Clustering":
        design = 0.7 if yform == "Y4_pattern_multilevel_pca" else 0.5
        goal = 1.0 if yform == "Y4_pattern_multilevel_pca" else 0.4
        robust = 0.6
        interp = 0.6
        # type16 as X doesn't drive clustering; more like post-hoc compare
        if x_is_cat:
            goal -= 0.2

    elif z == "Z10_Supervised_ML":
        # prediction-oriented; needs enough N and careful CV; interpretability lower
        design = 0.7
        goal = 0.8 if yform in ["Y1_raw_level","Y2_reactivity_delta","Y4_pattern_multilevel_pca"] else 0.6
        robust = 0.6
        interp = 0.4
        # type16 with many classes needs big N
        if x_is_cat:
            robust -= 0.2

    else:
        return 0.0

    # -------------------------
    # Universal penalties/adjustments
    # -------------------------
    # X interactions increases model complexity -> penalize for small sample
    if x_is_inter and n_subj < 80:
        robust -= 0.2
        interp -= 0.1

    # Type16 requires adequate group size
    if x_is_cat and min_group is not None:
        if min_group < 10:
            robust -= 0.4
            design -= 0.1
        elif min_group < 20:
            robust -= 0.2

    # Binary rare event: prefer rare-event methods
    if yform == "Y5_binary_multi" and min_ev is not None:
        if min_ev < 0.02:
            # extremely rare: Spearman etc basically no
            if z in ["Z1_Spearman","Z2_PartialCorr","Z3_LMM","Z4_GAMM","Z8_PCA_FA","Z9_Clustering"]:
                goal -= 0.5
            # GLMM less stable
            if z == "Z6_GLMM_logit":
                robust -= 0.2

    # clamp
    design = float(np.clip(design, 0, 1))
    goal   = float(np.clip(goal,   0, 1))
    robust = float(np.clip(robust, 0, 1))
    interp = float(np.clip(interp, 0, 1))

    return design + goal + robust + interp

# =========================================================
# BUILD SCORE MATRICES + PLOT
# =========================================================
def plot_score_heatmap(score_df: pd.DataFrame, title: str, out_png: str):
    data = score_df.to_numpy(dtype=float)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=0, vmax=4)

    ax.set_title(title)
    ax.set_yticks(np.arange(score_df.shape[0]))
    ax.set_yticklabels(score_df.index.tolist(), fontsize=10)
    ax.set_xticks(np.arange(score_df.shape[1]))
    ax.set_xticklabels(score_df.columns.tolist(), rotation=25, ha="right", fontsize=10)

    # annotate values
    for i in range(score_df.shape[0]):
        for j in range(score_df.shape[1]):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Method suitability score (0~4)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

def build_all_heatmaps():
    tables = {}

    # diagnostics per X/Y
    x_diags = {x: diag_xform(dfX, x) for x in X_FORMS}
    y_diags = {y: diag_yform(dfX, y) for y in Y_FORMS}

    tables["X_diagnostics"] = pd.DataFrame(x_diags).T
    tables["Y_diagnostics"] = pd.DataFrame(y_diags).T
    tables["BASE"] = pd.DataFrame([BASE_DIAG])

    # Z heatmaps
    for z in Z_MODELS:
        score = pd.DataFrame(index=Y_FORMS, columns=X_FORMS, dtype=float)
        for y in Y_FORMS:
            for x in X_FORMS:
                sc = score_cell(z, x, y, x_diags[x], y_diags[y], BASE_DIAG)
                score.loc[y, x] = sc

        tables[f"{z}_scores"] = score

        out_png = os.path.join(OUT_DIR, f"{z}_suitability_heatmap.png")
        plot_score_heatmap(score, f"{z}: suitability on (X-form × Y-form)", out_png)

    return tables

TABLES = build_all_heatmaps()

# =========================================================
# SAVE META TABLES + also save derived variable lists
# =========================================================
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    for name, obj in TABLES.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_excel(writer, sheet_name=name[:31])

    # Save Y variants inventory
    y_inv_rows = []
    for yk in Y_FORMS:
        variants = Y_META.get(yk, {}).get("variants", {})
        for vk, cols in variants.items():
            if isinstance(cols, list):
                y_inv_rows.append({"Y_form": yk, "variant": vk, "n_cols": len(cols), "cols": ", ".join(cols[:30])})
            elif isinstance(cols, dict):
                # PCA levels map -> list cols
                for kk, cc in cols.items():
                    y_inv_rows.append({"Y_form": yk, "variant": f"{vk}:{kk}", "n_cols": len(cc), "cols": ", ".join(cc)})
            else:
                y_inv_rows.append({"Y_form": yk, "variant": vk, "n_cols": None, "cols": str(cols)})
    pd.DataFrame(y_inv_rows).to_excel(writer, sheet_name="Y_variants_inventory", index=False)

print("DONE.")
print("Output directory:", OUT_DIR)
print("Excel tables:", OUT_XLSX)
print("Generated 1 heatmap per Z model.")