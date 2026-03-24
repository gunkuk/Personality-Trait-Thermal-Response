# 핵심 요약:
# - FFM / MBTI(연속형): global permutation profile test -> Welch ANOVA -> Holm -> Games-Howell
# - MBTI(이분형): global permutation binary-pattern test -> chi-square -> Holm -> pairwise proportion/Fisher
# - clustering에는 성격변수를 쓰지 않고, cluster 결과를 external validator 관점에서 post-hoc 분석한다.

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest

# Games-Howell p-value를 위한 선택적 import
try:
    from statsmodels.stats.libqsturng import psturng
    HAS_QSTURNG = True
except Exception:
    HAS_QSTURNG = False


# =========================================================
# 1. 기본 변수 정의
# =========================================================
FFM_VARS = ["o1", "c1", "e1", "a1", "n1"]
MBTI_CONT_VARS = ["e", "n", "t", "j", "a"]

MBTI_BINARY_RULES = {
    "e": ("E", "I"),
    "n": ("N", "S"),
    "t": ("T", "F"),
    "j": ("J", "P"),
    "a": ("A", "T"),   # Assertive / Turbulent
}


# =========================================================
# 2. 유틸
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cluster_path", type=str, required=True)
    parser.add_argument("--personality_xlsx_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--cluster_id_col", type=str, default="id")
    parser.add_argument("--cluster_label_col", type=str, default="cluster")

    parser.add_argument("--personality_sheet", type=str, default="BI")
    parser.add_argument("--personality_id_col", type=str, default="id")

    parser.add_argument(
        "--analysis_modes",
        nargs="+",
        default=["ffm", "mbti_cont", "mbti_bin"],
        choices=["ffm", "mbti_cont", "mbti_bin"],
        help="실행할 분석 모드"
    )

    parser.add_argument("--n_perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    # MBTI binary threshold
    parser.add_argument("--mbti_hi_threshold", type=float, default=51.0)
    parser.add_argument("--mbti_lo_threshold", type=float, default=49.0)
    parser.add_argument(
        "--mbti_tie_policy",
        type=str,
        default="drop",
        choices=["drop", "first", "second"],
        help="예: 49 < score < 51 또는 score==50일 때 처리 방식"
    )

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def holm_correction(pvals: List[float]) -> List[float]:
    """Holm step-down adjusted p-values."""
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, p in enumerate(ranked):
        val = (m - i) * p
        val = max(val, prev)
        adj[i] = min(val, 1.0)
        prev = adj[i]

    out = np.empty(m, dtype=float)
    out[order] = adj
    return out.tolist()


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_cluster_df(cluster_path: Path, id_col: str, cluster_col: str) -> pd.DataFrame:
    if cluster_path.suffix.lower() == ".csv":
        df = pd.read_csv(cluster_path)
    elif cluster_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(cluster_path)
    else:
        raise ValueError(f"Unsupported cluster file type: {cluster_path.suffix}")

    if id_col not in df.columns:
        raise KeyError(f"Cluster file missing id column: {id_col}")
    if cluster_col not in df.columns:
        raise KeyError(f"Cluster file missing cluster column: {cluster_col}")

    out = df[[id_col, cluster_col]].copy()
    out = out.rename(columns={id_col: "id", cluster_col: "cluster"})
    out["id"] = normalize_id(out["id"])
    out = out.dropna(subset=["id", "cluster"]).copy()
    out["id"] = out["id"].astype(int)
    return out


def load_personality_df(xlsx_path: Path, sheet_name: str, id_col: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)

    # 대소문자/공백 최소 대응
    col_map = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=col_map)

    if id_col not in df.columns:
        # fallback: 흔한 id 컬럼명 탐색
        candidates = ["id", "ID", "no.", "no", "subject_id"]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise KeyError(f"Personality sheet missing id column: {id_col}")
        id_col = found

    needed = list(set([id_col] + FFM_VARS + MBTI_CONT_VARS))
    existing = [c for c in needed if c in df.columns]

    out = df[existing].copy()
    out = out.rename(columns={id_col: "id"})
    out["id"] = normalize_id(out["id"])
    out = out.dropna(subset=["id"]).copy()
    out["id"] = out["id"].astype(int)

    for c in out.columns:
        if c != "id":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def merge_cluster_personality(cluster_df: pd.DataFrame, personality_df: pd.DataFrame) -> pd.DataFrame:
    df = cluster_df.merge(personality_df, on="id", how="inner")
    df = df.dropna(subset=["cluster"]).copy()
    return df


def group_arrays(df: pd.DataFrame, value_col: str) -> Dict[str, np.ndarray]:
    out = {}
    for g, sub in df.groupby("cluster"):
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            out[str(g)] = vals
    return out


# =========================================================
# 3. Continuous global permutation profile test
# =========================================================
def multivariate_pseudo_f(X: np.ndarray, groups: np.ndarray) -> float:
    """
    연속형 trait 벡터에 대한 permutation-based global profile test statistic.
    고전 MANOVA 대신 group centroid 기반 pseudo-F 사용.
    """
    valid = ~pd.isna(groups)
    X = X[valid]
    groups = groups[valid]

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    groups = groups[mask]

    uniq = np.unique(groups)
    n = len(groups)
    k = len(uniq)
    p = X.shape[1]

    if n <= k or k < 2 or p < 1:
        return np.nan

    overall_mean = X.mean(axis=0)

    ss_between = 0.0
    ss_within = 0.0

    for g in uniq:
        Xg = X[groups == g]
        mg = Xg.mean(axis=0)
        ss_between += len(Xg) * np.sum((mg - overall_mean) ** 2)
        ss_within += np.sum((Xg - mg) ** 2)

    if ss_within <= 0:
        return np.nan

    df1 = k - 1
    df2 = n - k
    return (ss_between / df1) / (ss_within / df2)


def permutation_profile_test(
    df: pd.DataFrame,
    trait_cols: List[str],
    n_perm: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    work = df[["cluster"] + trait_cols].dropna().copy()
    if work.shape[0] < 5:
        return {
            "n_used": work.shape[0],
            "pseudo_f": np.nan,
            "p_perm": np.nan,
        }

    X = work[trait_cols].to_numpy(dtype=float)
    # trait scale 차이를 줄이기 위해 z-score
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    groups = work["cluster"].to_numpy()

    obs = multivariate_pseudo_f(X, groups)

    rng = np.random.default_rng(seed)
    perm_stats = []
    for _ in range(n_perm):
        perm_groups = rng.permutation(groups)
        perm_stats.append(multivariate_pseudo_f(X, perm_groups))
    perm_stats = np.array(perm_stats, dtype=float)

    p_perm = (np.sum(perm_stats >= obs) + 1) / (len(perm_stats) + 1)

    return {
        "n_used": int(work.shape[0]),
        "pseudo_f": float(obs),
        "p_perm": float(p_perm),
    }


# =========================================================
# 4. Welch ANOVA
# =========================================================
def welch_anova_from_groups(groups: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Welch ANOVA manual implementation.
    """
    clean = {}
    for g, arr in groups.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 2 and np.var(arr, ddof=1) > 0:
            clean[g] = arr

    k = len(clean)
    if k < 2:
        return {"k_groups": k, "F": np.nan, "df1": np.nan, "df2": np.nan, "p": np.nan}

    ns = np.array([len(v) for v in clean.values()], dtype=float)
    means = np.array([np.mean(v) for v in clean.values()], dtype=float)
    vars_ = np.array([np.var(v, ddof=1) for v in clean.values()], dtype=float)

    if np.any(ns <= 1) or np.any(vars_ <= 0):
        return {"k_groups": k, "F": np.nan, "df1": np.nan, "df2": np.nan, "p": np.nan}

    w = ns / vars_
    w_sum = np.sum(w)
    mean_w = np.sum(w * means) / w_sum

    A = np.sum(w * (means - mean_w) ** 2) / (k - 1)

    term = np.sum((1 / (ns - 1)) * (1 - (w / w_sum)) ** 2)
    B = 1 + (2 * (k - 2) / (k**2 - 1)) * term

    F_stat = A / B
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * term)
    p = 1 - stats.f.cdf(F_stat, df1, df2)

    return {
        "k_groups": int(k),
        "F": float(F_stat),
        "df1": float(df1),
        "df2": float(df2),
        "p": float(p),
    }


def eta_squared_classical(groups: Dict[str, np.ndarray]) -> float:
    """
    보조 effect size: classical eta^2 (불균형/분산불균등에 완벽하진 않지만 해석용 보조지표)
    """
    clean = [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 0]
    if len(clean) < 2:
        return np.nan

    all_vals = np.concatenate(clean)
    grand = np.mean(all_vals)

    ss_between = 0.0
    ss_total = np.sum((all_vals - grand) ** 2)
    if ss_total <= 0:
        return np.nan

    for arr in clean:
        ss_between += len(arr) * (np.mean(arr) - grand) ** 2

    return float(ss_between / ss_total)


# =========================================================
# 5. Games-Howell
# =========================================================
def games_howell(groups: Dict[str, np.ndarray], trait_name: str) -> pd.DataFrame:
    rows = []

    keys = list(groups.keys())
    valid = {}
    for g in keys:
        arr = np.asarray(groups[g], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 2:
            valid[g] = arr

    keys = list(valid.keys())
    k = len(keys)
    if k < 2:
        return pd.DataFrame()

    for g1, g2 in itertools.combinations(keys, 2):
        x1 = valid[g1]
        x2 = valid[g2]

        n1, n2 = len(x1), len(x2)
        m1, m2 = np.mean(x1), np.mean(x2)
        v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

        se2 = (v1 / n1) + (v2 / n2)
        se = math.sqrt(se2) if se2 > 0 else np.nan
        mean_diff = m1 - m2

        if se <= 0 or np.isnan(se):
            p_val = np.nan
            df = np.nan
            t_stat = np.nan
        else:
            t_stat = abs(mean_diff) / se
            df = (se2 ** 2) / (((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1))

            if HAS_QSTURNG:
                # Games-Howell은 studentized range 기반
                q_stat = abs(mean_diff) / math.sqrt(0.5 * se2)
                p_val = float(psturng(q_stat, k, df))
            else:
                # fallback: Welch t 근사
                p_val = float(2 * (1 - stats.t.cdf(t_stat, df)))

        rows.append({
            "trait": trait_name,
            "group1": g1,
            "group2": g2,
            "n1": n1,
            "n2": n2,
            "mean1": m1,
            "mean2": m2,
            "mean_diff": mean_diff,
            "df": df,
            "stat": t_stat,
            "p_raw": p_val,
            "method": "Games-Howell" if HAS_QSTURNG else "Welch-t-approx",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_within_trait"] = holm_correction(out["p_raw"].fillna(1.0).tolist())
    return out


# =========================================================
# 6. Binary MBTI
# =========================================================
def score_to_binary_label(
    score: float,
    first_label: str,
    second_label: str,
    hi_threshold: float,
    lo_threshold: float,
    tie_policy: str,
) -> str | float:
    if pd.isna(score):
        return np.nan
    if score >= hi_threshold:
        return first_label
    if score <= lo_threshold:
        return second_label

    # ambiguous zone
    if tie_policy == "drop":
        return np.nan
    if tie_policy == "first":
        return first_label
    if tie_policy == "second":
        return second_label
    return np.nan


def add_mbti_binary_columns(
    df: pd.DataFrame,
    hi_threshold: float = 51.0,
    lo_threshold: float = 49.0,
    tie_policy: str = "drop",
) -> pd.DataFrame:
    out = df.copy()

    for axis, (first_label, second_label) in MBTI_BINARY_RULES.items():
        out[f"{axis}_bin"] = out[axis].apply(
            lambda x: score_to_binary_label(
                x, first_label, second_label, hi_threshold, lo_threshold, tie_policy
            )
        )
    return out


def binary_global_permutation_test(
    df: pd.DataFrame,
    binary_cols: List[str],
    n_perm: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    각 binary 축별 chi-square statistic을 합산한 global permutation test
    """
    work = df[["cluster"] + binary_cols].copy()
    work = work.dropna(subset=["cluster"]).copy()

    def summed_chi2(local_df: pd.DataFrame) -> float:
        total = 0.0
        for col in binary_cols:
            sub = local_df[["cluster", col]].dropna()
            if sub.empty or sub["cluster"].nunique() < 2 or sub[col].nunique() < 2:
                continue
            tab = pd.crosstab(sub["cluster"], sub[col])
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            chi2, _, _, _ = chi2_contingency(tab)
            total += chi2
        return total

    obs = summed_chi2(work)

    rng = np.random.default_rng(seed)
    perm_stats = []
    for _ in range(n_perm):
        perm_df = work.copy()
        perm_df["cluster"] = rng.permutation(perm_df["cluster"].to_numpy())
        perm_stats.append(summed_chi2(perm_df))

    perm_stats = np.array(perm_stats, dtype=float)
    p_perm = (np.sum(perm_stats >= obs) + 1) / (len(perm_stats) + 1)

    return {
        "n_used": int(work.shape[0]),
        "chi2_sum": float(obs),
        "p_perm": float(p_perm),
    }


def binary_trait_association(df: pd.DataFrame, col: str) -> Dict[str, float]:
    sub = df[["cluster", col]].dropna().copy()
    if sub.empty or sub["cluster"].nunique() < 2 or sub[col].nunique() < 2:
        return {
            "trait": col,
            "chi2": np.nan,
            "dof": np.nan,
            "p_raw": np.nan,
            "n_used": sub.shape[0],
        }

    tab = pd.crosstab(sub["cluster"], sub[col])
    chi2, p, dof, _ = chi2_contingency(tab)

    return {
        "trait": col,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_raw": float(p),
        "n_used": int(sub.shape[0]),
    }


def pairwise_binary_posthoc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = df[["cluster", col]].dropna().copy()
    if sub.empty or sub["cluster"].nunique() < 2 or sub[col].nunique() < 2:
        return pd.DataFrame()

    levels = sorted(sub["cluster"].astype(str).unique().tolist())
    pos_label = sorted(sub[col].unique().tolist())[0]  # arbitrary fixed positive label

    rows = []
    for g1, g2 in itertools.combinations(levels, 2):
        s1 = sub[sub["cluster"].astype(str) == g1][col]
        s2 = sub[sub["cluster"].astype(str) == g2][col]

        count1 = int((s1 == pos_label).sum())
        count2 = int((s2 == pos_label).sum())
        n1 = int(s1.notna().sum())
        n2 = int(s2.notna().sum())

        # 2x2 table
        table = np.array([
            [count1, n1 - count1],
            [count2, n2 - count2],
        ])

        # expected count가 작으면 Fisher
        expected = stats.contingency.expected_freq(table)
        if (expected < 5).any():
            _, p = fisher_exact(table)
            method = "Fisher-exact"
            stat = np.nan
        else:
            stat, p = proportions_ztest([count1, count2], [n1, n2])
            method = "Two-proportion-z"

        rows.append({
            "trait": col,
            "group1": g1,
            "group2": g2,
            "positive_label": pos_label,
            "count1": count1,
            "n1": n1,
            "prop1": count1 / n1 if n1 > 0 else np.nan,
            "count2": count2,
            "n2": n2,
            "prop2": count2 / n2 if n2 > 0 else np.nan,
            "stat": stat,
            "p_raw": float(p),
            "method": method,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_within_trait"] = holm_correction(out["p_raw"].tolist())
    return out


# =========================================================
# 7. Continuous pipeline runner
# =========================================================
def run_continuous_pipeline(
    df: pd.DataFrame,
    trait_cols: List[str],
    analysis_name: str,
    out_dir: Path,
    n_perm: int,
    seed: int,
) -> None:
    ensure_dir(out_dir)

    available = [c for c in trait_cols if c in df.columns]
    if len(available) == 0:
        raise ValueError(f"[{analysis_name}] No available trait columns found.")

    # ---------- global profile test ----------
    global_result = permutation_profile_test(df, available, n_perm=n_perm, seed=seed)
    pd.DataFrame([global_result]).to_csv(out_dir / "global_profile_test.csv", index=False, encoding="utf-8-sig")

    # ---------- trait-wise Welch ANOVA ----------
    trait_rows = []
    pairwise_all = []

    for trait in available:
        groups = group_arrays(df, trait)
        welch = welch_anova_from_groups(groups)
        eta2 = eta_squared_classical(groups)

        trait_rows.append({
            "trait": trait,
            "k_groups": welch["k_groups"],
            "F": welch["F"],
            "df1": welch["df1"],
            "df2": welch["df2"],
            "p_raw": welch["p"],
            "eta2_classical": eta2,
            "n_used": int(df[[trait, "cluster"]].dropna().shape[0]),
        })

    trait_df = pd.DataFrame(trait_rows)
    if not trait_df.empty:
        trait_df["p_holm"] = holm_correction(trait_df["p_raw"].fillna(1.0).tolist())
        trait_df["significant_holm_0.05"] = trait_df["p_holm"] < 0.05

    trait_df.to_csv(out_dir / "trait_level_tests.csv", index=False, encoding="utf-8-sig")

    # ---------- Games-Howell ----------
    if not trait_df.empty:
        sig_traits = trait_df.loc[trait_df["significant_holm_0.05"], "trait"].tolist()
        for trait in sig_traits:
            groups = group_arrays(df, trait)
            gh = games_howell(groups, trait_name=trait)
            if not gh.empty:
                pairwise_all.append(gh)

    if len(pairwise_all) > 0:
        pairwise_df = pd.concat(pairwise_all, ignore_index=True)
    else:
        pairwise_df = pd.DataFrame()

    pairwise_df.to_csv(out_dir / "pairwise_posthoc.csv", index=False, encoding="utf-8-sig")

    # ---------- summary ----------
    summary_lines = []
    summary_lines.append(f"[ANALYSIS] {analysis_name}")
    summary_lines.append(f"[N_TRAITS] {len(available)}")
    summary_lines.append(f"[GLOBAL] n_used={global_result['n_used']}, pseudo_f={global_result['pseudo_f']:.6f}, p_perm={global_result['p_perm']:.6f}")

    if not trait_df.empty:
        summary_lines.append("[TRAIT-LEVEL]")
        for _, row in trait_df.sort_values("p_holm").iterrows():
            summary_lines.append(
                f"- {row['trait']}: Welch F={row['F']:.4f}, "
                f"p_raw={row['p_raw']:.6f}, p_holm={row['p_holm']:.6f}, "
                f"eta2={row['eta2_classical']:.4f}, sig={bool(row['significant_holm_0.05'])}"
            )

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


# =========================================================
# 8. Binary MBTI pipeline runner
# =========================================================
def run_binary_mbti_pipeline(
    df: pd.DataFrame,
    out_dir: Path,
    n_perm: int,
    seed: int,
    hi_threshold: float,
    lo_threshold: float,
    tie_policy: str,
) -> None:
    ensure_dir(out_dir)

    if not all(c in df.columns for c in MBTI_CONT_VARS):
        raise ValueError("[mbti_bin] MBTI continuous axis columns missing.")

    work = add_mbti_binary_columns(
        df,
        hi_threshold=hi_threshold,
        lo_threshold=lo_threshold,
        tie_policy=tie_policy,
    )

    binary_cols = [f"{c}_bin" for c in MBTI_CONT_VARS]

    # ---------- global ----------
    global_result = binary_global_permutation_test(work, binary_cols, n_perm=n_perm, seed=seed)
    pd.DataFrame([global_result]).to_csv(out_dir / "global_profile_test.csv", index=False, encoding="utf-8-sig")

    # ---------- axis-wise chi-square ----------
    rows = []
    pairwise_all = []
    for col in binary_cols:
        res = binary_trait_association(work, col)
        rows.append(res)

    trait_df = pd.DataFrame(rows)
    if not trait_df.empty:
        trait_df["p_holm"] = holm_correction(trait_df["p_raw"].fillna(1.0).tolist())
        trait_df["significant_holm_0.05"] = trait_df["p_holm"] < 0.05

    trait_df.to_csv(out_dir / "trait_level_tests.csv", index=False, encoding="utf-8-sig")

    # ---------- pairwise ----------
    if not trait_df.empty:
        sig_traits = trait_df.loc[trait_df["significant_holm_0.05"], "trait"].tolist()
        for col in sig_traits:
            pw = pairwise_binary_posthoc(work, col)
            if not pw.empty:
                pairwise_all.append(pw)

    if len(pairwise_all) > 0:
        pairwise_df = pd.concat(pairwise_all, ignore_index=True)
    else:
        pairwise_df = pd.DataFrame()

    pairwise_df.to_csv(out_dir / "pairwise_posthoc.csv", index=False, encoding="utf-8-sig")

    # ---------- binary tables ----------
    for col in binary_cols:
        tab = pd.crosstab(work["cluster"], work[col], dropna=False)
        tab.to_csv(out_dir / f"contingency_{col}.csv", encoding="utf-8-sig")

    # ---------- summary ----------
    summary_lines = []
    summary_lines.append("[ANALYSIS] mbti_bin")
    summary_lines.append(
        f"[BINARY_RULE] hi_threshold={hi_threshold}, lo_threshold={lo_threshold}, tie_policy={tie_policy}"
    )
    summary_lines.append(
        f"[GLOBAL] n_used={global_result['n_used']}, chi2_sum={global_result['chi2_sum']:.6f}, p_perm={global_result['p_perm']:.6f}"
    )

    if not trait_df.empty:
        summary_lines.append("[AXIS-LEVEL]")
        for _, row in trait_df.sort_values("p_holm").iterrows():
            summary_lines.append(
                f"- {row['trait']}: chi2={row['chi2']:.4f}, "
                f"p_raw={row['p_raw']:.6f}, p_holm={row['p_holm']:.6f}, "
                f"sig={bool(row['significant_holm_0.05'])}"
            )

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


# =========================================================
# 9. Main
# =========================================================
def main():
    args = parse_args()

    cluster_path = Path(args.cluster_path)
    personality_xlsx_path = Path(args.personality_xlsx_path)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    cluster_df = load_cluster_df(
        cluster_path=cluster_path,
        id_col=args.cluster_id_col,
        cluster_col=args.cluster_label_col,
    )

    personality_df = load_personality_df(
        xlsx_path=personality_xlsx_path,
        sheet_name=args.personality_sheet,
        id_col=args.personality_id_col,
    )

    merged = merge_cluster_personality(cluster_df, personality_df)

    # config snapshot
    config_snapshot = {
        "cluster_path": str(cluster_path),
        "personality_xlsx_path": str(personality_xlsx_path),
        "out_dir": str(out_dir),
        "cluster_id_col": args.cluster_id_col,
        "cluster_label_col": args.cluster_label_col,
        "personality_sheet": args.personality_sheet,
        "personality_id_col": args.personality_id_col,
        "analysis_modes": args.analysis_modes,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "mbti_hi_threshold": args.mbti_hi_threshold,
        "mbti_lo_threshold": args.mbti_lo_threshold,
        "mbti_tie_policy": args.mbti_tie_policy,
        "n_merged_subjects": int(merged.shape[0]),
        "cluster_counts": merged["cluster"].value_counts(dropna=False).to_dict(),
    }

    with open(out_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    merged.to_csv(out_dir / "merged_cluster_personality.csv", index=False, encoding="utf-8-sig")

    if "ffm" in args.analysis_modes:
        run_continuous_pipeline(
            df=safe_numeric(merged, FFM_VARS),
            trait_cols=FFM_VARS,
            analysis_name="ffm",
            out_dir=out_dir / "ffm",
            n_perm=args.n_perm,
            seed=args.seed,
        )

    if "mbti_cont" in args.analysis_modes:
        run_continuous_pipeline(
            df=safe_numeric(merged, MBTI_CONT_VARS),
            trait_cols=MBTI_CONT_VARS,
            analysis_name="mbti_cont",
            out_dir=out_dir / "mbti_cont",
            n_perm=args.n_perm,
            seed=args.seed,
        )

    if "mbti_bin" in args.analysis_modes:
        run_binary_mbti_pipeline(
            df=safe_numeric(merged, MBTI_CONT_VARS),
            out_dir=out_dir / "mbti_bin",
            n_perm=args.n_perm,
            seed=args.seed,
            hi_threshold=args.mbti_hi_threshold,
            lo_threshold=args.mbti_lo_threshold,
            tie_policy=args.mbti_tie_policy,
        )

    print("[DONE] personality post-hoc analysis completed.")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()