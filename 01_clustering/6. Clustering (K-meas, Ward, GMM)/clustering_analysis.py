# 핵심 요약:
# - clustering 분석의 순수 로직 모듈이다. config 정의는 포함하지 않는다.
# - 모델 적합(KMeans, Ward, GMM), 내부지표(Silhouette, CH, DB), 안정성(seed/subsample ARI)을 담당한다.
# - AnalysisConfig는 run_clustering.py에서 정의하며, 이 모듈은 인자로 받아서만 사용한다.
# - feature별 설명력(F/eta²), outlier 탐지, tiny cluster 원인 분석 기능을 포함한다.
# - block 진단(상관, PCA, energy), co-clustering matrix, report 생성 함수도 여기에 모아둔다.
# - 이 파일 단독으로는 실행되지 않으며, run_clustering.py가 진입점이다.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

if TYPE_CHECKING:
    from run_clustering import AnalysisConfig


# =========================================================
# 데이터 클래스
# =========================================================

@dataclass
class ModelResult:
    model_name: str
    k: int
    labels: np.ndarray
    silhouette: float
    ch_index: float
    db_index: float
    bic: float | None = None
    aic: float | None = None
    covariance_type: str | None = None


# =========================================================
# 유틸리티
# =========================================================

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def format_cov_text(covariance_type: str | None) -> str:
    return f", covariance_type={covariance_type}" if covariance_type else ""


# =========================================================
# 입력 처리
# =========================================================

def find_id_col(df: pd.DataFrame, cfg: "AnalysisConfig") -> str | None:
    for c in cfg.id_col_candidates:
        if c in df.columns:
            return c
    return None


def load_input_dataframe(input_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Input file not found: {input_path}")
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if sheet_name is None:
        return pd.read_excel(input_path)
    return pd.read_excel(input_path, sheet_name=sheet_name)


def ensure_numeric_matrix(
    df: pd.DataFrame, id_col: str | None
) -> Tuple[pd.DataFrame, pd.Series | None]:
    work = df.copy()
    ids = None
    if id_col is not None:
        ids = work[id_col].copy()
        work = work.drop(columns=[id_col])

    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    na_count = int(work.isna().sum().sum())
    if na_count > 0:
        raise ValueError(
            f"[ERROR] numeric matrix 내 결측치가 {na_count}개 있습니다. "
            "현재 코드는 이미 정리된 input matrix를 가정합니다."
        )

    nunique = work.nunique(dropna=False)
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        print(f"[INFO] zero-variance columns removed: {zero_var_cols}")
        work = work.drop(columns=zero_var_cols)

    return work, ids


def filter_blocks_to_existing_columns(
    blocks: Dict[str, List[str]], feature_cols: List[str]
) -> Dict[str, List[str]]:
    return {
        name: [c for c in cols if c in feature_cols]
        for name, cols in blocks.items()
        if any(c in feature_cols for c in cols)
    }


# =========================================================
# 내부 지표
# =========================================================

def evaluate_partition(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan, np.nan, np.nan

    counts = pd.Series(labels).value_counts()
    try:
        sil = silhouette_score(X, labels) if not (counts <= 1).any() else silhouette_score(X, labels)
    except Exception:
        sil = np.nan

    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan

    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan

    return sil, ch, db


def is_candidate_usable(row: pd.Series, cfg: "AnalysisConfig") -> bool:
    min_ok = safe_float(row.get("min_cluster_size", np.nan)) >= cfg.usable_min_cluster_size
    subsample_ari = safe_float(row.get("subsample_stability_mean_ari", np.nan))
    stability_ok = np.isnan(subsample_ari) or (subsample_ari >= 0.60)
    return bool(min_ok and stability_ok)


# =========================================================
# Block 진단
# =========================================================

def summarize_block_correlations(X_df: pd.DataFrame, blocks: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for block_name, cols in blocks.items():
        sub = X_df[cols].copy()
        corr = sub.corr().abs()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        vals = corr.where(mask).stack()
        rows.append({
            "block": block_name,
            "n_features": len(cols),
            "mean_abs_corr_within_block": float(vals.mean()) if len(vals) else np.nan,
            "max_abs_corr_within_block": float(vals.max()) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_block_pca(X_df: pd.DataFrame, blocks: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for block_name, cols in blocks.items():
        sub = X_df[cols].copy()
        if len(cols) == 1:
            rows.append({
                "block": block_name, "n_features": 1,
                "pc1_explained_variance_ratio": 1.0,
                "pc2_explained_variance_ratio": np.nan,
            })
            continue
        pca = PCA(n_components=min(len(cols), sub.shape[0]))
        pca.fit(sub)
        evr = pca.explained_variance_ratio_
        rows.append({
            "block": block_name, "n_features": len(cols),
            "pc1_explained_variance_ratio": float(evr[0]),
            "pc2_explained_variance_ratio": float(evr[1]) if len(evr) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_block_energy(X_df: pd.DataFrame, blocks: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for block_name, cols in blocks.items():
        sub = X_df[cols].values
        norms = np.sqrt((sub ** 2).sum(axis=1))
        rows.append({
            "block": block_name, "n_features": len(cols),
            "mean_l2_norm": float(np.mean(norms)),
            "sd_l2_norm": float(np.std(norms)),
        })
    return pd.DataFrame(rows)


# =========================================================
# Feature 설명력 (신규)
# =========================================================

def compute_feature_discriminability(
    X_df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    feature별 군집 설명력을 one-way ANOVA F-통계량과 eta-squared로 계산한다.

    - F-statistic : 군집 간 분산 / 군집 내 분산 비율. 클수록 해당 feature가 군집을 잘 구분함.
    - eta_squared : SS_between / SS_total. 효과크기(0~1).
                    0.01=small, 0.06=medium, 0.14 이상=large.
    - p_value     : F-통계량의 유의확률. 0.05 미만이면 군집 간 유의한 차이.
    """
    unique_labels = np.unique(labels)
    rows = []

    for col in X_df.columns:
        groups = [X_df[col].values[labels == lab] for lab in unique_labels]
        groups_valid = [g for g in groups if len(g) > 1]

        try:
            f_stat, p_val = scipy_stats.f_oneway(*groups_valid)
        except Exception:
            f_stat, p_val = np.nan, np.nan

        grand_mean = float(X_df[col].mean())
        ss_total = float(((X_df[col] - grand_mean) ** 2).sum())
        ss_between = float(sum(
            len(g) * (float(g.mean()) - grand_mean) ** 2
            for g in groups if len(g) > 0
        ))
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        if np.isnan(eta_sq):
            effect_label = "unknown"
        elif eta_sq >= 0.14:
            effect_label = "large"
        elif eta_sq >= 0.06:
            effect_label = "medium"
        elif eta_sq >= 0.01:
            effect_label = "small"
        else:
            effect_label = "negligible"

        rows.append({
            "feature": col,
            "f_statistic": safe_float(f_stat),
            "p_value": safe_float(p_val),
            "eta_squared": safe_float(eta_sq),
            "effect_size": effect_label,
            "significant_p05": (
                bool(safe_float(p_val) < 0.05)
                if not np.isnan(safe_float(p_val)) else False
            ),
        })

    out = (
        pd.DataFrame(rows)
        .sort_values("f_statistic", ascending=False)
        .reset_index(drop=True)
    )
    out.insert(0, "discriminability_rank", range(1, len(out) + 1))
    return out


# =========================================================
# Outlier 탐지 (신규)
# =========================================================

def detect_outliers_by_centroid_distance(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    z_threshold: float = 2.5,
    id_col: str | None = None,
    ids: pd.Series | None = None,
) -> pd.DataFrame:
    """
    군집 내 centroid로부터의 Euclidean distance를 기반으로 outlier를 탐지한다.
    각 군집 내에서 distance의 z-score가 z_threshold를 초과하는 subject를 outlier로 표시한다.

    해석 기준:
    - z_within_cluster > 2.5 : 군집 내에서 비교적 먼 위치 → 잠재적 outlier
    - 작은 군집(n<=6)에서는 z-score 계산이 불안정하므로 결과 해석에 주의
    """
    rows = []
    for lab in np.unique(labels):
        mask = labels == lab
        sub_vals = X_df.values[mask]
        orig_indices = np.where(mask)[0]
        n_in_cluster = len(orig_indices)

        centroid = sub_vals.mean(axis=0)
        distances = np.linalg.norm(sub_vals - centroid, axis=1)
        dist_std = distances.std()
        z_scores = (distances - distances.mean()) / dist_std if dist_std > 0 else np.zeros_like(distances)

        for i, (orig_idx, dist, z) in enumerate(zip(orig_indices, distances, z_scores)):
            subject_id = ids.iloc[orig_idx] if ids is not None else int(orig_idx)

            feature_devs = np.abs(sub_vals[i] - centroid)
            top_dev_idx = np.argsort(feature_devs)[::-1][:3]
            top_dev_features = " | ".join(
                f"{X_df.columns[fi]}(val={sub_vals[i, fi]:.2f}, dev={feature_devs[fi]:.2f})"
                for fi in top_dev_idx
            )

            rows.append({
                "subject_id": subject_id,
                "cluster": int(lab),
                "n_in_cluster": int(n_in_cluster),
                "dist_from_centroid": float(dist),
                "z_within_cluster": float(z),
                "is_outlier": bool(z > z_threshold),
                "top_deviant_features": top_dev_features,
                "note": "z-score unstable (small cluster)" if n_in_cluster <= 6 else "",
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["cluster", "z_within_cluster"], ascending=[True, False])
        .reset_index(drop=True)
    )


# =========================================================
# Tiny cluster 원인 분석 (신규)
# =========================================================

def explain_tiny_cluster_causes(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    profile_z: pd.DataFrame,
    feature_discrim_df: pd.DataFrame,
    tiny_cluster_threshold: int = 6,
) -> pd.DataFrame:
    """
    tiny cluster(n <= threshold)가 형성된 원인을 분류하고 설명한다.

    원인 유형 (상대거리 = nearest_dist / mean_pair_dist):
    - borderline_group   : 상대거리 < 0.5
      → 특정 군집 경계에 있는 애매한 케이스. k를 줄이면 해당 군집에 흡수될 가능성 높음.
    - outlier_group      : 상대거리 > 1.5
      → 모든 군집과 멀리 떨어진 극단적/비전형적 subject 집합.
        실제 유형이라기보다 outlier 흡수일 가능성.
    - distinct_small_group : 중간 (0.5~1.5)
      → 뚜렷한 특성 패턴을 가지나 수가 적은 소수 유형. 실질적 의미 검토 필요.
    """
    unique_labels = np.unique(labels)
    counts = pd.Series(labels).value_counts()
    tiny_labels = counts[counts <= tiny_cluster_threshold].index.tolist()
    if not tiny_labels:
        return pd.DataFrame()

    centroids = {
        int(lab): X_df.values[labels == lab].mean(axis=0)
        for lab in unique_labels
    }
    labs_list = list(centroids.keys())
    all_pair_dists = [
        float(np.linalg.norm(centroids[labs_list[i]] - centroids[labs_list[j]]))
        for i in range(len(labs_list))
        for j in range(i + 1, len(labs_list))
    ]
    mean_pair_dist = float(np.mean(all_pair_dists)) if all_pair_dists else 1.0

    top_discrim_features = feature_discrim_df["feature"].head(5).tolist()

    rows = []
    for tiny_lab in tiny_labels:
        tiny_lab = int(tiny_lab)
        centroid_tiny = centroids[tiny_lab]
        n_members = int(counts[tiny_lab])

        other_dists = {
            other_lab: float(np.linalg.norm(centroid_tiny - other_c))
            for other_lab, other_c in centroids.items()
            if other_lab != tiny_lab
        }
        nearest_cluster = min(other_dists, key=other_dists.get) if other_dists else None
        nearest_dist = other_dists[nearest_cluster] if nearest_cluster is not None else np.nan
        relative_nearest = nearest_dist / mean_pair_dist if mean_pair_dist > 0 else np.nan

        if np.isnan(relative_nearest):
            cause = "unknown"
            cause_detail = "거리 계산 불가"
        elif relative_nearest < 0.5:
            cause = "borderline_group"
            cause_detail = (
                f"군집 {nearest_cluster}의 centroid에 매우 가까움 (상대거리={relative_nearest:.2f}). "
                "경계 영역의 애매한 케이스일 가능성. k를 줄이면 해당 군집에 흡수될 수 있음."
            )
        elif relative_nearest > 1.5:
            cause = "outlier_group"
            cause_detail = (
                f"모든 다른 군집 centroid와 멀리 떨어짐 (상대거리={relative_nearest:.2f}). "
                "극단적 feature 값을 가진 비전형적 subject 집합일 가능성. "
                "실제 유형이기보다 outlier 흡수 군집일 수 있음."
            )
        else:
            cause = "distinct_small_group"
            cause_detail = (
                f"가장 가까운 군집 {nearest_cluster}과 중간 거리 (상대거리={relative_nearest:.2f}). "
                "뚜렷한 특성 패턴을 가진 소수 유형일 수 있음. 해석 시 외부 검증 필요."
            )

        if tiny_lab in profile_z.index:
            tiny_profile = profile_z.loc[tiny_lab]
            extreme_high = " | ".join(
                f"{f}(z={v:.2f})" for f, v in tiny_profile.nlargest(3).items()
            )
            extreme_low = " | ".join(
                f"{f}(z={v:.2f})" for f, v in tiny_profile.nsmallest(3).items()
            )
            discrim_z = " | ".join(
                f"{f}(z={tiny_profile[f]:.2f})"
                for f in top_discrim_features if f in tiny_profile.index
            )
        else:
            extreme_high = extreme_low = discrim_z = ""

        rows.append({
            "cluster": tiny_lab,
            "n_members": n_members,
            "cause_type": cause,
            "nearest_cluster": nearest_cluster,
            "nearest_dist": round(nearest_dist, 4) if not np.isnan(nearest_dist) else np.nan,
            "relative_nearest_dist": round(relative_nearest, 4) if not np.isnan(relative_nearest) else np.nan,
            "mean_pair_dist_reference": round(mean_pair_dist, 4),
            "cause_detail": cause_detail,
            "extreme_high_features": extreme_high,
            "extreme_low_features": extreme_low,
            "top_discrim_feature_zscores": discrim_z,
        })

    return pd.DataFrame(rows)


# =========================================================
# 모델 적합
# =========================================================

def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> ModelResult:
    model = KMeans(n_clusters=k, n_init=30, random_state=random_state)
    labels = model.fit_predict(X)
    sil, ch, db = evaluate_partition(X, labels)
    return ModelResult("kmeans", k, labels, sil, ch, db)


def run_ward(X: np.ndarray, k: int) -> ModelResult:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    sil, ch, db = evaluate_partition(X, labels)
    return ModelResult("ward", k, labels, sil, ch, db)


def run_gmm_single(
    X: np.ndarray, k: int, covariance_type: str, random_state: int = 42
) -> ModelResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianMixture(
            n_components=k, covariance_type=covariance_type,
            n_init=10, random_state=random_state,
        )
        model.fit(X)
        labels = model.predict(X)
    sil, ch, db = evaluate_partition(X, labels)
    return ModelResult(
        model_name="gmm", k=k, labels=labels,
        silhouette=sil, ch_index=ch, db_index=db,
        bic=model.bic(X), aic=model.aic(X),
        covariance_type=covariance_type,
    )


def fit_model_by_name(
    X: np.ndarray,
    model_name: str,
    k: int,
    random_state: int = 42,
    covariance_type: str = "full",
) -> np.ndarray:
    if model_name == "kmeans":
        return KMeans(n_clusters=k, n_init=30, random_state=random_state).fit_predict(X)
    if model_name == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if model_name == "gmm":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianMixture(
                n_components=k, covariance_type=covariance_type,
                n_init=10, random_state=random_state,
            )
            return model.fit(X).predict(X)
    raise ValueError(f"Unknown model_name: {model_name}")


# =========================================================
# 안정성
# =========================================================

def compute_seed_stability(
    X: np.ndarray, model_name: str, k: int,
    n_runs: int = 30, covariance_type: str = "full",
) -> dict:
    labels_list = [
        fit_model_by_name(X, model_name, k, random_state=seed, covariance_type=covariance_type)
        for seed in range(100, 100 + n_runs)
    ]
    ref = labels_list[0]
    ari_vs_ref = [adjusted_rand_score(ref, lab) for lab in labels_list[1:]]
    pairwise_ari = [
        adjusted_rand_score(labels_list[i], labels_list[j])
        for i in range(len(labels_list))
        for j in range(i + 1, len(labels_list))
    ]
    return {
        "seed_stability_mean_ari_vs_ref": float(np.mean(ari_vs_ref)) if ari_vs_ref else np.nan,
        "seed_stability_min_ari_vs_ref": float(np.min(ari_vs_ref)) if ari_vs_ref else np.nan,
        "seed_stability_mean_pairwise_ari": float(np.mean(pairwise_ari)) if pairwise_ari else np.nan,
        "seed_stability_sd_pairwise_ari": float(np.std(pairwise_ari)) if pairwise_ari else np.nan,
    }


def compute_subsample_stability(
    X: np.ndarray, model_name: str, k: int,
    n_runs: int = 100, subsample_ratio: float = 0.8,
    covariance_type: str = "full", random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    memberships = []
    for run_idx in range(n_runs):
        idx = np.sort(rng.choice(n, size=int(np.floor(n * subsample_ratio)), replace=False))
        labels_sub = fit_model_by_name(
            X[idx], model_name, k, random_state=1000 + run_idx, covariance_type=covariance_type
        )
        memberships.append((idx, labels_sub))

    ari_scores = []
    for i in range(len(memberships)):
        idx_i, lab_i = memberships[i]
        map_i = dict(zip(idx_i, lab_i))
        for j in range(i + 1, len(memberships)):
            idx_j, lab_j = memberships[j]
            map_j = dict(zip(idx_j, lab_j))
            overlap = np.intersect1d(idx_i, idx_j)
            if len(overlap) < max(10, k * 2):
                continue
            ari_scores.append(adjusted_rand_score(
                [map_i[t] for t in overlap], [map_j[t] for t in overlap]
            ))

    arr = np.array(ari_scores, dtype=float) if ari_scores else np.array([])
    return {
        "subsample_stability_mean_ari": float(np.mean(arr)) if arr.size else np.nan,
        "subsample_stability_median_ari": float(np.median(arr)) if arr.size else np.nan,
        "subsample_stability_p10_ari": float(np.percentile(arr, 10)) if arr.size else np.nan,
        "subsample_stability_min_ari": float(np.min(arr)) if arr.size else np.nan,
        "subsample_stability_sd_ari": float(np.std(arr)) if arr.size else np.nan,
        "subsample_stability_n_pairs": int(len(arr)),
    }


def compute_coclustering_matrix(
    X: np.ndarray, model_name: str, k: int,
    n_runs: int = 100, subsample_ratio: float = 0.8,
    covariance_type: str = "full", random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    same_count = np.zeros((n, n), dtype=float)
    seen_count = np.zeros((n, n), dtype=float)

    for run_idx in range(n_runs):
        idx = np.sort(rng.choice(n, size=int(np.floor(n * subsample_ratio)), replace=False))
        labels_sub = fit_model_by_name(
            X[idx], model_name, k, random_state=2000 + run_idx, covariance_type=covariance_type
        )
        for a in range(len(idx)):
            ia = idx[a]
            for b in range(a, len(idx)):
                ib = idx[b]
                seen_count[ia, ib] += 1
                seen_count[ib, ia] += 1
                if labels_sub[a] == labels_sub[b]:
                    same_count[ia, ib] += 1
                    same_count[ib, ia] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        coclust = np.divide(
            same_count, seen_count,
            out=np.zeros_like(same_count), where=seen_count > 0
        )
    np.fill_diagonal(coclust, 1.0)
    return coclust


# =========================================================
# 요약 테이블
# =========================================================

def model_result_to_row(r: ModelResult) -> dict:
    counts = pd.Series(r.labels).value_counts().sort_index()
    return {
        "model": r.model_name, "k": r.k, "covariance_type": r.covariance_type,
        "silhouette": r.silhouette, "calinski_harabasz": r.ch_index,
        "davies_bouldin": r.db_index, "bic": r.bic, "aic": r.aic,
        "min_cluster_size": int(counts.min()), "max_cluster_size": int(counts.max()),
        "cluster_sizes": ", ".join([f"C{i}:{v}" for i, v in counts.items()]),
    }


def results_to_df(results: List[ModelResult]) -> pd.DataFrame:
    out = pd.DataFrame([model_result_to_row(r) for r in results])
    out["sil_rank"] = out["silhouette"].rank(ascending=False, method="min")
    out["ch_rank"] = out["calinski_harabasz"].rank(ascending=False, method="min")
    out["db_rank"] = out["davies_bouldin"].rank(ascending=True, method="min")
    out["bic_rank"] = out["bic"].rank(ascending=True, method="min")
    out["aic_rank"] = out["aic"].rank(ascending=True, method="min")
    out["internal_rank_score"] = (
        out["sil_rank"].fillna(out["sil_rank"].max())
        + out["ch_rank"].fillna(out["ch_rank"].max())
        + out["db_rank"].fillna(out["db_rank"].max())
    )
    return out


def attach_stability_to_summary(
    summary_df: pd.DataFrame, X: np.ndarray,
    cfg: "AnalysisConfig", random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        model_name = row["model"]
        k = int(row["k"])
        covariance_type = row["covariance_type"] if pd.notna(row["covariance_type"]) else "full"
        print(f"[STABILITY] {model_name} | k={k} | cov={covariance_type}")
        merged = row.to_dict()
        merged.update(compute_seed_stability(X, model_name, k, cfg.seed_stability_runs, covariance_type))
        merged.update(compute_subsample_stability(
            X, model_name, k, cfg.subsample_stability_runs,
            cfg.subsample_ratio, covariance_type, random_state
        ))
        rows.append(merged)

    out = pd.DataFrame(rows)
    out["seed_rank"] = out["seed_stability_mean_pairwise_ari"].rank(ascending=False, method="min")
    out["subsample_rank"] = out["subsample_stability_mean_ari"].rank(ascending=False, method="min")

    out["model_specific_rank_score"] = out["internal_rank_score"].copy()
    gmm_mask = out["model"] == "gmm"
    out.loc[gmm_mask, "model_specific_rank_score"] = (
        out.loc[gmm_mask, "internal_rank_score"]
        + out.loc[gmm_mask, "bic_rank"].fillna(out.loc[gmm_mask, "bic_rank"].max())
        + out.loc[gmm_mask, "aic_rank"].fillna(out.loc[gmm_mask, "aic_rank"].max())
    )

    out["tiny_cluster_penalty"] = (out["min_cluster_size"] < cfg.usable_min_cluster_size).astype(int)
    out["overall_rank_score"] = (
        out["model_specific_rank_score"]
        + out["seed_rank"].fillna(out["seed_rank"].max())
        + out["subsample_rank"].fillna(out["subsample_rank"].max())
        + out["tiny_cluster_penalty"] * 3
    )
    out["usable_candidate"] = out.apply(lambda row: is_candidate_usable(row, cfg), axis=1)

    return out.sort_values(
        ["overall_rank_score", "subsample_stability_mean_ari", "silhouette"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def choose_best_solution(summary_df: pd.DataFrame) -> pd.Series:
    return summary_df.iloc[0]


# =========================================================
# 프로파일 테이블
# =========================================================

def get_top_features_by_cluster(
    profile_z: pd.DataFrame, top_n: int = 3
) -> Dict[int, Dict[str, List[str]]]:
    return {
        int(cl): {
            "high": profile_z.loc[cl].sort_values(ascending=False).head(top_n).index.tolist(),
            "low": profile_z.loc[cl].sort_values(ascending=False).tail(top_n).index.tolist(),
        }
        for cl in profile_z.index
    }


def build_profile_table(
    X_df: pd.DataFrame, labels: np.ndarray,
    id_col: str | None = None, ids: pd.Series | None = None,
) -> pd.DataFrame:
    labeled_df = X_df.copy()
    if ids is not None and id_col is not None:
        labeled_df.insert(0, id_col, ids)
    labeled_df["cluster"] = labels
    profile = labeled_df.drop(columns=[id_col] if id_col is not None else []).groupby("cluster").mean(numeric_only=True)
    if "cluster" in profile.columns:
        profile = profile.drop(columns=["cluster"])
    return profile


def build_labeled_df(
    X_df: pd.DataFrame, labels: np.ndarray,
    id_col: str | None = None, ids: pd.Series | None = None,
) -> pd.DataFrame:
    out = X_df.copy()
    if ids is not None and id_col is not None:
        out.insert(0, id_col, ids)
    out["cluster"] = labels
    return out


def extract_tiny_cluster_members(
    labeled_df: pd.DataFrame, model_name: str, k: int,
    covariance_type: str | None, cfg: "AnalysisConfig",
) -> pd.DataFrame:
    if "cluster" not in labeled_df.columns:
        return pd.DataFrame()

    id_col = find_id_col(labeled_df, cfg)
    counts = labeled_df["cluster"].value_counts()
    tiny_clusters = counts[counts <= cfg.tiny_cluster_threshold].sort_index()

    rows = []
    for cluster_id, n_members in tiny_clusters.items():
        sub = labeled_df[labeled_df["cluster"] == cluster_id].copy()
        member_ids = sub[id_col].tolist() if id_col is not None else sub.index.tolist()
        for member_id in member_ids:
            rows.append({
                "model": model_name, "k": k, "covariance_type": covariance_type,
                "cluster": int(cluster_id), "n_members": int(n_members),
                "member_id": member_id,
            })
    return pd.DataFrame(rows)


# =========================================================
# 보고서 생성
# =========================================================

def write_short_txt_report(
    save_path: Path,
    model_name: str,
    k: int,
    covariance_type: str | None,
    metrics: pd.Series,
    profile_z: pd.DataFrame,
    top_feat_summary: Dict[int, Dict[str, List[str]]],
    feature_discrim_df: pd.DataFrame | None = None,
) -> None:
    lines = []
    cov_text = format_cov_text(covariance_type)

    lines.append("[1. SHORT REPORT RESULT]")
    lines.append(f"- Selected solution: {model_name.upper()} | k={k}{cov_text}")
    lines.append(
        f"- Cluster validity: silhouette={safe_float(metrics['silhouette']):.4f}, "
        f"CH={safe_float(metrics['calinski_harabasz']):.2f}, "
        f"DB={safe_float(metrics['davies_bouldin']):.4f}"
    )
    if pd.notna(metrics.get("bic", np.nan)):
        lines.append(
            f"- Model fit (GMM-specific): "
            f"BIC={safe_float(metrics['bic']):.2f}, AIC={safe_float(metrics['aic']):.2f}"
        )
    lines.append(
        f"- Stability: "
        f"seed_ARI={safe_float(metrics.get('seed_stability_mean_pairwise_ari', np.nan)):.4f}, "
        f"subsample_ARI={safe_float(metrics.get('subsample_stability_mean_ari', np.nan)):.4f}"
    )
    lines.append(f"- Cluster sizes: {metrics['cluster_sizes']}")
    lines.append("")
    lines.append("- Cluster profile summary")
    for cl in profile_z.index:
        high_txt = ", ".join(top_feat_summary[int(cl)]["high"])
        low_txt = ", ".join(top_feat_summary[int(cl)]["low"])
        lines.append(
            f"  - Cluster {int(cl)}: "
            f"relatively high in ({high_txt}) / relatively low in ({low_txt})"
        )

    if feature_discrim_df is not None and not feature_discrim_df.empty:
        lines.append("")
        lines.append("- Feature discriminability (top 5, * = p<.05)")
        for _, row in feature_discrim_df.head(5).iterrows():
            sig = "* " if row.get("significant_p05", False) else "  "
            lines.append(
                f"  {sig}[{int(row['discriminability_rank'])}] {row['feature']}: "
                f"F={safe_float(row['f_statistic']):.2f}, "
                f"eta²={safe_float(row['eta_squared']):.3f} ({row['effect_size']}), "
                f"p={safe_float(row['p_value']):.4f}"
            )

    lines.append("")
    lines.append("[2. METRIC DESCRIPTION]")
    lines.append("- Silhouette: 군집 내 응집도와 군집 간 분리도를 함께 보는 지표이며, 클수록 좋다.")
    lines.append("- Calinski-Harabasz (CH): 군집 간 분산 / 군집 내 분산 비율. 클수록 좋다.")
    lines.append("- Davies-Bouldin (DB): 군집 간 분리 대비 군집 내 퍼짐. 작을수록 좋다.")
    lines.append("- BIC / AIC: GMM 모형 적합도+복잡도 기준. 작을수록 유리하다.")
    lines.append("- Seed ARI: random seed 변화에 대한 안정성. 클수록 좋다.")
    lines.append("- Subsample ARI: 샘플 일부 변경에 대한 구조 유지성. 클수록 좋다.")
    lines.append("- F-statistic: 군집 간 분산/군집 내 분산 비율. 클수록 해당 feature가 군집을 잘 구분함.")
    lines.append("- eta² (효과크기): 0.01=small, 0.06=medium, 0.14 이상=large.")
    lines.append("")
    lines.append("[3. METHODOLOGY DESCRIPTION]")
    lines.append("- Input matrix는 subject-level standardized feature matrix를 사용하였다.")
    lines.append("- K-means와 Ward는 k=2~6 범위에서 비교하였다.")
    lines.append("- GMM은 k=2~6 및 covariance_type(spherical, diag, tied, full)을 함께 탐색하였다.")
    lines.append("- Feature 설명력은 one-way ANOVA F-통계량 및 eta-squared로 계산하였다.")
    lines.append("- 최종 해는 내부지표, 안정성, tiny cluster 여부를 함께 고려하여 선정하였다.")

    save_path.write_text("\n".join(lines), encoding="utf-8-sig")

def write_total_summary_short_report(
    save_path: Path,
    summary_df: pd.DataFrame,
    best_row: pd.Series,
    block_corr_df: pd.DataFrame,
    block_pca_df: pd.DataFrame,
    block_energy_df: pd.DataFrame,
    tiny_cluster_members_df: pd.DataFrame,
    cfg: "AnalysisConfig",
    feature_discrim_df: pd.DataFrame | None = None,
    outlier_df: pd.DataFrame | None = None,
    tiny_cause_df: pd.DataFrame | None = None,
) -> None:
    lines = []

    # =========================================================
    # 0) 10줄 이내 핵심요약
    # =========================================================
    cov_text = format_cov_text(
        best_row["covariance_type"] if pd.notna(best_row["covariance_type"]) else None
    )

    flagged_n = 0
    if outlier_df is not None and not outlier_df.empty and "is_outlier" in outlier_df.columns:
        flagged_n = int((outlier_df["is_outlier"] == True).sum())

    tiny_n = 0
    if tiny_cluster_members_df is not None and not tiny_cluster_members_df.empty:
        tiny_n = int(tiny_cluster_members_df["member_id"].nunique())

    top3_feats = []
    if feature_discrim_df is not None and not feature_discrim_df.empty:
        top3_feats = feature_discrim_df["feature"].head(3).tolist()

    usable_top = int((summary_df.head(cfg.top_summary_n)["usable_candidate"] == True).sum())

    lines.append("[핵심요약]")
    lines.append(f"1) Best solution: {best_row['model']} | k={int(best_row['k'])}{cov_text}")
    lines.append(
        f"2) Validity: sil={safe_float(best_row['silhouette']):.4f}, "
        f"CH={safe_float(best_row['calinski_harabasz']):.2f}, "
        f"DB={safe_float(best_row['davies_bouldin']):.4f}"
    )
    lines.append(
        f"3) Stability: seed_ARI={safe_float(best_row.get('seed_stability_mean_pairwise_ari', np.nan)):.4f}, "
        f"subsample_ARI={safe_float(best_row.get('subsample_stability_mean_ari', np.nan)):.4f}"
    )
    lines.append(f"4) Cluster sizes: {best_row['cluster_sizes']}")
    lines.append(
        f"5) Candidate status: {'USABLE' if bool(best_row.get('usable_candidate', False)) else 'NOT RECOMMENDED'} "
        f"(top{cfg.top_summary_n} usable={usable_top})"
    )
    lines.append(
        f"6) Top discriminating features: {', '.join(top3_feats) if top3_feats else 'N/A'}"
    )
    lines.append(f"7) Outliers flagged: {flagged_n}")
    lines.append(f"8) Tiny-cluster members (unique): {tiny_n}")
    lines.append(
        f"9) Tiny-cluster threshold={cfg.tiny_cluster_threshold}, usable_min_cluster_size={cfg.usable_min_cluster_size}"
    )
    lines.append("10) 아래부터 상세요약")

    # =========================================================
    # 1) 상세요약
    # =========================================================
    lines.append("")
    lines.append("[상세요약]")
    lines.append("")

    # 1) Best solution
    lines.append("1) Best overall solution")
    lines.append(
        f"- {best_row['model']} | k={int(best_row['k'])}{cov_text} | "
        f"sil={safe_float(best_row['silhouette']):.4f}, "
        f"CH={safe_float(best_row['calinski_harabasz']):.2f}, "
        f"DB={safe_float(best_row['davies_bouldin']):.4f}, "
        f"seed_ARI={safe_float(best_row.get('seed_stability_mean_pairwise_ari', np.nan)):.4f}, "
        f"subsample_ARI={safe_float(best_row.get('subsample_stability_mean_ari', np.nan)):.4f}"
    )
    if pd.notna(best_row.get("bic", np.nan)):
        lines.append(f"- BIC={safe_float(best_row['bic']):.2f}, AIC={safe_float(best_row['aic']):.2f}")
    lines.append(f"- cluster_sizes: {best_row['cluster_sizes']}")

    # 2) Top candidates
    lines.append("")
    lines.append("2) Top candidates")
    top_df = summary_df.head(cfg.top_summary_n).copy()
    usable_df = top_df[top_df["usable_candidate"] == True].copy()
    nonusable_df = top_df[top_df["usable_candidate"] == False].copy()

    lines.append("- Usable candidates")
    if usable_df.empty:
        lines.append("  - none")
    else:
        for i, (_, row) in enumerate(usable_df.iterrows(), start=1):
            ct = format_cov_text(row["covariance_type"] if pd.notna(row["covariance_type"]) else None)
            lines.append(
                f"  - #{i}: {row['model']} | k={int(row['k'])}{ct} | "
                f"sil={safe_float(row['silhouette']):.4f}, "
                f"subsample_ARI={safe_float(row['subsample_stability_mean_ari']):.4f}, "
                f"min_size={int(row['min_cluster_size'])}, sizes={row['cluster_sizes']}"
            )

    lines.append("- Not recommended candidates")
    if nonusable_df.empty:
        lines.append("  - none")
    else:
        for i, (_, row) in enumerate(nonusable_df.iterrows(), start=1):
            ct = format_cov_text(row["covariance_type"] if pd.notna(row["covariance_type"]) else None)
            reasons = []
            if int(row["min_cluster_size"]) < cfg.usable_min_cluster_size:
                reasons.append(f"min_cluster_size<{cfg.usable_min_cluster_size}")
            if safe_float(row.get("subsample_stability_mean_ari", np.nan)) < 0.60:
                reasons.append("low_subsample_stability")
            lines.append(
                f"  - #{i}: {row['model']} | k={int(row['k'])}{ct} | "
                f"sil={safe_float(row['silhouette']):.4f}, "
                f"subsample_ARI={safe_float(row['subsample_stability_mean_ari']):.4f}, "
                f"min_size={int(row['min_cluster_size'])}, sizes={row['cluster_sizes']} | "
                f"reason={', '.join(reasons) if reasons else 'low_priority'}"
            )

    # 3) Feature discriminability
    lines.append("")
    lines.append("3) Feature discriminability (best solution 기준, * = p<.05)")
    if feature_discrim_df is None or feature_discrim_df.empty:
        lines.append("- 계산되지 않음")
    else:
        for _, row in feature_discrim_df.iterrows():
            sig = "* " if row.get("significant_p05", False) else "  "
            lines.append(
                f"  {sig}[{int(row['discriminability_rank'])}] {row['feature']}: "
                f"F={safe_float(row['f_statistic']):.2f}, "
                f"eta²={safe_float(row['eta_squared']):.3f} ({row['effect_size']}), "
                f"p={safe_float(row['p_value']):.4f}"
            )

    # 4) Outlier detection
    lines.append("")
    lines.append("4) Outlier detection (best solution 기준)")
    if outlier_df is None or outlier_df.empty:
        lines.append("- 계산되지 않음")
    else:
        flagged = outlier_df[outlier_df["is_outlier"] == True].copy()
        lines.append(f"- 전체 subject 수: {len(outlier_df)} | Outlier 표시: {len(flagged)}명")
        if flagged.empty:
            lines.append("- 탐지된 outlier 없음")
        else:
            for _, row in flagged.iterrows():
                lines.append(
                    f"  - subject={row['subject_id']}, cluster={int(row['cluster'])}, "
                    f"z={safe_float(row['z_within_cluster']):.2f}, "
                    f"dist={safe_float(row['dist_from_centroid']):.3f}"
                )
                lines.append(f"    top deviant: {row['top_deviant_features']}")

    # 5) Tiny cluster 원인
    lines.append("")
    lines.append(f"5) Tiny cluster cause analysis (n <= {cfg.tiny_cluster_threshold})")
    if tiny_cause_df is None or tiny_cause_df.empty:
        lines.append("- tiny cluster 없음 또는 계산되지 않음")
    else:
        for _, row in tiny_cause_df.iterrows():
            lines.append(f"  [Cluster {int(row['cluster'])}] n={int(row['n_members'])}, 원인={row['cause_type']}")
            lines.append(f"  - {row['cause_detail']}")
            lines.append(f"  - 높은 feature: {row['extreme_high_features']}")
            lines.append(f"  - 낮은 feature: {row['extreme_low_features']}")
            lines.append(f"  - 주요 discriminating feature z: {row['top_discrim_feature_zscores']}")
            lines.append("")

    # 6) Block diagnostics
    lines.append("6) Block diagnostics")
    lines.append("- Within-block correlation")
    for _, row in block_corr_df.iterrows():
        lines.append(
            f"  - {row['block']}: n={int(row['n_features'])}, "
            f"mean_abs_corr={safe_float(row['mean_abs_corr_within_block']):.3f}, "
            f"max_abs_corr={safe_float(row['max_abs_corr_within_block']):.3f}"
        )
    lines.append("- Block PCA")
    for _, row in block_pca_df.iterrows():
        lines.append(
            f"  - {row['block']}: n={int(row['n_features'])}, "
            f"PC1={safe_float(row['pc1_explained_variance_ratio']):.3f}, "
            f"PC2={safe_float(row['pc2_explained_variance_ratio']):.3f}"
        )
    lines.append("- Block energy")
    for _, row in block_energy_df.iterrows():
        lines.append(
            f"  - {row['block']}: n={int(row['n_features'])}, "
            f"mean_L2={safe_float(row['mean_l2_norm']):.3f}, "
            f"sd_L2={safe_float(row['sd_l2_norm']):.3f}"
        )

    # 7) Tiny cluster members
    lines.append("")
    lines.append(f"7) Tiny cluster members (n <= {cfg.tiny_cluster_threshold})")
    if tiny_cluster_members_df.empty:
        lines.append("- none")
    else:
        summary_tiny = (
            tiny_cluster_members_df
            .groupby(["model", "k", "covariance_type", "cluster", "n_members"], dropna=False)["member_id"]
            .apply(list).reset_index()
        )
        for _, row in summary_tiny.iterrows():
            ct = format_cov_text(row["covariance_type"] if pd.notna(row["covariance_type"]) else None)
            lines.append(
                f"- {row['model']} | k={int(row['k'])}{ct} | "
                f"cluster={int(row['cluster'])} | n={int(row['n_members'])} | members={row['member_id']}"
            )

    # 8) Interpretation caution
    lines.append("")
    lines.append("8) Interpretation caution")
    lines.append("- best solution은 상대적 최적해이지 절대적 진실이 아님")
    lines.append("- block imbalance / feature redundancy 가능성은 계속 점검 필요")
    lines.append("- tiny cluster는 실제 subgroup보다 outlier 흡수일 수 있음")
    lines.append("- 낮은 설명력 feature는 해석 비중을 낮출 것")

    save_path.write_text("\n".join(lines), encoding="utf-8-sig")