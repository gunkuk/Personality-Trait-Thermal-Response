# 핵심 요약
#   2) 환경값을 정규화하고 분석 단위(person / scenario)에 맞는 feature matrix를 구성한다.
#   3) 결측치 대체, Robust Scaling, PCA를 적용해 군집에 적합한 입력 공간을 만든다.
#   4) GMM을 여러 설정으로 탐색하여 BIC/AIC와 내부 평가지표를 기준으로 최적 군집 모델을 선택한다.
#   5) 반복 초기화 기반 안정성(ARI)을 계산하고, 최종 군집표·평가지표·2D 시각화·프로파일 그림을 출력한다.다.


# mode 요약
# - person: 참가자 단위 군집화
# - scenario_dynamic: dynamic 시나리오 구간만 군집화
# - scenario_static: static 시나리오 구간만 군집화
# - scenario_all: 전체 시나리오 구간 군집화


# 데이터 / feature set 요약
#   1) F0: 전체 생리 feature(ST + EDA + ECG)
#   2) F1: delta_dynamic_minus_static가 포함된 생리 변화량 feature 우선 사용, 없으면 F0로 대체
#   3) F2: 피부온/온도 관련 블록(st_, at_)만 사용
#   4) F3: ECG/HRV 관련 블록(HR_, MeanRR_, SDNN_, RMSSD_, LF_, HF_, LF_HF_)만 사용
#   5) F4: EDA 관련 블록(scr_, tonic_, phasic_, eda_, log1p_)만 사용
#   6) F5: 생리 delta feature + 주관설문 delta feature를 결합한 융합 feature set

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

import matplotlib.pyplot as plt

try:
    import umap  # umap-learn
except Exception:
    umap = None

try:
    import hdbscan
except Exception:
    hdbscan = None


DEFAULT_PHY_PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT\PHY_processed.xlsx"
DEFAULT_BI_PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\data\BI+SUR.xlsx"
DEFAULT_OUT_ROOT = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT\CLS"


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    phy_path: str
    bi_path: str
    out_dir: str = DEFAULT_OUT_ROOT
    mode: str = "person"  # person | scenario_dynamic | scenario_static | scenario_all
    feature_set: str = "F1"  # F0..F5
    seed: int = 42

    # preprocessing
    impute_strategy: str = "median"
    use_pca: bool = True
    pca_n_components: int = 20  # or -1 for "min(n_samples-1, n_features)"

    # GMM search
    k_min: int = 2
    k_max: int = 8
    covariance_types: Tuple[str, ...] = ("diag", "tied", "full")
    reg_covar: float = 1e-6
    n_init: int = 10
    max_iter: int = 1000

    # HDBSCAN fallback
    hdb_min_cluster_size: int = 10
    hdb_min_samples: Optional[int] = None

    # stability
    stability_runs: int = 20

    # visualization
    top_profile_features: int = 25


FULLWIDTH = str.maketrans(
    {"Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F", "Ｇ": "G", "Ｈ": "H"}
)

SEG_TO_PAIR = {"A": "AB", "B": "AB", "C": "CD", "D": "CD", "E": "EF", "F": "EF", "G": "GH", "H": "GH"}
SEG_TO_PHASE = {
    "A": "static",
    "B": "dynamic",
    "C": "static",
    "D": "dynamic",
    "E": "static",
    "F": "dynamic",
    "G": "static",
    "H": "dynamic",
}

EXPECTED_SCENARIO = {
    "A": ("AB", "static", "1to10"),
    "B": ("AB", "dynamic", "11to20"),
    "C": ("CD", "static", "1to10"),
    "D": ("CD", "dynamic", "11to20"),
    "E": ("EF", "static", "1to10"),
    "F": ("EF", "dynamic", "11to20"),
    "G": ("GH", "static", "1to10"),
    "H": ("GH", "dynamic", "11to20"),
}

ID_COLS_PERSON = ["연번"]

SURVEY_COLS = ["TSV", "TCV", "TA", "TP", "PT"] + [f"P{i}" for i in range(1, 9)] + [f"M{i}" for i in range(1, 9)]
PERSONALITY_COLS = ["BMI", "나이", "성별", "e", "n", "t", "j", "a", "o1", "c1", "e1", "a1", "n1"]


def normalize_env(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).translate(FULLWIDTH).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def first_nonnull(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


# ----------------------------
# Loaders
# ----------------------------
def load_phy_summaries(phy_path: str) -> pd.DataFrame:
    st = pd.read_excel(phy_path, sheet_name="ST_processed")
    eda = pd.read_excel(phy_path, sheet_name="EDA_processed")
    ecg = pd.read_excel(phy_path, sheet_name="ECG_processed")

    st = st[st["연번"].notna()].copy()

    for df in (st, eda, ecg):
        df["환경"] = df["환경"].apply(normalize_env)

    st = st[st["환경"].isin(["AB", "CD", "EF", "GH"])].copy()
    eda = eda[eda["환경"].isin(["AB", "CD", "EF", "GH"])].copy()
    ecg = ecg[ecg["환경"].isin(["AB", "CD", "EF", "GH"])].copy()

    m = st.merge(eda, on=["연번", "이름", "성별", "MBTI", "환경"], how="outer")
    m = m.merge(ecg, on=["연번", "이름", "환경"], how="outer", suffixes=("", "_ecg"))
    return m


def load_phy_scenarios(phy_path: str) -> pd.DataFrame:
    dfs = []
    for seg in list("ABCDEFGH"):
        df = pd.read_excel(phy_path, sheet_name=seg)
        df["환경"] = df["환경"].apply(normalize_env)
        df["원본환경"] = df["원본환경"].astype(str).str.strip()

        pair, phase, window = EXPECTED_SCENARIO[seg]
        df = df[
            (df["원본환경"] == pair)
            & (df["시나리오유형"] == phase)
            & (df["분할구간"] == window)
        ].copy()

        df["segment"] = seg
        df["pair"] = pair
        df["phase"] = phase
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_bi_sur(bi_path: str) -> Dict[str, pd.DataFrame]:
    return {
        "PI+SUR": pd.read_excel(bi_path, sheet_name="PI+SUR"),
        "인지능력": pd.read_excel(bi_path, sheet_name="인지능력"),
        "사후설문": pd.read_excel(bi_path, sheet_name="사후설문"),
    }


# ----------------------------
# Feature builders
# ----------------------------
def build_personality_table(pi_sur: pd.DataFrame) -> pd.DataFrame:
    df = pi_sur.copy()
    df["환경_norm"] = df["환경"].apply(normalize_env)
    df["성별"] = df["성별"].astype(str).str.strip()

    for c in ["BMI", "나이", "e", "n", "t", "j", "a", "o1", "c1", "e1", "a1", "n1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = df.groupby("연번").agg({c: first_nonnull for c in (["이름"] + PERSONALITY_COLS)}).reset_index()
    return agg


def build_subjective_table(pi_sur: pd.DataFrame) -> pd.DataFrame:
    df = pi_sur.copy()
    df["segment"] = df["환경"].apply(normalize_env)
    df = df[df["segment"].isin(list(SEG_TO_PAIR.keys()))].copy()

    df["pair"] = df["segment"].map(SEG_TO_PAIR)
    df["phase"] = df["segment"].map(SEG_TO_PHASE)

    for c in SURVEY_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = df.groupby(["연번", "pair", "phase"])[SURVEY_COLS].mean().reset_index()
    wide = agg.pivot_table(index=["연번", "pair"], columns="phase", values=SURVEY_COLS, aggfunc="mean")

    static = (
        wide.xs("static", level="phase", axis=1, drop_level=True)
        if "static" in wide.columns.get_level_values("phase")
        else pd.DataFrame(index=wide.index)
    )
    dynamic = (
        wide.xs("dynamic", level="phase", axis=1, drop_level=True)
        if "dynamic" in wide.columns.get_level_values("phase")
        else pd.DataFrame(index=wide.index)
    )

    cols = sorted(set(static.columns).union(dynamic.columns))
    static = static.reindex(columns=cols)
    dynamic = dynamic.reindex(columns=cols)

    delta = dynamic - static

    static.columns = [f"{c}_static_mean" for c in static.columns]
    dynamic.columns = [f"{c}_dynamic_mean" for c in dynamic.columns]
    delta.columns = [f"{c}_delta_dynamic_minus_static" for c in delta.columns]

    pair_level = pd.concat([static, dynamic, delta], axis=1).reset_index()
    feat_cols = [c for c in pair_level.columns if c not in ["연번", "pair"]]
    person = pair_level.groupby("연번")[feat_cols].mean().reset_index()
    return person


def build_person_level_phy(phy_sum: pd.DataFrame) -> pd.DataFrame:
    df = phy_sum.copy()
    df["환경"] = df["환경"].apply(normalize_env)
    df = df[df["환경"].isin(["AB", "CD", "EF", "GH"])].copy()

    drop_cols = ["이름", "성별", "MBTI", "환경"]
    feat_cols = [c for c in df.columns if c not in (["연번"] + drop_cols)]

    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    person = df.groupby("연번")[feat_cols].mean().reset_index()
    return person


def build_scenario_level(phy_scen: pd.DataFrame, phase_filter: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = phy_scen.copy()
    if phase_filter in ("static", "dynamic"):
        df = df[df["phase"] == phase_filter].copy()

    id_cols = ["연번", "segment", "pair", "phase"]
    feat_cols = [
        c
        for c in df.columns
        if c not in id_cols and c not in ["이름", "성별", "MBTI", "환경", "원본환경", "분할구간", "시나리오유형"]
    ]

    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    meta = df[id_cols].copy()
    X = df[feat_cols].copy()
    return X, meta


def get_feature_sets(columns: List[str]) -> Dict[str, List[str]]:
    cols = columns

    def pick(prefixes: Tuple[str, ...]) -> List[str]:
        return [c for c in cols if c.startswith(prefixes)]

    phy_all = pick(
        (
            "st_",
            "at_",
            "scr_",
            "tonic_",
            "phasic_",
            "eda_",
            "log1p_",
            "HR_",
            "MeanRR_",
            "SDNN_",
            "RMSSD_",
            "LF_",
            "HF_",
            "LF_HF_",
        )
    )
    phy_delta = [c for c in phy_all if "delta_dynamic_minus_static" in c]

    st_block = pick(("st_", "at_"))
    ecg_block = pick(("HR_", "MeanRR_", "SDNN_", "RMSSD_", "LF_", "HF_", "LF_HF_"))
    eda_block = pick(("scr_", "tonic_", "phasic_", "eda_", "log1p_"))

    return {
        "F0": phy_all,
        "F1": phy_delta if len(phy_delta) else phy_all,
        "F2": st_block,
        "F3": ecg_block,
        "F4": eda_block,
        "F5": phy_delta,
    }


# ----------------------------
# Modeling + evaluation
# ----------------------------
def make_preprocessor(cfg: Config, n_samples: int, n_features: int):
    imputer = SimpleImputer(strategy=cfg.impute_strategy)
    scaler = RobustScaler()

    pca = None
    if cfg.use_pca:
        if cfg.pca_n_components == -1:
            n_comp = max(2, min(n_samples - 1, n_features))
        else:
            n_comp = max(2, min(cfg.pca_n_components, n_samples - 1, n_features))
        pca = PCA(n_components=n_comp, random_state=cfg.seed)

    return imputer, scaler, pca


def fit_gmm_grid(Z: np.ndarray, cfg: Config) -> Tuple[GaussianMixture, pd.DataFrame]:
    rows = []
    best = None
    best_bic = np.inf

    for cov in cfg.covariance_types:
        for k in range(cfg.k_min, cfg.k_max + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    reg_covar=cfg.reg_covar,
                    random_state=cfg.seed,
                    n_init=cfg.n_init,
                    max_iter=cfg.max_iter,
                    init_params="kmeans",
                ).fit(Z)

                labels = gmm.predict(Z)

                if len(np.unique(labels)) < 2:
                    continue

                sil = silhouette_score(Z, labels)
                ch = calinski_harabasz_score(Z, labels)
                db = davies_bouldin_score(Z, labels)
                bic = gmm.bic(Z)
                aic = gmm.aic(Z)

                rows.append(
                    {
                        "model": "GMM",
                        "cov": cov,
                        "k": k,
                        "silhouette": sil,
                        "calinski_harabasz": ch,
                        "davies_bouldin": db,
                        "bic": bic,
                        "aic": aic,
                    }
                )

                if bic < best_bic:
                    best_bic = bic
                    best = gmm
            except Exception:
                continue

    if best is None:
        raise RuntimeError("GMM grid search failed for all settings. Try PCA, diag covariance, or fewer features.")

    return best, pd.DataFrame(rows).sort_values(["bic", "aic"], ascending=True)


def stability_by_restarts(fit_fn, Z: np.ndarray, cfg: Config, n_runs: int) -> Dict[str, float]:
    labels_list = []
    for i in range(n_runs):
        seed = cfg.seed + 1000 + i
        labels = fit_fn(Z, seed)
        labels_list.append(labels)

    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    aris = np.array(aris) if len(aris) else np.array([np.nan])
    return {
        "ari_mean": float(np.nanmean(aris)),
        "ari_std": float(np.nanstd(aris)),
        "ari_min": float(np.nanmin(aris)),
        "ari_max": float(np.nanmax(aris)),
        "pairs": int(np.sum(~np.isnan(aris))),
    }


def plot_umap_or_pca(Z: np.ndarray, labels: np.ndarray, cfg: Config, out_path: str):
    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=cfg.seed, n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(Z)
        title = "UMAP (2D)"
    else:
        emb = PCA(n_components=2, random_state=cfg.seed).fit_transform(Z)
        title = "PCA (2D) — UMAP not installed"

    plt.figure()
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=25, alpha=0.85)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cluster_profile(X_scaled: np.ndarray, labels: np.ndarray, feature_names: List[str], cfg: Config, out_path: str):
    df = pd.DataFrame(X_scaled, columns=feature_names)
    df["cluster"] = labels

    means = df.groupby("cluster").mean(numeric_only=True)
    var = means.var(axis=0).sort_values(ascending=False)
    top = var.head(cfg.top_profile_features).index.tolist()

    plt.figure(figsize=(12, 4 + 0.4 * len(means)))
    mat = means[top].to_numpy()
    plt.imshow(mat, aspect="auto")
    plt.yticks(range(len(means.index)), [f"C{c}" for c in means.index])
    plt.xticks(range(len(top)), top, rotation=90)
    plt.title("Cluster mean profile (scaled)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# 핵심: 군집별 feature 평균과 전체 평균 대비 차이를 계산하고, 사람이 읽을 수 있는 군집 해석 문장을 생성한다.
def summarize_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    out_dir: str,
    top_n: int = 5,
):
    if X_scaled.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: X_scaled has {X_scaled.shape[1]} columns, "
            f"but feature_names has {len(feature_names)}."
        )

    df = pd.DataFrame(X_scaled, columns=feature_names)
    df["cluster"] = labels

    # 군집별 평균
    cluster_means = df.groupby("cluster").mean(numeric_only=True)
    cluster_sizes = df["cluster"].value_counts().sort_index()

    # 전체 평균 대비 차이
    global_mean = df[feature_names].mean(axis=0)
    zdiff = cluster_means.sub(global_mean, axis=1)

    # 저장
    means_path = os.path.join(out_dir, "cluster_feature_means.csv")
    zdiff_path = os.path.join(out_dir, "cluster_feature_zdiff.csv")
    summary_path = os.path.join(out_dir, "cluster_interpretation.txt")

    cluster_means.to_csv(means_path, encoding="utf-8-sig")
    zdiff.to_csv(zdiff_path, encoding="utf-8-sig")

    # 텍스트 요약 생성
    lines = []
    lines.append("Cluster interpretation summary")
    lines.append("=" * 80)

    for c in cluster_means.index:
        row = zdiff.loc[c].sort_values(ascending=False)
        high_feats = row.head(top_n)
        low_feats = row.tail(top_n)

        lines.append(f"\n[Cluster {c}]")
        lines.append(f"- size: {int(cluster_sizes.loc[c])}")

        high_str = ", ".join([f"{feat} ({val:+.3f})" for feat, val in high_feats.items()])
        low_str = ", ".join([f"{feat} ({val:+.3f})" for feat, val in low_feats.items()])

        lines.append(f"- relatively high features: {high_str}")
        lines.append(f"- relatively low features: {low_str}")

        # 간단 해석 문장
        lines.append(
            f"- interpretation: Cluster {c} is characterized by relatively elevated "
            f"{', '.join(high_feats.index[:3])} and relatively reduced "
            f"{', '.join(low_feats.index[:3])}."
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "means_path": means_path,
        "zdiff_path": zdiff_path,
        "summary_path": summary_path,
    }

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phy",
        default=DEFAULT_PHY_PATH,
        help="Path to PHY_processed.xlsx",
    )
    parser.add_argument(
        "--bi",
        default=DEFAULT_BI_PATH,
        help="Path to BI+SUR.xlsx",
    )
    parser.add_argument(
        "--mode",
        default="person",
        choices=["person", "scenario_dynamic", "scenario_static", "scenario_all"],
    )
    parser.add_argument("--feature_set", default="F1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_ROOT,
        help="Root output directory. Results will be saved under <out_dir>/<feature_set>/",
    )

    args = parser.parse_args()

    # feature_set + mode별 출력 폴더 자동 생성
    base_out_dir = args.out_dir
    run_out_dir = os.path.join(base_out_dir, args.feature_set, args.mode)

    cfg = Config(
        phy_path=args.phy,
        bi_path=args.bi,
        out_dir=run_out_dir,
        mode=args.mode,
        feature_set=args.feature_set,
        seed=args.seed,
    )
    ensure_out_dir(cfg.out_dir)

    print("PHY path:", cfg.phy_path)
    print("BI path :", cfg.bi_path)
    print("OUT dir :", cfg.out_dir)

    phy_sum = load_phy_summaries(cfg.phy_path)
    bi = load_bi_sur(cfg.bi_path)
    pi_sur = bi["PI+SUR"]

    phy_person = build_person_level_phy(phy_sum)
    sur_person = build_subjective_table(pi_sur)
    per_personality = build_personality_table(pi_sur)

    if cfg.mode == "person":
        df = (
            phy_person
            .merge(sur_person, on="연번", how="left")
            .merge(per_personality, on="연번", how="left", suffixes=("", "_bio"))
        )

        phy_feature_sets = get_feature_sets([c for c in phy_person.columns if c != "연번"])
        if cfg.feature_set not in phy_feature_sets and cfg.feature_set != "F5":
            raise ValueError(f"Unknown feature_set: {cfg.feature_set}")

        if cfg.feature_set == "F5":
            phy_delta = phy_feature_sets["F1"]
            sur_delta = [c for c in sur_person.columns if "delta_dynamic_minus_static" in c]
            features = phy_delta + sur_delta
        else:
            features = phy_feature_sets[cfg.feature_set]

        if len(features) == 0:
            raise ValueError(f"No features selected for feature_set={cfg.feature_set}. Check column naming.")

        X = df[features].copy()
        meta = df[["연번"]].copy()

    else:
        phy_scen = load_phy_scenarios(cfg.phy_path)
        phase = None
        if cfg.mode == "scenario_dynamic":
            phase = "dynamic"
        elif cfg.mode == "scenario_static":
            phase = "static"

        X, meta = build_scenario_level(phy_scen, phase_filter=phase)
        features = X.columns.tolist()

        if len(features) == 0:
            raise ValueError(f"No scenario-level features found for mode={cfg.mode}.")

    imputer, scaler, pca = make_preprocessor(cfg, n_samples=X.shape[0], n_features=X.shape[1])
    X_imp = imputer.fit_transform(X)
    X_scl = scaler.fit_transform(X_imp)
    Z = pca.fit_transform(X_scl) if pca is not None else X_scl

    best_gmm, grid_df = fit_gmm_grid(Z, cfg)
    labels = best_gmm.predict(Z)

    def fit_fn(z, seed):
        g = GaussianMixture(
            n_components=best_gmm.n_components,
            covariance_type=best_gmm.covariance_type,
            reg_covar=cfg.reg_covar,
            random_state=seed,
            n_init=1,
            max_iter=cfg.max_iter,
            init_params="kmeans",
        ).fit(z)
        return g.predict(z)

    stab = stability_by_restarts(fit_fn, Z, cfg, cfg.stability_runs)

    metrics = {
        "mode": cfg.mode,
        "feature_set": cfg.feature_set,
        "n_samples": int(Z.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_used": int(Z.shape[1]),
        "gmm_k": int(best_gmm.n_components),
        "gmm_cov": best_gmm.covariance_type,
        "silhouette": float(silhouette_score(Z, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
        "bic": float(best_gmm.bic(Z)),
        "aic": float(best_gmm.aic(Z)),
        "cluster_sizes": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "stability": stab,
        "config": asdict(cfg),
    }

    grid_path = os.path.join(cfg.out_dir, "gmm_grid.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")

    clusters_path = os.path.join(cfg.out_dir, "clusters.csv")
    out = meta.copy()
    out["cluster"] = labels
    out.to_csv(clusters_path, index=False, encoding="utf-8-sig")

    metrics_path = os.path.join(cfg.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fig_umap = os.path.join(cfg.out_dir, "fig_umap.png")
    plot_umap_or_pca(Z, labels, cfg, fig_umap)

    fig_profile = os.path.join(cfg.out_dir, "fig_profile.png")
    plot_cluster_profile(X_scl, labels, feature_names=features, cfg=cfg, out_path=fig_profile)

    # 군집 해석용 테이블/텍스트 저장
    summary_paths = summarize_clusters(
        X_scaled=X_scl,
        labels=labels,
        feature_names=features,
        out_dir=cfg.out_dir,
        top_n=5,
    )

    print("DONE")
    print(f"- grid: {grid_path}")
    print(f"- clusters: {clusters_path}")
    print(f"- metrics: {metrics_path}")
    print(f"- umap: {fig_umap}")
    print(f"- profile: {fig_profile}")
    print(f"- cluster means: {summary_paths['means_path']}")
    print(f"- cluster zdiff: {summary_paths['zdiff_path']}")
    print(f"- cluster summary: {summary_paths['summary_path']}")


if __name__ == "__main__":
    main()