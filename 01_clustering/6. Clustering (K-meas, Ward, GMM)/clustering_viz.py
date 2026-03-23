# 핵심 요약:
# - clustering 결과 시각화 전용 모듈이다. 분석 로직은 포함하지 않는다.
# - 모델 비교 지표, GMM BIC/AIC, 군집 프로파일 heatmap, distance/co-clustering heatmap, PCA scatter를 생성한다.
# - 모든 함수는 save_path를 인자로 받아 파일을 직접 저장하며, run_clustering.py에서 폴더를 지정해서 호출한다.
# - 이 파일 단독으로는 실행되지 않으며, run_clustering.py가 진입점이다.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def plot_metric_comparison(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for model_name in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_name].copy()

        if model_name == "gmm":
            rep = (
                sub.sort_values(["bic", "aic", "silhouette"], ascending=[True, True, False])
                .groupby("k", as_index=False)
                .first()
                .sort_values("k")
            )
            axes[0].plot(rep["k"], rep["silhouette"], marker="o", label=f"{model_name}(best-cov)")
            axes[1].plot(rep["k"], rep["calinski_harabasz"], marker="o", label=f"{model_name}(best-cov)")
            axes[2].plot(rep["k"], rep["davies_bouldin"], marker="o", label=f"{model_name}(best-cov)")
        else:
            sub = sub.sort_values("k")
            axes[0].plot(sub["k"], sub["silhouette"], marker="o", label=model_name)
            axes[1].plot(sub["k"], sub["calinski_harabasz"], marker="o", label=model_name)
            axes[2].plot(sub["k"], sub["davies_bouldin"], marker="o", label=model_name)

    axes[0].set_title("Silhouette")
    axes[1].set_title("Calinski-Harabasz")
    axes[2].set_title("Davies-Bouldin")

    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_gmm_bic_aic_grid(summary_df: pd.DataFrame, save_path: Path) -> None:
    gmm_df = summary_df[summary_df["model"] == "gmm"].copy()
    if gmm_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for cov in gmm_df["covariance_type"].dropna().unique():
        sub = gmm_df[gmm_df["covariance_type"] == cov].sort_values("k")
        axes[0].plot(sub["k"], sub["bic"], marker="o", label=cov)
        axes[1].plot(sub["k"], sub["aic"], marker="o", label=cov)

    axes[0].set_title("GMM BIC by covariance_type")
    axes[1].set_title("GMM AIC by covariance_type")

    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cluster_profile_heatmap(profile_z: pd.DataFrame, title: str, save_path: Path) -> None:
    data = profile_z.values
    fig, ax = plt.subplots(
        figsize=(max(8, profile_z.shape[1] * 0.7), max(3, profile_z.shape[0] * 0.8))
    )

    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(profile_z.shape[1]))
    ax.set_yticks(np.arange(profile_z.shape[0]))
    ax.set_xticklabels(profile_z.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"C{idx}" for idx in profile_z.index], fontsize=10)
    ax.set_title(title)

    for i in range(profile_z.shape[0]):
        for j in range(profile_z.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_distance_heatmap(
    X: np.ndarray, labels: np.ndarray, save_path: Path,
    title: str = "Distance heatmap",
) -> None:
    order = np.argsort(labels)
    X_ord = X[order]
    labels_ord = labels[order]
    D = pairwise_distances(X_ord, metric="euclidean")

    plt.figure(figsize=(7, 6))
    im = plt.imshow(D, aspect="auto")
    plt.title(title)
    plt.xlabel("Samples (ordered by cluster)")
    plt.ylabel("Samples (ordered by cluster)")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    boundaries = []
    prev = labels_ord[0]
    for i, lab in enumerate(labels_ord):
        if lab != prev:
            boundaries.append(i - 0.5)
            prev = lab
    for b in boundaries:
        plt.axhline(b, linewidth=1)
        plt.axvline(b, linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_coclustering_heatmap(
    coclust: np.ndarray, labels: np.ndarray, save_path: Path,
    title: str = "Co-clustering heatmap",
) -> None:
    order = np.argsort(labels)
    M = coclust[order][:, order]
    labels_ord = labels[order]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(M, aspect="auto", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Samples (ordered by cluster)")
    plt.ylabel("Samples (ordered by cluster)")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    boundaries = []
    prev = labels_ord[0]
    for i, lab in enumerate(labels_ord):
        if lab != prev:
            boundaries.append(i - 0.5)
            prev = lab
    for b in boundaries:
        plt.axhline(b, linewidth=1)
        plt.axvline(b, linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_pca_scatter(
    X: np.ndarray, labels: np.ndarray, title: str, save_path: Path
) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    exp1 = pca.explained_variance_ratio_[0] * 100
    exp2 = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=50, alpha=0.85)
    plt.xlabel(f"PC1 ({exp1:.1f}%)")
    plt.ylabel(f"PC2 ({exp2:.1f}%)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

    return pd.DataFrame({"pc1": Z[:, 0], "pc2": Z[:, 1], "cluster": labels})
