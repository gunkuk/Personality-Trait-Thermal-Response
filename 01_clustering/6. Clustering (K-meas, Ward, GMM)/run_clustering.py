# 요약:
# - YAML config를 읽어 subject-level clustering 실험을 실행하는 runner.
# - 결과는 항상 프로젝트 루트의 03_outputs/reports/clustering/<timestamp>__<experiment_name>/ 아래 저장.
# - 실제 사용된 config(used_config.yaml)와 실행 메타데이터(run_metadata.json)를 함께 저장해 추적 가능하게 한다.
# - 분석 로직은 clustering_analysis.py, 시각화는 clustering_viz.py에 위임한다.

from __future__ import annotations

import argparse
import json
import hashlib
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

# 이 파일이 어느 디렉토리에서 실행되더라도 같은 폴더의 모듈을 찾을 수 있도록 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clustering_analysis import (
    ModelResult,
    attach_stability_to_summary,
    build_labeled_df,
    build_profile_table,
    choose_best_solution,
    compute_coclustering_matrix,
    compute_feature_discriminability,
    detect_outliers_by_centroid_distance,
    ensure_numeric_matrix,
    explain_tiny_cluster_causes,
    extract_tiny_cluster_members,
    filter_blocks_to_existing_columns,
    find_id_col,
    get_top_features_by_cluster,
    load_input_dataframe,
    results_to_df,
    run_gmm_single,
    run_kmeans,
    run_ward,
    safe_float,
    summarize_block_correlations,
    summarize_block_energy,
    summarize_block_pca,
    write_short_txt_report,
    write_total_summary_short_report,
)
from clustering_viz import (
    plot_cluster_profile_heatmap,
    plot_coclustering_heatmap,
    plot_distance_heatmap,
    plot_gmm_bic_aic_grid,
    plot_metric_comparison,
    plot_pca_scatter,
)


# =========================================================
# 기본 설정 dataclass
# =========================================================

@dataclass(frozen=True)
class AnalysisConfig:
    id_col_candidates: List[str] = field(
        default_factory=lambda: ["id", "ID", "subject_id", "subj_id"]
    )
    seed_stability_runs: int = 30
    subsample_stability_runs: int = 100
    subsample_ratio: float = 0.8
    tiny_cluster_threshold: int = 6
    usable_min_cluster_size: int = 10
    top_summary_n: int = 10
    outlier_z_threshold: float = 2.5
    blocks: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "PHY": [
                "hr_dtst_env_sd",
                "tonic_dtst_env_sd",
                "pulse_amplitude_dtst_env_sd",
                "skt_dtst_env_sd",
            ],
            "PSY": [
                "tsv_20minus10_env_sd",
                "m2_20minus10_env_sd",
            ],
            "BHR": [
                "p7_minus_m7_overall_mean",
                "p7_minus_m7_env_sd",
            ],
            "TLX": [
                "tlx1_dtst_env_sd",
            ],
            "EUP": [
                "eup5",
                "eup16",
            ],
        }
    )


@dataclass(frozen=True)
class RunConfig:
    # 현재 파일:
    # MBTI/01_model/6. Clustering (K-meas, Ward, GMM)/run_clustering.py
    # -> parents[2] = MBTI 프로젝트 루트
    base_dir: Path = Path(__file__).resolve().parents[2]

    input_path: Path | None = None
    out_dir: Path | None = None
    sheet_name: str | None = None

    # list로 두어 YAML에서 비연속 k도 받을 수 있게 한다.
    k_values: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    gmm_covariance_types: List[str] = field(
        default_factory=lambda: ["spherical", "diag", "tied", "full"]
    )

    random_state: int = 42
    assume_already_standardized: bool = True
    save_pca_auxiliary: bool = True

    experiment_name: str = "manual"
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def __post_init__(self):
        if self.input_path is None:
            object.__setattr__(
                self,
                "input_path",
                self.base_dir / "00_data" / "02_processed" / "input_matrix_standardized.xlsx",
            )

        if self.out_dir is None:
            # 기본 실행 시에도 OUT 폴더가 아니라 03_outputs 아래로 간다.
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{self.experiment_name}"
            object.__setattr__(
                self,
                "out_dir",
                self.base_dir / "03_outputs" / run_name,
            )


# =========================================================
# Config 유틸
# =========================================================

def _default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_yaml_dict() -> dict:
    base_dir = _default_base_dir()
    _ = base_dir  # 의도 명시용

    return {
        "experiment_name": "default",
        "run": {
            "input_path": "00_data/02_processed/input_matrix_standardized.xlsx",
            "out_base_dir": "03_outputs",
            "sheet_name": None,
            "random_state": 42,
            "assume_already_standardized": True,
            "save_pca_auxiliary": True,
        },
        "model": {
            "k_values": [2, 3, 4, 5, 6],
            "gmm_covariance_types": ["spherical", "diag", "tied", "full"],
        },
        "analysis": {
            "id_col_candidates": ["id", "ID", "subject_id", "subj_id"],
            "seed_stability_runs": 30,
            "subsample_stability_runs": 100,
            "subsample_ratio": 0.8,
            "tiny_cluster_threshold": 6,
            "usable_min_cluster_size": 10,
            "top_summary_n": 10,
            "outlier_z_threshold": 2.5,
        },
        "blocks": {
            "PHY": [
                "hr_dtst_env_sd",
                "tonic_dtst_env_sd",
                "pulse_amplitude_dtst_env_sd",
                "skt_dtst_env_sd",
            ],
            "PSY": [
                "tsv_20minus10_env_sd",
                "m2_20minus10_env_sd",
            ],
            "BHR": [
                "p7_minus_m7_overall_mean",
                "p7_minus_m7_env_sd",
            ],
            "TLX": [
                "tlx1_dtst_env_sd",
            ],
            "EUP": [
                "eup5",
                "eup16",
            ],
        },
    }


def deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def config_to_dict(cfg: RunConfig) -> dict:
    return {
        "experiment_name": cfg.experiment_name,
        "run": {
            "base_dir": str(cfg.base_dir),
            "input_path": str(cfg.input_path),
            "out_dir": str(cfg.out_dir),
            "sheet_name": cfg.sheet_name,
            "random_state": cfg.random_state,
            "assume_already_standardized": cfg.assume_already_standardized,
            "save_pca_auxiliary": cfg.save_pca_auxiliary,
        },
        "model": {
            "k_values": list(cfg.k_values),
            "gmm_covariance_types": list(cfg.gmm_covariance_types),
        },
        "analysis": {
            "id_col_candidates": list(cfg.analysis.id_col_candidates),
            "seed_stability_runs": cfg.analysis.seed_stability_runs,
            "subsample_stability_runs": cfg.analysis.subsample_stability_runs,
            "subsample_ratio": cfg.analysis.subsample_ratio,
            "tiny_cluster_threshold": cfg.analysis.tiny_cluster_threshold,
            "usable_min_cluster_size": cfg.analysis.usable_min_cluster_size,
            "top_summary_n": cfg.analysis.top_summary_n,
            "outlier_z_threshold": cfg.analysis.outlier_z_threshold,
        },
        "blocks": cfg.analysis.blocks,
    }


def build_config_from_yaml(config_path: Path) -> tuple[RunConfig, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        user_raw = yaml.safe_load(f) or {}

    merged = deep_update(default_yaml_dict(), user_raw)

    base_dir = _default_base_dir()
    experiment_name = merged.get("experiment_name", config_path.stem)

    run_raw = merged["run"]
    model_raw = merged["model"]
    analysis_raw = merged["analysis"]
    blocks_raw = merged["blocks"]

    input_path = (base_dir / run_raw["input_path"]).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}__{experiment_name}"
    out_dir = ((base_dir / run_raw["out_base_dir"]).resolve() / run_name)

    analysis_cfg = AnalysisConfig(
        id_col_candidates=analysis_raw["id_col_candidates"],
        seed_stability_runs=int(analysis_raw["seed_stability_runs"]),
        subsample_stability_runs=int(analysis_raw["subsample_stability_runs"]),
        subsample_ratio=float(analysis_raw["subsample_ratio"]),
        tiny_cluster_threshold=int(analysis_raw["tiny_cluster_threshold"]),
        usable_min_cluster_size=int(analysis_raw["usable_min_cluster_size"]),
        top_summary_n=int(analysis_raw["top_summary_n"]),
        outlier_z_threshold=float(analysis_raw["outlier_z_threshold"]),
        blocks=blocks_raw,
    )

    cfg = RunConfig(
        base_dir=base_dir,
        input_path=input_path,
        out_dir=out_dir,
        sheet_name=run_raw.get("sheet_name"),
        k_values=[int(x) for x in model_raw["k_values"]],
        gmm_covariance_types=[str(x) for x in model_raw["gmm_covariance_types"]],
        random_state=int(run_raw["random_state"]),
        assume_already_standardized=bool(run_raw["assume_already_standardized"]),
        save_pca_auxiliary=bool(run_raw["save_pca_auxiliary"]),
        experiment_name=experiment_name,
        analysis=analysis_cfg,
    )

    resolved_config = config_to_dict(cfg)
    resolved_config["config_path"] = str(config_path.resolve())
    return cfg, resolved_config


def save_run_artifacts(
    cfg: RunConfig,
    resolved_config: dict,
    extra_metadata: dict | None = None,
) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    used_config_path = cfg.out_dir / "used_config.yaml"
    with open(used_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            resolved_config,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    config_str = json.dumps(resolved_config, sort_keys=True, ensure_ascii=False, default=str)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:10]

    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_path": str(cfg.input_path),
        "out_dir": str(cfg.out_dir),
        "experiment_name": cfg.experiment_name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(cfg.out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def write_sample_yaml_if_missing(sample_path: Path) -> None:
    if sample_path.exists():
        return
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            default_yaml_dict(),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


# =========================================================
# MLflow 결과 로깅
# =========================================================

def log_mlflow_results(
    cfg: "RunConfig",
    best: "pd.Series",
    summary_df: "pd.DataFrame",
    outlier_df: "pd.DataFrame",
    feature_discrim_df: "pd.DataFrame",
    n_subjects: int = 0,
    n_features: int = 0,
) -> None:
    """
    Run 종료 후 MLflow에 파라미터·메트릭·아티팩트 태그를 기록한다.
    MLFLOW_RUN_ID 환경변수가 있으면 nested run으로 기록하고,
    없으면 독립 run으로 기록한다.
    MLflow 실패는 클러스터링 결과에 영향을 주지 않도록 try/except로 보호한다.
    """
    try:
        import os as _os
        import math
        import mlflow

        tracking_uri = _os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        experiment_name = getattr(cfg, "experiment_name", "default")
        parent_run_id = _os.environ.get("MLFLOW_RUN_ID")
        data_structure = _os.environ.get("DATA_STRUCTURE", "")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        def _safe_float(val):
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except Exception:
                return None

        # Build params dict
        params = {
            "n_subjects": str(n_subjects) if n_subjects else None,
            "n_features": str(n_features) if n_features else None,
            "k_values": str(getattr(cfg, "k_values", "")),
            "random_state": str(getattr(cfg, "random_state", "")),
            "input_path": Path(cfg.input_path).stem if getattr(cfg, "input_path", None) else None,
            "data_structure": data_structure or None,
        }
        # Remove None/empty values
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        # Build metrics dict
        metrics = {}
        for col, key in [
            ("silhouette", "best_silhouette"),
            ("calinski_harabasz", "best_calinski_harabasz"),
            ("davies_bouldin", "best_davies_bouldin"),
            ("k", "best_k"),
            ("subsample_stability_mean_ari", "best_subsample_ari"),
            ("seed_stability_mean_ari", "best_seed_ari"),
        ]:
            if col in best.index:
                val = _safe_float(best[col])
                if val is not None:
                    metrics[key] = val

        n_flagged = int((outlier_df["is_outlier"] == True).sum()) if "is_outlier" in outlier_df.columns else 0
        metrics["n_flagged_outliers"] = float(n_flagged)

        # Context manager helper
        if parent_run_id:
            ctx = mlflow.start_run(run_id=parent_run_id, nested=True)
        else:
            ctx = mlflow.start_run(run_name=f"clustering__{experiment_name}")

        with ctx:
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.set_tag("output_dir", str(cfg.out_dir.resolve()))

        print(f"[MLflow] Logged clustering results (experiment={experiment_name})")

    except Exception as e:
        print(f"[MLflow] logging skipped: {e}")


# =========================================================
# 메인
# =========================================================

def main(cfg: RunConfig | None = None, resolved_config: dict | None = None) -> None:
    cfg = cfg or RunConfig()
    resolved_config = resolved_config or config_to_dict(cfg)

    # --- 출력 폴더 구조 ---
    # out_dir/          ← 전체 summary + config/metadata
    # out_dir/png/      ← 모든 시각화 파일
    # out_dir/txt/      ← 모델별 short report
    # out_dir/xlsx/     ← 세부 xlsx 파일
    png_dir = cfg.out_dir / "png"
    txt_dir = cfg.out_dir / "txt"
    xlsx_dir = cfg.out_dir / "xlsx"
    for d in [cfg.out_dir, png_dir, txt_dir, xlsx_dir]:
        d.mkdir(parents=True, exist_ok=True)

    save_run_artifacts(cfg, resolved_config)

    # --- 데이터 로드 ---
    print(f"[LOAD] {cfg.input_path.resolve()}")
    df = load_input_dataframe(cfg.input_path, cfg.sheet_name)

    id_col = find_id_col(df, cfg.analysis)
    X_df, ids = ensure_numeric_matrix(df, id_col=id_col)
    feature_cols = X_df.columns.tolist()
    X = X_df.values

    blocks = filter_blocks_to_existing_columns(cfg.analysis.blocks, feature_cols)

    print(f"[INFO] experiment_name={cfg.experiment_name}")
    print(f"[INFO] out_dir={cfg.out_dir.resolve()}")
    print(f"[INFO] n_subjects={X.shape[0]}, n_features={X.shape[1]}")
    print(f"[INFO] features={feature_cols}")
    print(f"[INFO] blocks used={list(blocks.keys())}")

    if not cfg.assume_already_standardized:
        print("[WARN] 현재 코드는 standardized matrix 입력을 가정합니다.")

    # --- Block 진단 ---
    block_corr_df = summarize_block_correlations(X_df, blocks)
    block_pca_df = summarize_block_pca(X_df, blocks)
    block_energy_df = summarize_block_energy(X_df, blocks)

    # --- 모델 적합 ---
    all_results: List[ModelResult] = []

    for k in cfg.k_values:
        print(f"[RUN] KMeans k={k}")
        all_results.append(run_kmeans(X, k, cfg.random_state))

        print(f"[RUN] Ward k={k}")
        all_results.append(run_ward(X, k))

    for cov in cfg.gmm_covariance_types:
        for k in cfg.k_values:
            print(f"[RUN] GMM k={k}, cov={cov}")
            try:
                all_results.append(run_gmm_single(X, k, cov, cfg.random_state))
            except Exception as e:
                print(f"[WARN] GMM failed | k={k}, cov={cov} | {e}")

    # --- 안정성 계산 및 랭킹 ---
    raw_summary_df = results_to_df(all_results)
    summary_df = attach_stability_to_summary(
        raw_summary_df, X, cfg.analysis, cfg.random_state
    )

    # metadata 업데이트
    save_run_artifacts(
        cfg,
        resolved_config,
        extra_metadata={
            "n_subjects": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "summary_n_rows": int(summary_df.shape[0]),
            "summary_n_cols": int(summary_df.shape[1]),
        },
    )

    # --- Best solution 결정 ---
    best = choose_best_solution(summary_df)
    best_model = best["model"]
    best_k = int(best["k"])
    best_cov = best["covariance_type"] if pd.notna(best["covariance_type"]) else None
    print(f"[BEST] model={best_model}, k={best_k}, cov={best_cov}")

    best_result = next(
        (
            r
            for r in all_results
            if r.model_name == best_model and r.k == best_k and r.covariance_type == best_cov
        ),
        None,
    )
    if best_result is None:
        raise RuntimeError("Best result lookup failed.")

    labeled_df = build_labeled_df(X_df, best_result.labels, id_col=id_col, ids=ids)
    profile_z = build_profile_table(X_df, best_result.labels, id_col=id_col, ids=ids)
    top_feat_summary = get_top_features_by_cluster(profile_z, top_n=3)

    # --- Feature 설명력 ---
    print("[RUN] feature discriminability")
    feature_discrim_df = compute_feature_discriminability(X_df, best_result.labels)

    # --- Outlier 탐지 ---
    print("[RUN] outlier detection")
    outlier_df = detect_outliers_by_centroid_distance(
        X_df,
        best_result.labels,
        z_threshold=cfg.analysis.outlier_z_threshold,
        id_col=id_col,
        ids=ids,
    )

    # --- Co-clustering ---
    print("[RUN] best coclustering matrix")
    best_coclust = compute_coclustering_matrix(
        X=X,
        model_name=best_model,
        k=best_k,
        n_runs=cfg.analysis.subsample_stability_runs,
        subsample_ratio=cfg.analysis.subsample_ratio,
        covariance_type=best_cov if best_cov else "full",
        random_state=cfg.random_state,
    )

    # --- 모델별 best 결과 수집 + tiny cluster ---
    model_best_rows: List[pd.Series] = []
    model_best_label_dfs: Dict[str, pd.DataFrame] = {}
    # 모든 (model × k) 조합 라벨 — post-hoc 전체 실행용
    all_label_dfs: Dict[str, pd.DataFrame] = {}
    tiny_cluster_members_all: List[pd.DataFrame] = []

    for model_name in ["kmeans", "ward", "gmm"]:
        sub = summary_df[summary_df["model"] == model_name].copy()
        if sub.empty:
            continue

        sub_best = sub.iloc[0]
        model_best_rows.append(sub_best)

        mk = int(sub_best["k"])
        mcov = sub_best["covariance_type"] if pd.notna(sub_best["covariance_type"]) else None

        target_result = next(
            (
                r
                for r in all_results
                if r.model_name == model_name and r.k == mk and r.covariance_type == mcov
            ),
            None,
        )
        if target_result is None:
            continue

        tmp_df = build_labeled_df(X_df, target_result.labels, id_col=id_col, ids=ids)
        model_best_label_dfs[f"{model_name}_k{mk}_{mcov}"] = tmp_df

        tiny_df = extract_tiny_cluster_members(tmp_df, model_name, mk, mcov, cfg.analysis)
        if not tiny_df.empty:
            tiny_cluster_members_all.append(tiny_df)

    # 모든 k 값에 대해 라벨 저장 (kmeans/ward: cov 없음, gmm: 해당 k의 best cov)
    for model_name in ["kmeans", "ward"]:
        for k_val in cfg.k_values:
            r = next(
                (r for r in all_results if r.model_name == model_name and r.k == k_val),
                None,
            )
            if r is None:
                continue
            sheet_key = f"{model_name}_k{k_val}"
            all_label_dfs[sheet_key] = build_labeled_df(
                X_df, r.labels, id_col=id_col, ids=ids
            )

    for k_val in cfg.k_values:
        sub_gmm_k = summary_df[
            (summary_df["model"] == "gmm") & (summary_df["k"] == k_val)
        ]
        if sub_gmm_k.empty:
            continue
        best_gmm_row = sub_gmm_k.iloc[0]
        mcov_k = (
            best_gmm_row["covariance_type"]
            if pd.notna(best_gmm_row["covariance_type"])
            else None
        )
        r = next(
            (
                r
                for r in all_results
                if r.model_name == "gmm" and r.k == k_val and r.covariance_type == mcov_k
            ),
            None,
        )
        if r is None:
            continue
        cov_tag = mcov_k if mcov_k else "None"
        sheet_key = f"gmm_k{k_val}_{cov_tag}"
        all_label_dfs[sheet_key] = build_labeled_df(
            X_df, r.labels, id_col=id_col, ids=ids
        )

    best_tiny_df = extract_tiny_cluster_members(
        labeled_df, best_model, best_k, best_cov, cfg.analysis
    )
    if not best_tiny_df.empty:
        tiny_cluster_members_all.append(best_tiny_df)

    tiny_cluster_members_df = (
        pd.concat(tiny_cluster_members_all, axis=0, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
        if tiny_cluster_members_all
        else pd.DataFrame(
            columns=["model", "k", "covariance_type", "cluster", "n_members", "member_id"]
        )
    )

    # --- Tiny cluster 원인 분석 ---
    print("[RUN] tiny cluster cause analysis")
    tiny_cause_df = explain_tiny_cluster_causes(
        X_df=X_df,
        labels=best_result.labels,
        profile_z=profile_z,
        feature_discrim_df=feature_discrim_df,
        tiny_cluster_threshold=cfg.analysis.tiny_cluster_threshold,
    )

    # =========================================================
    # 시각화 → png/
    # =========================================================
    plot_metric_comparison(summary_df, png_dir / "model_comparison_metrics.png")
    plot_gmm_bic_aic_grid(summary_df, png_dir / "gmm_bic_aic_grid.png")

    plot_cluster_profile_heatmap(
        profile_z=profile_z,
        title=f"Cluster profile heatmap: {best_model} (k={best_k}, cov={best_cov})",
        save_path=png_dir / f"best_{best_model}_k{best_k}_cov_{best_cov}_profile_heatmap.png",
    )
    plot_distance_heatmap(
        X=X,
        labels=best_result.labels,
        save_path=png_dir / f"best_{best_model}_k{best_k}_cov_{best_cov}_distance_heatmap.png",
        title=f"Distance heatmap: {best_model} (k={best_k}, cov={best_cov})",
    )
    plot_coclustering_heatmap(
        coclust=best_coclust,
        labels=best_result.labels,
        save_path=png_dir / f"best_{best_model}_k{best_k}_cov_{best_cov}_coclustering_heatmap.png",
        title=f"Co-clustering heatmap: {best_model} (k={best_k}, cov={best_cov})",
    )

    pca_df = None
    if cfg.save_pca_auxiliary:
        pca_df = plot_pca_scatter(
            X=X,
            labels=best_result.labels,
            title=f"Auxiliary PCA: {best_model} (k={best_k}, cov={best_cov})",
            save_path=png_dir / f"best_{best_model}_k{best_k}_cov_{best_cov}_pca_aux.png",
        )

    for sub_best in model_best_rows:
        mn = sub_best["model"]
        mk = int(sub_best["k"])
        mcov = sub_best["covariance_type"] if pd.notna(sub_best["covariance_type"]) else None
        target_result = next(
            (
                r
                for r in all_results
                if r.model_name == mn and r.k == mk and r.covariance_type == mcov
            ),
            None,
        )
        if target_result is None:
            continue

        plot_distance_heatmap(
            X=X,
            labels=target_result.labels,
            save_path=png_dir / f"{mn}_best_k{mk}_cov_{mcov}_distance_heatmap.png",
            title=f"{mn} best distance heatmap (k={mk}, cov={mcov})",
        )

    # =========================================================
    # txt/ → 모델별 short report
    # =========================================================
    for sub_best in model_best_rows:
        mn = sub_best["model"]
        mk = int(sub_best["k"])
        mcov = sub_best["covariance_type"] if pd.notna(sub_best["covariance_type"]) else None

        target_result = next(
            (
                r
                for r in all_results
                if r.model_name == mn and r.k == mk and r.covariance_type == mcov
            ),
            None,
        )
        if target_result is None:
            continue

        tmp_profile = build_profile_table(X_df, target_result.labels, id_col=id_col, ids=ids)
        tmp_top = get_top_features_by_cluster(tmp_profile, top_n=3)
        tmp_discrim = compute_feature_discriminability(X_df, target_result.labels)

        write_short_txt_report(
            save_path=txt_dir / f"{mn}_best_k{mk}_cov_{mcov}_short_report.txt",
            model_name=mn,
            k=mk,
            covariance_type=mcov,
            metrics=sub_best,
            profile_z=tmp_profile,
            top_feat_summary=tmp_top,
            feature_discrim_df=tmp_discrim,
        )

    write_short_txt_report(
        save_path=txt_dir / f"best_{best_model}_k{best_k}_cov_{best_cov}_short_report.txt",
        model_name=best_model,
        k=best_k,
        covariance_type=best_cov,
        metrics=best,
        profile_z=profile_z,
        top_feat_summary=top_feat_summary,
        feature_discrim_df=feature_discrim_df,
    )

    # =========================================================
    # out_dir 루트 → 전체 summary
    # =========================================================

    write_total_summary_short_report(
        save_path=cfg.out_dir / "TOTAL_SUMMARY_SHORT_REPORT.txt",
        summary_df=summary_df,
        best_row=best,
        block_corr_df=block_corr_df,
        block_pca_df=block_pca_df,
        block_energy_df=block_energy_df,
        tiny_cluster_members_df=tiny_cluster_members_df,
        cfg=cfg.analysis,
        feature_discrim_df=feature_discrim_df,
        outlier_df=outlier_df,
        tiny_cause_df=tiny_cause_df,
    )

    # =========================================================
    # xlsx/ → 세부 데이터
    # =========================================================
    tiny_cluster_path = xlsx_dir / "tiny_cluster_members.xlsx"
    with pd.ExcelWriter(tiny_cluster_path, engine="openpyxl") as writer:
        tiny_cluster_members_df.to_excel(writer, sheet_name="tiny_members_long", index=False)
        if not tiny_cluster_members_df.empty:
            tiny_summary = (
                tiny_cluster_members_df.groupby(
                    ["model", "k", "covariance_type", "cluster", "n_members"],
                    dropna=False,
                )["member_id"]
                .apply(list)
                .reset_index()
            )
            tiny_summary.to_excel(writer, sheet_name="tiny_members_grouped", index=False)
        if not tiny_cause_df.empty:
            tiny_cause_df.to_excel(writer, sheet_name="tiny_cause_analysis", index=False)

    feature_discrim_path = xlsx_dir / "feature_discriminability.xlsx"
    with pd.ExcelWriter(feature_discrim_path, engine="openpyxl") as writer:
        feature_discrim_df.to_excel(writer, sheet_name="feature_discrim_best", index=False)

    outlier_path = xlsx_dir / "outlier_detection.xlsx"
    with pd.ExcelWriter(outlier_path, engine="openpyxl") as writer:
        outlier_df.to_excel(writer, sheet_name="all_subjects", index=False)
        flagged = outlier_df[outlier_df["is_outlier"] == True].copy()
        flagged.to_excel(writer, sheet_name="flagged_outliers", index=False)

    # =========================================================
    # out_dir 루트 → 전체 summary xlsx
    # =========================================================
    cols_to_show = [
        "model",
        "k",
        "covariance_type",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "bic",
        "aic",
        "seed_stability_mean_pairwise_ari",
        "subsample_stability_mean_ari",
        "min_cluster_size",
        "cluster_sizes",
        "usable_candidate",
        "overall_rank_score",
    ]

    summary_path = cfg.out_dir / "summary.xlsx"
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_all", index=False)
        raw_summary_df.to_excel(writer, sheet_name="summary_raw_no_stability", index=False)
        labeled_df.to_excel(writer, sheet_name="best_labels", index=False)
        profile_z.to_excel(writer, sheet_name="best_profile_z")
        feature_discrim_df.to_excel(writer, sheet_name="feature_discriminability", index=False)
        outlier_df.to_excel(writer, sheet_name="outlier_detection", index=False)
        if not tiny_cause_df.empty:
            tiny_cause_df.to_excel(writer, sheet_name="tiny_cause_analysis", index=False)
        block_corr_df.to_excel(writer, sheet_name="block_corr", index=False)
        block_pca_df.to_excel(writer, sheet_name="block_pca", index=False)
        block_energy_df.to_excel(writer, sheet_name="block_energy", index=False)
        tiny_cluster_members_df.to_excel(writer, sheet_name="tiny_cluster_members", index=False)
        pd.DataFrame(best_coclust).to_excel(writer, sheet_name="best_coclustering", index=False)

        # config를 엑셀 안에서도 바로 볼 수 있게 저장
        config_rows = []
        for section, content in resolved_config.items():
            if isinstance(content, dict):
                for k, v in content.items():
                    config_rows.append({"section": section, "key": k, "value": str(v)})
            else:
                config_rows.append({"section": "root", "key": section, "value": str(content)})
        pd.DataFrame(config_rows).to_excel(writer, sheet_name="used_config", index=False)

        if pca_df is not None:
            pca_df.to_excel(writer, sheet_name="best_pca_aux", index=False)

        # 전체 (model × k) 조합 라벨 시트 — 시트명 접두사 "lbl__" 로 식별
        for key, tmp_df in all_label_dfs.items():
            sheet_name_safe = ("lbl__" + key)[:31]
            tmp_df.to_excel(writer, sheet_name=sheet_name_safe, index=False)

    # metadata 갱신
    save_run_artifacts(
        cfg,
        resolved_config,
        extra_metadata={
            "best_model": best_model,
            "best_k": best_k,
            "best_covariance_type": best_cov,
            "n_subjects": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_flagged_outliers": int((outlier_df["is_outlier"] == True).sum()),
            "n_tiny_cluster_rows": int(tiny_cluster_members_df.shape[0]),
        },
    )

    # =========================================================
    # MLflow 로깅
    # =========================================================
    log_mlflow_results(
        cfg, best, summary_df, outlier_df, feature_discrim_df,
        n_subjects=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )

    # =========================================================
    # 콘솔 출력
    # =========================================================
    top_summary_df = summary_df.head(cfg.analysis.top_summary_n).copy()

    print("\n[TOP SUMMARY]")
    print(top_summary_df.to_string(index=False))

    print("\n[TOP 5 - HUMAN READABLE]")
    for i, (_, row) in enumerate(summary_df.head(5).iterrows(), start=1):
        cov_text = f", cov={row['covariance_type']}" if pd.notna(row["covariance_type"]) else ""
        print(
            f"{i}. {row['model']} k={int(row['k'])}{cov_text} "
            f"(sil={safe_float(row['silhouette']):.3f}, "
            f"subsample_ARI={safe_float(row['subsample_stability_mean_ari']):.3f}, "
            f"min_size={int(row['min_cluster_size'])}, "
            f"usable={bool(row['usable_candidate'])})"
        )

    print("\n[FEATURE DISCRIMINABILITY - TOP 5]")
    for _, row in feature_discrim_df.head(5).iterrows():
        sig = "*" if row.get("significant_p05", False) else " "
        print(
            f"  {sig}[{int(row['discriminability_rank'])}] {row['feature']}: "
            f"F={safe_float(row['f_statistic']):.2f}, "
            f"eta²={safe_float(row['eta_squared']):.3f} ({row['effect_size']})"
        )

    flagged = outlier_df[outlier_df["is_outlier"] == True]
    print(f"\n[OUTLIERS] {len(flagged)} subjects flagged (z > {cfg.analysis.outlier_z_threshold})")
    if not flagged.empty:
        cols = [c for c in ["subject_id", "cluster", "z_within_cluster"] if c in flagged.columns]
        print(flagged[cols].to_string(index=False))

    if not tiny_cluster_members_df.empty:
        print(f"\n[TINY CLUSTERS <= {cfg.analysis.tiny_cluster_threshold}]")
        print(tiny_cluster_members_df.head(20).to_string(index=False))

    if not tiny_cause_df.empty:
        cols = [
            c
            for c in [
                "cluster",
                "n_members",
                "cause_type",
                "nearest_cluster",
                "relative_nearest_dist",
            ]
            if c in tiny_cause_df.columns
        ]
        print("\n[TINY CLUSTER CAUSES]")
        print(tiny_cause_df[cols].to_string(index=False))

    print(f"\n[SAVED] {cfg.out_dir.resolve()}")
    print("  ├── used_config.yaml")
    print("  ├── run_metadata.json")
    print("  ├── summary.xlsx")
    print("  ├── SUMMARY_REPORT.txt")
    print(f"  ├── png/   ({len(list(png_dir.glob('*.png')))} files)")
    print(f"  ├── txt/   ({len(list(txt_dir.glob('*.txt')))} files)")
    print(f"  └── xlsx/  ({len(list(xlsx_dir.glob('*.xlsx')))} files)")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subject-level clustering runner with YAML config + run tracking."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config path. Example: configs/clustering/default.yaml",
    )
    parser.add_argument(
        "--write-sample-config",
        type=str,
        default=None,
        help="샘플 YAML을 해당 경로에 생성하고 종료. Example: configs/clustering/default.yaml",
    )
    args = parser.parse_args()

    if args.write_sample_config:
        sample_path = Path(args.write_sample_config)
        if not sample_path.is_absolute():
            sample_path = _default_base_dir() / sample_path
        write_sample_yaml_if_missing(sample_path)
        print(f"[SAVED SAMPLE CONFIG] {sample_path.resolve()}")
        sys.exit(0)

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = _default_base_dir() / config_path
        cfg, resolved_config = build_config_from_yaml(config_path)
    else:
        cfg = RunConfig()
        resolved_config = config_to_dict(cfg)

    main(cfg, resolved_config)