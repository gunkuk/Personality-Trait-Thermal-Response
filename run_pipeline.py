# 핵심: build_dataset -> run_clustering -> post_hoc_analysis를 순차 실행하고,
# clustering 결과가 저장된 03_outputs의 최신 run 폴더를 찾아 post_hoc 결과를 그 아래에 연결한다.
#
# Usage (Hydra syntax):
#   python run_pipeline.py dataset_version=v1
#   python run_pipeline.py dataset_version=v1 skip_posthoc=true
#   python run_pipeline.py -m data_structure=full,no_tlx
'''
python run_pipeline.py dataset_version=v1

# 옵션
python run_pipeline.py dataset_version=v1 skip_posthoc=true
python run_pipeline.py dataset_version=v1 data_structure=no_tlx

# 멀티런 (자동으로 여러 번 실행)
python run_pipeline.py -m dataset_version=v1 data_structure=full,no_tlx,PHY,PSY

# MLflow UI
mlflow ui  # 브라우저에서 http://localhost:5000
'''

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent

INPUT_MATRIX_DIR = ROOT / "00_data" / "modules" / "5. Input Matrix"
CLUSTERING_DIR = ROOT / "01_clustering" / "6. Clustering (K-meas, Ward, GMM)"
POSTHOC_DIR = ROOT / "02_post_hoc"

BUILD_SCRIPT = INPUT_MATRIX_DIR / "build_dataset.py"
CLUSTER_SCRIPT = CLUSTERING_DIR / "run_clustering.py"
POSTHOC_SCRIPT = POSTHOC_DIR / "PER_posthoc.py"

PROCESSED_DIR = ROOT / "00_data" / "02_processed"
OUTPUTS_DIR = ROOT / "03_outputs"
CONFIG_DIR = ROOT / "configs"


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("[CMD]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env)


def build_temp_config(
    dataset_path: Path,
    experiment_name: str,
) -> Path:
    """
    run_clustering.py용 임시 YAML 생성:
    - input_path override
    - experiment_name override
    """
    default_config = CONFIG_DIR / "default.yaml"
    config_lines: list[str] = []

    if default_config.exists():
        config_lines = default_config.read_text(encoding="utf-8").splitlines()

    import re as _re

    new_lines = []
    in_run_block = False
    input_path_written = False

    for line in config_lines:
        if _re.match(r"^run\s*:", line):
            in_run_block = True
            new_lines.append(line)
            continue

        if in_run_block and _re.match(r"^\S", line):
            if not input_path_written:
                new_lines.append(f"  input_path: {dataset_path.as_posix()}")
                input_path_written = True
            in_run_block = False

        if in_run_block and _re.match(r"^\s+input_path\s*:", line):
            new_lines.append(f"  input_path: {dataset_path.as_posix()}")
            input_path_written = True
            continue

        new_lines.append(line)

    if not input_path_written:
        if not any(line.strip().startswith("run:") for line in new_lines):
            new_lines.append("run:")
        new_lines.append(f"  input_path: {dataset_path.as_posix()}")

    new_text = "\n".join(new_lines)
    new_text = _re.sub(
        r"^experiment_name:.*",
        f"experiment_name: {experiment_name}",
        new_text,
        flags=_re.MULTILINE,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        tmp.write(new_text)
        return Path(tmp.name)


def find_latest_clustering_run(outputs_dir: Path, experiment_name: str) -> Path:
    """
    run_clustering.py가 03_outputs/<timestamp>__<experiment_name> 형태로 저장하므로
    같은 experiment_name suffix를 가진 최신 폴더를 찾는다.
    """
    candidates = [
        p for p in outputs_dir.iterdir()
        if p.is_dir() and p.name.endswith(f"__{experiment_name}")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No clustering run dir found in {outputs_dir} matching *__{experiment_name}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_cluster_assignment_input(run_dir: Path) -> Path:
    """
    post_hoc_analysis.py에 넘길 cluster assignment 파일.
    우선순위:
    1) summary.xlsx
    2) xlsx/best_labels.xlsx
    3) xlsx/ 아래 best label 관련 파일
    """
    summary_xlsx = run_dir / "summary.xlsx"
    if summary_xlsx.exists():
        return summary_xlsx

    xlsx_dir = run_dir / "xlsx"
    if xlsx_dir.exists():
        direct = xlsx_dir / "best_labels.xlsx"
        if direct.exists():
            return direct

        candidates = list(xlsx_dir.glob("*best*label*.xlsx"))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"Could not find cluster assignment source in {run_dir}. "
        f"Expected summary.xlsx or best label xlsx."
    )


@hydra.main(config_path="configs", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert OmegaConf to plain dict for easy access
    c = OmegaConf.to_container(cfg, resolve=True)

    dataset_version = c["dataset_version"]
    data_structure = c["data_structure"]
    experiment_name = c["experiment_name"]
    tracking_uri = c["tracking_uri"]
    skip_build = c["skip_build"]
    skip_clustering = c["skip_clustering"]
    skip_posthoc = c["skip_posthoc"]
    use_existing_interim = c["use_existing_interim"]
    dataset_ext = c["dataset_ext"]
    posthoc_analysis_modes = c["posthoc_analysis_modes"]

    # personality_xlsx_path: auto-resolve if not set
    personality_xlsx_path = c.get("personality_xlsx_path") or str(
        ROOT / "00_data" / "00_raw" / "rawdata_wrong corrected_0319.xlsx"
    )

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    env["DATASET_VERSION"] = dataset_version
    env["DATA_STRUCTURE"] = data_structure

    dataset_stem = f"X_subject_{dataset_version}_{data_structure}"
    dataset_path = PROCESSED_DIR / f"{dataset_stem}.{dataset_ext}"

    # --- MLflow parent run ---
    parent_run_id = None
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        parent_run = mlflow.start_run(run_name=f"pipeline__{dataset_version}__{data_structure}")
        parent_run_id = parent_run.info.run_id
        env["MLFLOW_RUN_ID"] = parent_run_id
        print(f"[MLflow] Started parent run: {parent_run_id}")
    except Exception as e:
        print(f"[MLflow] Could not start parent run: {e}")

    try:
        # 1) build_dataset
        if not skip_build:
            build_cmd = [
                sys.executable,
                str(BUILD_SCRIPT),
                "--dataset_version", dataset_version,
                "--data_structure", data_structure,
                "--experiment_name", f"{experiment_name}_build",
                "--tracking_uri", tracking_uri,
            ]
            if use_existing_interim:
                build_cmd.append("--use_existing_interim")
            run_command(build_cmd, env)
        else:
            print("[SKIP] build_dataset.py")

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {dataset_path}\n"
                f"Check dataset_version/data_structure/dataset_ext or disable skip_build."
            )

        # 2) clustering
        if not skip_clustering:
            tmp_config = build_temp_config(
                dataset_path=dataset_path,
                experiment_name=experiment_name,
            )

            cluster_cmd = [
                sys.executable,
                str(CLUSTER_SCRIPT),
                "--config", str(tmp_config),
            ]

            run_command(cluster_cmd, env)
        else:
            print("[SKIP] run_clustering.py")

        # clustering 결과 run dir 찾기
        clustering_run_dir = find_latest_clustering_run(OUTPUTS_DIR, experiment_name)
        print(f"[INFO] clustering_run_dir = {clustering_run_dir}")

        # 3) post-hoc
        if not skip_posthoc:
            cluster_input_path = find_cluster_assignment_input(clustering_run_dir)
            posthoc_out_dir = clustering_run_dir / "post_hoc"

            posthoc_cmd = [
                sys.executable,
                str(POSTHOC_SCRIPT),
                "--cluster_path", str(cluster_input_path),
                "--personality_xlsx_path", str(Path(personality_xlsx_path)),
                "--out_dir", str(posthoc_out_dir),
                "--analysis_modes",
                *posthoc_analysis_modes,
            ]

            run_command(posthoc_cmd, env)
        else:
            print("[SKIP] PER_posthoc.py")

        print("[ALL DONE] Pipeline completed.")
        print(f"  dataset_version   : {dataset_version}")
        print(f"  data_structure    : {data_structure}")
        print(f"  dataset_path      : {dataset_path}")

        # Log pipeline-level params to the already-active parent MLflow run
        if parent_run_id:
            try:
                import mlflow
                # The parent run is still active; log directly without re-entering
                mlflow.log_params({
                    "dataset_version": dataset_version,
                    "data_structure": data_structure,
                    "experiment_name": experiment_name,
                    "skip_build": skip_build,
                    "skip_clustering": skip_clustering,
                    "skip_posthoc": skip_posthoc,
                })
            except Exception as e:
                print(f"[MLflow] Could not log params to parent run: {e}")

        try:
            clustering_run_dir_local = find_latest_clustering_run(OUTPUTS_DIR, experiment_name)
            print(f"  clustering_run    : {clustering_run_dir_local}")
            if not skip_posthoc:
                print(f"  post_hoc_out_dir  : {clustering_run_dir_local / 'post_hoc'}")
        except Exception:
            pass

    finally:
        # End parent MLflow run
        if parent_run_id:
            try:
                import mlflow
                mlflow.end_run()
                print(f"[MLflow] Ended parent run: {parent_run_id}")
            except Exception as e:
                print(f"[MLflow] Could not end parent run: {e}")


if __name__ == "__main__":
    main()
