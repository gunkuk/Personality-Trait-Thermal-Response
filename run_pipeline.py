# 핵심: build_dataset.py와 run_clustering.py를 순차 실행하면서 dataset version/structure를 일관되게 전달한다.
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

'''
전체 파이프라인
python run_pipeline.py --dataset_version v1 --data_structure full

dataset build만
python build_dataset.py --dataset_version v1 --data_structure full

clustering만
python run_pipeline.py --dataset_version v1 --data_structure full --skip_build

블록 커스텀
python build_dataset.py --dataset_version v2 --data_structure PHY,PSY,BHR
python build_dataset.py --dataset_version v2 --data_structure no_tlx


'''

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent
INPUT_MATRIX_DIR = ROOT / "00_data" / "modules" / "5. Input Matrix"   # 수정
CLUSTERING_DIR   = ROOT / "01_clustering" / "6. Clustering (K-meas, Ward, GMM)"  # 수정
BUILD_SCRIPT     = INPUT_MATRIX_DIR / "build_dataset.py"
CLUSTER_SCRIPT   = CLUSTERING_DIR / "run_clustering.py"
PROCESSED_DIR    = ROOT / "00_data" / "02_processed"   # 수정


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset-build + clustering pipeline.")
    parser.add_argument("--dataset_version", type=str, required=True, help="e.g. v1, v2")
    parser.add_argument("--data_structure", type=str, default="full", help="e.g. full, no_tlx, PHY,PSY")
    parser.add_argument("--experiment_name", type=str, default="subject_level_pipeline")
    parser.add_argument("--tracking_uri", type=str, default="mlruns")
    parser.add_argument("--skip_build", action="store_true", help="Skip build_dataset.py and reuse existing processed dataset.")
    parser.add_argument("--use_existing_interim", action="store_true", help="Pass-through option for build_dataset.py")
    parser.add_argument("--dataset_ext", type=str, default="csv", choices=["csv", "xlsx"], help="Processed dataset extension to pass downstream.")
    parser.add_argument("--cluster_extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to run_clustering.py. Prefix with --cluster_extra_args")
    return parser.parse_args()



def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)



def main() -> None:
    args = parse_args()

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = args.tracking_uri
    env["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name
    env["DATASET_VERSION"] = args.dataset_version
    env["DATA_STRUCTURE"] = args.data_structure

    dataset_stem = f"X_subject_{args.dataset_version}_{args.data_structure}"
    dataset_path = PROCESSED_DIR / f"{dataset_stem}.{args.dataset_ext}"

    if not args.skip_build:
        build_cmd = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--dataset_version", args.dataset_version,
            "--data_structure", args.data_structure,
            "--experiment_name", f"{args.experiment_name}_build",
            "--tracking_uri", args.tracking_uri,
        ]
        if args.use_existing_interim:
            build_cmd.append("--use_existing_interim")
        run_command(build_cmd, env)
    else:
        print("[SKIP] build_dataset.py")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}\n"
            f"Check dataset_version/data_structure/dataset_ext or disable --skip_build."
        )

    # run_clustering.py uses --config YAML; build a minimal override config
    default_config = ROOT / "configs" / "default.yaml"
    config_lines = []
    if default_config.exists():
        config_lines = default_config.read_text(encoding="utf-8").splitlines()

    # Replace / insert run.input_path and experiment_name
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
                new_lines.append(f"  input_path: {dataset_path}")
                input_path_written = True
            in_run_block = False
        if in_run_block and _re.match(r"^\s+input_path\s*:", line):
            new_lines.append(f"  input_path: {dataset_path}")
            input_path_written = True
            continue
        new_lines.append(line)
    if not input_path_written:
        new_lines.append(f"  input_path: {dataset_path}")

    # Set experiment_name
    new_text = "\n".join(new_lines)
    new_text = _re.sub(r"^experiment_name:.*", f"experiment_name: {args.experiment_name}", new_text, flags=_re.MULTILINE)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        tmp.write(new_text)
        tmp_config = tmp.name

    cluster_cmd = [
        sys.executable,
        str(CLUSTER_SCRIPT),
        "--config", tmp_config,
    ]
    if args.cluster_extra_args:
        cluster_cmd.extend(args.cluster_extra_args)

    run_command(cluster_cmd, env)

    print("[ALL DONE] Pipeline completed.")
    print(f"  dataset_version : {args.dataset_version}")
    print(f"  data_structure  : {args.data_structure}")
    print(f"  dataset_path    : {dataset_path}")


if __name__ == "__main__":
    main()
