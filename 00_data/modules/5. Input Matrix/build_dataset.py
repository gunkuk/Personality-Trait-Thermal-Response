# 핵심: raw data에서 최종 subject-level clustering 입력 dataset 버전을 생성한다.
from pathlib import Path
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "rawdata_wrong corrected.xlsx"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATASET_VERSION = "v1"


def build_subject_dataset() -> pd.DataFrame:
    # TODO:
    # 1. raw load
    # 2. input matrix 생성
    # 3. missing 처리
    # 4. scaling
    # 5. feature selection
    # 6. 최종 X 반환
    raise NotImplementedError


def save_manifest(df: pd.DataFrame, save_path: Path) -> None:
    manifest = pd.DataFrame([{
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": RAW_PATH.name,
        "n_subjects": df.shape[0],
        "n_features": df.shape[1] - 1 if "id" in df.columns else df.shape[1],
        "feature_names": ", ".join(df.columns.tolist()),
        "cleaning_version": "missing_data_v1",
        "scaling_version": "blockwise_standardization_v1",
        "selection_version": "fixed_feature_set_v1",
    }])

    manifest_path = save_path.with_name("dataset_manifest.csv")
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        manifest = pd.concat([old, manifest], ignore_index=True)

    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def main():
    df = build_subject_dataset()

    save_path = PROCESSED_DIR / f"X_subject_{DATASET_VERSION}.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    save_manifest(df, save_path)

    print(f"[DONE] Saved dataset: {save_path}")


if __name__ == "__main__":
    main()