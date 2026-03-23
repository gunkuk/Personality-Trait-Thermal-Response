# 요약: subject-level input matrix의 numeric feature 결측을 평균 대치하고 저장한다.

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "interim" / "input_matrix_subject_level.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "interim" / "input_matrix_subject_level_imputed.xlsx"

SHEET_NAME = "subject_features_only"


def main():
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)

    ids = df["id"].copy()
    X = df.drop(columns=["id"]).copy()

    # numeric 변환
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    print("=== 결측 개수(행 기준) ===")
    missing_rows = X[X.isna().any(axis=1)]
    print(missing_rows.shape[0])

    print("\n=== 결측 subject ===")
    print(df.loc[X.isna().any(axis=1), ["id"]])

    print("\n=== 컬럼별 결측 개수 ===")
    print(X.isna().sum())

    # 평균 대치
    X_imputed = X.copy()
    for col in X_imputed.columns:
        mean_val = X_imputed[col].mean()
        X_imputed[col] = X_imputed[col].fillna(mean_val)

    out = pd.concat([ids, X_imputed], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name=SHEET_NAME, index=False)

    print("\n[DONE] imputed matrix saved")
    print(f"Path: {OUTPUT_PATH}")
    print("\n[INFO] Missing after imputation:")
    print(out.isna().sum())


if __name__ == "__main__":
    main()