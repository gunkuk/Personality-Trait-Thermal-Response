from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_excel(
    BASE_DIR / "input_matrix_subject_level.xlsx",
    sheet_name="subject_features_only"
)

missing_rows = df[df.isna().any(axis=1)]

print("=== 결측 개수 ===")
print(missing_rows.shape[0])

print("\n=== 결측 subject ===")
print(missing_rows[["id"]])

print("\n=== 어떤 컬럼이 결측인지 ===")
print(missing_rows.isna().sum())