# 요약: subject-level input matrix에서 SEX를 제외하고,
# clustering에 사용할 numeric feature만 평균 대치 후
# block-wise standardization (z-score + sqrt(p) scaling)을 수행한다.


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "interim" / "input_matrix_subject_level_imputed.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "input_matrix_standardized.xlsx"

df = pd.read_excel(INPUT_PATH, sheet_name="subject_features_only")

ids = df["id"].copy()

blocks = {
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

feature_cols = [col for cols in blocks.values() for col in cols]
X = df[feature_cols].copy()

for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# 여기서는 결측 대치 안 함
if X.isna().sum().sum() > 0:
    raise ValueError("Imputed input expected, but missing values still remain.")

X_scaled_blocks = []

for block_name, cols in blocks.items():
    block_data = X[cols].copy()

    scaler = StandardScaler()
    Z = scaler.fit_transform(block_data)

    p = len(cols)
    Z = Z / np.sqrt(p)

    Z_df = pd.DataFrame(Z, columns=cols, index=X.index)
    X_scaled_blocks.append(Z_df)

X_final = pd.concat(X_scaled_blocks, axis=1)
X_final.insert(0, "id", ids)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
X_final.to_excel(OUTPUT_PATH, sheet_name="subject_features_only", index=False)

print("[DONE] standardized matrix saved")
print(f"Path: {OUTPUT_PATH}")
print(f"Shape: {X_final.shape}")