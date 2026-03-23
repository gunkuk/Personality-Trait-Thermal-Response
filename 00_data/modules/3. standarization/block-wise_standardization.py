# 요약: subject-level input matrix에서 SEX를 제외하고,
# clustering에 사용할 numeric feature만 평균 대치 후
# block-wise standardization (z-score + sqrt(p) scaling)을 수행한다.

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# 1. 경로
# =========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_matrix_subject_level.xlsx"
OUTPUT_PATH = BASE_DIR / "input_matrix_standardized.xlsx"

# =========================
# 2. 데이터 로드
# =========================
df = pd.read_excel(INPUT_PATH, sheet_name="subject_features_only")

ids = df["id"].copy()

# =========================
# 3. block 정의 (SEX 제외)
# =========================
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

# clustering에 실제 사용할 컬럼만 추출
feature_cols = [col for cols in blocks.values() for col in cols]
X = df[feature_cols].copy()

# 전부 numeric으로 강제 변환
for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# =========================
# 4. 결측 처리 (전체 평균 대치)
# =========================
X_imputed = X.copy()

for col in feature_cols:
    mean_val = X_imputed[col].mean()
    X_imputed[col] = X_imputed[col].fillna(mean_val)

# =========================
# 5. block-wise standardization
# =========================
X_scaled_blocks = []

for block_name, cols in blocks.items():
    block_data = X_imputed[cols].copy()

    # block 내부 z-score
    scaler = StandardScaler()
    Z = scaler.fit_transform(block_data)

    # block size 보정
    p = len(cols)
    Z = Z / np.sqrt(p)

    Z_df = pd.DataFrame(Z, columns=cols, index=X_imputed.index)
    X_scaled_blocks.append(Z_df)

# =========================
# 6. concat
# =========================
X_final = pd.concat(X_scaled_blocks, axis=1)
X_final.insert(0, "id", ids)

# =========================
# 7. 저장
# =========================
X_final.to_excel(OUTPUT_PATH, index=False)

print("[DONE] standardized matrix saved")
print(f"Path: {OUTPUT_PATH}")
print(f"Shape: {X_final.shape}")

print("\n[INFO] Missing after imputation:")
print(X_final.isna().sum())