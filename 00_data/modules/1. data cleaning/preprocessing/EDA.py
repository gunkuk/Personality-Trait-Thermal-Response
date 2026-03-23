import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, decimate, find_peaks, savgol_filter
from scipy.stats import linregress
from cvxeda import cvxEDA
import warnings

warnings.filterwarnings('ignore')

# =========================
# SETTINGS
# =========================
INPUT_DIR = r"C:\Project\mbti\01_data_normalized\eda"
FEATURE_DIR = r"C:\Project\mbti\03_features"
LOG_DIR = r"C:\Project\mbti\02_logs"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TARGET_FS = 10
MINUTES = 20

# =========================
# BIOPAC LOADER
# =========================
def load_acq_txt(file_path):

    with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        if len(parts) > 1 and parts[0].lower() == "sec":
            header_idx = i
            header_cols = parts
            break

    desc = lines[header_idx + 1].strip().split("\t")
    if desc[0] == "":
        desc = desc[1:]

    data_rows = []
    n_cols = len(header_cols)

    for line in lines[header_idx + 2:]:
        parts = line.strip().split("\t")
        if len(parts) == n_cols:
            try:
                float(parts[0])
                data_rows.append(parts)
            except:
                continue

    df = pd.DataFrame(data_rows, columns=header_cols)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df, desc


# =========================
# 추가 지표 계산 함수
# =========================
def calculate_additional_metrics(phasic, tonic, target_fs, phase_start_min, phase_end_min):
    """
    특정 시간 구간에서 EDA 지표를 계산합니다.
    """
    samples_per_min = target_fs * 60
    start_idx = phase_start_min * samples_per_min
    end_idx = phase_end_min * samples_per_min
    
    # 해당 구간 신호 추출
    phasic_phase = phasic[start_idx:end_idx]
    tonic_phase = tonic[start_idx:end_idx]
    
    # SCR (Skin Conductance Response) 피크 검출
    peaks, properties = find_peaks(phasic_phase, height=0.05, distance=10)
    scr_frequency = len(peaks) / (phase_end_min - phase_start_min) if len(peaks) > 0 else 0
    scr_amplitude = float(np.mean(properties['peak_heights'])) if len(peaks) > 0 else 0
    
    # Tonic Trend (선형 회귀)
    x = np.arange(len(tonic_phase))
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, tonic_phase)
        tonic_trend = float(slope)
    except:
        tonic_trend = 0.0
    
    # Phasic 통계
    phasic_mean = float(np.mean(phasic_phase))
    phasic_std = float(np.std(phasic_phase))
    phasic_max = float(np.max(phasic_phase))
    
    # EDA 변동성 (미분)
    eda_derivative = np.diff(phasic_phase) / (1 / target_fs)
    eda_variability = float(np.std(eda_derivative))
    
    return {
        "scr_frequency": scr_frequency,
        "scr_amplitude": scr_amplitude,
        "tonic_trend": tonic_trend,
        "phasic_mean": phasic_mean,
        "phasic_std": phasic_std,
        "phasic_max": phasic_max,
        "eda_variability": eda_variability
    }


# =========================
# cvxEDA 입력 전처리 함수
# =========================
def prepare_signal_for_cvxeda(eda_ds, target_fs):
    """
    cvxEDA가 안정적으로 작동하도록 신호를 전처리합니다.
    - 신호를 [0, 1] 범위로 정규화
    - NaN/Inf 제거
    - 신호 크기 확인
    """
    
    # NaN/Inf 제거
    eda_clean = eda_ds.copy()
    eda_clean = np.nan_to_num(eda_clean, nan=0.0, posinf=1e6, neginf=0.0)
    
    # 음수 제거
    eda_clean = np.maximum(eda_clean, 0)
    
    # 신호 범위 확인
    eda_min = np.min(eda_clean)
    eda_max = np.max(eda_clean)
    eda_range = eda_max - eda_min
    
    # 신호가 너무 작으면 스케일업
    if eda_range < 0.001:
        eda_clean = eda_clean * 1000
        scale_factor = 1000
    elif eda_range < 0.01:
        eda_clean = eda_clean * 100
        scale_factor = 100
    else:
        scale_factor = 1
    
    # [0, 1] 범위로 정규화
    eda_min = np.min(eda_clean)
    eda_max = np.max(eda_clean)
    eda_range = eda_max - eda_min
    
    if eda_range > 0:
        eda_norm = (eda_clean - eda_min) / eda_range
    else:
        eda_norm = eda_clean
    
    # 안정성을 위해 [0.001, 0.999] 범위로 클립
    eda_norm = np.clip(eda_norm, 0.001, 0.999)
    
    return eda_norm, scale_factor


# =========================
# cvxEDA 분해 함수 (입력 전처리 포함)
# =========================
def decompose_eda_robust(eda_ds, target_fs, filename, log_file):
    """
    여러 방법으로 EDA를 분해합니다.
    """
    
    # ===== 1차: cvxEDA (입력 전처리 포함) =====
    try:
        # 신호 전처리
        eda_norm, scale_factor = prepare_signal_for_cvxeda(eda_ds, target_fs)
        
        # cvxEDA 실행
        result = list(cvxEDA(eda_norm, 1.0/target_fs))
        
        if len(result) < 3:
            raise ValueError(f"cvxEDA returned {len(result)} elements")
        
        r = result[0]  # phasic
        t = result[2]  # tonic
        
        # NaN/Inf 처리
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 음수 제거
        tonic = np.maximum(t, 0)
        phasic = np.maximum(r, 0)
        
        # 원래 스케일로 복원
        if scale_factor != 1:
            tonic = tonic / scale_factor
            phasic = phasic / scale_factor
        
        # 검증: 신호가 충분한지 확인
        signal_energy = np.sum(phasic**2) + np.sum(tonic**2)
        if signal_energy < 1e-10:
            raise ValueError("cvxEDA: insufficient signal energy")
        
        signal_range = np.max(eda_ds) - np.min(eda_ds)
        log_file.write(f"{filename}: ✓ cvxEDA | scale={scale_factor} | energy={signal_energy:.2e}\n")
        
        return tonic, phasic, "cvxEDA", signal_range
    
    except Exception as e:
        log_file.write(f"{filename}: ✗ cvxEDA - {str(e)}\n")
        pass
    
    # ===== 2차: Simple Filter (0.5Hz) =====
    try:
        log_file.write(f"  → 2차: Simple Filter (0.5Hz)\n")
        
        sos_lp = butter(2, 0.5, fs=target_fs, output='sos')
        tonic = sosfiltfilt(sos_lp, eda_ds)
        tonic = np.maximum(tonic, 0)
        
        phasic = eda_ds - tonic
        phasic = np.maximum(phasic, 0)
        
        signal_energy = np.sum(phasic**2) + np.sum(tonic**2)
        if signal_energy < 1e-10:
            raise ValueError("Simple filter: insufficient energy")
        
        signal_range = np.max(eda_ds) - np.min(eda_ds)
        log_file.write(f"  → Simple Filter (0.5Hz) 성공 | energy={signal_energy:.2e}\n")
        
        return tonic, phasic, "Simple_Filter_05Hz", signal_range
    
    except Exception as e:
        log_file.write(f"  ✗ Simple Filter (0.5Hz) - {str(e)}\n")
        pass
    
    # ===== 3차: Simple Filter (0.1Hz) =====
    try:
        log_file.write(f"  → 3차: Simple Filter (0.1Hz)\n")
        
        sos_vlow = butter(2, 0.1, fs=target_fs, output='sos')
        tonic = sosfiltfilt(sos_vlow, eda_ds)
        tonic = np.maximum(tonic, 0)
        
        phasic = eda_ds - tonic
        phasic = np.maximum(phasic, 0)
        
        signal_energy = np.sum(phasic**2) + np.sum(tonic**2)
        if signal_energy < 1e-10:
            raise ValueError("Simple filter (0.1Hz): insufficient energy")
        
        signal_range = np.max(eda_ds) - np.min(eda_ds)
        log_file.write(f"  → Simple Filter (0.1Hz) 성공 | energy={signal_energy:.2e}\n")
        
        return tonic, phasic, "Simple_Filter_01Hz", signal_range
    
    except Exception as e:
        log_file.write(f"  ✗ Simple Filter (0.1Hz) - {str(e)}\n")
        pass
    
    # ===== 4차: Savitzky-Golay 필터 =====
    try:
        log_file.write(f"  → 4차: Savitzky-Golay 필터\n")
        
        window_length = min(101, len(eda_ds) // 2)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < 5:
            raise ValueError("Signal too short for Savitzky-Golay")
        
        tonic = savgol_filter(eda_ds, window_length, 3)
        tonic = np.maximum(tonic, 0)
        
        phasic = eda_ds - tonic
        phasic = np.maximum(phasic, 0)
        
        signal_energy = np.sum(phasic**2) + np.sum(tonic**2)
        if signal_energy < 1e-10:
            raise ValueError("Savitzky-Golay: insufficient energy")
        
        signal_range = np.max(eda_ds) - np.min(eda_ds)
        log_file.write(f"  → Savitzky-Golay 성공 | window={window_length} | energy={signal_energy:.2e}\n")
        
        return tonic, phasic, "Savitzky_Golay", signal_range
    
    except Exception as e:
        log_file.write(f"  ✗ Savitzky-Golay - {str(e)}\n")
        pass
    
    # ===== 5차: Moving Average =====
    try:
        log_file.write(f"  → 5차: Moving Average\n")
        
        window = min(600, len(eda_ds) // 3)
        if window < 2:
            raise ValueError("Signal too short for moving average")
        
        tonic = pd.Series(eda_ds).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        tonic = np.maximum(tonic, 0)
        
        phasic = eda_ds - tonic
        phasic = np.maximum(phasic, 0)
        
        signal_energy = np.sum(phasic**2) + np.sum(tonic**2)
        if signal_energy < 1e-10:
            raise ValueError("Moving average: insufficient energy")
        
        signal_range = np.max(eda_ds) - np.min(eda_ds)
        log_file.write(f"  → Moving Average 성공 | window={window} | energy={signal_energy:.2e}\n")
        
        return tonic, phasic, "Moving_Average", signal_range
    
    except Exception as e:
        log_file.write(f"  ✗ Moving Average - {str(e)}\n")
        pass
    
    # ===== 6차: Original EDA =====
    try:
        log_file.write(f"  → 6차: Original EDA\n")
        
        sos_ultra = butter(2, 0.05, fs=target_fs, output='sos')
        tonic = sosfiltfilt(sos_ultra, eda_ds)
        tonic = np.maximum(tonic, 0)
        
        phasic = eda_ds.copy()
        phasic = np.maximum(phasic, 0)
        
        signal_energy = np.sum(phasic**2) + np.sum(tonic**2)
        signal_range = np.max(eda_ds) - np.min(eda_ds)
        log_file.write(f"  → Original EDA | energy={signal_energy:.2e}\n")
        
        return tonic, phasic, "Original_EDA", signal_range
    
    except Exception as e:
        log_file.write(f"  ✗ Original EDA - {str(e)}\n")
        pass
    
    # ===== 모두 실패: Zero 반환 =====
    log_file.write(f"  ✗ 모든 방법 실패\n")
    signal_range = np.max(eda_ds) - np.min(eda_ds)
    
    return np.zeros_like(eda_ds), np.zeros_like(eda_ds), "Failed_All_Methods", signal_range


# =========================
# MAIN
# =========================
rows = []
failed_files = []
method_stats = {}

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")])

# 로그 파일 생성
log_path = os.path.join(LOG_DIR, "eda_processing_log.txt")

print(f"\n총 {len(files)}개 파일\n")

with open(log_path, "w", encoding="utf-8") as log_file:
    log_file.write("="*80 + "\n")
    log_file.write("EDA 전처리 로그 (안정성 강화)\n")
    log_file.write("="*80 + "\n\n")

for idx, fname in enumerate(files, 1):

    print(f"[{idx}/{len(files)}] {fname}", end=" ", flush=True)

    file_path = os.path.join(INPUT_DIR, fname)
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")

    subject = parts[0]
    name = "_".join(parts[1:-1])
    condition = parts[-1]

    try:
        df, desc = load_acq_txt(file_path)

        # -------- EDA channel 찾기 --------
        eda_idx = None
        for i, d in enumerate(desc):
            if "EDA" in d:
                eda_idx = i
                break

        if eda_idx is None:
            print("❌ EDA 없음")
            failed_files.append((fname, "EDA 채널 없음"))
            continue

        eda_col = df.columns[eda_idx + 1]
        eda_raw = df[eda_col].values
        time = df["sec"].values

        fs_raw = round(1 / np.mean(np.diff(time)))

        # =========================
        # 1. Artifact 제거
        # =========================
        eda = eda_raw.copy()
        
        eda[eda < 0] = np.nan
        
        Q1 = np.nanpercentile(eda, 25)
        Q3 = np.nanpercentile(eda, 75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            eda[(eda < lower_bound) | (eda > upper_bound)] = np.nan
        
        eda = pd.Series(eda).interpolate(method='linear').bfill().ffill().values

        # =========================
        # 2. 저주파 필터링
        # =========================
        sos = butter(4, 5, fs=fs_raw, output='sos')
        eda_filtered = sosfiltfilt(sos, eda)

        # =========================
        # 3. Downsample
        # =========================
        factor = int(fs_raw / TARGET_FS)
        if factor > 1:
            eda_ds = decimate(eda_filtered, factor, zero_phase=True)
        else:
            eda_ds = eda_filtered.copy()

        # =========================
        # 4. 수치 안정화
        # =========================
        eda_ds = eda_ds - np.min(eda_ds) + 1e-6

        # =========================
        # 5. EDA 분해 (안정성 강화)
        # =========================
        with open(log_path, "a", encoding="utf-8") as log_file:
            tonic, phasic, method, signal_range = decompose_eda_robust(
                eda_ds, TARGET_FS, fname, log_file
            )

        # 통계 기록
        if method not in method_stats:
            method_stats[method] = 0
        method_stats[method] += 1

        # 최종 NaN 방지
        tonic = np.nan_to_num(tonic, nan=0.0)
        phasic = np.nan_to_num(phasic, nan=0.0)

        # =========================
        # 6. 최소 샘플 검증
        # =========================
        samples_per_min = TARGET_FS * 60
        required_samples = MINUTES * samples_per_min
        if len(eda_ds) < required_samples:
            print(f"❌ 데이터 부족")
            failed_files.append((fname, f"데이터 부족: {len(eda_ds)}/{required_samples}"))
            continue

        # =========================
        # 7. 1분 평균
        # =========================
        row = {
            "subject": subject,
            "name": name,
            "condition": condition,
            "decomposition_method": method,
            "signal_range": signal_range
        }

        # tonic 1~20분
        for m in range(MINUTES):
            start = m * samples_per_min
            end = (m + 1) * samples_per_min
            row[f"tonic_{m+1}"] = float(np.mean(tonic[start:end]))

        # phasic 1~20분
        for m in range(MINUTES):
            start = m * samples_per_min
            end = (m + 1) * samples_per_min
            row[f"phasic_{m+1}"] = float(np.mean(phasic[start:end]))

        # =========================
        # 8. 구간별 추가 지표
        # =========================
        metrics_first_half = calculate_additional_metrics(phasic, tonic, TARGET_FS, 0, 10)
        for key, value in metrics_first_half.items():
            row[f"{key}_1to10"] = value

        metrics_second_half = calculate_additional_metrics(phasic, tonic, TARGET_FS, 10, 20)
        for key, value in metrics_second_half.items():
            row[f"{key}_11to20"] = value

        rows.append(row)
        
        print(f"✔ [{method}]")

    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        failed_files.append((fname, str(e)[:100]))
        continue

# =========================
# 통계 출력
# =========================
with open(log_path, "a", encoding="utf-8") as log_file:
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("처리 결과 요약\n")
    log_file.write("="*80 + "\n\n")
    log_file.write(f"✓ 성공: {len(rows)}개\n")
    log_file.write(f"✗ 실패: {len(failed_files)}개\n\n")
    
    log_file.write("분해 방법 통계:\n")
    for method, count in sorted(method_stats.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(rows) if len(rows) > 0 else 0
        log_file.write(f"  {method}: {count}개 ({pct:.1f}%)\n")
    
    if failed_files:
        log_file.write(f"\n실패 파일 목록 ({len(failed_files)}개):\n")
        for fname, reason in failed_files:
            log_file.write(f"  - {fname}: {reason}\n")

print("\n" + "="*80)
print("분해 방법 통계:")
print("="*80)
for method, count in sorted(method_stats.items(), key=lambda x: x[1], reverse=True):
    pct = 100 * count / len(rows) if len(rows) > 0 else 0
    print(f"  {method}: {count}개 ({pct:.1f}%)")

# =========================
# SAVE
# =========================
df_out = pd.DataFrame(rows)

tonic_cols = [f"tonic_{i}" for i in range(1, 21)]
phasic_cols = [f"phasic_{i}" for i in range(1, 21)]
metrics_1to10 = [
    "scr_frequency_1to10", "scr_amplitude_1to10", "tonic_trend_1to10",
    "phasic_mean_1to10", "phasic_std_1to10", "phasic_max_1to10", "eda_variability_1to10"
]
metrics_11to20 = [
    "scr_frequency_11to20", "scr_amplitude_11to20", "tonic_trend_11to20",
    "phasic_mean_11to20", "phasic_std_11to20", "phasic_max_11to20", "eda_variability_11to20"
]

df_out = df_out[
    ["subject", "name", "condition", "decomposition_method", "signal_range"] + 
    tonic_cols + 
    phasic_cols + 
    metrics_1to10 +
    metrics_11to20
]

output_path = os.path.join(FEATURE_DIR, "eda_cvxEDA_1min_error.xlsx")
df_out.to_excel(output_path, index=False)

print("\n" + "="*80)
print("=== ALL DONE ===")
print("="*80)
print(f"✓ Saved to: {output_path}")
print(f"✓ 처리 성공: {len(df_out)}개 파일")
print(f"✗ 처리 실패: {len(failed_files)}개 파일")
print(f"📋 로그 파일: {log_path}")
print("="*80)
