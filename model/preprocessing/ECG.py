import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
import scipy
warnings.filterwarnings('ignore')

# =========================
# SETTINGS
# =========================
INPUT_DIR = r"C:\Project\mbti\01_data_normalized\ecg"
FEATURE_DIR = r"C:\Project\mbti\03_features"
DEBUG_DIR = r"C:\Project\mbti\04_debug"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

MINUTES = 20
DEBUG_MODE = False
RESAMPLE_RATE = 4

# scipy 버전 확인
SCIPY_VERSION = tuple(map(int, scipy.__version__.split('.')[:2]))
print(f"SciPy 버전: {scipy.__version__}")
print(f"호환성 확인: scipy >= 1.8 ? {SCIPY_VERSION >= (1, 8)}\n")

# =========================
# ECG 신호 품질 진단
# =========================
def diagnose_signal_quality(ecg, fs):
    """
    ECG 신호의 품질을 다양한 지표로 평가
    """
    diagnostics = {}
    
    fft_vals = np.abs(fft(ecg))
    freqs = fftfreq(len(ecg), 1/fs)
    
    signal_band = (freqs > 5) & (freqs < 40)
    noise_band = ((freqs > 0.5) & (freqs < 3)) | ((freqs > 45) & (freqs < 100))
    
    signal_power = np.mean(fft_vals[signal_band]**2) if signal_band.sum() > 0 else 0
    noise_power = np.mean(fft_vals[noise_band]**2) if noise_band.sum() > 0 else 1
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    diagnostics['snr'] = float(snr)
    
    baseline = signal.savgol_filter(ecg, window_length=min(501, len(ecg)//10*2+1), polyorder=3)
    baseline_drift = np.std(baseline)
    diagnostics['baseline_drift'] = float(baseline_drift)
    
    kurtosis = ((ecg - np.mean(ecg))**4).mean() / (np.std(ecg)**4 + 1e-10)
    diagnostics['kurtosis'] = float(kurtosis)
    
    diagnostics['signal_range'] = float(np.max(np.abs(ecg)))
    diagnostics['signal_std'] = float(np.std(ecg))
    
    return diagnostics


# =========================
# 적응형 ECG 전처리
# =========================
def preprocess_ecg_advanced(ecg, fs, snr, debug_fname=""):
    """
    SNR에 따라 적응형 전처리 적용
    
    SNR < -15: 강력한 필터링
    SNR < -10: 중간 필터링
    SNR >= -10: 표준 필터링
    """
    
    ecg_processed = ecg.copy()
    
    # SNR에 따른 적응형 필터 설정
    if snr < -15:
        # 매우 낮은 SNR: 강력한 필터링
        filter_order = 5
        highpass_freq = 1.0
        lowpass_freq = 35
        smooth_window_factor = 10
    elif snr < -10:
        # 낮은 SNR: 중간 정도 필터링
        filter_order = 4
        highpass_freq = 0.7
        lowpass_freq = 40
        smooth_window_factor = 15
    else:
        # 보통 SNR: 표준 필터링
        filter_order = 4
        highpass_freq = 0.5
        lowpass_freq = 40
        smooth_window_factor = 20
    
    # Step 1: 고통과 필터 (기선 드리프트 제거)
    sos = signal.butter(filter_order, highpass_freq, 'hp', fs=fs, output='sos')
    ecg_detrend = signal.sosfilt(sos, ecg_processed)
    
    # Step 2: 전력선 잡음 제거 (50/60 Hz)
    for power_freq in [50, 60]:
        low_freq = max(1, power_freq - 2)
        high_freq = min(fs/2 - 1, power_freq + 2)
        
        if low_freq < high_freq:
            try:
                sos = signal.butter(4, [low_freq, high_freq], 'bs', fs=fs, output='sos')
                ecg_detrend = signal.sosfilt(sos, ecg_detrend)
            except Exception as e:
                pass
    
    # Step 3: 밴드패스 필터
    sos = signal.butter(filter_order, [5, lowpass_freq], 'bp', fs=fs, output='sos')
    ecg_filtered = signal.sosfilt(sos, ecg_detrend)
    
    # Step 4: 강화된 평활화 (SNR에 따라 조정)
    if len(ecg_filtered) > 51:
        window_len = len(ecg_filtered) // smooth_window_factor * 2 + 1
        window_len = min(window_len, 101)
        
        if window_len < 5:
            window_len = 5
        if window_len % 2 == 0:
            window_len += 1
        
        ecg_smooth = signal.savgol_filter(ecg_filtered, window_length=int(window_len), polyorder=2)
    else:
        ecg_smooth = ecg_filtered
    
    # Step 5: 정규화
    ecg_norm = (ecg_smooth - np.mean(ecg_smooth)) / (np.std(ecg_smooth) + 1e-10)
    
    # 디버그 시각화
    if DEBUG_MODE and debug_fname:
        fig, axes = plt.subplots(6, 1, figsize=(14, 10))
        time_axis = np.arange(len(ecg)) / fs
        
        axes[0].plot(time_axis, ecg, 'b-', linewidth=0.5)
        axes[0].set_title(f'{debug_fname} - Original ECG (SNR={snr:.1f})')
        axes[0].set_ylabel('Amplitude')
        
        axes[1].plot(time_axis, ecg_detrend, 'g-', linewidth=0.5)
        axes[1].set_title('After High-pass & Bandstop Filter')
        axes[1].set_ylabel('Amplitude')
        
        axes[2].plot(time_axis, ecg_filtered, 'r-', linewidth=0.5)
        axes[2].set_title(f'After Bandpass Filter (5-{lowpass_freq} Hz)')
        axes[2].set_ylabel('Amplitude')
        
        axes[3].plot(time_axis, ecg_smooth, 'purple', linewidth=0.5)
        axes[3].set_title('After Smoothing')
        axes[3].set_ylabel('Amplitude')
        
        axes[4].plot(time_axis, ecg_norm, 'brown', linewidth=0.5)
        axes[4].set_title('Normalized Signal')
        axes[4].set_ylabel('Z-score')
        
        fft_orig = np.abs(fft(ecg))
        fft_processed = np.abs(fft(ecg_norm))
        freqs = fftfreq(len(ecg), 1/fs)
        
        axes[5].semilogy(freqs[:len(freqs)//2], fft_orig[:len(fft_orig)//2], 'b-', label='Original', linewidth=0.8)
        axes[5].semilogy(freqs[:len(freqs)//2], fft_processed[:len(fft_processed)//2], 'r-', label='Processed', linewidth=0.8)
        axes[5].set_xlim([0, 100])
        axes[5].set_xlabel('Frequency (Hz)')
        axes[5].set_ylabel('Power')
        axes[5].legend()
        axes[5].set_title('FFT Comparison')
        
        plt.tight_layout()
        debug_path = os.path.join(DEBUG_DIR, f"{debug_fname}_preprocessing.png")
        plt.savefig(debug_path, dpi=100)
        plt.close()
    
    return ecg_norm


# =========================
# 적응형 R-peak 검출
# =========================
def detect_rpeaks_robust(ecg_clean, fs, snr):
    """
    SNR에 따라 적응형 R-peak 검출
    여러 방법을 시도하고 최적의 결과 선택
    """
    
    rpeaks_results = {}
    
    # 방법 1: neurokit
    try:
        signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method='neurokit')
        rpeaks_results['neurokit'] = info.get("ECG_R_Peaks", np.array([]))
    except Exception as e:
        rpeaks_results['neurokit'] = np.array([])
    
    # 방법 2: Pan-Tompkins
    try:
        signals_pt, info_pt = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method='pantompkins')
        rpeaks_results['pantompkins'] = info_pt.get("ECG_R_Peaks", np.array([]))
    except Exception as e:
        rpeaks_results['pantompkins'] = np.array([])
    
    # 방법 3: find_peaks (SNR이 낮으면 더 적극적)
    try:
        if len(ecg_clean) > 0:
            min_distance = int(fs * 0.4)
            
            # SNR에 따라 prominence 조정
            if snr < -15:
                # 매우 낮은 SNR: 낮은 prominence (더 많은 피크 감지)
                prominence = np.std(ecg_clean) * 0.15
            elif snr < -10:
                # 낮은 SNR: 중간 prominence
                prominence = np.std(ecg_clean) * 0.30
            else:
                # 보통: 높은 prominence
                prominence = np.std(ecg_clean) * 0.5
            
            peaks, _ = signal.find_peaks(ecg_clean, distance=min_distance, prominence=prominence)
            rpeaks_results['findpeaks'] = peaks
        else:
            rpeaks_results['findpeaks'] = np.array([])
    except Exception as e:
        rpeaks_results['findpeaks'] = np.array([])
    
    # 가장 많은 피크를 검출한 방법 선택
    best_method = max(rpeaks_results, key=lambda k: len(rpeaks_results[k]))
    rpeaks = rpeaks_results[best_method]
    
    return rpeaks, best_method


# =========================
# RR 간격 정제
# =========================
def validate_and_clean_rr(rr, fs):
    """
    RR 간격 검증 및 정제
    """
    
    rr_original_len = len(rr)
    
    # 1단계: 범위 필터링
    rr_valid = rr[(rr >= 300) & (rr <= 2000)]
    
    if len(rr_valid) < len(rr) * 0.6:
        rr_valid = rr[(rr >= 250) & (rr <= 2500)]
    
    if len(rr_valid) < 10:
        return rr_valid, {'removed': rr_original_len, 'reason': 'insufficient_data'}
    
    # 2단계: MAD 기반 이상치 제거
    median_rr = np.median(rr_valid)
    mad = np.median(np.abs(rr_valid - median_rr))
    
    threshold = median_rr + 3 * mad * 1.4826
    lower_threshold = median_rr - 3 * mad * 1.4826
    
    rr_cleaned = rr_valid[(rr_valid <= threshold) & (rr_valid >= lower_threshold)]
    
    if len(rr_cleaned) < 10:
        rr_cleaned = rr_valid
    
    stats = {
        'original': rr_original_len,
        'after_range_filter': len(rr_valid),
        'after_outlier_removal': len(rr_cleaned),
        'median_rr': float(median_rr),
        'mad': float(mad)
    }
    
    return rr_cleaned, stats


# =========================
# HRV 시간 영역 지표
# =========================
def calculate_hrv_time_domain(rr_seg):
    """
    HRV 시간 영역 지표 계산
    
    HR: 심박수 (bpm)
    MeanRR: 평균 RR 간격 (ms)
    SDNN: RR 간격의 표준편차 (ms)
    RMSSD: 연속 RR 간격 차이의 제곱 평균 제곱근 (ms)
    pNN50: 50ms 이상 차이나는 비율 (%)
    """
    
    if len(rr_seg) < 5:
        return {
            'HR': np.nan,
            'MeanRR': np.nan,
            'SDNN': np.nan,
            'RMSSD': np.nan,
            'pNN50': np.nan,
            'valid_beats': 0
        }
    
    # 기본 지표
    mean_rr = np.mean(rr_seg)
    hr = 60000 / mean_rr
    sdnn = np.std(rr_seg)
    
    # RR 간격 차이
    rr_diff = np.diff(rr_seg)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    
    # pNN50
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
    
    return {
        'HR': float(hr),
        'MeanRR': float(mean_rr),
        'SDNN': float(sdnn),
        'RMSSD': float(rmssd),
        'pNN50': float(pnn50),
        'valid_beats': len(rr_seg)
    }


# =========================
# HRV 주파수 영역 지표
# =========================
def calculate_hrv_frequency_domain(rr_seg, fs=4):
    """
    HRV 주파수 영역 지표 계산
    
    LF: 저주파 파워 (0.04-0.15 Hz)
    HF: 고주파 파워 (0.15-0.4 Hz)
    LF/HF: 교감/부교감 비율
    """
    
    if len(rr_seg) < 15:
        return {
            'LF': np.nan,
            'HF': np.nan,
            'LF_HF': np.nan,
            'LF_norm': np.nan,
            'HF_norm': np.nan
        }
    
    try:
        # RR을 균등 간격으로 리샘플
        time_rr = np.cumsum(rr_seg) / 1000  # 초 단위
        
        # 충분한 데이터 확인
        if len(time_rr) < 10 or time_rr[-1] < 10:
            return {
                'LF': np.nan,
                'HF': np.nan,
                'LF_HF': np.nan,
                'LF_norm': np.nan,
                'HF_norm': np.nan
            }
        
        f_interp = interp1d(time_rr, rr_seg, kind='linear', fill_value='extrapolate')
        time_uniform = np.arange(0, time_rr[-1], 1/fs)
        rr_resampled = f_interp(time_uniform)
        
        # 정규화
        rr_detrend = rr_resampled - np.mean(rr_resampled)
        
        # Welch 파워 스펙트럼 밀도
        freqs, psd = signal.welch(rr_detrend, fs=fs, nperseg=min(128, len(rr_detrend)))
        
        # 대역 정의
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.sum() > 0 else 0
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.sum() > 0 else 0
        
        # LF/HF 비율
        lf_hf_ratio = lf_power / (hf_power + 1e-10)
        
        # 정규화
        total_lf_hf = lf_power + hf_power
        lf_norm = lf_power / (total_lf_hf + 1e-10) * 100 if total_lf_hf > 0 else 0
        hf_norm = hf_power / (total_lf_hf + 1e-10) * 100 if total_lf_hf > 0 else 0
        
        return {
            'LF': float(lf_power),
            'HF': float(hf_power),
            'LF_HF': float(lf_hf_ratio),
            'LF_norm': float(lf_norm),
            'HF_norm': float(hf_norm)
        }
        
    except Exception as e:
        return {
            'LF': np.nan,
            'HF': np.nan,
            'LF_HF': np.nan,
            'LF_norm': np.nan,
            'HF_norm': np.nan
        }


# =========================
# BIOPAC LOADER
# =========================
def load_acq_txt(file_path):
    """
    BIOPAC .txt 파일 로드
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            lines = f.readlines()

        header_idx = None
        for i, line in enumerate(lines):
            parts = line.strip().split("\t")
            if len(parts) > 1 and parts[0].lower() == "sec":
                header_idx = i
                header_cols = parts
                break

        if header_idx is None:
            return None, None

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

        if not data_rows:
            return None, None

        df = pd.DataFrame(data_rows, columns=header_cols)
        df = df.apply(pd.to_numeric, errors="coerce")

        return df, desc
    
    except Exception as e:
        return None, None


# =========================
# MAIN
# =========================
rows = []
failed_files = []
debug_count = 0
low_snr_count = 0

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")])

print(f"\n{'='*80}")
print(f"총 {len(files)}개 파일 처리 시작")
print(f"{'='*80}\n")

for idx, fname in enumerate(files, 1):

    print(f"[{idx:3d}/{len(files)}] {fname:50s}", end=" → ", flush=True)

    try:
        file_path = os.path.join(INPUT_DIR, fname)
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")

        subject = parts[0]
        name = "_".join(parts[1:-1])
        condition = parts[-1]

        result = load_acq_txt(file_path)
        if result[0] is None:
            print("❌ 파일 로드 실패")
            failed_files.append((fname, "파일 형식 오류"))
            continue

        df, desc = result

        # ECG 채널 찾기
        ecg_idx = None
        for i, d in enumerate(desc):
            if "ECG" in d:
                ecg_idx = i
                break

        if ecg_idx is None:
            print("❌ ECG 없음")
            failed_files.append((fname, "ECG 채널 없음"))
            continue

        ecg_col = df.columns[ecg_idx + 1]
        ecg = df[ecg_col].values
        time = df["sec"].values

        # 샘플링 주파수 계산
        try:
            fs = round(1 / np.mean(np.diff(time)))
        except:
            fs = 250

        # 신호 품질 진단
        diagnostics = diagnose_signal_quality(ecg, fs)
        
        # SNR 기준 완화 (-25 이하만 제외)
        if diagnostics['snr'] < -25:
            print(f"❌ SNR={diagnostics['snr']:.1f} (심각) → skip")
            failed_files.append((fname, f"SNR={diagnostics['snr']:.1f}"))
            continue

        # SNR이 낮으면 카운트
        if diagnostics['snr'] < -10:
            quality_indicator = "⚠"
            low_snr_count += 1
        else:
            quality_indicator = "✔"

        # 적응형 고급 전처리
        debug_mode_for_file = DEBUG_MODE and debug_count < 5
        ecg_clean = preprocess_ecg_advanced(
            ecg, fs, 
            snr=diagnostics['snr'],
            debug_fname=stem if debug_mode_for_file else ""
        )
        
        if debug_mode_for_file:
            debug_count += 1

        # 적응형 R-peak 검출
        rpeaks, rpeak_method = detect_rpeaks_robust(ecg_clean, fs, snr=diagnostics['snr'])

        if len(rpeaks) < 10:
            print(f"{quality_indicator} R-peak 부족 ({len(rpeaks)}개) → skip")
            failed_files.append((fname, f"R-peak {len(rpeaks)}개"))
            continue

        # RR 간격 계산 및 정제
        rr_full = np.diff(rpeaks) / fs * 1000
        rr_cleaned, rr_stats = validate_and_clean_rr(rr_full, fs)

        if len(rr_cleaned) < 20:
            print(f"{quality_indicator} 정제된 RR 부족 → skip")
            failed_files.append((fname, f"Cleaned RR {len(rr_cleaned)}개"))
            continue

        # 메타정보
        row = {
            "subject": subject,
            "name": name,
            "condition": condition,
            "fs": fs,
            "total_rpeaks": len(rpeaks),
            "rpeak_method": rpeak_method,
            "snr": diagnostics['snr'],
            "baseline_drift": diagnostics['baseline_drift'],
            "kurtosis": diagnostics['kurtosis'],
            "total_valid_rr": len(rr_cleaned)
        }

        # 분 단위 HRV 계산
        for m in range(1, MINUTES + 1):

            start_sec = (m - 1) * 60
            end_sec = m * 60

            mask = (rpeaks / fs >= start_sec) & (rpeaks / fs < end_sec)
            r_seg = rpeaks[mask]

            if len(r_seg) < 5:
                # 데이터 부족시 NaN
                row[f"HR_{m}"] = np.nan
                row[f"MeanRR_{m}"] = np.nan
                row[f"SDNN_{m}"] = np.nan
                row[f"RMSSD_{m}"] = np.nan
                row[f"pNN50_{m}"] = np.nan
                row[f"LF_{m}"] = np.nan
                row[f"HF_{m}"] = np.nan
                row[f"LF_HF_{m}"] = np.nan
                row[f"valid_beats_{m}"] = 0
                continue

            rr_seg = np.diff(r_seg) / fs * 1000
            
            # 분 단위 범위 필터링
            rr_seg = rr_seg[(rr_seg >= 300) & (rr_seg <= 2000)]

            if len(rr_seg) < 5:
                row[f"HR_{m}"] = np.nan
                row[f"MeanRR_{m}"] = np.nan
                row[f"SDNN_{m}"] = np.nan
                row[f"RMSSD_{m}"] = np.nan
                row[f"pNN50_{m}"] = np.nan
                row[f"LF_{m}"] = np.nan
                row[f"HF_{m}"] = np.nan
                row[f"LF_HF_{m}"] = np.nan
                row[f"valid_beats_{m}"] = 0
                continue

            # ===== 시간 영역 분석 =====
            hrv_time = calculate_hrv_time_domain(rr_seg)
            row[f"HR_{m}"] = hrv_time['HR']
            row[f"MeanRR_{m}"] = hrv_time['MeanRR']
            row[f"SDNN_{m}"] = hrv_time['SDNN']
            row[f"RMSSD_{m}"] = hrv_time['RMSSD']
            row[f"pNN50_{m}"] = hrv_time['pNN50']
            row[f"valid_beats_{m}"] = hrv_time['valid_beats']
            
            # ===== 주파수 영역 분석 =====
            hrv_freq = calculate_hrv_frequency_domain(rr_seg, fs=RESAMPLE_RATE)
            row[f"LF_{m}"] = hrv_freq['LF']
            row[f"HF_{m}"] = hrv_freq['HF']
            row[f"LF_HF_{m}"] = hrv_freq['LF_HF']

        rows.append(row)
        print(f"{quality_indicator} 완료 (SNR={diagnostics['snr']:.1f})")

    except Exception as e:
        error_msg = str(e)[:30]
        print(f"❌ {error_msg}")
        failed_files.append((fname, str(e)[:50]))
        continue

# =========================
# 결과 저장
# =========================
if rows:
    df_out = pd.DataFrame(rows)

    # 컬럼 순서 정의
    meta_cols = [
        "subject", "name", "condition", "fs", "total_rpeaks", 
        "rpeak_method", "snr", "baseline_drift", "kurtosis", "total_valid_rr"
    ]
    
    # HRV 지표 컬럼 (분 단위)
    hr_cols = [f"HR_{i}" for i in range(1, MINUTES + 1)]
    meanrr_cols = [f"MeanRR_{i}" for i in range(1, MINUTES + 1)]
    sdnn_cols = [f"SDNN_{i}" for i in range(1, MINUTES + 1)]
    rmssd_cols = [f"RMSSD_{i}" for i in range(1, MINUTES + 1)]
    pnn50_cols = [f"pNN50_{i}" for i in range(1, MINUTES + 1)]
    lf_cols = [f"LF_{i}" for i in range(1, MINUTES + 1)]
    hf_cols = [f"HF_{i}" for i in range(1, MINUTES + 1)]
    lfhf_cols = [f"LF_HF_{i}" for i in range(1, MINUTES + 1)]
    valid_beats_cols = [f"valid_beats_{i}" for i in range(1, MINUTES + 1)]

    col_order = (meta_cols + hr_cols + meanrr_cols + sdnn_cols + rmssd_cols + 
                 pnn50_cols + lf_cols + hf_cols + lfhf_cols + valid_beats_cols)
    
    df_out = df_out[col_order]

    output_path = os.path.join(FEATURE_DIR, "ecg_hrv_1min_error.xlsx")
    df_out.to_excel(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"✔ {len(rows)}개 파일 저장 완료")
    print(f"경로: {output_path}")
    print(f"{'='*80}")
    
    # 통계 정보 출력
    print(f"\n📊 추출된 HRV 지표:")
    print(f"  ├─ 시간 영역: HR, MeanRR, SDNN, RMSSD, pNN50")
    print(f"  ├─ 주파수 영역: LF, HF, LF/HF")
    print(f"  ├─ 분석 단위: 1분 × {MINUTES}분")
    print(f"  └─ 총 {len(df_out.columns) - len(meta_cols)} 개의 HRV 컬럼")
    
    print(f"\n📈 데이터 통계:")
    print(f"  ├─ 총 행: {len(df_out)}")
    print(f"  ├─ 총 열: {len(df_out.columns)}")
    print(f"  ├─ 낮은 SNR 파일: {low_snr_count}개")
    
    # NaN 비율 계산
    total_cells = len(df_out) * len(df_out.columns)
    nan_cells = df_out.isna().sum().sum()
    nan_percent = (nan_cells / total_cells) * 100
    print(f"  ├─ NaN 비율: {nan_percent:.2f}%")
    print(f"  └─ 유효 데이터: {(100 - nan_percent):.2f}%")
    
    # 기본 통계
    print(f"\n📋 기본 통계:")
    hr_mean = df_out[[f"HR_{i}" for i in range(1, MINUTES + 1)]].values.flatten()
    hr_mean = hr_mean[~np.isnan(hr_mean)]
    
    sdnn_mean = df_out[[f"SDNN_{i}" for i in range(1, MINUTES + 1)]].values.flatten()
    sdnn_mean = sdnn_mean[~np.isnan(sdnn_mean)]
    
    rmssd_mean = df_out[[f"RMSSD_{i}" for i in range(1, MINUTES + 1)]].values.flatten()
    rmssd_mean = rmssd_mean[~np.isnan(rmssd_mean)]
    
    lfhf_mean = df_out[[f"LF_HF_{i}" for i in range(1, MINUTES + 1)]].values.flatten()
    lfhf_mean = lfhf_mean[~np.isnan(lfhf_mean)]
    
    print(f"  ├─ 평균 HR: {np.mean(hr_mean):.2f} ± {np.std(hr_mean):.2f} bpm")
    print(f"  ├─ 평균 SDNN: {np.mean(sdnn_mean):.2f} ± {np.std(sdnn_mean):.2f} ms")
    print(f"  ├─ 평균 RMSSD: {np.mean(rmssd_mean):.2f} ± {np.std(rmssd_mean):.2f} ms")
    print(f"  └─ 평균 LF/HF: {np.mean(lfhf_mean):.2f} ± {np.std(lfhf_mean):.2f}")

else:
    print(f"\n{'='*80}")
    print(f"❌ 처리된 파일이 없습니다")
    print(f"{'='*80}")

# 실패 리포트
if failed_files:
    print(f"\n⚠ 처리 실패: {len(failed_files)}개 파일")
    print(f"{'='*80}")
    for fname, reason in failed_files[:20]:
        print(f"  - {fname:50s} : {reason}")
    if len(failed_files) > 20:
        print(f"  ... 외 {len(failed_files)-20}개")
    print(f"{'='*80}")
else:
    print(f"\n{'='*80}")
    print(f"✔ 모든 파일이 성공적으로 처리되었습니다!")
    print(f"{'='*80}")

print(f"\n처리 완료!\n")
