# Biosignal Preprocessing Summary

이 문서는 각 생체신호가 **어느 수준의 전처리 단계까지 수행되었는지**를 기준으로 정리한 것이다.  
(구체적 알고리즘 설명이 아니라 **처리 수준(level)** 중심)

---

# 1. 전처리 수준 비교

| 단계 | ECG | EDA | SKT |
|---|---|---|---|
| Artifact 제거 | ✓ | ✓ | 일부 |
| Outlier 제거 | ✓ | ✓ | ✗ |
| Missing value 보간 | ✗ | ✓ | ✗ |
| Baseline drift 처리 | ✓ | ✓ | ✗ |
| Filtering | ✓ | ✓ | ✗ |
| Downsampling / Resampling | ✗ | ✓ | ✓ |
| Smoothing | ✓ | 일부 | ✓ |
| Normalization | ✓ | ✗ | ✗ |
| Signal decomposition | ✗ | ✓ | ✗ |
| Interval 계산 | ✓ | ✗ | ✗ |
| Physiological validation | ✓ | ✗ | ✗ |
| Time-window summary | ✓ | ✓ | ✓ |

---

# 2. ECG 처리 수준

## Signal Cleaning
- baseline drift 제거
- powerline noise 제거
- bandpass filtering
- smoothing
- normalization

## Signal Quality
- SNR 계산
- baseline drift 측정
- kurtosis 측정
- low quality signal 제외

## Event Detection
- R-peak detection

## Interval Processing
- RR interval 계산
- physiological range filtering
- outlier 제거

## Feature Level
- HRV 계산

### 시간 해상도
- **1 minute window**
- **20 minute duration**

---

# 3. EDA 처리 수준

## Artifact Cleaning
- 음수값 제거
- ##### 얘가 문제 같은데요?
- extreme outlier 제거
- missing value interpolation

## Signal Conditioning
- low-pass filtering
- downsampling
- numerical stabilization

## Signal Decomposition
EDA → 두 성분으로 분해

- **tonic**
- **phasic**

## Event Detection
- SCR peak detection

## Feature Level

### 1분 단위
- tonic mean
- phasic mean

### 10분 단위
- SCR frequency
- SCR amplitude
- tonic trend
- phasic statistics
- EDA variability

---

# 4. SKT 처리 수준

## Signal Conditioning
- resampling
- moving average smoothing

## Feature Level

전체 신호를 **4구간으로 분할**

- quarter 1
- quarter 2
- quarter 3
- quarter 4

각 구간 평균 온도 계산

### 출력
- ST quarter mean
- AT quarter mean

---
