# META Method Suitability Heatmaps (MBTI/FFM × Thermal Responses)

이 프로젝트는 “어떤 통계/ML 방법론을 먼저 테스트할지”를 빠르게 정하기 위해,  
**X(성격 데이터 형태)** × **Y(반응 데이터 형태)** 매트릭스 위에  
**Z(모델/방법론)** 별 **적합도 점수(0~4)** 를 부여하고, Z마다 히트맵 1장을 생성한다.

> 핵심: 유의성(p-value) 히트맵이 아니라, **방법론 적합성(suitability)** 히트맵이다.

---

## 1) 데이터 가정 및 용어

### 1.1 시트0(반복측정 온열 실험)
- `연번`: subject ID
- `환경`: A~H → 세트 `AB/CD/EF/GH`로 묶음

**각자 봐야 되는디...?**

- `시간`: 0, 5, 10, 15, 20
- `phase`: 
  - static: 0/5/10
  - dynamic: 10/15/20  
  (시간=10이 static/dynamic 양쪽에 포함되는 설계; 평균에서는 겹치지만 “전환점” 의미가 있으므로 그대로 두는 정책)

### 1.2 온도 시나리오(세트/방향)
- set:
  - AB: 23 → (ramp) 18 (cooling)
  - CD: 18 → (ramp) 23 (heating)
  - EF: 23 → (ramp) 28 (heating)
  - GH: 28 → (ramp) 23 (cooling)
- dir01:
  - cooling(AB/GH) = 0
  - heating(CD/EF) = 1

---

## 2) 목적: X × Y × Z 프레임

- **X (성격 특성 형태)**
  - X1_continuous: MBTI/FFM 점수(표준화 z-score)
  - X2_binary_cut50: 각 축 점수 ≥50 vs <50 (0/1)
  - X3_type16: (e,n,t,j) cut50로 16유형(ENxJ 등) 생성
  - X4_interactions: 축×축 상호작용, 축×dir 상호작용

- **Y (반응 데이터 형태)**
  - Y1_raw_level: 원자료(반복측정 수준) (TSV,TCV,TA,TP,PT,P1~8,M1~8)
  - Y2_reactivity_delta: 정적→동적 반응성 Δ = dynamic_mean - static_mean
  - Y3_mismatch_multi: 불일치/미스매치(연속형 + Δgap 등 엔지니어드 조합)
  - Y4_pattern_multilevel_pca: 패턴(다단 레벨 PCA에서 나온 PC들)
  - Y5_binary_multi: 이항 사건(열적 무기력 등 여러 정의)

- **Z (모델/방법론)**
  - Z1 Spearman
  - Z2 Partial correlation
  - Z3 LMM (Linear Mixed Model)
  - Z4 GAMM (Generalized Additive Mixed Model)
  - Z5 GEE
  - Z6 GLMM logistic
  - Z7 Rare-event logistic (Firth 보정)
  - Z8 PCA/FA
  - Z9 Clustering
  - Z10 Supervised ML

---

## 3) 적합도(0~4) 판단 기준

각 셀(= X형태 × Y형태)에 대해, Z모델이 얼마나 “먼저 시도할 가치가 있는지”를 평가한다.

### 3.1 점수 구조(통일)
각 셀 점수 = 아래 4가지 하위 점수의 합 (각 0~1) → **총 0~4**
1) **Design fit**: 반복측정/계층구조/온도시나리오 같은 설계를 모델이 제대로 소화하는가?
2) **Goal fit**: 현재 목표(관계 탐색, 반응성, 미스매치, 패턴, 이항 사건)에 잘 맞는가?
3) **Robustness**: 결측/희귀사건/분리(separation)/불균형/다중공선성에 얼마나 견고한가?
4) **Interpretability**: 논문화/설득력(계수 해석, 효과 크기, 신뢰구간, 도메인 의미 부여)이 쉬운가?

> 중요: 이 점수는 “통계적 유의성”이 아니라,  
> **데이터 구조를 기준으로 한 방법론 우선순위**이다.

### 3.2 공통 패널티(데이터 구조 기반)
- **type16(범주형 X3)**에서 최소 그룹 크기(min_group)가 작으면:
  - GLMM/LMM 등 집단효과 추정이 불안정 → robust 감소
- **Y5(이항)**에서 사건률(event rate)이 매우 낮으면(예: < 2~3%):
  - GLMM 로지스틱이 수렴 문제/분리로 불안정해질 수 있음
  - Spearman/단순 상관은 의미 거의 없음 → goal 급감
  - Firth/rare-event 로지스틱이 상대적으로 유리

---

## 4) 각 Z 모델 소개 + 언제 쓰는지

### Z1 Spearman
- **무엇**: 단조(monotonic) 상관(순위 기반)
- **장점**: 빠른 스크리닝, 이상치에 상대적으로 둔감
- **한계**: 반복측정 구조/혼란변수(set, time, dir) 통제 불가
- **추천 용도**: “관계 후보가 있는지 대충 훑기” (초기 단계)

### Z2 Partial correlation
- **무엇**: 특정 공변량(예: dir, set, 성별 등)을 통제한 상관
- **장점**: confounding 일부 제거 가능
- **한계**: 반복측정/랜덤효과 처리 부족(구현에 따라)
- **추천 용도**: 스크리닝+통제변수 고려

### Z3 LMM (Linear Mixed Model)
- **무엇**: 연속형 Y에 대해 subject 랜덤효과 포함
- **장점**: 반복측정/개인차를 자연스럽게 모델링
- **특히 유리**: Spearman에서 무유의라도,
  - set/time/dir 같은 구조적 변동을 설명하고
  - personality 효과를 “잔차 위에서” 추정 → 살아나는 경우가 많음
- **추천 용도**: Y1/Y2/Y3 연속형의 **주력 1순위**

### Z4 GAMM
- **무엇**: 비선형(온도 ramp, 시간) + mixed
- **장점**: 시간/온도 변화의 곡선 패턴을 잘 잡음
- **한계**: 해석 복잡, 과적합 위험, 구현/튜닝 필요
- **추천 용도**: raw level에서 시간/온도 반응이 비선형일 때

### Z5 GEE
- **무엇**: 반복측정 상관구조를 “마진널(평균)” 관점으로 처리
- **장점**: 상관구조 지정으로 견고, 표준오차 robust
- **한계**: 개인차(랜덤효과)를 직접 추정하지 않음
- **추천 용도**: 이항/연속 반복측정에서 robust한 평균효과 추정

### Z6 GLMM logistic
- **무엇**: 이항 사건 + subject 랜덤효과
- **장점**: 개인차 고려한 사건 발생 확률 모델링
- **한계**: 희귀사건/분리 → 수렴 문제 가능
- **추천 용도**: Y5 이항이 “희귀하지 않을 때” 또는 표본/사건 충분할 때

### Z7 Rare-event logistic (Firth)
- **무엇**: 희귀사건/분리 상황에서 편향 감소(특히 Firth)
- **장점**: 사건이 적어도 안정적으로 계수/OR 추정 가능
- **한계**: mixed 구조와 결합이 까다로울 수 있음(구현 의존)
- **추천 용도**: 열적 무기력처럼 사건이 2~5% 수준인 경우 **우선 고려**

### Z8 PCA / FA
- **무엇**: 고차원 행동/설문을 소수 잠재축(PC)으로 압축
- **장점**: 패턴 탐색(phenotype), 차원 축소
- **한계**: PC의 의미 부여(해석) 필요
- **추천 용도**: Y4 패턴/타입화를 만들기 위한 전처리/1차 분석

### Z9 Clustering
- **무엇**: 행동/반응 패턴을 군집으로 묶어 archetype 도출
- **장점**: “comfort phenotype” 제시 가능
- **한계**: 안정성/재현성 검증 필요(bootstrap, silhouette 등)
- **추천 용도**: PCA→클러스터→성격과 연결(후속 검정) 흐름

### Z10 Supervised ML
- **무엇**: 예측 중심(랜덤포레스트, XGB, SVR 등)
- **장점**: 비선형/상호작용 자동 포착, 예측력 평가 가능
- **한계**: 해석력 낮아질 수 있음, CV 설계 중요
- **추천 용도**: “성격을 넣으면 예측이 개선되는가?” (superior tool 검증)

---

## 5) Y3/Y5/Y4 파생 변수(핵심)

### 5.1 Y3: mismatch (연속형)
코드가 자동으로 생성하는 대표 지표:
- `m_minus_p`: 다수재실(M) 행동평균 - 혼자(P) 행동평균  
- `mis_apathy_score`: max(0, -TCV) × max(0, -행동평균)
- `mis_social_inhib_score`: max(0, -TCV) × max(0, -(M-P))
- `mis_temp_perception_abs`: |TSV - PT|
- `mis_MP_L1`: Σ |Mi - Pi| (행동 패턴 차이 크기)
- Δ레벨:
  - `d_gap1..8`: Δ(Mi-Pi) = (M-P)_dynamic - (M-P)_static
  - `d_m_minus_p`: 평균 기반 social gap의 Δ

### 5.2 Y5: binary events (이항)
여러 정의를 동시에 생성해 “희귀사건” 민감도를 확인한다:
- `apathy_strict`: (TCV<0) & (모든 P,M ≤ 0)
- `apathy_relaxed`: (TCV<0) & (행동평균 ≤ 0)
- `social_inhib`: (TCV<0) & (M-P ≤ 0)
- `misperception_top20`: |TSV-PT| 상위 20%
- `MP_shift_top20`: Σ|Mi-Pi| 상위 20%

### 5.3 Y4: multi-level PCA
여러 레벨에서 PC1~PC3 생성:
- P_only: P1~8
- M_only: M1~8
- PM: P1~8 + M1~8
- Survey+Behavior: TSV/TCV/TA/TP/PT + P + M
- Delta_Behavior: d_p*, d_m*
- Delta_Gaps: d_gap1..8 등

---

## 6) 코드 흐름(전체 구조)

### 6.1 입력
- `PATH`, `SHEET0`에서 시트0 로딩
- 컬럼명 소문자화/공백 제거
- `set`, `dir01`, `phase` 파생

### 6.2 X 만들기
- X1: z-score
- X2: cut50 binary
- X3: type16 (e,n,t,j 필요)
- X4: interactions (axis×axis, axis×dir)

### 6.3 Y 만들기
- Y1: raw
- Y2: subject×set×dir 단위 Δ(dynamic-static)
- Y3: mismatch / gap / engineered
- Y4: 다레벨 PCA
- Y5: 여러 binary 사건 정의

### 6.4 진단(diagnostics)
- X 진단: 결측률, 0분산 비율, type16 최소 그룹
- Y 진단: 연속/이항/패턴 여부, 희귀사건률(min/max)

### 6.5 스코어링(rule-based)
- `score_cell(z, xform, yform, dx, dy, base)`에서 0~4 계산
- 공통 패널티: type16 희귀 그룹, binary 희귀 사건 등

### 6.6 출력
- 폴더: `META_METHOD_HEATMAPS/`
- 파일:
  - `meta_method_heatmaps_tables.xlsx`
    - Z모델별 점수 테이블
    - X/Y diagnostics
    - Y 파생변수 인벤토리
  - `Z*_suitability_heatmap.png` (모델별 히트맵 1장)

---

## 7) 해석 가이드(어떻게 읽나)

- 점수 0~4 중:
  - **3.0 이상**: “1~2순위로 먼저 돌릴 가치가 큼”
  - **2.0~3.0**: 상황/목표에 따라 보조적으로 사용
  - **2.0 미만**: 현재 구조에선 비효율(혹은 리스크 큼)

예:
- (X1 연속) × (Y1 raw)에서:
  - LMM/GAMM/GEE가 높은 점수를 받으면  
    → “반복측정 구조를 모델로 소화한 뒤 성격 효과를 보는 전략”이 우선
- (X3 type16) × (Y5 binary)에서:
  - 그룹이 작고 사건이 희귀하면  
    → GLMM보다 rare-event/Firth가 상대적으로 우선

---

## 8) 실행 방법

1) Python 환경 준비
- numpy, pandas, matplotlib, openpyxl, statsmodels 권장

2) 실행
```bash
python meta_method_heatmaps.py