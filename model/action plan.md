---
<action plan>

해당 액션 플랜은 설문 데이터, 생체 데이터, 기본 인적사항 및 성격 특성에 대해 최대한 다각도의 통계적 조사를 이어가기 위함.

그러한 방법으로, 성격 데이터 x축, 그 외 데이터 (기본 인적사항, 설문, 생체 데이터) y축, 분석 모델을 z축으로 하여 가장 각 조합의 적합성을 따져보고, 가장 적합한 조합들에 대해서 차례로 분석 진행

---
MTD 적합성 heatmap 결과:
1. All x - raw, responsibility, mismatch, pattern y: LMM
2. All x - binary(희귀조건) y: logit_firth
3. All x - responsibility: GAMM
4. All x - pattern y: PCA, Clustering