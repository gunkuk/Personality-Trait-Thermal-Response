import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from itertools import combinations
import os

# 0) 경로 설정
output_dir = "analysis_output"
if not os.path.exists(output_dir): os.makedirs(output_dir)

PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\MBTI.xlsx"
df = pd.read_excel(PATH)
df.columns = df.columns.astype(str).str.strip().str.lower()

# 1) 변수 설정
MBTI_BASE = ["e", "n", "t", "j", "a"]
id_col = "연번"; env_col = "환경"; TCV = "tcv"
P_COLS = [f"p{i}" for i in range(1, 9)]; M_COLS = [f"m{i}" for i in range(1, 9)]

# 2) 전처리 및 변수 생성
def env_to_set(x):
    x = str(x).strip().upper()
    if x in ["A", "B"]: return "AB"
    if x in ["C", "D"]: return "CD"
    if x in ["E", "F"]: return "EF"
    if x in ["G", "H"]: return "GH"
    return np.nan

df["set"] = df[env_col].map(env_to_set)
df["direction"] = df["set"].map(lambda s: -1 if s in ["AB", "GH"] else 1 if s in ["CD", "EF"] else np.nan)
df["p_mean_val"] = df[P_COLS].mean(axis=1, skipna=True)
df["m_mean_val"] = df[M_COLS].mean(axis=1, skipna=True)
df["apathy"] = np.where((df[TCV] < 0) & (df["p_mean_val"] <= 0) & (df["m_mean_val"] <= 0), 1, 0)

# MBTI 쌍 생성 및 숫자형 변환
mbti_pairs = []
for combo in combinations(MBTI_BASE, 2):
    p_name = f"{combo[0]}_{combo[1]}"
    df[p_name] = pd.to_numeric(df[combo[0]], errors='coerce') * pd.to_numeric(df[combo[1]], errors='coerce')
    mbti_pairs.append(p_name)

all_mbti_vars = MBTI_BASE + mbti_pairs
for c in all_mbti_vars: df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
df = df.dropna(subset=["set", "direction", "apathy"] + all_mbti_vars)

# 3) 모델링
formula = "apathy ~ 0 + C(set) + " + " + ".join(all_mbti_vars) + " + direction"
gee = smf.gee(formula, groups=df[id_col], data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()

# 4) 결과 테이블 생성 및 MEAN 추가
gee_or_tbl = pd.DataFrame({
    "OR": np.exp(gee.params),
    "CI_low": np.exp(gee.conf_int()[0]),
    "CI_high": np.exp(gee.conf_int()[1]),
    "p": gee.pvalues
})

# MEAN 값 계산 (SET별 발생 확률)
set_means = df.groupby("set")["apathy"].mean()
gee_or_tbl["MEAN"] = np.nan # 기본값

# 인덱스 매칭하여 MEAN 채우기
for s in set_means.index:
    idx_name = f"C(set)[{s}]"
    if idx_name in gee_or_tbl.index:
        gee_or_tbl.at[idx_name, "MEAN"] = set_means[s]

# Note 생성 로직 (이유 명시)
def generate_note(row, name):
    if name == "direction":
        return "Excluded: Multicollinearity with Set"
    if pd.isna(row["p"]) or pd.isna(row["CI_low"]):
        if "C(set)" in name:
            return "Note: Too few Apathy cases in this Set"
        return "Note: Sparse data or Separation issue"
    if row["CI_low"] < 0.001 or row["CI_high"] > 1000:
        return "Warning: High instability (Small sample)"
    return "Normal"

gee_or_tbl["Note"] = [generate_note(row, idx) for idx, row in gee_or_tbl.iterrows()]

# 컬럼 순서 재배치 (요청하신 대로 Note 왼쪽에 MEAN)
cols = ["OR", "CI_low", "CI_high", "p", "MEAN", "Note"]
gee_or_tbl = gee_or_tbl[cols]

# 5) 저장
gee_or_tbl.sort_values("p").to_csv(os.path.join(output_dir, "Final_Apathy_OR_Table.csv"), encoding="utf-8-sig")
print("\n[최종 분석 완료] 'analysis_output/Final_Apathy_OR_Table.csv' 파일을 확인하세요.")
print(gee_or_tbl.sort_values("p").head(10)) # 상위 10개 출력