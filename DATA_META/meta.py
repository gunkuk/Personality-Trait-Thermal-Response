import numpy as np
import pandas as pd
import os

# 0) 경로 설정 및 폴더 생성
output_dir = "analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\MBTI.xlsx"
df = pd.read_excel(PATH)
df.columns = df.columns.astype(str).str.strip().str.lower()

# 1) 대상 열(Columns) 설정
target_cols = ["tsv", "tcv", "ta", "tp", "pt"] + [f"p{i}" for i in range(1, 9)] + [f"m{i}" for i in range(1, 9)]
available_cols = [c for c in target_cols if c in df.columns]

# 데이터 숫자형 변환
for c in available_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 2) Mean ± SD 포맷 함수
def get_mean_sd(x):
    m = x.mean()
    s = x.std()
    if pd.isna(m) or pd.isna(s):
        return "-"
    return f"{m:.2f} ± {s:.2f}"

summary_data = []

# 3) 행(Rows) 생성: 개별 환경별 (A, B, C, D, E, F, G, H)
# 환경 열의 값을 대문자로 정규화하여 개별적으로 처리합니다.
df["env_single"] = df["환경"].astype(str).str.strip().str.upper()

# 분석할 알파벳 리스트
env_list = ["A", "B", "C", "D", "E", "F", "G", "H"]

for env in env_list:
    temp_df = df[df["env_single"] == env]
    row = {"Category": f"ENV_{env}"}
    for c in available_cols:
        row[c] = get_mean_sd(temp_df[c])
    summary_data.append(row)

# 4) 행(Rows) 생성: MBTI 개별 축별 (High/Low) - 기준점 50점
mbti_axes = ["e", "n", "t", "j", "a"]
split_point = 50

for axis in mbti_axes:
    if axis in df.columns:
        df[axis] = pd.to_numeric(df[axis], errors='coerce')
        
        # High 그룹 (>= 50)
        high_df = df[df[axis] >= split_point]
        row_h = {"Category": f"{axis.upper()}_High (>= {split_point})"}
        for c in available_cols:
            row_h[c] = get_mean_sd(high_df[c])
        summary_data.append(row_h)
        
        # Low 그룹 (< 50)
        low_df = df[df[axis] < split_point]
        row_l = {"Category": f"{axis.upper()}_Low (< {split_point})"}
        for c in available_cols:
            row_l[c] = get_mean_sd(low_df[c])
        summary_data.append(row_l)

# 5) 최종 표 구성 및 저장
summary_df = pd.DataFrame(summary_data)
final_cols = ["Category"] + available_cols
summary_df = summary_df[final_cols]

summary_df.to_csv(os.path.join(output_dir, "Sensory_Summary_Individual_Env.csv"), encoding="utf-8-sig", index=False)

print(f"\n[작업 완료] 'analysis_output/Sensory_Summary_Individual_Env.csv' 파일이 생성되었습니다.")
print("개별 환경(A-H)과 MBTI 50점 기준 분류가 모두 포함되었습니다.")