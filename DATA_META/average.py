import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 0) 경로 설정
output_dir = "analysis_output"
if not os.path.exists(output_dir): os.makedirs(output_dir)

PATH = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\MBTI.xlsx"
df = pd.read_excel(PATH)
df.columns = df.columns.astype(str).str.strip().str.lower()

# 1) 대상 컬럼 정의
id_col = "연번"
env_col = "환경"
mbti_cols = ["e", "n", "t", "j", "a"]
ffm_cols = ["o1", "c1", "e1", "a1", "n1"]
p_cols = [f"p{i}" for i in range(1, 9)]
m_cols = [f"m{i}" for i in range(1, 9)]
sensory_cols = ["tsv", "tcv", "ta", "tp", "pt"]

# 데이터 숫자형 변환
all_target = mbti_cols + ffm_cols + sensory_cols + p_cols + m_cols
for c in all_target:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# 행 단위 기초 지표 미리 계산
df['p_mean_row'] = df[p_cols].mean(axis=1)
df['m_mean_row'] = df[m_cols].mean(axis=1)
df['gap_row'] = df['m_mean_row'] - df['p_mean_row']

# 시각화 함수 정의 (기존 스타일 유지)
def save_scatter_matrix(x_cols, y_vars, data, filename, title_prefix):
    # 유효한 y_vars만 필터링
    valid_y = [v for v in y_vars if v in data.columns]
    fig, axes = plt.subplots(len(valid_y), len(x_cols), figsize=(22, 4 * len(valid_y)))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, y_var in enumerate(valid_y):
        for j, x_var in enumerate(x_cols):
            ax = axes[i, j]
            if x_var in data.columns:
                sns.regplot(x=x_var, y=y_var, data=data, ax=ax,
                            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                ax.set_title(f"{y_var.upper()} by {title_prefix}_{x_var.upper()}")
                ax.set_xlabel(f"{x_var.upper()} Score")
                ax.set_ylabel(f"Mean {y_var.upper()}")
            else:
                ax.set_axis_off()

    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# [기존 OUTPUT 유지] 전체 시나리오 평균 데이터 기반
# ---------------------------------------------------------
agg_dict = {col: 'mean' for col in (mbti_cols + ffm_cols + sensory_cols) if col in df.columns}
person_df = df.groupby(id_col).agg(agg_dict).reset_index()

person_df['p_mean'] = df.groupby(id_col)['p_mean_row'].mean().values
person_df['m_mean'] = df.groupby(id_col)['m_mean_row'].mean().values
person_df['m_minus_p_mean'] = person_df['m_mean'] - person_df['p_mean']

y_vars_global = sensory_cols + ['p_mean', 'm_mean', 'm_minus_p_mean']

save_scatter_matrix(mbti_cols, y_vars_global, person_df, "MBTI_Sensory_ScatterPlots.png", "MBTI")
save_scatter_matrix(ffm_cols, y_vars_global, person_df, "FFM_Sensory_ScatterPlots.png", "FFM")
person_df.to_csv(os.path.join(output_dir, "Person_Level_Combined_Averages.csv"), encoding="utf-8-sig", index=False)

# ---------------------------------------------------------
# [신규 추가] 시나리오별(A-H) 데이터 기반 (총 16개 이미지)
# ---------------------------------------------------------
# 환경 열 정규화 (A, B, C... 추출)
df['env_clean'] = df[env_col].astype(str).str.strip().str.upper()
scenarios = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

for env in scenarios:
    # 해당 환경의 데이터만 추출
    env_df_raw = df[df['env_clean'] == env]
    
    if env_df_raw.empty:
        continue
    
    # 해당 환경 내에서의 개인별 평균 계산
    env_person_df = env_df_raw.groupby(id_col).agg(agg_dict).reset_index()
    env_person_df['p_mean'] = env_df_raw.groupby(id_col)['p_mean_row'].mean().values
    env_person_df['m_mean'] = env_df_raw.groupby(id_col)['m_mean_row'].mean().values
    env_person_df['m_minus_p_mean'] = env_person_df['m_mean'] - env_person_df['p_mean']
    
    # MBTI 산점도 생성
    save_scatter_matrix(mbti_cols, y_vars_global, env_person_df, 
                        f"MBTI_Scenario_{env}_ScatterPlots.png", f"MBTI_Env_{env}")
    
    # FFM 산점도 생성
    save_scatter_matrix(ffm_cols, y_vars_global, env_person_df, 
                        f"FFM_Scenario_{env}_ScatterPlots.png", f"FFM_Env_{env}")

print(f"\n[분석 완료] '{output_dir}' 폴더 내에 기존 파일 및 시나리오별 이미지 16개가 추가되었습니다.")