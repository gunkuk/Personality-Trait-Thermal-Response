import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIG (경로 및 파일명 설정)
# ==========================================
path = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\data\BI+SUR.xlsx"
out_dir = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\OUT"
out_name = "PCA_PM+EUP"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

excel_out_path = os.path.join(out_dir, f"{out_name}.xlsx")
plot_out_path = os.path.join(out_dir, f"{out_name}_dashboard.png")

# ==========================================
# 2. 신뢰도 검사 함수 (Cronbach's Alpha)
# ==========================================
def calculate_cronbach_alpha(df):
    items_count = df.shape[1]
    variance_sum = df.var(axis=0, ddof=1).sum()
    total_var = df.sum(axis=1).var(ddof=1)
    alpha = (items_count / (items_count - 1)) * (1 - (variance_sum / total_var))
    return alpha

# ==========================================
# 3. 데이터 로드 및 전처리
# ==========================================
path_MP_raw = pd.read_excel(path, sheet_name=0)
path_EUP_raw = pd.read_excel(path, sheet_name=2, header=1)

p_cols = [f'P{i}' for i in range(1, 9)]
m_cols = [f'M{i}' for i in range(1, 9)]

# P 그룹
df_p = path_MP_raw[p_cols].apply(pd.to_numeric, errors='coerce').dropna()
# M 그룹
df_m = path_MP_raw[m_cols].apply(pd.to_numeric, errors='coerce').dropna()
# M - P 그룹
df_diff_raw = (path_MP_raw[m_cols].values - path_MP_raw[p_cols].values)
df_diff = pd.DataFrame(df_diff_raw, columns=[f'Diff_{i}' for i in range(1, 9)]).dropna()
# EUP 그룹 (인덱스 14~31 컬럼)
df_eup = path_EUP_raw.iloc[:, 14:32].apply(pd.to_numeric, errors='coerce').dropna()
df_eup.columns = [f'EUP{i}' for i in range(1, 19)]

# ==========================================
# 4. 분석 수행 (PCA 및 신뢰도)
# ==========================================
groups = {
    "P": df_p,
    "M": df_m,
    "M-P": df_diff,
    "EUP": df_eup
}

results_loadings = {}
results_scores = {}
results_stats = {}
alpha_dict = {} # 그룹별 신뢰도 수치를 이름으로 매칭하기 위해 딕셔너리 사용

for name, data in groups.items():
    # 신뢰도 및 PCA 실행
    alpha = calculate_cronbach_alpha(data)
    alpha_dict[name] = round(alpha, 4)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA()
    pca_scores = pca.fit_transform(scaled)
    
    results_loadings[name] = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(data.shape[1])], index=data.columns)
    results_scores[name] = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(data.shape[1])])
    results_stats[name] = (pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_))

# ==========================================
# 5. 통합 시각화 (Scree Plot Dashboard)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"PCA Analysis Dashboard: {out_name}", fontsize=18)

# 시각화 및 데이터 처리 순서 동일하게 유지
plot_order = ["M-P", "EUP", "M", "P"]
ax_flat = axes.flatten()

for i, name in enumerate(plot_order):
    ev, cv = results_stats[name]
    ax = ax_flat[i]
    
    ax.bar(range(1, len(ev)+1), ev, alpha=0.5, align='center', label='Individual Variance')
    ax.step(range(1, len(cv)+1), cv, where='mid', label='Cumulative Variance', color='red')
    
    # 딕셔너리에서 이름을 직접 찾아 신뢰도 수치 입력 (바뀔 염려 없음)
    ax.set_title(f"Scree Plot: {name} (Cronbach's Alpha: {alpha_dict[name]})", fontsize=14)
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(plot_out_path)

# ==========================================
# 6. 엑셀 파일 통합 저장 (지정된 시트 순서 준수)
# ==========================================
group_order = ["M-P", "EUP", "M", "P"]

# 요약 데이터프레임 생성
summary_df = pd.DataFrame([
    {"Group": g, "Cronbach_Alpha": alpha_dict[g], "PC1_Variance": round(results_stats[g][0][0], 4)} 
    for g in group_order
])

with pd.ExcelWriter(excel_out_path, engine='xlsxwriter') as writer:
    # Summary 시트
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    # Loadings 시트들 (M-P, EUP, M, P 순)
    for g in group_order:
        results_loadings[g].to_excel(writer, sheet_name=f"loadings_{g}")
        
    # Scores 시트들 (M-P, EUP, M, P 순)
    for g in group_order:
        results_scores[g].to_excel(writer, sheet_name=f"scores_{g}", index=False)

print(f"분석 완료: 엑셀({excel_out_path}) 및 대시보드 이미지({plot_out_path}) 저장 성공")