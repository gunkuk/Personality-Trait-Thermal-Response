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
out_name = "PCA_PM+EUP+SPC_SPH_P8_M8"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

excel_out_path = os.path.join(out_dir, f"{out_name}.xlsx")
plot_out_path = os.path.join(out_dir, f"{out_name}_dashboard.png")

ID_COL = "연번"  # LMM merge key (필수)

# ==========================================
# 2. 신뢰도 검사 함수 (Cronbach's Alpha)
# ==========================================
def calculate_cronbach_alpha(df_items: pd.DataFrame) -> float:
    # item 2개 미만이면 alpha 해석 의미 약함 -> NaN
    if df_items.shape[1] < 2:
        return np.nan
    items_count = df_items.shape[1]
    variance_sum = df_items.var(axis=0, ddof=1).sum()
    total_var = df_items.sum(axis=1).var(ddof=1)
    if total_var == 0 or np.isnan(total_var):
        return np.nan
    alpha = (items_count / (items_count - 1)) * (1 - (variance_sum / total_var))
    return alpha

# ==========================================
# 3. PCA 실행(=fit/transform) + LMM용 score 저장 유틸
# ==========================================
def pca_fit_transform_for_lmm(
    df_raw: pd.DataFrame,
    cols: list,
    id_col: str,
    group_name: str,
    n_components: int | None = None,
):
    """
    - PCA fit: complete-case(해당 cols 모두 non-null)로 수행
    - scores: complete-case 행에만 계산하고, 원본 df index 기준 확장(나머지 NaN)
    - 반환:
        loadings_df: index=변수명, columns=PC1..PCk
        scores_df:   columns=[id_col, PC1..PCk] (id 포함)
        stats:       (explained_variance_ratio, cumulative_variance)
        alpha:       Cronbach alpha (가능 시)
        n_fit:       PCA fit에 사용된 complete-case 행 수
    """
    d = df_raw[[id_col] + cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    fit_mask = d[cols].notna().all(axis=1)
    d_fit = d.loc[fit_mask, [id_col] + cols].copy()
    n_fit = int(d_fit.shape[0])

    # 최소 표본 체크
    if n_fit < 3:
        loadings_df = pd.DataFrame(index=cols)
        scores_df = pd.DataFrame({id_col: d[id_col]})
        stats = (np.array([]), np.array([]))
        alpha = calculate_cronbach_alpha(d_fit[cols]) if n_fit >= 2 else np.nan
        return loadings_df, scores_df, stats, alpha, n_fit

    alpha = calculate_cronbach_alpha(d_fit[cols])

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(d_fit[cols].values)

    pca = PCA(n_components=n_components)
    S_fit = pca.fit_transform(X_fit)

    n_pc = pca.components_.shape[0]
    pc_cols = [f"PC{i+1}" for i in range(n_pc)]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=pc_cols
    )

    scores_all = pd.DataFrame(index=d.index, columns=pc_cols, dtype=float)
    scores_all.loc[d_fit.index, pc_cols] = S_fit

    scores_df = pd.concat([d[[id_col]].copy(), scores_all], axis=1)

    ev = pca.explained_variance_ratio_
    cv = np.cumsum(ev)
    stats = (ev, cv)

    return loadings_df, scores_df, stats, alpha, n_fit

# ==========================================
# 4. 데이터 로드
# ==========================================
df0 = pd.read_excel(path, sheet_name=0)                 # PI+SUR
df2 = pd.read_excel(path, sheet_name=2, header=1)       # EUP 슬라이드

if ID_COL not in df0.columns:
    raise ValueError(f"[Sheet0] '{ID_COL}' 컬럼이 없습니다.")
if ID_COL not in df2.columns:
    raise ValueError(f"[Sheet2] '{ID_COL}' 컬럼이 없습니다.")

# ==========================================
# 5. PCA 대상 그룹 구성 (LMM 목적)
# ==========================================
# (1) P/M 1~7 (기존 유지)
p_cols_1_7 = [f'P{i}' for i in range(1, 8)]
m_cols_1_7 = [f'M{i}' for i in range(1, 8)]

# (2) M-P 1~7
diff_cols_1_7 = [f'Diff_{i}' for i in range(1, 8)]
df0_diff = df0[[ID_COL] + p_cols_1_7 + m_cols_1_7].copy()
for c in p_cols_1_7 + m_cols_1_7:
    df0_diff[c] = pd.to_numeric(df0_diff[c], errors='coerce')
for i in range(1, 8):
    df0_diff[f'Diff_{i}'] = df0_diff[f'M{i}'] - df0_diff[f'P{i}']

# (3) EUP 18문항 (기존 slice 유지: 14~31 컬럼 = 18개)
df2_eup = df2[[ID_COL]].copy()
eup_block = df2.iloc[:, 14:32].apply(pd.to_numeric, errors='coerce')
eup_block.columns = [f'EUP{i}' for i in range(1, 19)]
df2_eup = pd.concat([df2_eup, eup_block], axis=1)
eup_cols = [f'EUP{i}' for i in range(1, 19)]

# ✅ (4) 요청: SP_C, SP_H (sheet2) + P8, M8 (sheet0) 네 개를 "하나의 PCA"로 통합
required_sheet2 = ["SP_C", "SP_H"]
required_sheet0 = ["P8", "M8"]

missing2 = [c for c in required_sheet2 if c not in df2.columns]
missing0 = [c for c in required_sheet0 if c not in df0.columns]

if missing2:
    raise ValueError(f"[Sheet2] PCA 통합을 위해 {missing2} 컬럼이 필요합니다.")
if missing0:
    raise ValueError(f"[Sheet0] PCA 통합을 위해 {missing0} 컬럼이 필요합니다.")

# row-level 기준으로 맞추기: df0의 (연번, P8, M8)에 df2의 (연번, SP_C, SP_H)를 merge
df0_p8m8 = df0[[ID_COL, "P8", "M8"]].copy()
df2_sp = df2[[ID_COL, "SP_C", "SP_H"]].copy()

df_sp_p8m8 = df0_p8m8.merge(df2_sp, on=ID_COL, how="left")
sp_p8m8_cols = ["SP_C", "SP_H", "P8", "M8"]

# 그룹 정의
groups = {
    "P": df0[[ID_COL] + p_cols_1_7].copy(),
    "M": df0[[ID_COL] + m_cols_1_7].copy(),
    "M-P": df0_diff[[ID_COL] + diff_cols_1_7].copy(),
    "EUP": df2_eup[[ID_COL] + eup_cols].copy(),
    "SP_C+SP_H+P8+M8": df_sp_p8m8[[ID_COL] + sp_p8m8_cols].copy(),  # ✅ 통합 PCA
}

group_cols = {
    "P": p_cols_1_7,
    "M": m_cols_1_7,
    "M-P": diff_cols_1_7,
    "EUP": eup_cols,
    "SP_C+SP_H+P8+M8": sp_p8m8_cols,
}

# ==========================================
# 6. PCA 실행
# ==========================================
results_loadings = {}
results_scores = {}
results_stats = {}
alpha_dict = {}
nfit_dict = {}

for name, dfg in groups.items():
    cols = group_cols[name]
    loadings, scores, stats, alpha, n_fit = pca_fit_transform_for_lmm(
        df_raw=dfg,
        cols=cols,
        id_col=ID_COL,
        group_name=name,
        n_components=None
    )
    results_loadings[name] = loadings
    results_scores[name] = scores
    results_stats[name] = stats
    alpha_dict[name] = round(alpha, 4) if pd.notna(alpha) else np.nan
    nfit_dict[name] = n_fit

# ==========================================
# 7. 시각화 (Scree Plot Dashboard)
# ==========================================
plot_order = ["M-P", "EUP", "M", "P", "SP_C+SP_H+P8+M8"]

# 5개 그룹 -> 3x2 (마지막 1칸 비움)
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle(
    f"PCA Dashboard for LMM Inputs: {out_name}\n"
    f"(P/M/M-P: 1~7, EUP: 1~18, Combined: SP_C+SP_H+P8+M8)",
    fontsize=16
)
ax_flat = axes.flatten()

for i in range(len(ax_flat)):
    ax = ax_flat[i]
    if i >= len(plot_order):
        ax.axis("off")
        continue

    name = plot_order[i]
    ev, cv = results_stats[name]

    if len(ev) == 0:
        ax.set_title(f"Scree Plot: {name} (n_fit<3)", fontsize=12)
        ax.text(0.5, 0.5, "Not enough complete cases to fit PCA", ha="center", va="center")
        ax.axis("off")
        continue

    ax.bar(range(1, len(ev) + 1), ev, alpha=0.5, align="center", label="Individual Variance")
    ax.step(range(1, len(cv) + 1), cv, where="mid", label="Cumulative Variance", color="red")

    ax.set_title(
        f"Scree Plot: {name} | alpha={alpha_dict[name]} | n_fit={nfit_dict[name]}",
        fontsize=12
    )
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(plot_out_path, dpi=200)

# ==========================================
# 8. 엑셀 저장 (LMM merge 위한 ID 포함 scores 저장)
# ==========================================
summary_rows = []
for g in plot_order:
    ev, cv = results_stats[g]
    pc1 = float(ev[0]) if len(ev) > 0 else np.nan
    pc2 = float(ev[1]) if len(ev) > 1 else np.nan
    summary_rows.append({
        "Group": g,
        "Cronbach_Alpha": alpha_dict[g],
        "n_fit_complete_case": nfit_dict[g],
        "PC1_Variance": round(pc1, 4) if pd.notna(pc1) else np.nan,
        "PC2_Variance": round(pc2, 4) if pd.notna(pc2) else np.nan,
    })
summary_df = pd.DataFrame(summary_rows)

with pd.ExcelWriter(excel_out_path, engine="xlsxwriter") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    for g in plot_order:
        # 시트명 길이 제한/특수문자 방지
        safe_g = g.replace("+", "_").replace(" ", "")[:28]
        results_loadings[g].to_excel(writer, sheet_name=f"loadings_{safe_g}")

    for g in plot_order:
        safe_g = g.replace("+", "_").replace(" ", "")[:28]
        results_scores[g].to_excel(writer, sheet_name=f"scores_{safe_g}", index=False)

print(f"완료: 엑셀 저장 -> {excel_out_path}")
print(f"완료: 대시보드 저장 -> {plot_out_path}")