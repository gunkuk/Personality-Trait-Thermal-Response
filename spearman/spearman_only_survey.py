import pandas as pd
import numpy as np

# =========================
# 1) Load (first sheet)
# =========================
path = r'C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\MBTI.xlsx'
df = pd.read_excel(path, sheet_name=0)

# =========================
# 2) Choose variables for Spearman
#    (edit this list if needed)
# =========================
vars_for_corr = [
    "TSV", "TCV", "TA", "TP", "PT",
    "P1","P2","P3","P4","P5","P6","P7","P8",
    "M1","M2","M3","M4","M5","M6","M7","M8",
]

# Keep only existing columns (safety)
vars_for_corr = [c for c in vars_for_corr if c in df.columns]

# =========================
# 3) Spearman correlation matrix
#    pairwise complete observations (default)
# =========================
spearman_r = df[vars_for_corr].corr(method="spearman")

# Optional: show rounded matrix
print("Spearman correlation (rho) matrix:")
print(spearman_r.round(3))

# =========================
# 4) (Optional) Save to Excel
# =========================
out_path = r"C:\Users\rjs11\Desktop\1\paper\00_SCI\MBTI\model\spearman\OUT\spearman_rho.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

    spearman_r.to_excel(writer, sheet_name="spearman_rho")

print(f"\nSaved: {out_path}")

# =========================
# 5) (Optional) Long-form sorted pairs
#    useful to find strongest associations quickly
# =========================
rho_long = (
    spearman_r.where(np.triu(np.ones(spearman_r.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
rho_long.columns = ["var1", "var2", "rho"]
rho_long["abs_rho"] = rho_long["rho"].abs()
rho_long = rho_long.sort_values("abs_rho", ascending=False)

print("\nTop 20 absolute correlations:")
print(rho_long.head(20).to_string(index=False))