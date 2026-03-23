# 요약: 업로드된 rawdata_wrong corrected.xlsx에서 지정한 블록별 PCA를 수행하고,
# 방법론적 타당성(KMO/Bartlett/상관/표본수/반복측정 구조)을 점검한 뒤,
# Excel 요약본, 블록별 PNG, 해석 TXT를 상대경로 기준으로 model/OUT/<현재폴더명>/에 저장한다.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# =========================
# 1) 경로 설정 (상대 위치)
# =========================
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
INPUT_XLSX = PROJECT_ROOT / "model" / "1. data cleaning" / "domain knowledge" / "rawdata_wrong corrected_0319.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "model" / "OUT" / SCRIPT_DIR.name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = OUTPUT_DIR / "png"
PNG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2) 사용자 지정 블록 정의
# =========================
BLOCKS = {
    "SUR_TSV_TCV_TA_TP": ["TSV", "TCV", "TA", "TP"],
    "SUR_P_M": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "M1", "M2", "M3", "M4", "M5", "M6", "M7"],
    "SUR_P_minus_M": ["P1-M1", "P2-M2", "P3-M3", "P4-M4", "P5-M5", "P6-M6", "P7-M7"],
    "EUP_ONLY": [f"EUP{i}" for i in range(1, 19)],
    "EUP_SP_ONLY": ["SP1", "SP2", "SP3", "SP4"],
    "BI_SEX_BMI_AGE": ["SEX", "BMI", "AGE"],
    "BI_SEX_BMI": ["SEX", "BMI"],
    "TLX_only": ["TLX1", "TLX2", "TLX3", "TLX4", "TLX5", "TLX6"],
    "TLX_S": ["TLX1", "TLX2", "TLX3", "TLX4", "TLX5", "TLX6", "S1", "S2"],
    "BI_MBTI": ["E", "N", "T", "J", "A"],
    "BI_FFM": ["o1", "c1", "e1", "a1", "n1"],
    "BI_MBTI_FFM": ["E", "N", "T", "J", "A", "o1", "c1", "e1", "a1", "n1"],
}


# =========================
# 3) 데이터 로드/정리
# =========================

def _upper_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() if pd.notna(c) else f"unnamed_{i}" for i, c in enumerate(df.columns)]
    rename_map = {
        "sex": "SEX",
        "age": "AGE",
        "name": "name",
        "no.": "no.",
        "ID": "ID",
        "env": "env",
        "time (min)": "time_min",
    }
    new_cols = []
    for c in df.columns:
        if c in rename_map:
            new_cols.append(rename_map[c])
        elif c.lower() in {"bmi"}:
            new_cols.append("BMI")
        else:
            new_cols.append(c)
    df.columns = new_cols
    return df


def load_sheets(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    sheets = {
        "BI": _upper_cols(pd.read_excel(xlsx_path, sheet_name="BI", header=1)),
        "SUR": _upper_cols(pd.read_excel(xlsx_path, sheet_name="SUR", header=1)),
        "EUP": _upper_cols(pd.read_excel(xlsx_path, sheet_name="EUP", header=1)),
        "TLX": _upper_cols(pd.read_excel(xlsx_path, sheet_name="TLX", header=1)),
    }

    # sex 숫자화
    if "SEX" in sheets["BI"].columns:
        sheets["BI"]["SEX"] = (
            sheets["BI"]["SEX"].astype(str).str.strip().str.upper().map({"M": 1, "F": 0, "MALE": 1, "FEMALE": 0})
        )

    # SUR: 반복측정이므로 ID-env 단위 평균으로 축약 (time-level 중복으로 인한 N inflation 완화)
    sur = sheets["SUR"].copy()
    sur_num_cols = [c for c in sur.columns if c not in {"ID", "name", "env", "time_min"}]
    for c in sur_num_cols:
        sur[c] = pd.to_numeric(sur[c], errors="coerce")
    sur_agg = sur.groupby(["ID", "name", "env"], dropna=False)[sur_num_cols].mean().reset_index()
    for i in range(1, 8):
        p = f"P{i}"
        m = f"M{i}"
        if p in sur_agg.columns and m in sur_agg.columns:
            sur_agg[f"P{i}-M{i}"] = sur_agg[p] - sur_agg[m]
    sheets["SUR_AGG"] = sur_agg

    # EUP / TLX / BI 숫자화
    for key in ["BI", "EUP", "TLX"]:
        df = sheets[key]
        for c in df.columns:
            if c not in {"ID", "no.", "name", "env", "scenerio"}:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        sheets[key] = df

    return sheets


# =========================
# 4) 방법론 점검 함수
# =========================

def corr_matrix_psd(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr().replace([np.inf, -np.inf], np.nan)
    return corr


def bartlett_sphericity_test(df: pd.DataFrame) -> Tuple[float, float, int]:
    """Returns chi-square, p-value, dof."""
    X = df.dropna(axis=0)
    n, p = X.shape
    if n <= p or p < 2:
        return np.nan, np.nan, 0
    R = X.corr().values
    detR = np.linalg.det(R)
    if detR <= 0 or np.isnan(detR):
        return np.nan, np.nan, int(p * (p - 1) / 2)
    chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(detR)
    dof = int(p * (p - 1) / 2)
    p_value = chi2.sf(chi_square, dof)
    return float(chi_square), float(p_value), dof


def kmo_test(df: pd.DataFrame) -> Tuple[float, pd.Series]:
    X = df.dropna(axis=0)
    if X.shape[1] < 2:
        return np.nan, pd.Series(dtype=float)
    R = X.corr().values
    try:
        invR = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return np.nan, pd.Series(np.nan, index=X.columns)
    partial = -invR / np.sqrt(np.outer(np.diag(invR), np.diag(invR)))
    np.fill_diagonal(partial, 0.0)
    r2 = R**2
    p2 = partial**2
    np.fill_diagonal(r2, 0.0)
    np.fill_diagonal(p2, 0.0)
    denom = r2 + p2
    kmo_overall = np.sum(r2) / np.sum(denom) if np.sum(denom) != 0 else np.nan
    kmo_vars = np.sum(r2, axis=0) / np.sum(denom, axis=0)
    return float(kmo_overall), pd.Series(kmo_vars, index=X.columns)


def validity_judgement(df: pd.DataFrame, block_name: str, native_unit: str) -> Dict[str, object]:
    x = df.copy()
    x = x.loc[:, x.notna().sum(axis=0) >= max(3, int(0.3 * len(x)))]
    n, p = x.shape
    unique_counts = x.nunique(dropna=True)
    binary_like = int((unique_counts <= 2).sum())
    ordinal_like = int(((unique_counts >= 3) & (unique_counts <= 7)).sum())
    low_var = unique_counts[unique_counts <= 1].index.tolist()

    avg_abs_r = np.nan
    corr = x.corr()
    if p >= 2:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        avg_abs_r = float(upper.abs().mean()) if len(upper) else np.nan

    bart_chi2, bart_p, bart_dof = bartlett_sphericity_test(x)
    kmo, kmo_vars = kmo_test(x)

    cautions = []
    if native_unit == "SUR_ID_ENV_MEAN":
        cautions.append("SUR는 원래 time-level 반복측정이므로, 시간행을 그대로 쓰면 독립성 위반/N inflation 위험이 커서 ID-env 평균으로 축약함.")
    if binary_like > 0:
        cautions.append(f"이 블록에는 이진형/준이진형 변수가 {binary_like}개 포함되어 Pearson PCA 해석이 불안정할 수 있음.")
    if ordinal_like == p:
        cautions.append("모든 문항이 Likert/ordinal일 가능성이 높아, 엄밀하게는 polychoric 기반 차원축소가 더 적절할 수 있음.")
    if p < 3:
        cautions.append("변수 수가 2개 이하인 블록은 PCA보다는 상관/단일축 비교에 가까움.")
    if n < max(50, 5 * p):
        cautions.append(f"유효 표본수 n={n}가 충분히 크지 않을 수 있음. 보수 기준 5p 또는 50 미만 여부를 확인할 것.")
    if not np.isnan(kmo) and kmo < 0.60:
        cautions.append(f"KMO={kmo:.3f} < 0.60: 공통요인/축약 적합도가 낮음.")
    if not np.isnan(bart_p) and bart_p >= 0.05:
        cautions.append(f"Bartlett p={bart_p:.4g} >= 0.05: 상관구조가 약해 PCA 근거가 약함.")
    if not np.isnan(avg_abs_r) and avg_abs_r < 0.20:
        cautions.append(f"평균 |r|={avg_abs_r:.3f} < 0.20: 블록 내부 상관이 약함.")
    if low_var:
        cautions.append("상수/준상수 변수 존재: " + ", ".join(low_var))

    valid = True
    if p < 2 or n < 10 or (not np.isnan(kmo) and kmo < 0.50) or (not np.isnan(bart_p) and bart_p >= 0.10):
        valid = False

    return {
        "n": n,
        "p": p,
        "binary_like": binary_like,
        "ordinal_like": ordinal_like,
        "avg_abs_r": avg_abs_r,
        "kmo": kmo,
        "bartlett_chi2": bart_chi2,
        "bartlett_p": bart_p,
        "bartlett_dof": bart_dof,
        "valid_for_exploratory_pca": valid,
        "cautions": cautions,
    }


# =========================
# 5) PCA 실행/해석
# =========================

@dataclass
class PCAResult:
    block_name: str
    native_unit: str
    used_columns: List[str]
    n: int
    p: int
    validity: Dict[str, object]
    explained_variance_ratio: List[float]
    cumulative_variance_ratio: List[float]
    n_components_80: int
    n_components_90: int
    n_components_95: int
    reducible_to_2pc_80: bool
    reducible_to_2pc_90: bool
    one_raw_proxy_ok: bool
    one_raw_proxy_name: str | None
    one_raw_proxy_r2: float | None
    one_raw_proxy_weight_note: str
    loadings: pd.DataFrame
    scores: pd.DataFrame
    summary_text: str


def choose_native_dataframe(sheets: Dict[str, pd.DataFrame], block_name: str) -> Tuple[pd.DataFrame, str]:
    if block_name.startswith("SUR_"):
        return sheets["SUR_AGG"].copy(), "SUR_ID_ENV_MEAN"
    if block_name.startswith("EUP_"):
        return sheets["EUP"].copy(), "PERSON"
    if block_name.startswith("TLX_"):
        return sheets["TLX"].copy(), "ID_ENV"
    if block_name.startswith("BI_"):
        return sheets["BI"].copy(), "PERSON"
    raise ValueError(f"Unknown block: {block_name}")


def build_numeric_block(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    x = df[cols].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    nunique = x.nunique(dropna=True) 
    x = x.loc[:, nunique > 1]
    return x


def run_block_pca(block_name: str, cols: List[str], sheets: Dict[str, pd.DataFrame]) -> PCAResult:
    source_df, native_unit = choose_native_dataframe(sheets, block_name)
    x = build_numeric_block(source_df, [c for c in cols if c in source_df.columns])
    validity = validity_judgement(x, block_name, native_unit)

    n, p = x.shape
    if p < 2:
        summary = f"[{block_name}] usable variable count < 2, PCA skipped."
        return PCAResult(
            block_name=block_name,
            native_unit=native_unit,
            used_columns=list(x.columns),
            n=n,
            p=p,
            validity=validity,
            explained_variance_ratio=[],
            cumulative_variance_ratio=[],
            n_components_80=0,
            n_components_90=0,
            n_components_95=0,
            reducible_to_2pc_80=False,
            reducible_to_2pc_90=False,
            one_raw_proxy_ok=False,
            one_raw_proxy_name=None,
            one_raw_proxy_r2=None,
            one_raw_proxy_weight_note="usable variable count < 2",
            loadings=pd.DataFrame(),
            scores=pd.DataFrame(),
            summary_text=summary,
        )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(x)
    X_std = scaler.fit_transform(X_imp)

    pca = PCA(n_components=min(n, p))
    scores = pca.fit_transform(X_std)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    def n_for(threshold: float) -> int:
        return int(np.searchsorted(cum, threshold) + 1)

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=x.columns,
        columns=[f"PC{i+1}" for i in range(len(evr))],
    )
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])

    # 하나의 raw variable proxy 여부 판단
    # retained PCs: 80% 설명력까지, 단 최소 1개 최대 2개만 요약 기준으로 확인
    retained = min(2, n_for(0.80))
    target = scores_df[[f"PC{i+1}" for i in range(retained)]].values
    best_var, best_r2, best_note = None, -np.inf, ""
    if retained == 1:
        y = target.ravel()
        for c in x.columns:
            lr = LinearRegression().fit(X_std[:, [list(x.columns).index(c)]], y)
            r2 = lr.score(X_std[:, [list(x.columns).index(c)]], y)
            if r2 > best_r2:
                best_var = c
                best_r2 = float(r2)
                best_note = f"{c}가 PC1을 단일 변수로 설명하는 R²={r2:.3f}; PC1 loading={loadings.loc[c, 'PC1']:.3f}"
        proxy_ok = best_r2 >= 0.70
    else:
        for c in x.columns:
            lr = LinearRegression().fit(X_std[:, [list(x.columns).index(c)]], target)
            pred = lr.predict(X_std[:, [list(x.columns).index(c)]])
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - target.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            if r2 > best_r2:
                best_var = c
                best_r2 = float(r2)
                contrib = ", ".join([f"PC{i+1} loading={loadings.loc[c, f'PC{i+1}']:.3f}" for i in range(retained)])
                best_note = f"{c}가 retained PCs({retained}개)를 단일 변수로 설명하는 pseudo-R²={r2:.3f}; {contrib}"
        proxy_ok = best_r2 >= 0.70

    summary_lines = [
        f"[BLOCK] {block_name}",
        f"- native unit: {native_unit}",
        f"- n={n}, p={p}",
        f"- usable columns: {', '.join(x.columns.tolist())}",
        f"- exploratory PCA valid? {'YES' if validity['valid_for_exploratory_pca'] else 'CAUTION/NO'}",
        f"- KMO={validity['kmo']:.3f}" if not pd.isna(validity['kmo']) else "- KMO=NA",
        f"- Bartlett p={validity['bartlett_p']:.4g}" if not pd.isna(validity['bartlett_p']) else "- Bartlett p=NA",
        f"- avg |r|={validity['avg_abs_r']:.3f}" if not pd.isna(validity['avg_abs_r']) else "- avg |r|=NA",
        f"- cumulative variance: PC1={cum[0]:.3f}, PC2={cum[1]:.3f}" if len(cum) >= 2 else f"- cumulative variance: PC1={cum[0]:.3f}",
        f"- n_components for 80/90/95% = {n_for(0.80)}/{n_for(0.90)}/{n_for(0.95)}",
        f"- reducible to <=2 PCs? 80%={'YES' if n_for(0.80) <= 2 else 'NO'}, 90%={'YES' if n_for(0.90) <= 2 else 'NO'}",
        f"- one raw variable can represent retained structure? {'YES' if proxy_ok else 'NO'}",
        f"- best raw proxy: {best_var} (R²={best_r2:.3f})" if best_var else "- best raw proxy: NA",
        f"- detail: {best_note}",
    ]
    if validity["cautions"]:
        summary_lines.append("- cautions:")
        summary_lines.extend([f"  * {c}" for c in validity["cautions"]])

    return PCAResult(
        block_name=block_name,
        native_unit=native_unit,
        used_columns=x.columns.tolist(),
        n=n,
        p=p,
        validity=validity,
        explained_variance_ratio=evr.tolist(),
        cumulative_variance_ratio=cum.tolist(),
        n_components_80=n_for(0.80),
        n_components_90=n_for(0.90),
        n_components_95=n_for(0.95),
        reducible_to_2pc_80=n_for(0.80) <= 2,
        reducible_to_2pc_90=n_for(0.90) <= 2,
        one_raw_proxy_ok=proxy_ok,
        one_raw_proxy_name=best_var,
        one_raw_proxy_r2=best_r2,
        one_raw_proxy_weight_note=best_note,
        loadings=loadings,
        scores=scores_df,
        summary_text="\n".join(summary_lines),
    )


# =========================
# 6) 그림 출력
# =========================

def save_block_png(result: PCAResult) -> None:
    if result.loadings.empty:
        return

    max_pc_to_plot = min(5, len(result.explained_variance_ratio))
    pcs = [f"PC{i+1}" for i in range(max_pc_to_plot)]
    loadings_plot = result.loadings[pcs]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Scree
    ax1 = fig.add_subplot(gs[0, 0])
    xs = np.arange(1, len(result.explained_variance_ratio) + 1)
    ax1.plot(xs, result.explained_variance_ratio, marker="o")
    ax1.set_title(f"{result.block_name} - Scree Plot")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.grid(alpha=0.3)

    # cumulative
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(xs, result.cumulative_variance_ratio, marker="o")
    ax2.axhline(0.80, linestyle="--", linewidth=1)
    ax2.axhline(0.90, linestyle="--", linewidth=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"{result.block_name} - Cumulative Variance")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.grid(alpha=0.3)

    # loadings heatmap-like
    ax3 = fig.add_subplot(gs[1, :])
    im = ax3.imshow(loadings_plot.values, aspect="auto")
    ax3.set_title(f"{result.block_name} - Loadings")
    ax3.set_xticks(range(len(pcs)))
    ax3.set_xticklabels(pcs)
    ax3.set_yticks(range(len(loadings_plot.index)))
    ax3.set_yticklabels(loadings_plot.index)
    for i in range(loadings_plot.shape[0]):
        for j in range(loadings_plot.shape[1]):
            ax3.text(j, i, f"{loadings_plot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax3, fraction=0.025, pad=0.01)

    valid_text = "YES" if result.validity["valid_for_exploratory_pca"] else "CAUTION/NO"
    note = (
        f"n={result.n}, p={result.p}, valid={valid_text}, "
        f"KMO={result.validity['kmo']:.3f} " if not pd.isna(result.validity['kmo']) else f"n={result.n}, p={result.p}, valid={valid_text}, KMO=NA "
    )
    fig.suptitle(note, fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(PNG_DIR / f"{result.block_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# =========================
# 7) Excel/TXT 출력
# =========================

def write_outputs(results: List[PCAResult]) -> None:
    summary_rows = []
    txt_lines = []

    excel_path = OUTPUT_DIR / "PCA_block_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for res in results:
            summary_rows.append({
                "block": res.block_name,
                "native_unit": res.native_unit,
                "n": res.n,
                "p": res.p,
                "valid_for_exploratory_pca": res.validity["valid_for_exploratory_pca"],
                "kmo": res.validity["kmo"],
                "bartlett_p": res.validity["bartlett_p"],
                "avg_abs_r": res.validity["avg_abs_r"],
                "n_components_80": res.n_components_80,
                "n_components_90": res.n_components_90,
                "n_components_95": res.n_components_95,
                "reducible_to_2pc_80": res.reducible_to_2pc_80,
                "reducible_to_2pc_90": res.reducible_to_2pc_90,
                "one_raw_proxy_ok": res.one_raw_proxy_ok,
                "one_raw_proxy_name": res.one_raw_proxy_name,
                "one_raw_proxy_r2": res.one_raw_proxy_r2,
                "used_columns": ", ".join(res.used_columns),
                "cautions": " | ".join(res.validity["cautions"]),
            })

            evr_df = pd.DataFrame({
                "PC": [f"PC{i+1}" for i in range(len(res.explained_variance_ratio))],
                "explained_variance_ratio": res.explained_variance_ratio,
                "cumulative_variance_ratio": res.cumulative_variance_ratio,
            })
            load_df = res.loadings.copy()
            if load_df.empty:
                load_df = pd.DataFrame({"message": ["PCA skipped: usable variable count < 2"]})

            block_sheet = res.block_name[:31]
            evr_df.to_excel(writer, sheet_name=block_sheet, startrow=0, index=False)
            load_df.to_excel(writer, sheet_name=block_sheet, startrow=len(evr_df) + 3, index=True)

            txt_lines.append(res.summary_text)
            txt_lines.append("=" * 100)

        summary_df = pd.DataFrame(summary_rows).sort_values(by=["valid_for_exploratory_pca", "block"], ascending=[False, True])
        summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)

    txt_path = OUTPUT_DIR / "PCA_block_interpretation.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")


# =========================
# 8) 실행
# =========================

def main() -> None:
    print(f"[INFO] INPUT: {INPUT_XLSX}")
    print(f"[INFO] OUTPUT DIR: {OUTPUT_DIR}")
    sheets = load_sheets(INPUT_XLSX)
    results: List[PCAResult] = []

    for block_name, cols in BLOCKS.items():
        print(f"[RUN] {block_name}")
        res = run_block_pca(block_name, cols, sheets)
        results.append(res)
        save_block_png(res)

    write_outputs(results)
    print("[DONE] Files saved:")
    print(f"- {OUTPUT_DIR / 'PCA_block_summary.xlsx'}")
    print(f"- {OUTPUT_DIR / 'PCA_block_interpretation.txt'}")
    print(f"- {PNG_DIR}")


if __name__ == "__main__":
    main()
