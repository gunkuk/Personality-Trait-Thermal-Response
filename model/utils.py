from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# =========================
# Basic DF helpers
# =========================
def norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cands = [c.lower() for c in candidates]
    for c in cands:
        if c in df.columns:
            return c
    return None


def keep_existing(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def ensure_columns(df: pd.DataFrame, cols: List[str], fill_value=np.nan) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def to_numeric_df(df: pd.DataFrame, cols: List[str], missing_tokens: Tuple[str, ...]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out = out.replace(list(missing_tokens), np.nan).infer_objects(copy=False)
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def to_numeric_series(s: pd.Series, missing_tokens: Tuple[str, ...]) -> pd.Series:
    x = s.replace(list(missing_tokens), np.nan).infer_objects(copy=False)
    return pd.to_numeric(x, errors="coerce")


def select_cols_regex(df: pd.DataFrame, pattern: str) -> List[str]:
    rx = re.compile(pattern)
    return [c for c in df.columns if rx.match(str(c))]


# =========================
# Encoders
# =========================
def encode_sex(series: pd.Series) -> pd.Series:
    """
    sex_bin: 남/남자/M/male -> 1, 여/여자/F/female -> 0, 그 외 NaN
    """
    s = series.astype(str).str.strip().str.lower()
    male = {"m", "male", "남", "남자", "man", "1"}
    female = {"f", "female", "여", "여자", "woman", "0"}
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s.isin(male)] = 1.0
    out[s.isin(female)] = 0.0
    return out


def mbti_to_bin(series: pd.Series, threshold: float = 50.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (s >= threshold).astype("float64")


# =========================
# Scenario helpers
# =========================
def env_to_set(env: str) -> str:
    e = str(env).strip().upper()
    if e in ("A", "B"):
        return "AB"
    if e in ("C", "D"):
        return "CD"
    if e in ("E", "F"):
        return "EF"
    if e in ("G", "H"):
        return "GH"
    if e == "HG":
        return "GH"
    return np.nan


def set_to_direction(set_code: str) -> float:
    s = str(set_code).strip().upper()
    if s in ("AB", "GH"):
        return -1.0  # cooling
    if s in ("CD", "EF"):
        return 1.0   # heating
    return np.nan


def make_phase(times: pd.Series, static_times: Tuple[int, ...], dynamic_times: Tuple[int, ...]) -> pd.Series:
    return np.where(
        times.isin(static_times),
        "static",
        np.where(times.isin(dynamic_times), "dynamic", pd.NA),
    )


# =========================
# Correlation helpers
# =========================
def lower_triangle_only(corr: pd.DataFrame, keep_diag: bool = False) -> pd.DataFrame:
    n = corr.shape[0]
    k = 1 if keep_diag else 0
    mask = np.triu(np.ones((n, n), dtype=bool), k=k)
    return corr.mask(mask)


def reorder_square(corr: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    cols = [c for c in order if c in corr.columns]
    return corr.loc[cols, cols]


# =========================
# Stats helpers
# =========================
def zscore_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if np.isfinite(sd) and sd > 0:
            df[c] = (x - mu) / sd


# =========================
# Excel writer
# =========================
def write_excel(out_path: str, sheets: Dict[str, pd.DataFrame], index: bool = True) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=index)