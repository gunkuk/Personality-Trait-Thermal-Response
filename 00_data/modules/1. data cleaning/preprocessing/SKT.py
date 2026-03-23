import os
import numpy as np
import pandas as pd
from scipy.signal import resample

# =========================
# SETTINGS
# =========================
FS_ST = 1       # ST target Hz
FS_AT = 0.2     # AT target Hz (5 sec)

ST_SMOOTH_SEC = 60
AT_SMOOTH_SEC = 120

INPUT_DIR = r"C:\Project\mbti\01_data_normalized"          # 20분 정규화된 txt 폴더
PREPROCESS_DIR = r"C:\Project\mbti\02_preprocessing"
FEATURE_DIR = r"C:\Project\mbti\03_features"

os.makedirs(PREPROCESS_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# =========================
# UTIL FUNCTIONS
# =========================
def moving_average(signal, window):
    if window <= 1 or len(signal) < window:
        return signal

    pad = window // 2
    padded = np.pad(signal, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:-pad]


def load_acq_txt(file_path):

    with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    header_cols = None

    for i, line in enumerate(lines):

        clean_line = line.replace("\ufeff", "").strip()
        parts = clean_line.split("\t")

        if len(parts) > 1 and parts[0].strip().lower() == "sec":
            header_idx = i
            header_cols = parts
            break

    if header_idx is None:
        print("DEBUG first line:", repr(lines[0]))
        raise ValueError("HEADER NOT FOUND")

    # description 줄
    desc_line = lines[header_idx + 1].replace("\ufeff", "").rstrip("\n")
    desc_parts = desc_line.split("\t")

    if desc_parts[0] == "":
        desc_parts = desc_parts[1:]

    if len(desc_parts) == len(header_cols) - 1:
        channel_order = desc_parts
    else:
        channel_order = header_cols[1:]


    data_rows = []
    n_cols = len(header_cols)

    for line in lines[header_idx + 2:]:

        clean_line = line.replace("\ufeff", "").strip()
        parts = clean_line.split("\t")

        if len(parts) == n_cols:
            try:
                float(parts[0])
                data_rows.append(parts)
            except:
                continue

    df = pd.DataFrame(data_rows, columns=header_cols)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df, channel_order

def quarter_means(signal):
    if signal is None or len(signal) < 4:
        return np.nan, np.nan, np.nan, np.nan

    splits = np.array_split(signal, 4)
    return tuple(np.nanmean(s) for s in splits)


# =========================
# MAIN PROCESS
# =========================
summary_rows = []

for fname in sorted(os.listdir(INPUT_DIR)):

    if not fname.lower().endswith(".txt"):
        continue

    print(f"\n[PROCESSING] {fname}")

    file_path = os.path.join(INPUT_DIR, fname)
    stem = os.path.splitext(fname)[0]

    parts = stem.split("_")
    subject = parts[0]
    condition = parts[1] if len(parts) > 1 else "NA"

    try:
        df, channel_order = load_acq_txt(file_path)
    except Exception as e:
        print(f"[SKIP] {fname} → {e}")
        continue

    # ---------------- RAW Sampling Rate 자동 계산 ----------------
    time = df["sec"].values
    fs_raw = round(1.0 / np.mean(np.diff(time)))
    print("  fs_raw =", fs_raw)

    data_cols = df.columns[1:]

    channel_map = {}
    for i, ch_name in enumerate(channel_order):
        if i < len(data_cols):
            channel_map[ch_name] = data_cols[i]

    st_key = next((k for k in channel_map if "SKT A" in k), None)
    at_key = next((k for k in channel_map if "SKT B" in k), None)

    if st_key is None or at_key is None:
        print(f"[SKIP] {fname}: SKT A/B not found")
        continue

    st_raw = df[channel_map[st_key]].values.astype(float)
    at_raw = df[channel_map[at_key]].values.astype(float)

    # ---------------- ST ----------------
    n_st = max(4, int(round(len(st_raw) * FS_ST / fs_raw)))
    st_ds = resample(st_raw, n_st)
    st_smooth = moving_average(st_ds, int(ST_SMOOTH_SEC * FS_ST))

    np.save(os.path.join(PREPROCESS_DIR, f"{stem}_st.npy"), st_smooth)

    # ---------------- AT ----------------
    n_at = max(4, int(round(len(at_raw) * FS_AT / fs_raw)))
    at_ds = resample(at_raw, n_at)
    at_smooth = moving_average(at_ds, int(AT_SMOOTH_SEC * FS_AT))

    np.save(os.path.join(PREPROCESS_DIR, f"{stem}_at.npy"), at_smooth)

    print("  ST length:", len(st_smooth))
    print("  AT length:", len(at_smooth))

    # ---------------- Quarter Mean ----------------
    st_q1, st_q2, st_q3, st_q4 = quarter_means(st_smooth)
    at_q1, at_q2, at_q3, at_q4 = quarter_means(at_smooth)

    summary_rows.append({
        "subject": subject,
        "condition": condition,
        "st_q1_mean": st_q1,
        "st_q2_mean": st_q2,
        "st_q3_mean": st_q3,
        "st_q4_mean": st_q4,
        "at_q1_mean": at_q1,
        "at_q2_mean": at_q2,
        "at_q3_mean": at_q3,
        "at_q4_mean": at_q4
    })

# =========================
# =========================
EXPECTED_COLS = [
    "subject","condition",
    "st_q1_mean","st_q2_mean","st_q3_mean","st_q4_mean",
    "at_q1_mean","at_q2_mean","at_q3_mean","at_q4_mean"
]

summary_df = pd.DataFrame(summary_rows)

for col in EXPECTED_COLS:
    if col not in summary_df.columns:
        summary_df[col] = np.nan

summary_df = summary_df[EXPECTED_COLS]

summary_df.to_csv(
    os.path.join(FEATURE_DIR, "temperature_quarter_means_all.csv"),
    sep="\t",             
    index=False,
    encoding="utf-8-sig"  
)

print("\n=== DONE ===")