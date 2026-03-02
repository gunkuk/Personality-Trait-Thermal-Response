#사용법
## GUI 창이 총 3번 뜸
# 1. 앞부분이 될 ACQ 파일 선택
# 2. 뒷부분이 될 ACQ 파일 선택
# 3. 저장할 폴더 선택

## 주의사항
# 선택된 두 파일을 [0,600) SEC로 CROP하므로, ACQKNOWLEGE에서 미리 선택할 파일이 적합한 구간 (600초 조금 넘게) 으로 잘라놔야 함. 





import os
import numpy as np
import pandas as pd
import bioread
import tkinter as tk
from tkinter import filedialog
from io import StringIO

# =========================
# CONSTANTS (same as M1/M2/M3)
# =========================
TIME_COL_NAME = "sec"
CUTOFF_SEC = 600.0
TIME_OFFSET = 600.0

FLOAT_FMT = "%.6f"
DELIMITER = "\t"
LINE_TERMINATOR = "\r\n"

# =========================
# GUI
# =========================
def pick_acq_files(title: str):
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title=title,
        filetypes=[("ACQ files", "*.acq"), ("All files", "*.*")]
    )
    root.destroy()
    if not paths:
        raise SystemExit(f"[CANCELLED] {title}")
    return list(paths)

def pick_dir(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    if not path:
        raise SystemExit(f"[CANCELLED] {title}")
    return path

# =========================
# ACQ -> TXT STRING (M1/M2 behavior)
# =========================
def acq_to_txt(acq_path: str, mode: str) -> str:
    """
    mode:
      - "front": trim <600, no shift
      - "back" : trim <600, +600 shift
    returns: 2-line header + numeric data (tab, CRLF, float 6)
    """
    acq = bioread.read_file(acq_path)
    fs = acq.samples_per_second

    channels = [ch for ch in acq.channels if ch.data is not None]
    if len(channels) == 0:
        return ""

    min_len = min(len(ch.data) for ch in channels)
    time_sec = np.arange(min_len) / fs

    data = {TIME_COL_NAME: time_sec}
    for idx, ch in enumerate(channels, start=1):
        data[f"CH{idx}"] = ch.data[:min_len]

    df = pd.DataFrame(data)

    # TRIM
    df = df[df[TIME_COL_NAME] < CUTOFF_SEC].copy()
    if len(df) == 0:
        return ""

    # SHIFT (back)
    if mode == "back":
        df[TIME_COL_NAME] = df[TIME_COL_NAME] + TIME_OFFSET

    # HEADER 2 lines
    desc_row = [""]
    for ch in channels:
        name = ch.name.strip() if ch.name else "unknown"
        unit = ch.units.strip() if ch.units else "unknown"
        desc_row.append(f"{name} ({unit})")

    header1 = DELIMITER.join(df.columns)
    header2 = DELIMITER.join(desc_row)

    data_str = df.to_csv(
        sep=DELIMITER,
        index=False,
        header=False,
        float_format=FLOAT_FMT,
        lineterminator=LINE_TERMINATOR
    )

    return header1 + LINE_TERMINATOR + header2 + LINE_TERMINATOR + data_str

def numeric_df_from_txt(txt: str) -> pd.DataFrame:
    if not txt:
        return pd.DataFrame()
    return pd.read_csv(StringIO(txt), sep=DELIMITER, header=None, skiprows=2)

def header2_from_txt(txt: str) -> str:
    if not txt:
        return ""
    lines = txt.splitlines()
    if len(lines) < 2:
        return ""
    return LINE_TERMINATOR.join(lines[:2]) + LINE_TERMINATOR

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    front_files = pick_acq_files("Select FRONT .acq files (M1 input)")
    back_files  = pick_acq_files("Select BACK .acq files (M2 input)")
    out_dir     = pick_dir("Select OUTPUT folder (save merged txt here)")
    os.makedirs(out_dir, exist_ok=True)

    if len(front_files) != len(back_files):
        raise ValueError(
            f"Selection count mismatch: front={len(front_files)} vs back={len(back_files)}. "
            "Select the same number of files in the same order."
        )

    merged_count = 0
    skipped_count = 0

    for front_path, back_path in zip(front_files, back_files):
        front_base = os.path.splitext(os.path.basename(front_path))[0]
        back_base  = os.path.splitext(os.path.basename(back_path))[0]

        print(f"[PAIR] front={front_base}.acq | back={back_base}.acq")

        txt_front = acq_to_txt(front_path, mode="front")
        txt_back  = acq_to_txt(back_path,  mode="back")

        if not txt_front:
            print(f"  -> front empty after trim (<{CUTOFF_SEC}s), skipped")
            skipped_count += 1
            continue
        if not txt_back:
            print(f"  -> back empty after trim (<{CUTOFF_SEC}s), skipped")
            skipped_count += 1
            continue

        df1 = numeric_df_from_txt(txt_front)
        df2 = numeric_df_from_txt(txt_back)

        if df1.empty or df2.empty:
            print("  -> parsed empty numeric data, skipped")
            skipped_count += 1
            continue

        if df1.shape[1] != df2.shape[1]:
            raise ValueError(
                f"Column count mismatch:\n"
                f"  front={os.path.basename(front_path)} cols={df1.shape[1]}\n"
                f"  back ={os.path.basename(back_path)} cols={df2.shape[1]}"
            )

        TIME_COL_IDX = 0
        df = (
            pd.concat([df1, df2], ignore_index=True)
            .sort_values(TIME_COL_IDX)
            .reset_index(drop=True)
        )

        # Output naming: based on front file name
        out_name = f"{front_base}_merged.txt"
        out_path = os.path.join(out_dir, out_name)

        # Re-attach 2-line header (use FRONT header as canonical)
        header_str = header2_from_txt(txt_front)

        data_str = df.to_csv(
            sep=DELIMITER,
            index=False,
            header=False,
            float_format=FLOAT_FMT,
            lineterminator=LINE_TERMINATOR
        )

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(header_str)
            f.write(data_str)

        print(
            f"  -> saved: {out_name} | "
            f"range={df.iloc[0, TIME_COL_IDX]:.6f}~{df.iloc[-1, TIME_COL_IDX]:.6f} | "
            f"samples={len(df)}"
        )
        merged_count += 1

    print(f"=== DONE === merged={merged_count}, skipped={skipped_count}")