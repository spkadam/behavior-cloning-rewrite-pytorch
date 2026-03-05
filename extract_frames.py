"""
extract_frames.py — Extract frames from a video + CSV into image files.
It reads a video and a CSV with steering data, extracts one JPEG per frame, and writes a new
CSV (`<name>_keras.csv`) containing only the columns needed for training:

    image_idx, time, propel_cmd, steer_cmd

USAGE:
  python extract_frames.py \
      --video  /path/to/data1_2018_07_05_15_01_28.avi \
      --csv    /path/to/data1_2018_07_05_15_01_28.csv \
      --outdir /path/to/data1_2018_07_05_15_01_28/
"""

import os
import csv
import cv2
import pandas as pd
import argparse


def _read_vehicle_csv(csv_path: str) -> pd.DataFrame:
    """
    Read the raw CSV exported by the vehicle logger.

    These CSVs have two quirks:
      1. A '%' prefix on the header line (e.g. "%image_idx,...").
      2. Sometimes the header and first data row are fused on the same
         line with NO newline between them.

    We fix both by reading the raw text, splitting the header from the
    data, and feeding clean lines to pandas.
    """
    EXPECTED_HEADER = "image_idx,time,propel_cmd,steer_cmd,height_cmd,tilt_cmd,propel_est,steer_est"
    NUM_COLS = len(EXPECTED_HEADER.split(","))  # 8

    with open(csv_path, "r") as f:
        raw_lines = f.readlines()

    # Strip the '%' prefix from the first line
    first_line = raw_lines[0].lstrip("%").strip()

    # Check if the header and first data row are fused (more fields than
    # expected on the first line).
    parts = first_line.split(",")
    if len(parts) > NUM_COLS:
        # Header is the first NUM_COLS fields; the rest is the first data row
        header = ",".join(parts[:NUM_COLS])
        first_data = ",".join(parts[NUM_COLS:])
        clean_lines = [header + "\n", first_data + "\n"] + raw_lines[1:]
    else:
        clean_lines = [first_line + "\n"] + raw_lines[1:]

    from io import StringIO
    df = pd.read_csv(StringIO("".join(clean_lines)))
    df.columns = [c.strip().lstrip("%") for c in df.columns]
    return df


def extract(video_path: str, csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Read source CSV (handles the vehicle logger quirks) ----
    df = _read_vehicle_csv(csv_path)
    image_idx = df["image_idx"].values
    time_vals = df["time"].values
    propel = df["propel_cmd"].values
    steer = df["steer_cmd"].values

    # ---- Prepare output CSV ----
    out_csv_path = os.path.splitext(csv_path)[0] + "_keras.csv"
    out_csv = open(out_csv_path, "w", newline="")
    writer = csv.writer(out_csv)
    writer.writerow(["image_idx", "time", "propel_cmd", "steer_cmd"])

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    row_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if row_idx >= len(df):
            break

        idx = int(image_idx[row_idx])
        if idx != row_idx:
            print(f"WARNING: frame index mismatch: expected {row_idx}, got {idx}")
            break

        # Save frame as JPEG
        img_path = os.path.join(out_dir, f"{idx}.jpg")
        cv2.imwrite(img_path, frame)

        # Write CSV row
        writer.writerow([idx, time_vals[row_idx], propel[row_idx], steer[row_idx]])
        row_idx += 1

    cap.release()
    out_csv.close()
    print(f"Extracted {row_idx} frames → {out_dir}")
    print(f"Output CSV → {out_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames + write training CSV")
    parser.add_argument("--video", required=True, help="Path to .avi video file")
    parser.add_argument("--csv", required=True, help="Path to source .csv file")
    parser.add_argument("--outdir", required=True, help="Directory for extracted JPEG images")
    args = parser.parse_args()

    extract(args.video, args.csv, args.outdir)
