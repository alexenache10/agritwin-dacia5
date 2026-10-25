import json
import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

DATASET_ROOT = Path("/mnt/d/Master/AN2/SEM1/IC3/data/Dataset_DACIA5")
P1_TRAIN_DIR = DATASET_ROOT / "32x32_patches/32x32_SAR+MSI_patches/problem1/training_2020-2023"
P1_TEST_DIR = DATASET_ROOT / "32x32_patches/32x32_SAR+MSI_patches/problem1/test_2024"
OUTPUT_DIR = Path("docs")

def parse_filename(name):
    pattern = r"S[12]_patch(?:RGB)?_(\d{8})_(\d+)_(\d+)\.(mat|tif|png)"
    m = re.match(pattern, name)
    if not m:
        return None
    d_str, target, patch_id = m.groups()[:3]
    date = datetime.strptime(d_str, "%Y%m%d")
    return {
        "date": date,
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "doy": date.timetuple().tm_yday,
        "target": int(target),
        "patch_id": int(patch_id),
        "parcel_uid": f"{date.year}_{target}_{patch_id}"
    }

def inspect_sample_arrays(train_dir):
    s1_mat_path = next((train_dir / "sentinel1_patches_mat").glob("*.mat"))
    s2_mat_path = next((train_dir / "sentinel2_patches_mat").glob("*.mat"))

    s1_mat = loadmat(str(s1_mat_path))
    s2_mat = loadmat(str(s2_mat_path))

    s1_k = [k for k in s1_mat.keys() if not k.startswith("__")][0]
    s2_k = [k for k in s2_mat.keys() if not k.startswith("__")][0]

    s1_arr = s1_mat[s1_k]
    s2_arr = s2_mat[s2_k]

    return {
        "s1_key": s1_k,
        "s1_shape": list(s1_arr.shape),
        "s1_dtype": str(s1_arr.dtype),
        "s1_min": float(np.min(s1_arr)),
        "s1_max": float(np.max(s1_arr)),
        "s2_key": s2_k,
        "s2_shape": list(s2_arr.shape),
        "s2_dtype": str(s2_arr.dtype),
        "s2_min": float(np.min(s2_arr)),
        "s2_max": float(np.max(s2_arr)),
    }

def process_split_csv(csv_path):
    df = pd.read_csv(csv_path)
    s1_meta = [parse_filename(f) for f in df["filenameS1"]]
    s2_meta = [parse_filename(f) for f in df["filenameS2"]]

    df_meta = pd.DataFrame(s1_meta)
    df_meta["s2_date"] = [m["date"] for m in s2_meta]
    df_meta["delta_days"] = (df_meta["date"] - df_meta["s2_date"]).abs().dt.days

    return df, df_meta

def run_audit():
    train_csv = P1_TRAIN_DIR / "train.csv"
    test_csv = P1_TEST_DIR / "test.csv"

    sample_meta = inspect_sample_arrays(P1_TRAIN_DIR)
    train_df, train_meta = process_split_csv(train_csv)
    test_df, test_meta = process_split_csv(test_csv)

    train_parcels = train_meta.groupby("parcel_uid")
    test_parcels = test_meta.groupby("parcel_uid")

    train_seq_len = train_parcels.size()
    test_seq_len = test_parcels.size()

    train_class_counts = train_meta.groupby("target")["parcel_uid"].nunique().to_dict()
    test_class_counts = test_meta.groupby("target")["parcel_uid"].nunique().to_dict()

    train_years = train_meta.groupby("year")["parcel_uid"].nunique().to_dict()
    test_years = test_meta.groupby("year")["parcel_uid"].nunique().to_dict()

    month_dist_train = train_meta["month"].value_counts().sort_index().to_dict()
    month_dist_test = test_meta["month"].value_counts().sort_index().to_dict()

    delta_days_train = train_meta["delta_days"].value_counts().to_dict()
    delta_days_test = test_meta["delta_days"].value_counts().to_dict()

    audit_report = {
        "sample_meta": sample_meta,
        "train_pairs_count": len(train_df),
        "test_pairs_count": len(test_df),
        "train_parcels_unique": len(train_parcels),
        "test_parcels_unique": len(test_parcels),
        "train_years_parcels": train_years,
        "test_years_parcels": test_years,
        "train_class_parcels": train_class_counts,
        "test_class_parcels": test_class_counts,
        "train_seq_len_stats": {
            "min": int(train_seq_len.min()),
            "max": int(train_seq_len.max()),
            "mean": float(train_seq_len.mean()),
            "median": float(train_seq_len.median()),
        },
        "test_seq_len_stats": {
            "min": int(test_seq_len.min()),
            "max": int(test_seq_len.max()),
            "mean": float(test_seq_len.mean()),
            "median": float(test_seq_len.median()),
        },
        "month_distribution_train": month_dist_train,
        "month_distribution_test": month_dist_test,
        "s1_s2_temporal_delta_days_train": delta_days_train,
        "s1_s2_temporal_delta_days_test": delta_days_test,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "datacard_stats.json", "w") as f:
        json.dump(audit_report, f, indent=2)

    print("=" * 60)
    print("DACIA5 AUDIT SUMMARY (Problem 1)")
    print("=" * 60)
    print(f"Train Pairs: {len(train_df)} | Unique Parcels: {len(train_parcels)}")
    print(f"Test Pairs:  {len(test_df)} | Unique Parcels: {len(test_parcels)}")
    print("-" * 60)
    print(f"Array Shapes: S1 {sample_meta['s1_shape']} ({sample_meta['s1_dtype']}) | S2 {sample_meta['s2_shape']} ({sample_meta['s2_dtype']})")
    print(f"S1 Value Range: [{sample_meta['s1_min']:.2f}, {sample_meta['s1_max']:.2f}]")
    print(f"S2 Value Range: [{sample_meta['s2_min']:.2f}, {sample_meta['s2_max']:.2f}]")
    print("-" * 60)
    print("Train Class Distribution (Unique field time series):", train_class_counts)
    print("Test Class Distribution (Unique field time series):", test_class_counts)
    print("-" * 60)
    print("Parcels per Year (Train):", train_years)
    print("Parcels per Year (Test):", test_years)
    print("-" * 60)
    print("Sequence Length Train (Min/Mean/Max):", f"{train_seq_len.min()} / {train_seq_len.mean():.1f} / {train_seq_len.max()}")
    print("Sequence Length Test  (Min/Mean/Max):", f"{test_seq_len.min()} / {test_seq_len.mean():.1f} / {test_seq_len.max()}")
    print("-" * 60)
    print("Monthly Acquisition Dist (Train):", month_dist_train)
    print("Monthly Acquisition Dist (Test):", month_dist_test)
    print("-" * 60)
    print("S1-S2 Temporal Delays in Days (Train):", delta_days_train)
    print("=" * 60)

if __name__ == "__main__":
    run_audit()