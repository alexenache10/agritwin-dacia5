import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.utils.config import config


def harmonize_target(raw_target: int) -> int:
    if raw_target == 9747:
        return 9748
    if raw_target == 254:
        return 3017
    return raw_target


def parse_patch_filename(name: str) -> Optional[Dict[str, Any]]:
    pattern = r"S[12]_patch(?:RGB)?_(\d{8})_(\d+)_(\d+)\.(mat|tif|png)"
    match = re.match(pattern, name)
    if not match:
        return None
    d_str, target_str, patch_id_str = match.groups()[:3]
    date = datetime.strptime(d_str, "%Y%m%d")
    raw_t = int(target_str)
    harm_t = harmonize_target(raw_t)
    return {
        "date": date,
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "doy": date.timetuple().tm_yday,
        "raw_target": raw_t,
        "target": harm_t,
        "patch_id": int(patch_id_str),
        "parcel_uid": f"{date.year}_{raw_t}_{patch_id_str}",
    }


def _load_mat(path: Path) -> np.ndarray:
    data = loadmat(str(path))
    key = [k for k in data.keys() if not k.startswith("__")][0]
    return data[key].astype(np.float32)


def _compute_step_descriptors(s1: np.ndarray, s2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if s1.ndim == 3 and s1.shape[0] == 2 and s1.shape[-1] != 2:
        s1 = np.transpose(s1, (1, 2, 0))
    if s2.ndim == 3 and s2.shape[0] == 12 and s2.shape[-1] != 12:
        s2 = np.transpose(s2, (1, 2, 0))

    s1_raw_mean = s1.mean(axis=(0, 1))
    s2_raw_mean = s2.mean(axis=(0, 1)) / 10000.0
    raw_vec = np.concatenate([s1_raw_mean, s2_raw_mean], axis=0).astype(np.float32)

    eps = 1e-6
    s2_norm = s2 / 10000.0
    b2 = s2_norm[:, :, 1]
    b3 = s2_norm[:, :, 2]
    b4 = s2_norm[:, :, 3]
    b5 = s2_norm[:, :, 4]
    b8 = s2_norm[:, :, 7]
    b11 = s2_norm[:, :, 10]

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndre = (b8 - b5) / (b8 + b5 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndmi = (b8 - b11) / (b8 + b11 + eps)
    bsi = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)

    vv = s1[:, :, 0]
    vh = s1[:, :, 1]
    cr = vh - vv

    bio_maps = np.stack([vv, vh, cr, ndvi, ndre, ndwi, ndmi, bsi], axis=-1)
    bio_vec = bio_maps.mean(axis=(0, 1)).astype(np.float32)

    return bio_vec, raw_vec


def _build_split_records(split_dir: Path, csv_path: Path, split_name: str) -> List[Dict[str, Any]]:
    s1_dir = split_dir / "sentinel1_patches_mat"
    s2_dir = split_dir / "sentinel2_patches_mat"

    raw_df = pd.read_csv(csv_path)
    meta_records = [parse_patch_filename(f) for f in raw_df["filenameS1"]]
    meta_df = pd.DataFrame(meta_records)
    df = pd.concat([raw_df[["filenameS1", "filenameS2"]], meta_df], axis=1)
    df = df.sort_values(by=["parcel_uid", "date"]).reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    for uid, group in df.groupby("parcel_uid"):
        if len(group) == 0:
            continue
        bio_steps = []
        raw_steps = []
        for s1_file, s2_file in zip(group["filenameS1"], group["filenameS2"]):
            s1_arr = _load_mat(s1_dir / s1_file)
            s2_arr = _load_mat(s2_dir / s2_file)
            b_vec, r_vec = _compute_step_descriptors(s1_arr, s2_arr)
            bio_steps.append(b_vec)
            raw_steps.append(r_vec)

        records.append(
            {
                "parcel_uid": str(uid),
                "split": split_name,
                "year": int(group["year"].iloc[0]),
                "target": int(group["target"].iloc[0]),
                "raw_target": int(group["raw_target"].iloc[0]),
                "patch_id": int(group["patch_id"].iloc[0]),
                "months": group["month"].to_numpy(dtype=np.int32),
                "doys": group["doy"].to_numpy(dtype=np.float32),
                "bio_seq": np.stack(bio_steps, axis=0),
                "raw_seq": np.stack(raw_steps, axis=0),
            }
        )
    return records


def load_or_build_cache(force_rebuild: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    if config.cache_path.exists() and not force_rebuild:
        with open(config.cache_path, "rb") as f:
            return pickle.load(f)

    config.cache_path.parent.mkdir(parents=True, exist_ok=True)
    train_records = _build_split_records(config.train_dir, config.train_csv, "train")
    test_records = _build_split_records(config.test_dir, config.test_csv, "test")

    payload = {"train": train_records, "test": test_records}
    with open(config.cache_path, "wb") as f:
        pickle.dump(payload, f)
    return payload


def build_feature_matrix(
    records: List[Dict[str, Any]],
    max_month: int = 10,
    representation: str = "R1",
    modality: str = "s1_s2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    years = []
    active_mask = []

    for rec in records:
        idx = np.where(rec["months"] <= max_month)[0]
        is_active = len(idx) > 0

        if representation == "R1":
            seq = rec["bio_seq"]
            if modality == "s1_only":
                seq = seq[:, :3]
            elif modality == "s2_only":
                seq = seq[:, 3:]
            n_ch = seq.shape[1]
            feat_dim = n_ch * 8
        else:
            seq = rec["raw_seq"]
            if modality == "s1_only":
                seq = seq[:, :2]
            elif modality == "s2_only":
                seq = seq[:, 2:]
            n_ch = seq.shape[1]
            feat_dim = n_ch * 4

        if not is_active:
            feat_vec = np.zeros(feat_dim, dtype=np.float32)
        else:
            sub_seq = seq[idx]
            f_mean = np.mean(sub_seq, axis=0)
            f_std = np.std(sub_seq, axis=0)
            f_min = np.min(sub_seq, axis=0)
            f_max = np.max(sub_seq, axis=0)

            if representation == "R1":
                f_med = np.median(sub_seq, axis=0)
                f_q25 = np.percentile(sub_seq, 25, axis=0)
                f_q75 = np.percentile(sub_seq, 75, axis=0)
                f_delta = sub_seq[-1] - sub_seq[0]
                feat_vec = np.concatenate(
                    [f_mean, f_std, f_min, f_max, f_med, f_q25, f_q75, f_delta],
                    axis=0,
                ).astype(np.float32)
            else:
                feat_vec = np.concatenate(
                    [f_mean, f_std, f_min, f_max],
                    axis=0,
                ).astype(np.float32)

        features.append(feat_vec)
        targets.append(rec["target"])
        years.append(rec["year"])
        active_mask.append(is_active)

    return (
        np.stack(features, axis=0),
        np.array(targets, dtype=np.int64),
        np.array(years, dtype=np.int64),
        np.array(active_mask, dtype=bool),
    )