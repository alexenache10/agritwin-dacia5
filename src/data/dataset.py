import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Inject project root into sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import config


def parse_patch_filename(name: str) -> Optional[Dict[str, Any]]:
    pattern = r"S[12]_patch(?:RGB)?_(\d{8})_(\d+)_(\d+)\.(mat|tif|png)"
    match = re.match(pattern, name)
    if not match:
        return None
    d_str, target_str, patch_id_str = match.groups()[:3]
    date = datetime.strptime(d_str, "%Y%m%d")
    return {
        "date": date,
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "doy": date.timetuple().tm_yday,
        "target": int(target_str),
        "patch_id": int(patch_id_str),
        "parcel_uid": f"{date.year}_{target_str}_{patch_id_str}",
    }


class DACIA5TimeSeriesDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_month: Optional[int] = None,
        spatial_aggregation: str = "mean",
        root_dir: Optional[Path] = None,
    ):
        self.split = split
        self.max_month = max_month
        self.spatial_aggregation = spatial_aggregation

        base_dir = (
            root_dir or config.data_root
        ) / "32x32_patches/32x32_SAR+MSI_patches/problem1"
        if self.split == "train":
            self.split_dir = base_dir / "training_2020-2023"
            self.csv_path = self.split_dir / "train.csv"
        else:
            self.split_dir = base_dir / "test_2024"
            self.csv_path = self.split_dir / "test.csv"

        self.s1_dir = self.split_dir / "sentinel1_patches_mat"
        self.s2_dir = self.split_dir / "sentinel2_patches_mat"

        raw_df = pd.read_csv(self.csv_path)
        meta_records = [parse_patch_filename(f) for f in raw_df["filenameS1"]]
        meta_df = pd.DataFrame(meta_records)

        # Merge metadata without duplicating target columns
        df = pd.concat([raw_df[["filenameS1", "filenameS2"]], meta_df], axis=1)

        # Apply early season temporal cutoff if requested
        if self.max_month is not None:
            df = df[df["month"] <= self.max_month].copy()

        df = df.sort_values(by=["parcel_uid", "date"]).reset_index(drop=True)

        self.parcels: List[Dict[str, Any]] = []
        for uid, group in df.groupby("parcel_uid"):
            if len(group) == 0:
                continue
            target_val = int(group["target"].iloc[0])
            self.parcels.append(
                {
                    "parcel_uid": str(uid),
                    "target": target_val,
                    "year": int(group["year"].iloc[0]),
                    "patch_id": int(group["patch_id"].iloc[0]),
                    "s1_files": group["filenameS1"].tolist(),
                    "s2_files": group["filenameS2"].tolist(),
                    "doys": group["doy"].tolist(),
                }
            )

        unique_targets = sorted(list({int(p["target"]) for p in self.parcels}))
        self.target_to_idx = {t: idx for idx, t in enumerate(unique_targets)}
        self.idx_to_target = {idx: t for t, idx in self.target_to_idx.items()}

    def __len__(self) -> int:
        return len(self.parcels)

    def _load_mat(self, path: Path) -> np.ndarray:
        data = loadmat(str(path))
        key = [k for k in data.keys() if not k.startswith("__")][0]
        return data[key].astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.parcels[idx]
        seq_steps = []

        for s1_file, s2_file in zip(item["s1_files"], item["s2_files"]):
            s1 = self._load_mat(self.s1_dir / s1_file)
            s2 = self._load_mat(self.s2_dir / s2_file)

            # Ensure channel dimension is first: (H, W, C) -> (C, H, W)
            if s1.ndim == 3 and s1.shape[-1] == 2:
                s1 = np.transpose(s1, (2, 0, 1))
            if s2.ndim == 3 and s2.shape[-1] == 12:
                s2 = np.transpose(s2, (2, 0, 1))

            if self.spatial_aggregation == "mean":
                # Average spatial dimensions to build feature vectors
                s1_vec = s1.mean(axis=(-2, -1))
                s2_vec = s2.mean(axis=(-2, -1))
                step_feat = np.concatenate([s1_vec, s2_vec], axis=0)
            else:
                step_feat = np.concatenate([s1, s2], axis=0)

            seq_steps.append(step_feat)

        x = torch.from_numpy(np.stack(seq_steps, axis=0))
        doys = torch.tensor(item["doys"], dtype=torch.float32)
        y = torch.tensor(self.target_to_idx[item["target"]], dtype=torch.long)

        return {
            "x": x,
            "doys": doys,
            "y": y,
            "raw_target": item["target"],
            "parcel_uid": item["parcel_uid"],
            "seq_len": len(item["doys"]),
        }


def pad_collate_sits(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [b["seq_len"] for b in batch]
    max_len = max(lengths)
    batch_size = len(batch)
    feat_dim = batch[0]["x"].shape[1:]

    padded_x = torch.zeros(
        (batch_size, max_len, *feat_dim), dtype=batch[0]["x"].dtype
    )
    padded_doys = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)

    for i, b in enumerate(batch):
        l = b["seq_len"]
        padded_x[i, :l] = b["x"]
        padded_doys[i, :l] = b["doys"]
        mask[i, :l] = True
        targets[i] = b["y"]

    return {
        "x": padded_x,
        "doys": padded_doys,
        "mask": mask,
        "y": targets,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "parcel_uids": [b["parcel_uid"] for b in batch],
    }


if __name__ == "__main__":
    train_ds = DACIA5TimeSeriesDataset(
        split="train", spatial_aggregation="mean"
    )
    early_ds = DACIA5TimeSeriesDataset(
        split="train", max_month=5, spatial_aggregation="mean"
    )
    test_ds = DACIA5TimeSeriesDataset(split="test", spatial_aggregation="mean")

    loader = DataLoader(
        early_ds, batch_size=16, shuffle=True, collate_fn=pad_collate_sits
    )
    batch_sample = next(iter(loader))

    print("Pipeline check passed:")
    print(f" - Train parcels (full season): {len(train_ds)}")
    print(f" - Train parcels (cutoff May / month <= 5): {len(early_ds)}")
    print(f" - Test parcels (2024 season): {len(test_ds)}")
    print(
        f" - Padded Batch X shape: {tuple(batch_sample['x'].shape)} (Batch, MaxSeq, FeatDim)"
    )
    print(f" - DOY shape: {tuple(batch_sample['doys'].shape)}")
    print(f" - Mask shape: {tuple(batch_sample['mask'].shape)}")
    print(f" - Target batch: {batch_sample['y'].tolist()}")