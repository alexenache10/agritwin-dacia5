import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DACIA5Config:
    data_root: Path = Path(
        os.getenv(
            "DACIA5_DATA_ROOT",
            "/mnt/d/Master/AN2/SEM1/IC3/data/Dataset_DACIA5",
        )
    )

    patches_dir: Path = (
        data_root / "32x32_patches" / "32x32_SAR+MSI_patches" / "problem1"
    )
    train_dir: Path = patches_dir / "training_2020-2023"
    test_dir: Path = patches_dir / "test_2024"

    train_csv: Path = train_dir / "train.csv"
    test_csv: Path = test_dir / "test.csv"

    cache_path: Path = Path("data/processed/dacia5_cp2_cache.pkl")
    results_json_path: Path = Path("docs/cp2_metrics.json")
    figure_cp2_path: Path = Path("docs/cp2_baseline_representation.png")

    crop_names = {
        101: "Winter wheat",
        108: "Corn",
        131: "Sunflower",
        151: "Winter rape",
        202: "Lucerne",
        253: "Fodder crops",
        1010: "Winter barley",
        2037: "Permanent grassland",
        3017: "Arable land / Other",
        9748: "Fallow land",
    }

    spatial_size: int = 32
    s1_channels: int = 2
    s2_channels: int = 12
    fused_channels: int = 14
    eval_months: tuple = (5, 6, 7, 8, 9, 10)
    seeds: tuple = (42, 43, 44, 45, 46)


config = DACIA5Config()