import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

STATS_PATH = PROJECT_ROOT / "docs" / "datacard_stats.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "cp1_dataset_overview.png"


def plot_eda():
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)

    # 1. Monthly acquisition distribution (Train vs Test)
    months = np.arange(1, 13)
    train_months = stats["month_distribution_train"]
    test_months = stats["month_distribution_test"]

    train_vals = [train_months.get(str(m), 0) for m in months]
    test_vals = [test_months.get(str(m), 0) for m in months]

    w = 0.4
    axes[0].bar(months - w / 2, train_vals, width=w, label="Train (2020-2023)", color="#1F4E79")
    axes[0].bar(months + w / 2, test_vals, width=w, label="Test (2024)", color="#2CA02C")
    axes[0].set_xlabel("Month of Year")
    axes[0].set_ylabel("Acquisition Pairs (S1 + S2)")
    axes[0].set_title("Temporal Distribution of Satellite Scenes")
    axes[0].set_xticks(months)
    axes[0].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True)

    # Highlight early decision window
    axes[0].axvspan(4.5, 6.5, color="#FFA500", alpha=0.15, label="Early Window (May-Jun)")

    # 2. S1-S2 Temporal Delays
    delays_dict = stats["s1_s2_temporal_delta_days_train"]
    delays = sorted([int(k) for k in delays_dict.keys()])
    delay_counts = [delays_dict[str(d)] for d in delays]

    axes[1].bar([f"{d} days" for d in delays], delay_counts, color="#D9534F", width=0.5)
    axes[1].set_xlabel("S1 vs. S2 Temporal Offset")
    axes[1].set_ylabel("Number of Paired Observations")
    axes[1].set_title("Sentinel-1 / Sentinel-2 Asynchrony (Train)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(delay_counts):
        axes[1].text(i, v + 40, f"{v / sum(delay_counts) * 100:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Figure successfully saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_eda()