import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models.baselines import (
    evaluate_cv_protocol,
    evaluate_loyo_protocol,
    evaluate_temporal_2024_protocol,
)
from src.models.representations import load_or_build_cache
from src.utils.config import config


def plot_cp2_results(results: dict, output_path: Path) -> None:
    months = list(config.eval_months)
    month_labels = ["May", "Jun", "Jul", "Aug", "Sep", "Oct (Full)"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    cv_r1_fused = [results["cv_2020_2023"]["R1_s1_s2"][str(m)]["rf_macro_f1_mean"] * 100 for m in months]
    cv_r1_std = [results["cv_2020_2023"]["R1_s1_s2"][str(m)]["rf_macro_f1_std"] * 100 for m in months]
    cv_r2_fused = [results["cv_2020_2023"]["R2_s1_s2"][str(m)]["rf_macro_f1_mean"] * 100 for m in months]
    cv_r1_s2 = [results["cv_2020_2023"]["R1_s2_only"][str(m)]["rf_macro_f1_mean"] * 100 for m in months]
    cv_r1_s1 = [results["cv_2020_2023"]["R1_s1_only"][str(m)]["rf_macro_f1_mean"] * 100 for m in months]
    cv_clim = [results["cv_2020_2023"]["R1_s1_s2"][str(m)]["clim_macro_f1_mean"] * 100 for m in months]

    x = np.arange(len(months))
    axes[0].plot(x, cv_r1_fused, marker="o", linewidth=2.2, color="#1F4E79", label="RF + R1 (Biophysical S1+S2)")
    axes[0].fill_between(
        x,
        np.array(cv_r1_fused) - np.array(cv_r1_std),
        np.array(cv_r1_fused) + np.array(cv_r1_std),
        color="#1F4E79",
        alpha=0.15,
    )
    axes[0].plot(x, cv_r2_fused, marker="s", linewidth=2.0, linestyle="--", color="#D9534F", label="RF + R2 (Raw Bands S1+S2)")
    axes[0].plot(x, cv_r1_s2, marker="^", linewidth=1.8, linestyle="-.", color="#2CA02C", label="RF + R1 (Optical S2-only)")
    axes[0].plot(x, cv_r1_s1, marker="v", linewidth=1.8, linestyle=":", color="#FF7F0E", label="RF + R1 (Radar S1-only)")
    axes[0].plot(x, cv_clim, linewidth=1.5, linestyle="--", color="#7F7F7F", label="Baseline: Climatology (Prior)")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(month_labels)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[0].set_xlabel("Decision Cutoff Month (In-Season)")
    axes[0].set_title("Parcel-Grouped CV (2020-2023): Accuracy vs. Earliness")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=True)

    t24_r1_fused = [results["test_2024"]["R1_s1_s2"][str(m)]["rf_macro_f1_mean"] * 100 for m in months]
    t24_r1_std = [results["test_2024"]["R1_s1_s2"][str(m)]["rf_macro_f1_std"] * 100 for m in months]
    t24_r2_fused = [results["test_2024"]["R2_s1_s2"][str(m)]["rf_macro_f1_mean"] * 100 for m in months]
    t24_r1_s2 = [results["test_2024"]["R1_s2_only"][str(m)]["rf_macro_f1_mean"] * 100 for m in months]
    t24_clim = [results["test_2024"]["R1_s1_s2"][str(m)]["clim_macro_f1"] * 100 for m in months]

    w = 0.25
    axes[1].bar(x - w, t24_r1_fused, width=w, yerr=t24_r1_std, capsize=3, color="#1F4E79", label="RF + R1 (S1+S2)")
    axes[1].bar(x, t24_r2_fused, width=w, color="#D9534F", label="RF + R2 (Raw 14ch)")
    axes[1].bar(x + w, t24_r1_s2, width=w, color="#2CA02C", label="RF + R1 (S2-only)")
    axes[1].plot(x, t24_clim, linestyle="--", marker="x", color="#555555", label="Climatology Baseline")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(month_labels)
    axes[1].set_ylabel("Macro-F1 (%) on Unseen 2024 Year")
    axes[1].set_xlabel("Decision Cutoff Month (2024 Campaign)")
    axes[1].set_title("Out-of-Year Test (2024): R1 vs. R2 Generalization")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_cp2_pipeline() -> None:
    print("Loading or building DACIA5 time series cache...")
    data_cache = load_or_build_cache(force_rebuild=False)
    train_records = data_cache["train"]
    test_records = data_cache["test"]
    print(f"Loaded {len(train_records)} train parcels (2020-2023) and {len(test_records)} test parcels (2024).")

    configs_to_run = [
        ("R1", "s1_s2"),
        ("R2", "s1_s2"),
        ("R1", "s2_only"),
        ("R1", "s1_only"),
    ]

    results = {"cv_2020_2023": {}, "test_2024": {}, "loyo_full_season": {}}

    for rep, mod in configs_to_run:
        key = f"{rep}_{mod}"
        results["cv_2020_2023"][key] = {}
        results["test_2024"][key] = {}

        for m in config.eval_months:
            cv_res = evaluate_cv_protocol(train_records, max_month=m, representation=rep, modality=mod)
            t24_res = evaluate_temporal_2024_protocol(
                train_records, test_records, max_month=m, representation=rep, modality=mod
            )
            results["cv_2020_2023"][key][str(m)] = cv_res
            results["test_2024"][key][str(m)] = t24_res

        loyo_res = evaluate_loyo_protocol(train_records, max_month=10, representation=rep, modality=mod)
        results["loyo_full_season"][key] = loyo_res

    config.results_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_cp2_results(results, config.figure_cp2_path)

    print("\n" + "=" * 85)
    print("CP2 SUMMARY: PARCEL-GROUPED 5-FOLD CV (2020-2023) - MACRO-F1 (%)")
    print("=" * 85)
    header = f"{'Config':<15} | " + " | ".join([f"M{m:02d}" for m in config.eval_months])
    print(header)
    print("-" * 85)
    for rep, mod in configs_to_run:
        key = f"{rep}_{mod}"
        row_str = f"{key:<15} | " + " | ".join(
            [
                f"{results['cv_2020_2023'][key][str(m)]['rf_macro_f1_mean']*100:5.1f}%"
                for m in config.eval_months
            ]
        )
        print(row_str)
    clim_row = f"{'Climatology':<15} | " + " | ".join(
        [
            f"{results['cv_2020_2023']['R1_s1_s2'][str(m)]['clim_macro_f1_mean']*100:5.1f}%"
            for m in config.eval_months
        ]
    )
    print(clim_row)

    print("\n" + "=" * 85)
    print("CP2 SUMMARY: OUT-OF-YEAR TEST (2024 UNSEEN SEASON) - MACRO-F1 (%)")
    print("=" * 85)
    print(header)
    print("-" * 85)
    for rep, mod in configs_to_run:
        key = f"{rep}_{mod}"
        row_str = f"{key:<15} | " + " | ".join(
            [
                f"{results['test_2024'][key][str(m)]['rf_macro_f1_mean']*100:5.1f}%"
                for m in config.eval_months
            ]
        )
        print(row_str)
    clim_t24 = f"{'Climatology':<15} | " + " | ".join(
        [
            f"{results['test_2024']['R1_s1_s2'][str(m)]['clim_macro_f1']*100:5.1f}%"
            for m in config.eval_months
        ]
    )
    print(clim_t24)

    print("\n" + "=" * 85)
    print("LOYO (2020-2023) FULL SEASON MACRO-F1 & OA")
    print("=" * 85)
    for rep, mod in configs_to_run:
        key = f"{rep}_{mod}"
        loyo_f1 = results["loyo_full_season"][key]["loyo_macro_f1_mean"] * 100
        loyo_oa = results["loyo_full_season"][key]["loyo_oa_mean"] * 100
        print(f"{key:<15} -> LOYO Macro-F1: {loyo_f1:5.1f}% | LOYO OA: {loyo_oa:5.1f}%")
    print("=" * 85)
    print(f"Saved metrics to {config.results_json_path} and figure to {config.figure_cp2_path}")


if __name__ == "__main__":
    run_cp2_pipeline()