from typing import Any, Dict, List

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.models.representations import build_feature_matrix
from src.utils.config import config


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    present_labels = np.unique(y_true)
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)
    )
    oa = float(accuracy_score(y_true, y_pred))
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "oa": oa}


def evaluate_cv_protocol(
    train_records: List[Dict[str, Any]],
    max_month: int,
    representation: str,
    modality: str,
    n_splits: int = 5,
) -> Dict[str, Any]:
    X, y, _, active = build_feature_matrix(
        train_records, max_month=max_month, representation=representation, modality=modality
    )
    X_act, y_act = X[active], y[active]

    rf_f1s, rf_oas = [], []
    clim_f1s, clim_oas = [], []

    for seed in config.seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_rf_f1, fold_rf_oa = [], []
        fold_clim_f1, fold_clim_oa = [], []

        for tr_idx, val_idx in skf.split(X_act, y_act):
            X_tr, y_tr = X_act[tr_idx], y_act[tr_idx]
            X_val, y_val = X_act[val_idx], y_act[val_idx]

            clim = DummyClassifier(strategy="prior")
            clim.fit(X_tr, y_tr)
            y_pred_clim = clim.predict(X_val)
            m_clim = _compute_metrics(y_val, y_pred_clim)
            fold_clim_f1.append(m_clim["macro_f1"])
            fold_clim_oa.append(m_clim["oa"])

            rf = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            y_pred_rf = rf.predict(X_val)
            m_rf = _compute_metrics(y_val, y_pred_rf)
            fold_rf_f1.append(m_rf["macro_f1"])
            fold_rf_oa.append(m_rf["oa"])

        rf_f1s.append(np.mean(fold_rf_f1))
        rf_oas.append(np.mean(fold_rf_oa))
        clim_f1s.append(np.mean(fold_clim_f1))
        clim_oas.append(np.mean(fold_clim_oa))

    return {
        "active_parcels": int(active.sum()),
        "total_parcels": int(len(active)),
        "rf_macro_f1_mean": float(np.mean(rf_f1s)),
        "rf_macro_f1_std": float(np.std(rf_f1s)),
        "rf_oa_mean": float(np.mean(rf_oas)),
        "clim_macro_f1_mean": float(np.mean(clim_f1s)),
        "clim_oa_mean": float(np.mean(clim_oas)),
    }


def evaluate_temporal_2024_protocol(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    max_month: int,
    representation: str,
    modality: str,
) -> Dict[str, Any]:
    X_tr_all, y_tr_all, _, act_tr = build_feature_matrix(
        train_records, max_month=max_month, representation=representation, modality=modality
    )
    X_te_all, y_te_all, _, act_te = build_feature_matrix(
        test_records, max_month=max_month, representation=representation, modality=modality
    )

    X_tr, y_tr = X_tr_all[act_tr], y_tr_all[act_tr]
    X_te, y_te = X_te_all[act_te], y_te_all[act_te]

    clim = DummyClassifier(strategy="prior")
    clim.fit(X_tr, y_tr)
    m_clim = _compute_metrics(y_te, clim.predict(X_te))

    rf_f1s, rf_oas, rf_wf1s = [], [], []
    for seed in config.seeds:
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        m_rf = _compute_metrics(y_te, y_pred)
        rf_f1s.append(m_rf["macro_f1"])
        rf_wf1s.append(m_rf["weighted_f1"])
        rf_oas.append(m_rf["oa"])

    return {
        "active_test_parcels": int(act_te.sum()),
        "total_test_parcels": int(len(act_te)),
        "rf_macro_f1_mean": float(np.mean(rf_f1s)),
        "rf_macro_f1_std": float(np.std(rf_f1s)),
        "rf_weighted_f1_mean": float(np.mean(rf_wf1s)),
        "rf_oa_mean": float(np.mean(rf_oas)),
        "rf_oa_std": float(np.std(rf_oas)),
        "clim_macro_f1": m_clim["macro_f1"],
        "clim_oa": m_clim["oa"],
    }


def evaluate_loyo_protocol(
    train_records: List[Dict[str, Any]],
    max_month: int = 10,
    representation: str = "R1",
    modality: str = "s1_s2",
) -> Dict[str, Any]:
    X, y, years, active = build_feature_matrix(
        train_records, max_month=max_month, representation=representation, modality=modality
    )
    X, y, years = X[active], y[active], years[active]

    unique_years = sorted(np.unique(years).tolist())
    year_metrics = {}

    for held_out_year in unique_years:
        tr_mask = years != held_out_year
        val_mask = years == held_out_year

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        f1_seeds, oa_seeds = [], []
        for seed in config.seeds:
            rf = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            preds = rf.predict(X_val)
            m = _compute_metrics(y_val, preds)
            f1_seeds.append(m["macro_f1"])
            oa_seeds.append(m["oa"])

        year_metrics[str(held_out_year)] = {
            "macro_f1": float(np.mean(f1_seeds)),
            "oa": float(np.mean(oa_seeds)),
            "n_parcels": int(val_mask.sum()),
        }

    mean_loyo_f1 = float(np.mean([v["macro_f1"] for v in year_metrics.values()]))
    mean_loyo_oa = float(np.mean([v["oa"] for v in year_metrics.values()]))
    return {
        "per_year": year_metrics,
        "loyo_macro_f1_mean": mean_loyo_f1,
        "loyo_oa_mean": mean_loyo_oa,
    }