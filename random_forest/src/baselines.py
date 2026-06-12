from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

DEFAULT_RF_KWARGS = {"n_estimators": 100, "n_jobs": 4}


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
    **rf_kwargs,
) -> RandomForestClassifier:
    kwargs = {**DEFAULT_RF_KWARGS, "random_state": seed, **rf_kwargs}
    rf = RandomForestClassifier(**kwargs)
    rf.fit(X_train, y_train)
    return rf


def select_features_mrmr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
) -> list[str]:
    from mrmr import mrmr_classif

    return mrmr_classif(X=X_train, y=y_train, K=k, show_progress=False)


def evaluate_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence,
) -> dict:
    overall = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_species = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(overall),
        "macro_f1": float(macro),
        "per_species_recall": {
            str(label): float(value) for label, value in zip(labels, per_species)
        },
    }


def macro_f1(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence,
) -> float:
    return float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


def split_features(
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "primary_label",
) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        sub = features_df[features_df["split"] == split]
        out[split] = {
            "X": sub[list(feature_cols)],
            "y": sub[label_col],
        }
    return out


def run_baseline_all_features(
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "primary_label",
    seed: int = 42,
) -> dict:
    parts = split_features(features_df, feature_cols, label_col)
    rf = train_rf(parts["train"]["X"], parts["train"]["y"], seed=seed)
    y_pred = rf.predict(parts["test"]["X"])
    labels = sorted(features_df[label_col].unique())
    metrics = evaluate_predictions(parts["test"]["y"], y_pred, labels)
    metrics["method"] = "all_features"
    metrics["selected_features"] = list(feature_cols)
    return metrics


def run_baseline_mrmr(
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "primary_label",
    k: int = 15,
    seed: int = 42,
) -> dict:
    parts = split_features(features_df, feature_cols, label_col)
    selected = select_features_mrmr(parts["train"]["X"], parts["train"]["y"], k=k)
    rf = train_rf(parts["train"]["X"][selected], parts["train"]["y"], seed=seed)
    y_pred = rf.predict(parts["test"]["X"][selected])
    labels = sorted(features_df[label_col].unique())
    metrics = evaluate_predictions(parts["test"]["y"], y_pred, labels)
    metrics["method"] = f"mrmr_top{k}"
    metrics["selected_features"] = selected
    return metrics
