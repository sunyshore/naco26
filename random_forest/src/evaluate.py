from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.baselines import (
    evaluate_predictions,
    split_features,
    train_rf,
)
from src.ec import committee_predict


def evaluate_mask_on_test(
    mask: np.ndarray,
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "primary_label",
    method_name: str = "method",
    seed: int = 42,
) -> dict:
    cols = [c for c, b in zip(feature_cols, mask) if b]
    if not cols:
        raise ValueError("Mask selects zero features")
    parts = split_features(features_df, feature_cols, label_col)
    rf = train_rf(parts["train"]["X"][cols], parts["train"]["y"], seed=seed)
    y_pred = rf.predict(parts["test"]["X"][cols])
    labels = sorted(features_df[label_col].unique())
    metrics = evaluate_predictions(parts["test"]["y"], y_pred, labels)
    metrics["method"] = method_name
    metrics["selected_features"] = cols
    metrics["n_selected"] = len(cols)
    return metrics


def evaluate_committee_on_test(
    masks: list[np.ndarray],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "primary_label",
    method_name: str = "ec_niched_committee",
    seed: int = 42,
) -> dict:
    y_true, y_pred, _ = committee_predict(
        masks,
        features_df,
        feature_cols,
        label_col=label_col,
        train_split="train",
        eval_split="test",
        rf_seed=seed,
    )
    labels = sorted(features_df[label_col].unique())
    metrics = evaluate_predictions(y_true, y_pred, labels)
    metrics["method"] = method_name
    metrics["committee_size"] = len(masks)
    metrics["selected_features_per_member"] = [
        [c for c, b in zip(feature_cols, m) if b] for m in masks
    ]
    return metrics


def comparison_table(method_results: list[dict]) -> pd.DataFrame:
    rows = []
    for m in method_results:
        row = {
            "method": m["method"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
        }
        for label, recall in m["per_species_recall"].items():
            row[f"recall_{label}"] = recall
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def plot_comparison(comparison_df: pd.DataFrame, save_path: Path | None = None):
    import matplotlib.pyplot as plt

    label_map = {
        "all_features": "All features",
        "mrmr_top15": "mRMR top-15",
        "ec_plain_best": "Plain EC\nbest",
        "ec_plain_committee": "Plain EC\ncommittee",
        "ec_niched_best": "Niched EC\nbest",
        "ec_niched_committee": "Niched EC\ncommittee",
        "ec_niched_weighted_best": "Niched EC\nweighted best",
        "ec_niched_weighted_committee": "Niched EC\nweighted committee",
    }

    df = comparison_df.copy()
    df.index = [label_map.get(i, i) for i in df.index]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    headline = df[["accuracy", "macro_f1"]]
    headline.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("Overall accuracy and macro-F1 by method")
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel("score")
    axes[0].set_xlabel("")
    axes[0].legend(loc="lower right")
    axes[0].tick_params(axis='x', labelsize=8)

    recall_cols = [c for c in df.columns if c.startswith("recall_")]
    per_species = df[recall_cols].copy()
    per_species.columns = [c.replace("recall_", "") for c in recall_cols]
    per_species.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Per-species recall by method")
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel("recall")
    axes[1].set_xlabel("")
    axes[1].legend(loc="lower right", fontsize="small")
    axes[1].tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig