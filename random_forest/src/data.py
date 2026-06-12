from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = [
    "clip_id",
    "primary_label",
    "common_name",
    "class_name",
    "filename",
    "split",
]


def build_manifest(
    raw_data_dir: Path,
    species_specs: list[tuple[str, int]],
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

    train_csv = pd.read_csv(Path(raw_data_dir) / "train.csv")
    rng = np.random.default_rng(seed)

    parts: list[pd.DataFrame] = []
    for label, target in species_specs:
        species_df = train_csv[train_csv["primary_label"] == label].reset_index(drop=True)
        if len(species_df) == 0:
            raise ValueError(f"No clips found for primary_label '{label}'")
        if len(species_df) < target:
            raise ValueError(
                f"Species '{label}' has only {len(species_df)} clips, "
                f"requested {target}"
            )

        idx = rng.permutation(len(species_df))[:target]
        sampled = species_df.iloc[idx].reset_index(drop=True)
        sampled["split"] = _assign_splits(target, split_ratios, rng)
        parts.append(sampled)

    manifest = pd.concat(parts, ignore_index=True)
    manifest["clip_id"] = manifest["filename"].apply(lambda f: Path(f).stem)
    return manifest[MANIFEST_COLUMNS]


def _assign_splits(
    n: int,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    labels = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test,
        dtype=object,
    )
    rng.shuffle(labels)
    return labels


def split_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["primary_label", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
