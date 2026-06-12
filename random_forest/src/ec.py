from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.baselines import macro_f1, split_features, train_rf

DEFAULT_POP_SIZE = 50
DEFAULT_N_GENERATIONS = 50
DEFAULT_TOURNAMENT_K = 3
DEFAULT_SHARING_THRESHOLD = 0.7
DEFAULT_TOP_K = 5


def run_ga(
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "primary_label",
    pop_size: int = DEFAULT_POP_SIZE,
    n_generations: int = DEFAULT_N_GENERATIONS,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    mutation_rate: float | None = None,
    sharing: bool = False,
    sharing_threshold: float = DEFAULT_SHARING_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 42,
    rf_seed: int = 42,
    progress: bool = True,
    class_weights: dict | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_features = len(feature_cols)
    if mutation_rate is None:
        mutation_rate = 1.0 / n_features

    parts = split_features(features_df, feature_cols, label_col)
    X_train, y_train = parts["train"]["X"], parts["train"]["y"]
    X_val, y_val = parts["val"]["X"], parts["val"]["y"]
    labels = sorted(features_df[label_col].unique())

    population = _init_population(pop_size, n_features, rng)

    history: list[dict] = []
    raw_fitness = np.zeros(pop_size)

    iterator = range(n_generations)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc="GA generations")

    for gen in iterator:
        raw_fitness = np.array(
            [
                _fitness(
                    population[i], X_train, y_train, X_val, y_val, labels, rf_seed,
                    class_weights=class_weights,
                )
                for i in range(pop_size)
            ]
        )

        if sharing:
            shared_fitness = apply_fitness_sharing(
                raw_fitness, population, sharing_threshold
            )
        else:
            shared_fitness = raw_fitness

        history.append(
            {
                "generation": gen,
                "best_raw": float(raw_fitness.max()),
                "mean_raw": float(raw_fitness.mean()),
                "best_shared": float(shared_fitness.max()),
                "mean_shared": float(shared_fitness.mean()),
                "diversity": _diversity(population),
            }
        )

        if gen == n_generations - 1:
            break

        new_population = np.empty_like(population)
        for i in range(pop_size):
            p1 = _tournament(shared_fitness, tournament_k, rng)
            p2 = _tournament(shared_fitness, tournament_k, rng)
            child = uniform_crossover(population[p1], population[p2], rng)
            child = bit_flip_mutate(child, mutation_rate, rng)
            if child.sum() == 0:
                child[rng.integers(n_features)] = 1
            new_population[i] = child
        population = new_population

    best_idx = int(np.argmax(raw_fitness))
    top_indices = np.argsort(raw_fitness)[-top_k:][::-1]

    return {
        "history": history,
        "final_population": population,
        "raw_fitness": raw_fitness,
        "best_mask": population[best_idx].copy(),
        "best_fitness": float(raw_fitness[best_idx]),
        "top_k_masks": [population[i].copy() for i in top_indices],
        "top_k_fitness": [float(raw_fitness[i]) for i in top_indices],
        "config": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "tournament_k": tournament_k,
            "mutation_rate": mutation_rate,
            "sharing": sharing,
            "sharing_threshold": sharing_threshold,
            "top_k": top_k,
            "seed": seed,
            "rf_seed": rf_seed,
            "feature_cols": list(feature_cols),
            "class_weights": class_weights,
        },
    }


def _init_population(
    pop_size: int, n_features: int, rng: np.random.Generator
) -> np.ndarray:
    pop = (rng.random((pop_size, n_features)) < 0.5).astype(np.int8)
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i, rng.integers(n_features)] = 1
    return pop


def _fitness(
    mask: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list,
    rf_seed: int,
    class_weights: dict | None = None,
) -> float:
    if mask.sum() == 0:
        return 0.0
    cols = X_train.columns[mask.astype(bool)]
    rf = train_rf(X_train[cols], y_train, seed=rf_seed)
    y_pred = rf.predict(X_val[cols])
    if class_weights is None:
        return macro_f1(y_val, y_pred, labels=labels)
    per_species_f1 = f1_score(
        y_val, y_pred, labels=labels, average=None, zero_division=0
    )
    weights = np.array([class_weights.get(label, 1.0) for label in labels], dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(weights, per_species_f1))


def _tournament(
    fitness: np.ndarray, k: int, rng: np.random.Generator
) -> int:
    contestants = rng.integers(0, len(fitness), size=k)
    return int(contestants[np.argmax(fitness[contestants])])


def uniform_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child


def bit_flip_mutate(
    individual: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    flips = rng.random(len(individual)) < rate
    return np.where(flips, 1 - individual, individual)


def apply_fitness_sharing(
    raw_fitness: np.ndarray,
    population: np.ndarray,
    threshold: float,
) -> np.ndarray:
    # Pairwise agreement: fraction of bits that match between two masks.
    diffs = population[:, None, :] != population[None, :, :]
    agreement = 1.0 - diffs.mean(axis=2)
    n_similar = (agreement >= threshold).sum(axis=1)
    return raw_fitness / n_similar


def _diversity(population: np.ndarray) -> float:
    pop_size = len(population)
    if pop_size < 2:
        return 0.0
    diffs = population[:, None, :] != population[None, :, :]
    pairwise = diffs.mean(axis=2)
    iu = np.triu_indices(pop_size, k=1)
    return float(pairwise[iu].mean())


def committee_predict(
    masks: list[np.ndarray],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "primary_label",
    train_split: str = "train",
    eval_split: str = "test",
    rf_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = split_features(features_df, feature_cols, label_col)
    X_train, y_train = parts[train_split]["X"], parts[train_split]["y"]
    X_eval, y_eval = parts[eval_split]["X"], parts[eval_split]["y"]

    proba_sum: np.ndarray | None = None
    classes_: np.ndarray | None = None
    used = 0
    for mask in masks:
        cols = [c for c, b in zip(feature_cols, mask) if b]
        if not cols:
            continue
        rf = train_rf(X_train[cols], y_train, seed=rf_seed)
        proba = rf.predict_proba(X_eval[cols])
        if proba_sum is None:
            proba_sum = proba
            classes_ = rf.classes_
        else:
            proba_sum = proba_sum + proba
        used += 1

    if used == 0:
        raise ValueError("All committee masks are empty")
    proba_avg = proba_sum / used
    y_pred = classes_[np.argmax(proba_avg, axis=1)]
    return y_eval.values, y_pred, proba_avg
