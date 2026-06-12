from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

DEFAULT_SR = 22050
DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_N_MFCC = 13
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def feature_columns(n_mfcc: int = DEFAULT_N_MFCC) -> list[str]:
    return (
        [f"mfcc_{i}_mean" for i in range(n_mfcc)]
        + [f"mfcc_{i}_std" for i in range(n_mfcc)]
        + [
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "spectral_flux_mean",
            "spectral_flux_std",
            "zcr_mean",
            "zcr_std",
        ]
    )


METADATA_COLUMNS = [
    "window_id",
    "clip_id",
    "primary_label",
    "common_name",
    "class_name",
    "split",
    "window_index",
]


def extract_features_for_clip(
    audio_path: Path,
    sr: int = DEFAULT_SR,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> list[dict]:
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    n_samples = int(window_seconds * sr)
    if len(y) < n_samples:
        return []

    out = []
    n_windows = len(y) // n_samples
    for i in range(n_windows):
        chunk = y[i * n_samples : (i + 1) * n_samples]
        feats = _features_for_window(chunk, sr, n_mfcc, n_fft, hop_length)
        feats["window_index"] = i
        out.append(feats)
    return out


def _features_for_window(
    y: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int
) -> dict:
    out: dict = {}

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    for i in range(n_mfcc):
        out[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        out[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    out["spectral_centroid_mean"] = float(np.mean(centroid))
    out["spectral_centroid_std"] = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    out["spectral_rolloff_mean"] = float(np.mean(rolloff))
    out["spectral_rolloff_std"] = float(np.std(rolloff))

    # Spectral flux: L2 norm of frame-to-frame differences in the magnitude
    # spectrum. librosa has no direct call, so compute from the STFT.
    spectrum = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    flux = np.linalg.norm(np.diff(spectrum, axis=1), axis=0)
    out["spectral_flux_mean"] = float(np.mean(flux))
    out["spectral_flux_std"] = float(np.std(flux))

    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=n_fft, hop_length=hop_length
    )
    out["zcr_mean"] = float(np.mean(zcr))
    out["zcr_std"] = float(np.std(zcr))

    return out


def extract_all(
    manifest: pd.DataFrame,
    raw_data_dir: Path,
    audio_subdir: str = "train_audio",
    progress: bool = True,
    sr: int = DEFAULT_SR,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> pd.DataFrame:
    audio_root = Path(raw_data_dir) / audio_subdir
    iterator = list(manifest.itertuples(index=False))

    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Extracting features")

    rows: list[dict] = []
    skipped: list[str] = []
    for row in iterator:
        path = audio_root / row.filename
        try:
            window_feats = extract_features_for_clip(
                path,
                sr=sr,
                window_seconds=window_seconds,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )
        except Exception as exc:
            warnings.warn(f"Failed to load {row.filename}: {exc}")
            skipped.append(row.clip_id)
            continue
        if not window_feats:
            skipped.append(row.clip_id)
            continue
        for w in window_feats:
            w["clip_id"] = row.clip_id
            w["primary_label"] = row.primary_label
            w["common_name"] = row.common_name
            w["class_name"] = row.class_name
            w["split"] = row.split
            w["window_id"] = f"{row.clip_id}_w{w['window_index']}"
            rows.append(w)

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} clips (shorter than {window_seconds}s "
            f"or failed to load)"
        )

    df = pd.DataFrame(rows)
    return df[METADATA_COLUMNS + feature_columns(n_mfcc)]
