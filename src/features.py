from __future__ import annotations

import numpy as np
import librosa


def extract_mfcc_stats(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def _stack_stats(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([mat.mean(axis=1), mat.std(axis=1)], axis=0)

    return np.concatenate([_stack_stats(mfcc), _stack_stats(delta), _stack_stats(delta2)], axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)