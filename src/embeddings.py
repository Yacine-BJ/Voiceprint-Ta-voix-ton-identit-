"""Embedding backends for Week 2.

Baseline project starts with MFCC features. In Week 2, replace/add functions here
to extract embeddings from ECAPA-TDNN (SpeechBrain/NeMo/WeSpeaker).
"""

from __future__ import annotations

import numpy as np


def placeholder_embedding(audio: np.ndarray) -> np.ndarray:
    """Very lightweight placeholder vector.

    Replace this with a real pre-trained speaker embedding model.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.array(
        [
            float(np.mean(audio)),
            float(np.std(audio)),
            float(np.max(audio)),
            float(np.min(audio)),
        ],
        dtype=np.float32,
    )