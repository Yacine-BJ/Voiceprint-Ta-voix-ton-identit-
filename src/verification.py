from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

from src.evaluation import compute_eer, compute_mindcf
from src.features import cosine_similarity, extract_mfcc_stats


def _load_audio_map(sample_indices: set[int]) -> dict[int, tuple[np.ndarray, int]]:
    ds = load_dataset("gaunernst/voxceleb2-dev-wds", streaming=True)["train"]
    out: dict[int, tuple[np.ndarray, int]] = {}
    max_idx = max(sample_indices)

    for idx, sample in enumerate(ds):
        if idx > max_idx:
            break
        if idx not in sample_indices:
            continue
        audio = sample["audio"]
        out[idx] = (np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"]))
        if len(out) == len(sample_indices):
            break

    missing = sample_indices.difference(out.keys())
    if missing:
        raise RuntimeError(f"Missing streamed audio for sample indices: {sorted(missing)[:10]}...")
    return out


def run(records_path: str, pairs_path: str, metric: str = "cosine") -> None:
    records = pd.read_csv(records_path)
    pairs = pd.read_csv(pairs_path)

    sample_ids = set(records["sample_idx"].astype(int).tolist())
    audio_map = _load_audio_map(sample_ids)

    feat_map = {}
    for sid in sample_ids:
        x, sr = audio_map[sid]
        feat_map[sid] = extract_mfcc_stats(x, sr)

    y_true = pairs["label"].to_numpy().astype(int)

    if metric == "cosine":
        y_score = np.array(
            [
                cosine_similarity(feat_map[int(a)], feat_map[int(b)])
                for a, b in pairs[["sample_idx_a", "sample_idx_b"]].to_numpy()
            ]
        )
    elif metric == "logistic":
        X = np.array(
            [
                np.abs(feat_map[int(a)] - feat_map[int(b)])
                for a, b in pairs[["sample_idx_a", "sample_idx_b"]].to_numpy()
            ]
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y_true)
        y_score = clf.predict_proba(X)[:, 1]
    else:
        raise ValueError("metric must be one of: cosine, logistic")

    eer, eer_thr = compute_eer(y_true, y_score)
    mindcf, dcf_thr = compute_mindcf(y_true, y_score)

    print(f"metric={metric}")
    print(f"pairs={len(pairs)}")
    print(f"EER={eer:.4f} at threshold={eer_thr:.4f}")
    print(f"minDCF={mindcf:.4f} at threshold={dcf_thr:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline speaker verification.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--metric", choices=["cosine", "logistic"], default="cosine")
    args = parser.parse_args()
    run(args.records, args.pairs, args.metric)


if __name__ == "__main__":
    main()