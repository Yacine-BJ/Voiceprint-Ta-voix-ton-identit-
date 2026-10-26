from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


@dataclass
class AudioRecord:
    sample_idx: int
    speaker_id: str
    duration_s: float


def _extract_speaker_id(sample: dict) -> str | None:
    if "json" in sample and isinstance(sample["json"], dict):
        for key in ("speaker_id", "speaker", "spk_id", "id"):
            if key in sample["json"]:
                return str(sample["json"][key])
    for key in ("speaker_id", "speaker", "spk_id"):
        if key in sample:
            return str(sample[key])
    return None


def _extract_duration(sample: dict) -> float:
    audio = sample.get("audio")
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        arr = np.asarray(audio["array"])
        sr = float(audio["sampling_rate"])
        if sr > 0:
            return float(len(arr) / sr)
    return 0.0


def collect_streamed_records(max_samples: int, min_speaker_samples: int) -> pd.DataFrame:
    dataset = load_dataset("gaunernst/voxceleb2-dev-wds", streaming=True)
    records: list[AudioRecord] = []

    for idx, sample in enumerate(dataset["train"]):
        if idx >= max_samples:
            break
        speaker_id = _extract_speaker_id(sample)
        if not speaker_id:
            continue
        records.append(
            AudioRecord(sample_idx=idx, speaker_id=speaker_id, duration_s=_extract_duration(sample))
        )

    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        raise RuntimeError("No valid speaker IDs were found in streamed samples.")

    counts = df["speaker_id"].value_counts()
    keep_speakers = counts[counts >= min_speaker_samples].index
    df = df[df["speaker_id"].isin(keep_speakers)].reset_index(drop=True)
    return df


def build_pairs(df_records: pd.DataFrame, max_pairs: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = []

    by_speaker = {
        spk: group["sample_idx"].tolist()
        for spk, group in df_records.groupby("speaker_id", sort=False)
    }
    speakers = list(by_speaker.keys())

    # Positive pairs
    for spk, sample_ids in by_speaker.items():
        if len(sample_ids) < 2:
            continue
        local = []
        for i in range(len(sample_ids)):
            for j in range(i + 1, len(sample_ids)):
                local.append((sample_ids[i], sample_ids[j], 1, spk, spk))
        rng.shuffle(local)
        pairs.extend(local[: max(1, max_pairs // (2 * max(1, len(speakers))))])

    # Negative pairs
    target_neg = max_pairs - len(pairs)
    for _ in range(max(0, target_neg)):
        s1, s2 = rng.choice(speakers, size=2, replace=False)
        a = int(rng.choice(by_speaker[s1]))
        b = int(rng.choice(by_speaker[s2]))
        pairs.append((a, b, 0, s1, s2))

    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]

    return pd.DataFrame(
        pairs,
        columns=["sample_idx_a", "sample_idx_b", "label", "speaker_a", "speaker_b"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small streamed record and pair set.")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--min-speaker-samples", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_streamed_records(args.max_samples, args.min_speaker_samples)
    pairs = build_pairs(records, max_pairs=args.max_pairs, seed=args.seed)

    records.to_csv(args.output_dir / "records.csv", index=False)
    pairs.to_csv(args.output_dir / "pairs.csv", index=False)

    print(f"Saved {len(records)} records to {args.output_dir / 'records.csv'}")
    print(f"Saved {len(pairs)} pairs to {args.output_dir / 'pairs.csv'}")
    print("Label distribution:")
    print(pairs["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()