# Voiceprint — Your Voice, Your Identity

Starter repository for speaker verification ("same speaker or not") using:
- **Streaming subset** of VoxCeleb2 from Hugging Face (`gaunernst/voxceleb2-dev-wds`)
- **Explainable baseline** (MFCC + cosine similarity / logistic regression)
- **Core metrics** (EER, minDCF, ROC)
- **Demo app skeleton** (enroll → verify → delete voiceprint)

---

## 1) Project scope (3-week plan)

### Week 1 — Data + classical baseline
- Stream a limited sample from VoxCeleb2 (no full download).
- Build positive and negative pairs from speaker IDs.
- Extract MFCC/acoustic features.
- Train baseline verifier and compute EER.

### Week 2 — Modern embeddings
- Add a pre-trained embedding backend (ECAPA-TDNN via SpeechBrain or similar).
- Compare baseline vs pre-trained embeddings on identical pair splits.
- Add a lightweight metric-learning experiment (contrastive/triplet).

### Week 3 — Robustness + demo
- Evaluate impact of duration/noise/mic conditions.
- Calibrate threshold and report failure cases.
- Finalize interactive app and reproducibility documentation.

---

## 2) Ethical and legal checklist

- Document dataset source, license, and usage terms.
- Keep train/validation/test speaker-disjoint to avoid leakage.
- Treat voiceprints as sensitive biometric data.
- Provide opt-in enrollment and deletion in the demo.
- Do **not** redistribute audio unless license permits it.

---

## 3) Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build a small streamed sample and pairs

```bash
python -m src.preprocessing \
  --max-samples 400 \
  --min-speaker-samples 2 \
  --max-pairs 1500 \
  --output-dir data/processed
```

Outputs:
- `data/processed/records.csv` (sample index, speaker id, duration)
- `data/processed/pairs.csv` (pair list + label)

### Download a few real clips for app testing

```bash
python -m src.download_samples \
  --max-clips 20 \
  --max-scan-samples 50000 \
  --output-dir data/raw/test_clips
```

Outputs:
- WAV files in `data/raw/test_clips/`
- `data/raw/test_clips/downloaded_samples.csv` (clip path + speaker metadata)

### Train and evaluate baseline

```bash
python -m src.verification \
  --records data/processed/records.csv \
  --pairs data/processed/pairs.csv \
  --metric cosine
```

Optional logistic model:

```bash
python -m src.verification \
  --records data/processed/records.csv \
  --pairs data/processed/pairs.csv \
  --metric logistic
```

### Launch demo app (baseline prototype)

```bash
python app.py
```

---

## 4) Repository structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
└── src/
    ├── preprocessing.py   # streaming sample + pair generation
    ├── features.py        # MFCC/stat feature extraction
    ├── verification.py    # baseline training + evaluation run
    ├── evaluation.py      # EER/minDCF/ROC utilities
    └── embeddings.py      # placeholder for modern embeddings
```

---

## 5) What to deliver in final report

- Dataset provenance + licenses + composition stats.
- Pair protocol and leakage prevention strategy.
- Comparison table: classical baseline vs neural baseline vs pre-trained embeddings.
- Ablation studies (duration/noise/normalization/thresholding).
- Validation metrics: EER, minDCF, ROC (+ failure case analysis).
- Demo flow: enroll, verify, delete.