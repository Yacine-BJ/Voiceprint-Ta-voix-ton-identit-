from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from src.features import cosine_similarity, extract_mfcc_stats


DB_PATH = Path("data/processed/enrolled_voiceprints.npz")


def _load_db() -> dict[str, np.ndarray]:
    if not DB_PATH.exists():
        return {}
    data = np.load(DB_PATH, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _save_db(db: dict[str, np.ndarray]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DB_PATH, **db)


def _embed_from_audio(audio_tuple: tuple[int, np.ndarray]) -> np.ndarray:
    sr, audio = audio_tuple
    if audio is None:
        raise ValueError("No audio provided")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    return extract_mfcc_stats(audio, sr)


def enroll(speaker_name: str, audio: tuple[int, np.ndarray]) -> str:
    if not speaker_name.strip():
        return "Please provide a speaker name."
    emb = _embed_from_audio(audio)
    db = _load_db()
    db[speaker_name.strip()] = emb
    _save_db(db)
    return f"Speaker '{speaker_name}' enrolled."


def verify(speaker_name: str, audio: tuple[int, np.ndarray], threshold: float) -> str:
    db = _load_db()
    name = speaker_name.strip()
    if name not in db:
        return f"Speaker '{name}' not found in enrolled voiceprints."
    emb_new = _embed_from_audio(audio)
    score = cosine_similarity(db[name], emb_new)
    decision = "MATCH" if score >= threshold else "NO MATCH"
    return f"Score={score:.4f} | Threshold={threshold:.4f} | Decision={decision}"


def delete_speaker(speaker_name: str) -> str:
    db = _load_db()
    name = speaker_name.strip()
    if name not in db:
        return f"Speaker '{name}' not found."
    del db[name]
    _save_db(db)
    return f"Speaker '{name}' deleted."


with gr.Blocks(title="Voiceprint Demo") as demo:
    gr.Markdown("## Voiceprint Demo (Baseline)\nEnroll, verify, and delete a voiceprint.")

    name = gr.Textbox(label="Speaker name")
    audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio clip")
    threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Decision threshold")

    out = gr.Textbox(label="Result")

    with gr.Row():
        btn_enroll = gr.Button("Enroll")
        btn_verify = gr.Button("Verify")
        btn_delete = gr.Button("Delete")

    btn_enroll.click(enroll, inputs=[name, audio], outputs=[out])
    btn_verify.click(verify, inputs=[name, audio, threshold], outputs=[out])
    btn_delete.click(delete_speaker, inputs=[name], outputs=[out])


if __name__ == "__main__":
    demo.launch()