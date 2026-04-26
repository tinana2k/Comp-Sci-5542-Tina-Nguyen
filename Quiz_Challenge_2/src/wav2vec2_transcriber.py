"""
wav2vec2_transcriber.py — Alternative STT via Facebook wav2vec2
===============================================================
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Uses facebook/wav2vec2-base-960h (fine-tuned on LibriSpeech 960h) as a
secondary speech-to-text model to compare against OpenAI Whisper.

Key characteristics of wav2vec2-base-960h vs Whisper:
  - Outputs ALL CAPS with no punctuation (raw character sequence)
  - Requires 16 kHz mono WAV input (auto-resampled here)
  - Faster inference than Whisper on CPU; less accurate on noisy audio
  - Does NOT support multilingual or translation — English only
  - Open-source Apache-2.0 license
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_ID   = "facebook/wav2vec2-base-960h"
TARGET_SR  = 16_000   # wav2vec2 requires 16 kHz


class Wav2Vec2Transcriber:
    """
    Wraps facebook/wav2vec2-base-960h for automatic speech recognition.

    Parameters
    ----------
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Wav2Vec2] Loading {MODEL_ID} on {self.device} …")
        self._processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self._model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self.device)
        self._model.eval()
        print("[Wav2Vec2] Model ready.")

    def transcribe(self, audio_path: str | pathlib.Path) -> dict:
        """
        Transcribe an audio file with wav2vec2-base-960h.

        Returns a dict with keys:
          model, text (ALL CAPS, no punctuation), word_count, duration_s, wer_note
        """
        audio_path = pathlib.Path(audio_path)
        print(f"[Wav2Vec2] Transcribing: {audio_path.name}")
        t0 = time.perf_counter()

        # ── Load & resample ──────────────────────────────────────────────────
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform  = resampler(waveform)
        if waveform.shape[0] > 1:          # convert stereo → mono
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_np = waveform.squeeze().numpy()

        # ── Inference ────────────────────────────────────────────────────────
        inputs = self._processor(
            audio_np,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self._model(inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self._processor.decode(predicted_ids[0])

        elapsed = time.perf_counter() - t0
        result = {
            "model":       MODEL_ID,
            "text":        text,                    # ALL CAPS, no punctuation
            "word_count":  len(text.split()),
            "duration_s":  round(elapsed, 3),
            "note":        "wav2vec2 output is ALL-CAPS with no punctuation — "
                           "requires post-processing for downstream NLP.",
        }
        print(f"[Wav2Vec2] Done in {elapsed:.1f}s  |  {result['word_count']} words")
        return result


def save_wav2vec2_result(result: dict, output_path: str | pathlib.Path) -> None:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"[Wav2Vec2] Saved → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="wav2vec2-base-960h transcription")
    parser.add_argument("audio", help="Path to audio file (WAV, 16 kHz preferred)")
    parser.add_argument("--output-dir", default="outputs/transcriptions")
    args = parser.parse_args()

    t = Wav2Vec2Transcriber()
    result = t.transcribe(args.audio)

    stem = pathlib.Path(args.audio).stem
    save_wav2vec2_result(result, f"{args.output_dir}/{stem}_wav2vec2.json")

    print("\n── WAV2VEC2 TRANSCRIPT ──────────────────────────────────────────────")
    print(result["text"][:400])
