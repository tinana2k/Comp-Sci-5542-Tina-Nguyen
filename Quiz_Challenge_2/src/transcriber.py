"""
transcriber.py — Speech-to-Text via OpenAI Whisper
===================================================
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Provides two transcription modes:
  - Baseline  : no initial prompt, default Whisper decoding
  - Improved  : domain-aware initial_prompt + beam-search tuning
"""

from __future__ import annotations

import os
import json
import time
import pathlib
from typing import Optional

import whisper
import torch


# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]
DEFAULT_MODEL = "base"

IMPROVED_INITIAL_PROMPT = (
    "This is a lecture or business meeting transcript. "
    "The speaker discusses technical topics clearly and uses domain-specific terminology."
)


# ── TranscriptionResult dataclass ─────────────────────────────────────────────
class TranscriptionResult:
    """Container returned by both baseline and improved transcription calls."""

    def __init__(
        self,
        text: str,
        segments: list[dict],
        language: str,
        duration_s: float,
        model_size: str,
        strategy: str,
    ):
        self.text = text
        self.segments = segments
        self.language = language
        self.duration_s = duration_s
        self.model_size = model_size
        self.strategy = strategy

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "model_size": self.model_size,
            "language": self.language,
            "duration_s": round(self.duration_s, 3),
            "word_count": len(self.text.split()),
            "text": self.text,
            "segments": self.segments,
        }


# ── Transcriber ───────────────────────────────────────────────────────────────
class Transcriber:
    """
    Wraps OpenAI Whisper and exposes baseline / improved transcription methods.

    Parameters
    ----------
    model_size : str
        Whisper model variant (tiny | base | small | medium | large).
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        if model_size not in SUPPORTED_MODELS:
            raise ValueError(f"model_size must be one of {SUPPORTED_MODELS}")

        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Transcriber] Loading Whisper '{model_size}' on {self.device} …")
        self._model = whisper.load_model(model_size, device=self.device)
        print("[Transcriber] Model ready.")

    # ── public API ────────────────────────────────────────────────────────────

    def transcribe_baseline(self, audio_path: str | pathlib.Path) -> TranscriptionResult:
        """
        Baseline transcription: no initial prompt, Whisper defaults.
        Equivalent to running `whisper audio.mp3` from the CLI.
        """
        audio_path = str(audio_path)
        print(f"[Transcriber] Baseline transcription: {audio_path}")
        t0 = time.perf_counter()

        result = self._model.transcribe(
            audio_path,
            verbose=False,
        )

        elapsed = time.perf_counter() - t0
        return TranscriptionResult(
            text=result["text"].strip(),
            segments=result.get("segments", []),
            language=result.get("language", "unknown"),
            duration_s=elapsed,
            model_size=self.model_size,
            strategy="baseline",
        )

    def transcribe_improved(self, audio_path: str | pathlib.Path) -> TranscriptionResult:
        """
        Improved transcription:
          - domain-aware initial_prompt
          - beam_size=5 for better hypothesis search
          - temperature=0 for deterministic decoding
          - condition_on_previous_text=True for long-form coherence
        """
        audio_path = str(audio_path)
        print(f"[Transcriber] Improved transcription: {audio_path}")
        t0 = time.perf_counter()

        result = self._model.transcribe(
            audio_path,
            initial_prompt=IMPROVED_INITIAL_PROMPT,
            beam_size=5,
            temperature=0,
            condition_on_previous_text=True,
            verbose=False,
        )

        elapsed = time.perf_counter() - t0
        return TranscriptionResult(
            text=result["text"].strip(),
            segments=result.get("segments", []),
            language=result.get("language", "unknown"),
            duration_s=elapsed,
            model_size=self.model_size,
            strategy="improved",
        )

    def transcribe_both(
        self, audio_path: str | pathlib.Path
    ) -> tuple[TranscriptionResult, TranscriptionResult]:
        """Convenience wrapper: returns (baseline_result, improved_result)."""
        return self.transcribe_baseline(audio_path), self.transcribe_improved(audio_path)


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_transcription(result: TranscriptionResult, output_path: str | pathlib.Path) -> None:
    """Persist a TranscriptionResult to a JSON file."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    print(f"[Transcriber] Saved → {output_path}")


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper Speech-to-Text (baseline & improved)")
    parser.add_argument("audio", help="Path to audio file (MP3 / WAV / FLAC …)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=SUPPORTED_MODELS)
    parser.add_argument("--output-dir", default="outputs/transcriptions")
    args = parser.parse_args()

    transcriber = Transcriber(model_size=args.model)
    audio_name = pathlib.Path(args.audio).stem

    baseline, improved = transcriber.transcribe_both(args.audio)

    save_transcription(baseline, f"{args.output_dir}/{audio_name}_baseline.json")
    save_transcription(improved, f"{args.output_dir}/{audio_name}_improved.json")

    print("\n── BASELINE ──────────────────────────────────────────────────────────")
    print(baseline.text[:500])
    print("\n── IMPROVED ──────────────────────────────────────────────────────────")
    print(improved.text[:500])
