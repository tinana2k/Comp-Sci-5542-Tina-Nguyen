"""
tts_synthesizer.py — Text-to-Speech via Microsoft SpeechT5
===========================================================
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Converts text (summaries, action items, translated content) back to speech
using microsoft/speecht5_tts — a transformer-based TTS model.

Pipeline:
  Text → SpeechT5 → WAV audio file

This closes the full speech intelligence loop:
  Audio IN → Whisper STT → NLP Analysis → MarianMT Translation → SpeechT5 TTS → Audio OUT

Use Cases:
  - Read summaries aloud to students after a lecture
  - Generate audio action-item reminders
  - Synthesize translated meeting summaries
  - Accessibility: audio output for visual-impairment users
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


MODEL_ID     = "microsoft/speecht5_tts"
VOCODER_ID   = "microsoft/speecht5_hifigan"
EMBEDDINGS_DS = "Matthijs/cmu-arctic-xvectors"  # speaker embeddings
DEFAULT_SR   = 16_000


class SpeechSynthesizer:
    """
    Text-to-Speech using microsoft/speecht5_tts + HiFi-GAN vocoder.

    Parameters
    ----------
    speaker_id : int
        Index into the CMU-Arctic xvectors dataset (0–7004).
        Each index represents a different synthetic speaker voice.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        speaker_id: int = 7306,    # default: neutral female voice
        device: Optional[str] = None,
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker_id = speaker_id

        print(f"[SpeechT5] Loading {MODEL_ID} on {self.device} …")
        self._processor = SpeechT5Processor.from_pretrained(MODEL_ID)
        self._model     = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID).to(self.device)
        self._vocoder   = SpeechT5HifiGan.from_pretrained(VOCODER_ID).to(self.device)

        print(f"[SpeechT5] Loading speaker embeddings (speaker_id={speaker_id}) …")
        embeddings_ds = load_dataset(EMBEDDINGS_DS, split="validation")
        self._speaker_embeddings = (
            torch.tensor(embeddings_ds[speaker_id]["xvector"])
            .unsqueeze(0)
            .to(self.device)
        )
        print("[SpeechT5] Ready.")

    def synthesize(
        self,
        text: str,
        output_path: str | pathlib.Path,
        max_chars: int = 600,
    ) -> dict:
        """
        Synthesize speech from text and save as WAV.

        Parameters
        ----------
        text : str
            Input text (keep under ~600 chars for best quality).
        output_path : str | pathlib.Path
            Where to save the output WAV file.
        max_chars : int
            Hard truncation limit (SpeechT5 token limit ~600 chars).

        Returns
        -------
        dict with keys: text, output_path, sample_rate, num_samples, duration_s
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Hard-truncate if needed
        text_in = text[:max_chars].strip()
        print(f"[SpeechT5] Synthesizing {len(text_in)} chars → {output_path.name}")

        t0 = time.perf_counter()
        inputs = self._processor(text=text_in, return_tensors="pt").to(self.device)

        with torch.no_grad():
            speech = self._model.generate_speech(
                inputs["input_ids"],
                self._speaker_embeddings,
                vocoder=self._vocoder,
            )

        audio_np = speech.cpu().numpy()
        sf.write(str(output_path), audio_np, samplerate=DEFAULT_SR)
        elapsed = time.perf_counter() - t0

        result = {
            "model":       MODEL_ID,
            "text":        text_in,
            "output_path": str(output_path),
            "sample_rate": DEFAULT_SR,
            "num_samples": len(audio_np),
            "duration_s":  round(len(audio_np) / DEFAULT_SR, 2),
            "inference_s": round(elapsed, 3),
            "speaker_id":  self.speaker_id,
        }
        print(f"[SpeechT5] Generated {result['duration_s']:.1f}s audio in {elapsed:.1f}s")
        return result


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpeechT5 Text-to-Speech synthesis")
    parser.add_argument("text_input", help="Text string or path to .txt / .json (summary field)")
    parser.add_argument("--output", default="outputs/tts/summary_speech.wav")
    parser.add_argument("--speaker-id", type=int, default=7306)
    args = parser.parse_args()

    inp = pathlib.Path(args.text_input)
    if inp.exists():
        if inp.suffix == ".json":
            data = json.loads(inp.read_text(encoding="utf-8"))
            text = data.get("summary") or data.get("text", "")
        else:
            text = inp.read_text(encoding="utf-8")
    else:
        text = args.text_input  # treat as literal string

    synth  = SpeechSynthesizer(speaker_id=args.speaker_id)
    result = synth.synthesize(text, args.output)

    meta_path = pathlib.Path(args.output).with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SpeechT5] Metadata saved → {meta_path}")
