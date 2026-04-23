"""
translator.py — Multilingual Translation using Helsinki-NLP MarianMT
=====================================================================
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Translates English text (transcripts, summaries, action items) into:
  - Spanish  (ES)  — Helsinki-NLP/opus-mt-en-es
  - French   (FR)  — Helsinki-NLP/opus-mt-en-fr
  - Vietnamese (VI) — Helsinki-NLP/opus-mt-en-vi
  - Chinese  (ZH)  — Helsinki-NLP/opus-mt-en-zh

Additionally documents Whisper's built-in translation capability:
  - Whisper can translate non-English audio → English text directly
  - task="translate" in whisper.transcribe()

Use Cases:
  - International lecture distribution
  - Multilingual meeting summaries
  - Customer call localization
  - Medical dictation for non-English speakers
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

from transformers import MarianMTModel, MarianTokenizer


# ── Supported language pairs ───────────────────────────────────────────────────
LANGUAGE_CONFIG = {
    "es": {
        "name": "Spanish",
        "model_id": "Helsinki-NLP/opus-mt-en-es",
        "flag": "ES",
    },
    "fr": {
        "name": "French",
        "model_id": "Helsinki-NLP/opus-mt-en-fr",
        "flag": "FR",
    },
    "vi": {
        "name": "Vietnamese",
        "model_id": "Helsinki-NLP/opus-mt-en-vi",
        "flag": "VI",
    },
    "zh": {
        "name": "Chinese (Simplified)",
        "model_id": "Helsinki-NLP/opus-mt-en-zh",
        "flag": "ZH",
    },
}


class Translator:
    """
    Multilingual EN → {ES, FR, VI, ZH} translator using MarianMT.

    Parameters
    ----------
    target_langs : list[str]
        Language codes to load. Default: ['es', 'fr', 'vi']
    device : int
        -1 for CPU; 0+ for CUDA GPU index.
    """

    def __init__(
        self,
        target_langs: Optional[list[str]] = None,
        device: int = -1,
    ):
        self.target_langs = target_langs or ["es", "fr", "vi"]
        self.device = device
        self._models: dict = {}
        self._tokenizers: dict = {}

        for lang in self.target_langs:
            if lang not in LANGUAGE_CONFIG:
                raise ValueError(f"Unsupported lang '{lang}'. Choose from: {list(LANGUAGE_CONFIG)}")
            cfg = LANGUAGE_CONFIG[lang]
            print(f"[Translator] Loading {cfg['model_id']} ({cfg['name']}) …")
            self._tokenizers[lang] = MarianTokenizer.from_pretrained(cfg["model_id"])
            self._models[lang]     = MarianMTModel.from_pretrained(cfg["model_id"])
            if device >= 0:
                self._models[lang] = self._models[lang].cuda(device)
        print("[Translator] Ready.")

    def translate(self, text: str, target_lang: str, max_chunk_words: int = 100) -> dict:
        """
        Translate English text to a target language.

        Handles long texts by chunking at sentence boundaries.

        Parameters
        ----------
        text : str
            English input text.
        target_lang : str
            One of 'es', 'fr', 'vi', 'zh'.
        max_chunk_words : int
            Maximum words per translation chunk (avoids token-length overflow).

        Returns
        -------
        dict with keys: target_lang, language_name, model_id, original_text,
                        translated_text, word_count_original, word_count_translated,
                        duration_s
        """
        if target_lang not in self._models:
            raise ValueError(f"'{target_lang}' was not loaded. Re-init Translator with this lang.")

        cfg       = LANGUAGE_CONFIG[target_lang]
        tokenizer = self._tokenizers[target_lang]
        model     = self._models[target_lang]

        t0 = time.perf_counter()

        # ── Chunk text ────────────────────────────────────────────────────────
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        chunks, current_chunk, current_words = [], [], 0
        for sent in sentences:
            word_count = len(sent.split())
            if current_words + word_count > max_chunk_words and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk, current_words = [sent], word_count
            else:
                current_chunk.append(sent)
                current_words += word_count
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        # ── Translate each chunk ──────────────────────────────────────────────
        translated_chunks = []
        for chunk in chunks:
            inputs = tokenizer(
                [chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            if self.device >= 0:
                inputs = {k: v.cuda(self.device) for k, v in inputs.items()}

            translated_ids = model.generate(**inputs, num_beams=4, early_stopping=True)
            translated_chunk = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
            translated_chunks.append(translated_chunk)

        translated_text = " ".join(translated_chunks)
        elapsed = time.perf_counter() - t0

        return {
            "target_lang":          target_lang,
            "language_name":        cfg["name"],
            "model_id":             cfg["model_id"],
            "original_text":        text,
            "translated_text":      translated_text,
            "word_count_original":  len(text.split()),
            "word_count_translated": len(translated_text.split()),
            "duration_s":           round(elapsed, 3),
        }

    def translate_all(self, text: str) -> dict[str, dict]:
        """
        Translate text into all loaded target languages.

        Returns dict: {lang_code: translation_result}
        """
        return {lang: self.translate(text, lang) for lang in self.target_langs}


# ── Persistence ────────────────────────────────────────────────────────────────

def save_translation(result: dict, output_path: str | pathlib.Path) -> None:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"[Translator] Saved → {output_path}")


# ── Note on Whisper's built-in translation ─────────────────────────────────────
WHISPER_TRANSLATION_NOTE = """
Whisper also supports built-in translation:
    result = model.transcribe(audio_path, task="translate")
    # Translates non-English speech → English text in one step

This is complementary to MarianMT:
  - Whisper:   foreign audio → English text  (speech translation)
  - MarianMT:  English text  → foreign text  (text translation)
"""


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate English text to multiple languages")
    parser.add_argument("input", help="Path to .txt or .json file with English text")
    parser.add_argument("--langs", default="es,fr,vi", help="Comma-separated target lang codes")
    parser.add_argument("--output-dir", default="outputs/translations")
    args = parser.parse_args()

    inp_path = pathlib.Path(args.input)
    if inp_path.suffix == ".json":
        data = json.loads(inp_path.read_text(encoding="utf-8"))
        # Try to get summary text; fall back to full transcript
        text = data.get("summary") or data.get("text", "")
    else:
        text = inp_path.read_text(encoding="utf-8")

    langs = [l.strip() for l in args.langs.split(",")]
    translator = Translator(target_langs=langs)

    for lang in langs:
        result = translator.translate(text, lang)
        stem   = inp_path.stem
        save_translation(result, f"{args.output_dir}/{stem}_{lang}.json")
        cfg = LANGUAGE_CONFIG[lang]
        print(f"\n── {cfg['name'].upper()} ({lang.upper()}) ──────────────────────")
        print(result["translated_text"][:300])
