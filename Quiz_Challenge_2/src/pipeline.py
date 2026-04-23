"""
pipeline.py — End-to-End Speech Intelligence Pipeline
======================================================
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Full pipeline covering all assignment requirements:
  1. Whisper STT          — Baseline + Improved transcription
  2. wav2vec2 STT         — Alternative model comparison
  3. NLP Analysis         — Summary / Action Items / Sentiment / Keywords
  4. MarianMT Translation — EN → ES / FR / VI
  5. SpeechT5 TTS         — Summary audio synthesis (closes the loop)
  6. Evaluation           — WER / ROUGE / BLEU / Coverage comparison

Use Case Modes:
  --use-case lecture    (default) — Smart lecture assistant
  --use-case meeting              — Meeting summarizer
  --use-case call                 — Customer call analyzer
  --use-case medical              — Medical dictation assistant

Usage
-----
    python src/pipeline.py --audio data/sample_lecture.mp3
    python src/pipeline.py --audio data/sample_lecture.mp3 --model small --translate es fr vi
    python src/pipeline.py --audio data/meeting.mp3 --use-case meeting --tts
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import torch

# ── Add src/ to path so relative imports work ─────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from transcriber import Transcriber, save_transcription
from analyzer    import Analyzer, save_analysis
from evaluator   import Evaluator


# ── Use-case personas ──────────────────────────────────────────────────────────
USE_CASE_PROMPTS = {
    "lecture": (
        "This is a university lecture on computer science or artificial intelligence. "
        "The speaker uses technical terminology related to machine learning and NLP."
    ),
    "meeting": (
        "This is a business meeting or team standup. "
        "Participants discuss project status, deadlines, and action items."
    ),
    "call": (
        "This is a customer service or sales call. "
        "The conversation involves product questions, complaints, and resolutions."
    ),
    "medical": (
        "This is a medical dictation by a physician. "
        "The speaker uses clinical terminology including drug names, diagnoses, and procedures."
    ),
}

USE_CASE_LABELS = {
    "lecture":  "Smart Lecture Assistant",
    "meeting":  "Meeting Summarizer",
    "call":     "Customer Call Analyzer",
    "medical":  "Medical Dictation Assistant",
}

DEFAULT_OUTPUT = pathlib.Path("outputs")


def run_pipeline(
    audio_path: str,
    model_size: str      = "base",
    device: str | None   = None,
    use_case: str        = "lecture",
    translate_langs: list[str] | None = None,
    run_tts: bool        = False,
    run_wav2vec2: bool   = True,
    output_root: pathlib.Path = DEFAULT_OUTPUT,
) -> dict:
    """
    Execute the full 5-stage Speech Intelligence Pipeline.

    Returns the complete evaluation report dict.
    """
    audio_path  = pathlib.Path(audio_path)
    audio_stem  = audio_path.stem
    device_str  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_case    = use_case if use_case in USE_CASE_PROMPTS else "lecture"
    t_langs     = translate_langs or []

    print(f"\n{'='*65}")
    print(f"  {USE_CASE_LABELS[use_case].upper()}")
    print(f"  Audio   : {audio_path.name}")
    print(f"  Whisper : {model_size}  |  Device: {device_str}")
    print(f"  Langs   : {t_langs or '—'}")
    print(f"{'='*65}\n")

    # ── Stage 1: Whisper Transcription ────────────────────────────────────────
    print("[Stage 1/5] Whisper STT — baseline & improved …")
    transcriber = Transcriber(model_size=model_size, device=device)
    transcriber._model.transcribe.__func__  # confirm loaded
    domain_prompt = USE_CASE_PROMPTS[use_case]

    from transcriber import IMPROVED_INITIAL_PROMPT
    # Temporarily override improved prompt with use-case-specific one
    import transcriber as _tm
    _tm.IMPROVED_INITIAL_PROMPT = domain_prompt

    baseline_t, improved_t = transcriber.transcribe_both(audio_path)

    t_dir = output_root / "transcriptions"
    save_transcription(baseline_t, t_dir / f"{audio_stem}_baseline.json")
    save_transcription(improved_t, t_dir / f"{audio_stem}_improved.json")

    # ── Stage 2: wav2vec2 Transcription (optional) ────────────────────────────
    wav2vec2_result = None
    if run_wav2vec2:
        print("\n[Stage 2/5] wav2vec2 STT — model comparison …")
        try:
            from wav2vec2_transcriber import Wav2Vec2Transcriber, save_wav2vec2_result
            w2v2 = Wav2Vec2Transcriber(device=device_str)
            wav2vec2_result = w2v2.transcribe(audio_path)
            save_wav2vec2_result(wav2vec2_result, t_dir / f"{audio_stem}_wav2vec2.json")
        except Exception as e:
            print(f"  [wav2vec2] Skipped — {e}")
    else:
        print("\n[Stage 2/5] wav2vec2 STT — skipped (--no-wav2vec2)")

    # ── Stage 3: NLP Analysis ─────────────────────────────────────────────────
    print("\n[Stage 3/5] NLP Analysis (Summary / Actions / Sentiment / Keywords) …")
    device_id = 0 if (device_str == "cuda" and torch.cuda.is_available()) else -1
    analyzer  = Analyzer(device=device_id)
    baseline_a, improved_a = analyzer.analyze_both(improved_t.text)

    a_dir = output_root / "analysis"
    save_analysis(baseline_a, a_dir / f"{audio_stem}_baseline.json")
    save_analysis(improved_a, a_dir / f"{audio_stem}_improved.json")

    # ── Stage 4: Translation ─────────────────────────────────────────────────
    translations = {}
    if t_langs:
        print(f"\n[Stage 4/5] MarianMT Translation → {t_langs} …")
        try:
            from translator import Translator, save_translation
            translator = Translator(target_langs=t_langs, device=device_id)
            # Translate the improved summary (most useful artifact)
            summary_text = improved_a.summary
            tr_dir = output_root / "translations"
            for lang in t_langs:
                result = translator.translate(summary_text, lang)
                save_translation(result, tr_dir / f"{audio_stem}_{lang}.json")
                translations[lang] = result
        except Exception as e:
            print(f"  [Translation] Skipped — {e}")
    else:
        print("\n[Stage 4/5] Translation — skipped (pass --translate es fr vi)")

    # ── Stage 5: SpeechT5 TTS ────────────────────────────────────────────────
    tts_result = None
    if run_tts:
        print("\n[Stage 5/5] SpeechT5 TTS — synthesizing summary audio …")
        try:
            from tts_synthesizer import SpeechSynthesizer
            synth     = SpeechSynthesizer(device=device_str)
            tts_out   = output_root / "tts" / f"{audio_stem}_summary.wav"
            tts_result = synth.synthesize(improved_a.summary, tts_out)
        except Exception as e:
            print(f"  [TTS] Skipped — {e}")
    else:
        print("\n[Stage 5/5] TTS — skipped (pass --tts to enable)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[Eval] Computing metrics …")
    evaluator = Evaluator()
    report = evaluator.evaluate(
        baseline_transcription=baseline_t.text,
        improved_transcription=improved_t.text,
        baseline_analysis=baseline_a.to_dict(),
        improved_analysis=improved_a.to_dict(),
    )

    if translations:
        report["translations"] = {
            lang: {
                "language":   res["language_name"],
                "model":      res["model_id"],
                "word_count": res["word_count_translated"],
                "preview":    res["translated_text"][:200],
            }
            for lang, res in translations.items()
        }

    if tts_result:
        report["tts"] = {
            "model":      tts_result["model"],
            "output_wav": tts_result["output_path"],
            "duration_s": tts_result["duration_s"],
        }

    e_dir = output_root / "evaluation"
    evaluator.save_report(report, e_dir / f"{audio_stem}_report.json")
    evaluator.print_table(report)

    # ── Pretty-print results ──────────────────────────────────────────────────
    _print_results(improved_t, improved_a, translations)
    return report


def _print_results(improved_t, improved_a, translations):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  IMPROVED TRANSCRIPT (first 400 chars)")
    print(sep)
    print(improved_t.text[:400], "…")

    print(f"\n{sep}")
    print("  ABSTRACTIVE SUMMARY")
    print(sep)
    print(improved_a.summary)

    print(f"\n{sep}")
    print("  ACTION ITEMS")
    print(sep)
    if improved_a.action_items:
        for i, item in enumerate(improved_a.action_items, 1):
            print(f"  {i}. {item}")
    else:
        print("  (none detected)")

    print(f"\n{sep}")
    print("  SENTIMENT")
    print(sep)
    print(f"  {improved_a.sentiment}")

    print(f"\n{sep}")
    print("  TOP KEYWORDS")
    print(sep)
    for kw, score in improved_a.keywords[:6]:
        print(f"  {kw:<35} {score:.4f}")

    if translations:
        print(f"\n{sep}")
        print("  TRANSLATIONS (Summary preview)")
        print(sep)
        for lang, result in translations.items():
            print(f"\n  [{result['language_name']}]")
            print(f"  {result['translated_text'][:200]}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Speech Intelligence Pipeline")
    parser.add_argument("--audio",      default="data/sample_lecture.mp3",
                        help="Path to input audio file")
    parser.add_argument("--model",      default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device",     default=None, choices=["cpu", "cuda"])
    parser.add_argument("--use-case",   default="lecture",
                        choices=["lecture", "meeting", "call", "medical"])
    parser.add_argument("--translate",  nargs="*", default=[],
                        help="Target language codes: es fr vi zh")
    parser.add_argument("--tts",        action="store_true",
                        help="Enable SpeechT5 TTS synthesis of summary")
    parser.add_argument("--no-wav2vec2", action="store_true",
                        help="Skip wav2vec2 transcription stage")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    run_pipeline(
        audio_path      = args.audio,
        model_size      = args.model,
        device          = args.device,
        use_case        = args.use_case,
        translate_langs = args.translate,
        run_tts         = args.tts,
        run_wav2vec2    = not args.no_wav2vec2,
        output_root     = pathlib.Path(args.output_root),
    )
