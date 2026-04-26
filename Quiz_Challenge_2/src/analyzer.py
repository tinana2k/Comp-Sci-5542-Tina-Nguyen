"""
analyzer.py — NLP Intelligence Layer
=====================================
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Given a raw transcript text this module produces:
  1. Abstractive Summary           (facebook/bart-large-cnn)
  2. Action Item Extraction        (rule-based + keyword detection)
  3. Sentiment Analysis            (distilbert-base-uncased-finetuned-sst-2-english)
  4. Keyword Extraction            (KeyBERT — BERT-based keyphrase scoring)

Baseline vs. Improved comparison:
  - Baseline  : single short summary, simple word-frequency keywords
  - Improved  : structured summary + action items + granular sentiment + KeyBERT
"""

from __future__ import annotations

import re
import json
import pathlib
from collections import Counter
from typing import Optional

import nltk
from transformers import pipeline
from keybert import KeyBERT


# ── NLTK bootstrap ────────────────────────────────────────────────────────────
for _res in ("punkt", "stopwords", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(f"tokenizers/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

from nltk.corpus import stopwords  # noqa: E402
from nltk.tokenize import sent_tokenize, word_tokenize  # noqa: E402

_STOPWORDS = set(stopwords.words("english"))

# ── Action-item trigger patterns ──────────────────────────────────────────────
_ACTION_PATTERNS = re.compile(
    r"\b(we (need|should|must|will|have to)|"
    r"(todo|to-do|action item)[:\s]|"
    r"(follow[- ]up|follow up)|"
    r"(make sure|ensure|please|don't forget|remember to)|"
    r"(assign|schedule|send|review|check|confirm|update|prepare|complete|submit))\b",
    re.IGNORECASE,
)


# ── AnalysisResult ─────────────────────────────────────────────────────────────
class AnalysisResult:
    """Holds all NLP outputs for one transcript × strategy pair."""

    def __init__(
        self,
        strategy: str,
        summary: str,
        action_items: list[str],
        sentiment: dict,
        keywords: list[tuple[str, float]],
    ):
        self.strategy = strategy
        self.summary = summary
        self.action_items = action_items
        self.sentiment = sentiment
        self.keywords = keywords

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "summary": self.summary,
            "action_items": self.action_items,
            "sentiment": self.sentiment,
            "keywords": [
                {"phrase": kw, "score": round(score, 4)}
                for kw, score in self.keywords
            ],
        }


# ── Analyzer ──────────────────────────────────────────────────────────────────
class Analyzer:
    """
    Orchestrates all NLP tasks.

    Parameters
    ----------
    device : int
        -1 for CPU; 0+ for GPU index.
    """

    def __init__(self, device: int = -1):
        self.device = device
        print("[Analyzer] Loading summarisation model (BART-large-CNN) …")
        self._summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device,
        )
        print("[Analyzer] Loading sentiment model (DistilBERT-SST-2) …")
        self._sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )
        print("[Analyzer] Loading KeyBERT …")
        self._kw_model = KeyBERT()
        print("[Analyzer] Ready.")

    # ── public API ────────────────────────────────────────────────────────────

    def analyze_baseline(self, text: str) -> AnalysisResult:
        """
        Baseline analysis:
          - Summary  : first 3 sentences (extractive, no model)
          - Actions  : none extracted
          - Sentiment: single label for full text (truncated)
          - Keywords : top-5 TF-IDF word frequencies
        """
        sentences = sent_tokenize(text)
        summary = " ".join(sentences[:3])

        sentiment = self._get_sentiment_safe(text)

        words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in _STOPWORDS]
        freq = Counter(words).most_common(5)
        keywords = [(w, round(c / len(words), 4)) for w, c in freq]

        return AnalysisResult(
            strategy="baseline",
            summary=summary,
            action_items=[],
            sentiment=sentiment,
            keywords=keywords,
        )

    def analyze_improved(self, text: str) -> AnalysisResult:
        """
        Improved analysis:
          - Summary  : abstractive (BART-large-CNN), max 180 tokens
          - Actions  : sentence-level regex pattern extraction
          - Sentiment: per-sentence rolling + aggregate label
          - Keywords : top-10 KeyBERT keyphrases (MMR diversity)
        """
        # 1. Abstractive summary (BART handles up to 1024 tokens internally)
        safe_text = text[:3000]  # guard against very long transcripts
        try:
            summary_out = self._summarizer(
                safe_text,
                max_length=180,
                min_length=40,
                do_sample=False,
            )
            summary = summary_out[0]["summary_text"]
        except Exception:
            sentences = sent_tokenize(text)
            summary = " ".join(sentences[:5])

        # 2. Action items
        sentences = sent_tokenize(text)
        action_items = [
            s.strip()
            for s in sentences
            if _ACTION_PATTERNS.search(s)
        ][:10]  # cap at 10

        # 3. Per-sentence sentiment → aggregate
        sentiment = self._get_aggregate_sentiment(sentences)

        # 4. KeyBERT keyphrases with MMR diversity
        try:
            raw_kw = self._kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                use_mmr=True,
                diversity=0.5,
                top_n=10,
            )
        except Exception:
            raw_kw = []
        keywords = [(kw, score) for kw, score in raw_kw]

        return AnalysisResult(
            strategy="improved",
            summary=summary,
            action_items=action_items,
            sentiment=sentiment,
            keywords=keywords,
        )

    def analyze_both(self, text: str) -> tuple[AnalysisResult, AnalysisResult]:
        """Returns (baseline_result, improved_result)."""
        return self.analyze_baseline(text), self.analyze_improved(text)

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_sentiment_safe(self, text: str) -> dict:
        """Run sentiment on first 512 chars (model token limit guard)."""
        try:
            out = self._sentiment(text[:512])[0]
            return {"label": out["label"], "score": round(out["score"], 4)}
        except Exception:
            return {"label": "UNKNOWN", "score": 0.0}

    def _get_aggregate_sentiment(self, sentences: list[str]) -> dict:
        """
        Run sentiment per sentence, then aggregate:
          - dominant label by count
          - average confidence score
        """
        labels, scores = [], []
        for sent in sentences[:20]:  # cap to avoid slow inference
            res = self._get_sentiment_safe(sent)
            labels.append(res["label"])
            scores.append(res["score"])

        if not labels:
            return {"label": "UNKNOWN", "score": 0.0}

        dominant = Counter(labels).most_common(1)[0][0]
        avg_score = round(sum(scores) / len(scores), 4)
        positive_ratio = round(labels.count("POSITIVE") / len(labels), 4)
        return {
            "label": dominant,
            "score": avg_score,
            "positive_ratio": positive_ratio,
            "sentence_count": len(labels),
        }


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_analysis(result: AnalysisResult, output_path: str | pathlib.Path) -> None:
    """Persist an AnalysisResult to a JSON file."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    print(f"[Analyzer] Saved → {output_path}")


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP analysis of a transcript")
    parser.add_argument("transcript", help="Path to transcript .txt or JSON (with 'text' field)")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    t_path = pathlib.Path(args.transcript)
    if t_path.suffix == ".json":
        import json as _json
        text = _json.loads(t_path.read_text(encoding="utf-8"))["text"]
    else:
        text = t_path.read_text(encoding="utf-8")

    analyzer = Analyzer()
    baseline, improved = analyzer.analyze_both(text)

    stem = t_path.stem.replace("_baseline", "").replace("_improved", "")
    save_analysis(baseline, f"{args.output_dir}/{stem}_baseline.json")
    save_analysis(improved, f"{args.output_dir}/{stem}_improved.json")

    print("\n── BASELINE SUMMARY ──────────────────────────────────────────────────")
    print(baseline.summary)
    print("\n── IMPROVED SUMMARY ──────────────────────────────────────────────────")
    print(improved.summary)
    print("\n── ACTION ITEMS ──────────────────────────────────────────────────────")
    for i, item in enumerate(improved.action_items, 1):
        print(f"  {i}. {item}")
    print("\n── SENTIMENT ─────────────────────────────────────────────────────────")
    print(f"  Baseline : {baseline.sentiment}")
    print(f"  Improved : {improved.sentiment}")
    print("\n── KEYWORDS ──────────────────────────────────────────────────────────")
    for kw, score in improved.keywords:
        print(f"  {kw:<30} {score:.4f}")
