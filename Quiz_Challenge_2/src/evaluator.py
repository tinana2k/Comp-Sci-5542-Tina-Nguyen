"""
evaluator.py — Quantitative Evaluation for Speech Intelligence System
======================================================================
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Computes and compares evaluation metrics across baseline vs. improved:

Transcription Quality:
  - Word Error Rate (WER)       — requires reference transcript
  - Word Count delta            — proxy for verbosity

Summarisation Quality:
  - ROUGE-1, ROUGE-2, ROUGE-L   — requires reference summary
  - Summary Compression Ratio   — summary_len / transcript_len

Keyword Quality:
  - Coverage Recall             — fraction of reference keywords matched

Outputs a structured JSON report and prints a formatted console table.
"""

from __future__ import annotations

import json
import re
import pathlib
from typing import Optional

import pandas as pd
from rouge_score import rouge_scorer
import numpy as np


# ── WER helper ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) using dynamic programming.
    WER = (S + D + I) / N   where N = number of reference words.
    """
    ref_words = _tokenize(reference)
    hyp_words = _tokenize(hypothesis)

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return round(dp[n][m] / n, 4)


# ── ROUGE helper ──────────────────────────────────────────────────────────────

def rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """Return ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ── Keyword coverage ──────────────────────────────────────────────────────────

def keyword_coverage(reference_keywords: list[str], extracted_keywords: list[str]) -> float:
    """Fraction of reference keywords found in the extracted keyword list."""
    if not reference_keywords:
        return 0.0
    ref_set = {k.lower().strip() for k in reference_keywords}
    ext_set = {k.lower().strip() for k in extracted_keywords}
    matched = ref_set & ext_set
    return round(len(matched) / len(ref_set), 4)


# ── Main Evaluator ────────────────────────────────────────────────────────────

class Evaluator:
    """
    Compares baseline and improved outputs across all metrics.

    Usage
    -----
    evaluator = Evaluator()
    report = evaluator.evaluate(
        baseline_transcription="...",
        improved_transcription="...",
        baseline_analysis=baseline_result.to_dict(),
        improved_analysis=improved_result.to_dict(),
        reference_transcript="...",     # optional ground truth
        reference_summary="...",        # optional
        reference_keywords=["AI", ...], # optional
    )
    evaluator.save_report(report, "outputs/evaluation/report.json")
    evaluator.print_table(report)
    """

    def evaluate(
        self,
        baseline_transcription: str,
        improved_transcription: str,
        baseline_analysis: dict,
        improved_analysis: dict,
        reference_transcript: Optional[str] = None,
        reference_summary: Optional[str] = None,
        reference_keywords: Optional[list[str]] = None,
    ) -> dict:
        report: dict = {
            "transcription": {},
            "summarisation": {},
            "sentiment": {},
            "keywords": {},
            "action_items": {},
        }

        # ── Transcription ─────────────────────────────────────────────────────
        report["transcription"]["baseline_word_count"] = len(baseline_transcription.split())
        report["transcription"]["improved_word_count"] = len(improved_transcription.split())

        if reference_transcript:
            report["transcription"]["baseline_wer"] = word_error_rate(
                reference_transcript, baseline_transcription
            )
            report["transcription"]["improved_wer"] = word_error_rate(
                reference_transcript, improved_transcription
            )

        # ── Summarisation ─────────────────────────────────────────────────────
        b_summary = baseline_analysis.get("summary", "")
        i_summary = improved_analysis.get("summary", "")
        transcript_len = len(baseline_transcription.split())

        report["summarisation"]["baseline_length_words"] = len(b_summary.split())
        report["summarisation"]["improved_length_words"] = len(i_summary.split())
        report["summarisation"]["baseline_compression_ratio"] = (
            round(len(b_summary.split()) / max(transcript_len, 1), 4)
        )
        report["summarisation"]["improved_compression_ratio"] = (
            round(len(i_summary.split()) / max(transcript_len, 1), 4)
        )

        if reference_summary:
            report["summarisation"]["baseline_rouge"] = rouge_scores(reference_summary, b_summary)
            report["summarisation"]["improved_rouge"] = rouge_scores(reference_summary, i_summary)

        # ── ROUGE between strategies (cross-comparison) ───────────────────────
        if b_summary and i_summary:
            report["summarisation"]["cross_rouge"] = rouge_scores(b_summary, i_summary)

        # ── Sentiment ─────────────────────────────────────────────────────────
        report["sentiment"]["baseline"] = baseline_analysis.get("sentiment", {})
        report["sentiment"]["improved"] = improved_analysis.get("sentiment", {})

        # ── Keywords ──────────────────────────────────────────────────────────
        b_kws = [k["phrase"] for k in baseline_analysis.get("keywords", [])]
        i_kws = [k["phrase"] for k in improved_analysis.get("keywords", [])]
        report["keywords"]["baseline_count"] = len(b_kws)
        report["keywords"]["improved_count"] = len(i_kws)

        if reference_keywords:
            report["keywords"]["baseline_coverage"] = keyword_coverage(reference_keywords, b_kws)
            report["keywords"]["improved_coverage"] = keyword_coverage(reference_keywords, i_kws)

        # ── Action items ──────────────────────────────────────────────────────
        report["action_items"]["baseline_count"] = len(baseline_analysis.get("action_items", []))
        report["action_items"]["improved_count"] = len(improved_analysis.get("action_items", []))
        report["action_items"]["items"] = improved_analysis.get("action_items", [])

        return report

    def save_report(self, report: dict, output_path: str | pathlib.Path) -> None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[Evaluator] Report saved → {output_path}")

    def print_table(self, report: dict) -> None:
        """Print a formatted comparison table to console."""
        rows = []

        t = report.get("transcription", {})
        rows.append(["Transcription Word Count", t.get("baseline_word_count", "—"), t.get("improved_word_count", "—")])
        if "baseline_wer" in t:
            rows.append(["Word Error Rate ↓", t["baseline_wer"], t["improved_wer"]])

        s = report.get("summarisation", {})
        rows.append(["Summary Length (words)", s.get("baseline_length_words", "—"), s.get("improved_length_words", "—")])
        rows.append(["Compression Ratio ↓", s.get("baseline_compression_ratio", "—"), s.get("improved_compression_ratio", "—")])

        b_rouge = s.get("baseline_rouge", {})
        i_rouge = s.get("improved_rouge", {})
        if b_rouge:
            rows.append(["ROUGE-1 ↑", b_rouge.get("rouge1", "—"), i_rouge.get("rouge1", "—")])
            rows.append(["ROUGE-2 ↑", b_rouge.get("rouge2", "—"), i_rouge.get("rouge2", "—")])
            rows.append(["ROUGE-L ↑", b_rouge.get("rougeL", "—"), i_rouge.get("rougeL", "—")])

        sent = report.get("sentiment", {})
        rows.append(["Sentiment Label", sent.get("baseline", {}).get("label", "—"), sent.get("improved", {}).get("label", "—")])

        kw = report.get("keywords", {})
        rows.append(["Keywords Extracted", kw.get("baseline_count", "—"), kw.get("improved_count", "—")])
        if "baseline_coverage" in kw:
            rows.append(["Keyword Coverage ↑", kw["baseline_coverage"], kw["improved_coverage"]])

        ai = report.get("action_items", {})
        rows.append(["Action Items Found", ai.get("baseline_count", "—"), ai.get("improved_count", "—")])

        df = pd.DataFrame(rows, columns=["Metric", "Baseline", "Improved"])
        print("\n" + "=" * 60)
        print("  SPEECH INTELLIGENCE SYSTEM — EVALUATION REPORT")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Evaluate baseline vs. improved Speech Intelligence outputs")
    parser.add_argument("--baseline-transcription", required=True, help="JSON from transcriber (baseline)")
    parser.add_argument("--improved-transcription", required=True, help="JSON from transcriber (improved)")
    parser.add_argument("--baseline-analysis", required=True, help="JSON from analyzer (baseline)")
    parser.add_argument("--improved-analysis", required=True, help="JSON from analyzer (improved)")
    parser.add_argument("--output", default="outputs/evaluation/report.json")
    args = parser.parse_args()

    def _load(p):
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))

    bt = _load(args.baseline_transcription)
    it = _load(args.improved_transcription)
    ba = _load(args.baseline_analysis)
    ia = _load(args.improved_analysis)

    ev = Evaluator()
    report = ev.evaluate(
        baseline_transcription=bt["text"],
        improved_transcription=it["text"],
        baseline_analysis=ba,
        improved_analysis=ia,
    )
    ev.save_report(report, args.output)
    ev.print_table(report)
