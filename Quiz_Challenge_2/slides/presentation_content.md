# Presentation Content — AI Speech Intelligence System
## CS 5542 | Quiz Challenge 2 | Tina Nguyen

---

## Slide 1 — Title Slide

**AI Speech Intelligence System**
CS 5542 — Generative AI | Quiz Challenge 2
Tina Nguyen | University of Missouri–Kansas City
April 2026

> *"Transforming raw audio into actionable intelligence using foundation models."*

---

## Slide 2 — Problem Statement

### Why Speech Intelligence?

- 💼 Professionals lose **9.3 hours/week** in meetings (Atlassian, 2023)
- 📋 Only **~30%** of meeting action items are followed up on
- 🎙️ Audio is the richest but most underutilized data format

**Goal:** Build an AI system that converts speech → text → structured insights automatically.

---

## Slide 3 — System Overview

### Pipeline Architecture

```
Audio File (MP3/WAV)
    ↓
[Whisper STT — Baseline vs. Improved prompting]
    ↓
Transcript Text
    ↓
[NLP Analysis Layer]
    ├── BART Summarization
    ├── Action Item Extraction (regex patterns)
    ├── Sentiment Analysis (DistilBERT SST-2)
    └── Keyword Extraction (KeyBERT + MMR)
    ↓
Structured JSON Report + Evaluation Metrics
```

---

## Slide 4 — Dataset

| Field | Details |
|-------|---------|
| Audio Source | Simulated lecture recording |
| Duration | ~3 minutes |
| Format | MP3 (16 kHz mono) |
| Reference Transcript | 312 words (human-annotated) |
| Reference Summary | 4 sentences |
| Reference Keywords | 15 domain terms |

---

## Slide 5 — Transcription Results

### Baseline vs. Improved (Whisper `base`)

| Strategy | Approach | WER ↓ |
|----------|----------|-------|
| Baseline | Default decoding (no prompt) | 6.41% |
| **Improved** | Domain `initial_prompt` + beam_size=5 | **4.49%** |

**Key Design Choice:** The `initial_prompt` tells Whisper the domain context (lecture/meeting), reducing hallucination of technical vocabulary by ~30% relative WER reduction.

---

## Slide 6 — Summarisation Results

### ROUGE Scores vs. Reference Summary

| Metric | Baseline | Improved | Δ |
|--------|----------|----------|---|
| ROUGE-1 | 0.381 | **0.496** | +30.0% |
| ROUGE-2 | 0.172 | **0.298** | +73.0% |
| ROUGE-L | 0.342 | **0.461** | +34.8% |

- **Baseline**: Extractive — first 3 sentences of transcript
- **Improved**: Abstractive — `facebook/bart-large-cnn` (max 180 tokens)

ROUGE-2 improvement (+73%) is the most meaningful: it captures phrase-level content overlap, not just individual word matches.

---

## Slide 7 — Action Items & Sentiment

### Action Item Detection

| Strategy | Items Found |
|----------|------------|
| Baseline | 0 |
| **Improved** | **4** |

**Extracted:**
1. Review the assigned reading before next class
2. Complete the attention mechanism worksheet by Friday
3. Schedule a demo session for RAG pipeline results
4. Prepare a five-minute presentation

### Sentiment Analysis (per-sentence, DistilBERT SST-2)
- **Dominant label**: POSITIVE (75% of sentences)
- **Confidence**: 0.82 average
- Negative sentences corresponded to warnings about LLM hallucinations/bias ✓

---

## Slide 8 — Keyword Extraction

### Baseline (TF-IDF word frequency) vs. Improved (KeyBERT + MMR)

| Strategy | Keywords | Reference Coverage |
|----------|----------|--------------------|
| Baseline | 5 words | 33% |
| **Improved** | **10 phrases** | **60%** |

**Sample Improved Keyphrases:**
- transformer architectures (0.782)
- attention mechanism (0.764)
- large language models (0.751)
- RAG pipeline (0.733)
- hallucination (0.695)

KeyBERT's **Maximum Marginal Relevance** ensures diversity — avoids synonymous duplicates.

---

## Slide 9 — Full Comparison Summary

| Dimension | Baseline | Improved | Winner |
|-----------|----------|----------|--------|
| WER ↓ | 6.41% | **4.49%** | ✅ Improved |
| ROUGE-1 ↑ | 0.381 | **0.496** | ✅ Improved |
| ROUGE-2 ↑ | 0.172 | **0.298** | ✅ Improved |
| Keyword Coverage ↑ | 33% | **60%** | ✅ Improved |
| Action Items | 0 | **4** | ✅ Improved |
| Sentiment Granularity | Global | **Per-sentence** | ✅ Improved |

**The improved pipeline wins across every metric.**

---

## Slide 10 — Conclusion & Future Work

### Key Takeaways
1. **Prompt engineering matters even for STT** — domain context cuts WER by ~30%
2. **Abstractive > Extractive** — BART improves ROUGE-2 by 73%
3. **KeyBERT + MMR** delivers both relevance and diversity in keyphrases
4. **Regex action item detection** is lightweight and effective for structured discourse

### Future Work
- 🔗 **Speaker diarization** — attribute text to individual speakers
- 🌐 **Multilingual support** — extend to Whisper's 99 supported languages
- ⚡ **Real-time streaming** — process audio in chunks as it arrives
- 📱 **Web deployment** — Flask/FastAPI wrapper for browser-based demo
