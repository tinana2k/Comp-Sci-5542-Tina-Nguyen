# AI Speech Intelligence System
**CS 5542 — Quiz Challenge 2**
**Student:** Tina Nguyen | **Semester:** Spring 2026 | **Use Case:** Smart Lecture Assistant

---

## Overview

This project implements an end-to-end **AI Speech Intelligence System** that takes audio as input, converts speech to text using multiple state-of-the-art models, and automatically produces five intelligent outputs:

| Output | Technology |
|--------|-----------|
| **Transcription** | OpenAI Whisper (baseline + improved) & Facebook wav2vec2 |
| **Summarization** | facebook/bart-large-cnn (abstractive) |
| **Action Items** | Regex + linguistic trigger detection |
| **Sentiment Analysis** | DistilBERT SST-2 (per-sentence) |
| **Translation** | Helsinki-NLP MarianMT (EN → ES / FR / VI) |
| **Keyword Extraction** | KeyBERT + MMR diversity (multi-word keyphrases) |
| **TTS Output** | microsoft/speecht5_tts (closes the speech loop) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUDIO INPUT (.mp3 / .wav)                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    ┌─────────▼─────────┐   ┌──────────▼──────────┐
    │  OpenAI Whisper   │   │  Facebook wav2vec2  │
    │  (base, improved) │   │  (base-960h)        │
    │  WER: 3.5%        │   │  WER: 11.2%         │
    └─────────┬─────────┘   └──────────┬──────────┘
              │                        │
              │  Best transcript        │  Model comparison
              └────────────┬───────────┘
                           │
              ┌────────────▼────────────┐
              │    NLP ANALYSIS LAYER   │
              │  ┌────────────────────┐ │
              │  │ BART Summarizer    │ │  → Abstractive 4-sentence summary
              │  │ Action Detector    │ │  → 5 deadlined action items
              │  │ DistilBERT SST-2   │ │  → Per-sentence sentiment (21 sents)
              │  │ KeyBERT + MMR      │ │  → 12 multi-word keyphrases
              │  └────────────────────┘ │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  TRANSLATION LAYER      │
              │  Helsinki-NLP MarianMT  │
              │  EN → ES / FR / VI      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  TTS SYNTHESIS          │
              │  microsoft/speecht5_tts │  → Summary audio (WAV output)
              └─────────────────────────┘
```

---

## Models Used

| Model | Source | Task |
|-------|--------|------|
| `openai/whisper-base` | [HuggingFace](https://huggingface.co/openai/whisper-small) | Primary STT (baseline + improved) |
| `facebook/wav2vec2-base-960h` | [HuggingFace](https://huggingface.co/facebook/wav2vec2-base-960h) | Secondary STT (model comparison) |
| `microsoft/speecht5_tts` | [HuggingFace](https://huggingface.co/microsoft/speecht5_tts) | Text-to-Speech synthesis |
| `facebook/bart-large-cnn` | HuggingFace | Abstractive summarization |
| `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace | Sentiment analysis (SST-2) |
| `all-MiniLM-L6-v2` (via KeyBERT) | HuggingFace | Keyword extraction |
| `Helsinki-NLP/opus-mt-en-{es,fr,vi}` | HuggingFace | Multilingual translation |

---

## Project Structure

```
Quiz_Challenge_2/
├── data/
│   ├── sample_lecture.mp3                    ← Input audio (CS lecture, ~5 min)
│   ├── sample_reference_transcript.txt       ← Ground-truth transcript (548 words)
│   └── sample_reference_summary.txt          ← Reference summary (4 sentences)
│
├── src/
│   ├── transcriber.py           ← Whisper STT (baseline + improved)
│   ├── wav2vec2_transcriber.py  ← Facebook wav2vec2 STT (comparison model)
│   ├── analyzer.py              ← NLP: summary, actions, sentiment, keywords
│   ├── translator.py            ← MarianMT translation (EN → ES/FR/VI)
│   ├── tts_synthesizer.py       ← SpeechT5 text-to-speech
│   ├── evaluator.py             ← WER, ROUGE, BLEU, keyword coverage metrics
│   └── pipeline.py              ← End-to-end orchestrator (CLI)
│
├── notebooks/
│   └── speech_intelligence.ipynb  ← Colab notebook (all 5 stages)
│
├── outputs/
│   ├── transcriptions/
│   │   ├── sample_lecture_baseline.json    ← Whisper baseline (WER 8.9%)
│   │   ├── sample_lecture_improved.json    ← Whisper improved (WER 3.5%)
│   │   └── sample_lecture_wav2vec2.json    ← wav2vec2 (WER 11.2%, ALL-CAPS)
│   ├── analysis/
│   │   ├── sample_lecture_baseline.json    ← Extractive summary + TF-IDF keywords
│   │   └── sample_lecture_improved.json    ← BART summary + KeyBERT + DistilBERT
│   ├── translations/
│   │   ├── sample_lecture_es.json          ← Spanish (Helsinki-NLP)
│   │   ├── sample_lecture_fr.json          ← French  (Helsinki-NLP)
│   │   └── sample_lecture_vi.json          ← Vietnamese (Helsinki-NLP)
│   ├── tts/
│   │   └── sample_lecture_summary.wav      ← SpeechT5 audio of improved summary
│   └── evaluation/
│       ├── sample_lecture_report.json      ← Full metric report (all stages)
│       ├── comparison_charts.png           ← ROUGE / WER / Feature counts
│       ├── sentiment_keywords_chart.png    ← Sentiment timeline + KeyBERT
│       ├── radar_chart.png                 ← Multi-metric radar
│       ├── error_analysis_chart.png        ← Error taxonomy + ROUGE deltas
│       └── model_translation_chart.png     ← Whisper vs wav2vec2 + translation
│
├── generate_charts.py   ← Standalone visualization script (CPU-safe)
├── requirements.txt     ← Pinned dependencies
└── README.md
```

---

## Example Use Cases

| Use Case | `--use-case` flag | Domain Prompt |
|----------|------------------|---------------|
| **Smart Lecture Assistant** | `lecture` (default) | CS / AI / ML lecture vocabulary |
| **Meeting Summarizer** | `meeting` | Project status, deadlines, standups |
| **Customer Call Analyzer** | `call` | Product questions, complaints, resolutions |
| **Medical Dictation Assistant** | `medical` | Clinical terms, drug names, diagnoses |

---

## Quantitative Results

### STT Model Comparison

| Model | WER ↓ | Strengths | Limitations |
|-------|-------|-----------|------------|
| Whisper base (baseline) | 8.9% | Punctuation, multilingual | Acronym errors without domain prompt |
| **Whisper base (improved)** | **3.5%** | Domain prompt + beam_size=5 | GPU recommended for real-time |
| wav2vec2-base-960h | 11.2% | Fast CPU inference, Apache-2.0 | ALL-CAPS, no punctuation, English-only |

### Summarization Quality (ROUGE vs. Reference)

| Metric | Baseline (Extractive) | Improved (BART) | Delta |
|--------|-----------------------|-----------------|-------|
| ROUGE-1 | 0.312 | **0.563** | +80.3% |
| ROUGE-2 | 0.125 | **0.381** | +205.6% |
| ROUGE-L | 0.289 | **0.522** | +80.5% |

### NLP Feature Extraction

| Feature | Baseline | Improved |
|---------|----------|----------|
| Keywords extracted | 5 (single-word TF-IDF) | **12 (multi-word KeyBERT+MMR)** |
| Keyword coverage | 20.0% | **73.3%** |
| Action items detected | 0 | **5 (with deadlines)** |
| Sentiment granularity | Global | **Per-sentence (21 sentences)** |

### Translation Output

| Language | Model | Words (from 122 EN) | Inference |
|----------|-------|---------------------|-----------|
| Spanish | `opus-mt-en-es` | 130 | 3.4s |
| French | `opus-mt-en-fr` | 135 | 3.2s |
| Vietnamese | `opus-mt-en-vi` | 145 | 4.0s |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Lecture assistant (default)
python src/pipeline.py --audio data/sample_lecture.mp3

# With translation + TTS
python src/pipeline.py --audio data/sample_lecture.mp3 \
  --model small \
  --translate es fr vi \
  --tts

# Meeting summarizer use case
python src/pipeline.py --audio data/meeting.mp3 --use-case meeting

# Customer call analyzer
python src/pipeline.py --audio data/call.mp3 --use-case call
```

### 3. Generate Evaluation Charts

```bash
python generate_charts.py
```

### 4. Run in Google Colab

Open `notebooks/speech_intelligence.ipynb` in Google Colab (T4 GPU recommended).

---

## Google Colab Notebook Sections

| Section | Description |
|---------|-------------|
| **0. Setup** | Install deps, mount Drive, verify GPU |
| **1. Audio Input** | Upload / generate demo audio (gTTS) |
| **2. Whisper STT** | Baseline vs. improved transcription |
| **3. wav2vec2 STT** | Secondary model comparison |
| **4. NLP Analysis** | BART summary, action items, sentiment, keywords |
| **5. Translation** | MarianMT EN → ES / FR / VI |
| **6. SpeechT5 TTS** | Synthesize summary audio |
| **7. Evaluation** | WER, ROUGE, keyword coverage, model comparison |
| **8. Visualization** | All 5 evaluation charts |

---

## Technical Notes

### Why Whisper over wav2vec2 for this project?
- Whisper outputs **properly cased text with punctuation** — critical for downstream NLP (BART summarization needs sentence boundaries)
- Whisper supports `initial_prompt` for domain priming, reducing technical acronym errors by 61%
- Whisper natively supports `task="translate"` for **direct speech translation** (non-English audio → English text)
- wav2vec2's ALL-CAPS output requires additional post-processing (CTC alignment, punctuation restoration)

### Why MarianMT for translation?
- MarianMT runs **fully locally** — no API keys, no rate limits, offline-capable
- Available for 1,000+ language pairs via Helsinki-NLP
- Complements Whisper's built-in translation (audio→EN) with text-level EN→{ES,FR,VI} capability

### Whisper's Built-in Translation
```python
# Whisper can translate non-English speech → English text in one pass
model = whisper.load_model("base")
result = model.transcribe("lecture_french.mp3", task="translate")
# Returns English text regardless of input language
```

---

## Dependencies

See [`requirements.txt`](requirements.txt) for pinned versions.

Key packages:
- `openai-whisper` — Whisper STT
- `transformers` — wav2vec2, BART, DistilBERT, MarianMT, SpeechT5
- `keybert` — Keyword extraction
- `rouge-score` — ROUGE evaluation
- `sacrebleu` — BLEU metric for translation
- `matplotlib` — Visualization
- `gtts` — Google TTS for synthetic demo audio generation

---

*CS 5542 -- Quiz Challenge 2 | Spring 2026*
