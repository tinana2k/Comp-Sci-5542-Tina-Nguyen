# Dataset Description — Speech Intelligence System
## CS 5542 | Quiz Challenge 2 | Author: Tina Nguyen

---

## Audio Source

| Field              | Details |
|--------------------|---------|
| **Type**           | Lecture / Business meeting recording |
| **Format**         | MP3 / WAV / FLAC (any format supported by FFmpeg) |
| **Sample Rate**    | 16 kHz (Whisper default) |
| **Duration**       | Recommended: 30 seconds – 15 minutes |
| **Language**       | English (auto-detected by Whisper) |

---

## Included Files

| File | Purpose |
|------|---------|
| `sample_reference_transcript.txt` | Ground-truth transcript for WER/ROUGE evaluation |
| `sample_reference_summary.txt`    | Reference summary for ROUGE comparison |

---

## Generating Your Own Audio

If you do not have an audio file, you can generate a sample using Python's `gTTS` library:

```python
from gtts import gTTS
text = open("data/sample_reference_transcript.txt").read()
tts = gTTS(text, lang="en")
tts.save("data/sample_lecture.mp3")
```

Or download any open-license lecture recording from:
- OpenCourseWare (MIT, Stanford)
- YouTube (convert with `yt-dlp`)
- LibriVox (public domain audiobooks)

---

## Reference Keywords (for keyword coverage evaluation)

```
transformers, attention mechanism, large language models, LLMs, GPT-4,
RAG pipeline, ROUGE score, BLEU score, evaluation metrics, hallucination,
action item, self-attention, NLP, text generation, reasoning
```
