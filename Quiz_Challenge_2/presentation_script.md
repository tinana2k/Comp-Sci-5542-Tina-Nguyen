# Presentation Script: AI Speech Intelligence System (2-Minute Version)
**CS 5542 | Quiz Challenge 2 | Tina Nguyen**

*Pacing: Speak naturally. This is ~260 words, which will take exactly 1.5 to 2 minutes to read. You don't need to click through every slide individually; just flow through them naturally as you speak.*

---

**[Slides 1, 2, 3: Intro, Problem & Business Value]**
"Hello, my name is Tina Nguyen. Today I’m presenting my AI Speech Intelligence Pipeline.

Our core problem is that unstructured audio—like meetings and lectures—is the richest but most inaccessible data format we have. The business value here is massive: automating the extraction of action items and summaries reclaims countless hours of lost productivity while making content globally accessible."

**[Slides 4, 5, 6: Dataset & Architecture]**
"To solve this, my system processes a simulated technical lecture audio file, evaluated against a human-annotated reference transcript and summary.

The architecture uses an ensemble of HuggingFace models, functioning sequentially: audio goes into Whisper for transcription, the text flows through BART and DistilBERT for advanced NLP extraction, then into MarianMT for translation, and finally SpeechT5 synthesizes the text back into playable audio."

**[Slide 7: Prompt & Input Design]**
"A critical design choice was prompt engineering. By giving Whisper a domain-aware initial prompt—telling it to expect technical terms like 'LLMs'—the Word Error Rate plummeted by 61%, dropping from 8.9% down to 3.5% without any retraining."

**[Slides 8 & 9: Results & Evaluation]**
"Looking at the results and evaluation, the improved pipeline dominated the baseline across the board. The BART abstractive summarizer increased ROUGE-2 scores by over 200%. Concrete action items were accurately extracted, and the system instantly translated the intelligence into Spanish, French, and Vietnamese locally."

**[Slides 10 & 11: Links & Limitations]**
"As for limitations, running 7 distinct Transformer models sequentially requires significant GPU VRAM, and the system currently runs synchronously rather than continuously streaming.

You can find the full GitHub repository linked directly in the slide deck."

**[Slide 12: AI Disclosure & Conclusion]**
"Finally, as disclosed on the final slide, generative AI tools including Google DeepMind and HuggingFace models were utilized during the development process. 

Thank you, and I’d be happy to take any questions."
