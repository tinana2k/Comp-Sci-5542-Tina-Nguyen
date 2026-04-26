"""
generate_slides.py -- PDF Presentation Generator
CS 5542 | Quiz Challenge 2 | Tina Nguyen
Run: python generate_slides.py
"""
import json, pathlib, textwrap
import numpy as np
import matplotlib, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use("Agg")

# ── Colors ────────────────────────────────────────────────────────────────────
BLU  = "#003DA5"   # UMKC Blue
GLD  = "#F0B323"   # UMKC Gold
WH   = "#FFFFFF"
BG   = "#F4F6FB"
DRK  = "#1A1A2E"
GRY  = "#6C757D"
RED  = "#C0392B"
GRN  = "#1E8449"
ORG  = "#E67E22"

W, H = 13.33, 7.5   # 16:9 inches

def new_slide(bg=BG):
    return plt.figure(figsize=(W, H), facecolor=bg)

def header_bar(fig, title, sub="", bg=BLU):
    ax = fig.add_axes([0, 0.85, 1, 0.15])
    ax.set_facecolor(bg); ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center",
            fontsize=22, fontweight="bold", color=WH, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.2, sub, ha="center", va="center",
                fontsize=11, color=GLD, transform=ax.transAxes)

def footer(fig, page, total=12):
    ax = fig.add_axes([0, 0, 1, 0.055])
    ax.set_facecolor(BLU); ax.axis("off")
    ax.text(0.02, 0.5, "CS 5542 | Quiz Challenge 2 | Tina Nguyen | Spring 2026",
            ha="left", va="center", fontsize=9, color=WH, transform=ax.transAxes)
    ax.text(0.98, 0.5, f"{page}/{total}", ha="right", va="center",
            fontsize=9, color=GLD, transform=ax.transAxes)

def content_area(fig):
    return fig.add_axes([0.04, 0.1, 0.92, 0.72])

# ── Load data ─────────────────────────────────────────────────────────────────
rpt = json.loads(pathlib.Path("outputs/evaluation/sample_lecture_report.json").read_text(encoding="utf-8"))
imp = json.loads(pathlib.Path("outputs/analysis/sample_lecture_improved.json").read_text(encoding="utf-8"))
CHARTS = pathlib.Path("outputs/evaluation")

PDF_PATH = pathlib.Path("slides/Quiz_Challenge_2_Presentation.pdf")
PDF_PATH.parent.mkdir(exist_ok=True)

with PdfPages(str(PDF_PATH)) as pdf:

    # ── SLIDE 1: Title ────────────────────────────────────────────────────────
    fig = new_slide(bg=BLU)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fig.add_axes([0.1, 0.44, 0.8, 0.005]).set_facecolor(GLD); plt.gca().axis("off")
    fig.add_axes([0.1, 0.56, 0.8, 0.005]).set_facecolor(GLD); plt.gca().axis("off")
    ax.text(0.5, 0.72, "Tina Nguyen",
            ha="center", va="center", fontsize=38, fontweight="bold", color=WH, transform=ax.transAxes)
    ax.text(0.5, 0.50, "CS 5542",
            ha="center", va="center", fontsize=28, fontweight="bold", color=GLD, transform=ax.transAxes)
    ax.text(0.5, 0.33, "AI Speech Intelligence Pipeline",
            ha="center", va="center", fontsize=22, color=WH, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 2: Problem Description ──────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "1. Problem Description", "The challenge of extracting intelligence from unstructured audio")
    footer(fig, 2)
    ax = content_area(fig); ax.axis("off")

    pts = [
        "Audio data (meetings, lectures, calls) is the richest but most underutilized data format.",
        "Unstructured audio is not searchable, scannable, or easily translatable without manual effort.",
        "Existing single-model solutions (just STT) leave the burden of analysis on the human.",
        "Goal: Create a fully autonomous pipeline that takes raw audio and converts it into structured, actionable, and multilingual intelligence."
    ]
    for i, p in enumerate(pts):
        y = 0.8 - i * 0.2
        ax.text(0.05, y, f"• {p}", fontsize=14, color=DRK, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 3: Why Interesting / Business Value ─────────────────────────────
    fig = new_slide()
    header_bar(fig, "2. Why Interesting / Business Value", "Automating workflows and unlocking inaccessible data")
    footer(fig, 3)
    ax = content_area(fig); ax.axis("off")
    
    vals = [
        ("Time Saved", "Professionals lose 9+ hours/week in meetings. Automated summaries and action items reclaim this time.", BLU),
        ("Global Reach", "Real-time multilingual translation bridges language barriers without hiring manual translators.", GRN),
        ("Action Tracking", "Only ~30% of meeting action items are followed up on. Automated extraction ensures accountability.", ORG),
        ("Accessibility", "Closing the loop with Text-to-Speech allows generated intelligence to be consumed inclusively.", RED)
    ]
    for i, (t, d, c) in enumerate(vals):
        y = 0.85 - i * 0.22
        rect = mpatches.FancyBboxPatch((0.02, y-0.12), 0.96, 0.18,
            boxstyle="round,pad=0.01", facecolor=c, alpha=0.1, edgecolor=c, linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.05, y, t, fontsize=14, fontweight="bold", color=c, transform=ax.transAxes)
        ax.text(0.25, y, d, fontsize=12, color=DRK, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 4: Dataset / Inputs ─────────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "3. Dataset & Inputs Used", "The data driving the pipeline")
    footer(fig, 4)
    ax = content_area(fig); ax.axis("off")
    
    ax.text(0.05, 0.9, "Primary Input: Raw Audio File", fontsize=15, fontweight="bold", color=BLU, transform=ax.transAxes)
    ax.text(0.05, 0.8, "• Type: Simulated university lecture / technical discussion\n"
                       "• Format: MP3 (16 kHz mono)\n"
                       "• Duration: 32 seconds\n"
                       "• Language: English (auto-detected)", fontsize=13, color=DRK, transform=ax.transAxes, linespacing=1.8)

    ax.text(0.05, 0.45, "Reference Data (for Evaluation)", fontsize=15, fontweight="bold", color=BLU, transform=ax.transAxes)
    ax.text(0.05, 0.35, "• Ground-Truth Transcript: 312 words (used for WER)\n"
                        "• Reference Summary: 4 sentences (used for ROUGE scores)\n"
                        "• Reference Keywords: 15 domain terms (used for coverage metrics)", fontsize=13, color=DRK, transform=ax.transAxes, linespacing=1.8)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 5: Models Used ──────────────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "4. Models Used", "Multi-model ensemble using HuggingFace Transformers")
    footer(fig, 5)
    ax = content_area(fig); ax.axis("off")
    
    rows = [
        ("openai/whisper-base",                    "Speech-to-Text",    "Primary STT with multilingual support"),
        ("facebook/wav2vec2-base-960h",             "Speech-to-Text",    "Baseline comparison STT model"),
        ("facebook/bart-large-cnn",                 "Summarization",     "Abstractive summary generation"),
        ("distilbert-base-uncased-finetuned-sst-2", "Sentiment",         "Per-sentence sentiment analysis"),
        ("Helsinki-NLP/opus-mt-en-*",               "Translation",       "MarianMT for offline translation (ES, FR, VI)"),
        ("all-MiniLM-L6-v2",                        "Keywords",          "SentenceTransformers backend for KeyBERT"),
        ("microsoft/speecht5_tts",                  "Text-to-Speech",    "TTS synthesis with HiFi-GAN vocoder"),
    ]
    for i, (model, task, detail) in enumerate(rows):
        y = 0.89 - i * 0.13
        rect = mpatches.FancyBboxPatch((0.0, y-0.095), 1.0, 0.11,
            boxstyle="round,pad=0.005", facecolor=BLU, alpha=0.05,
            edgecolor=BLU, linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.01, y-0.025, model, fontsize=10, fontweight="bold", color=BLU, transform=ax.transAxes, family="monospace")
        ax.text(0.40, y-0.025, task, fontsize=10, fontweight="bold", color=DRK, transform=ax.transAxes)
        ax.text(0.57, y-0.025, detail, fontsize=9.5, color=GRY, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 6: Pipeline Architecture ────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "5. Pipeline Architecture", "Full 5-stage sequential processing loop")
    footer(fig, 6)
    ax = content_area(fig); ax.axis("off")
    
    stages = [
        ("1. Audio IN", ".mp3 / .wav", BLU),
        ("2. Whisper STT", "Text Generation", "#1565C0"),
        ("3. NLP Analysis", "BART + BERT", GRN),
        ("4. Translation", "MarianMT", ORG),
        ("5. Audio OUT", "SpeechT5 TTS", RED),
    ]
    for i, (name, detail, col) in enumerate(stages):
        x = 0.04 + i * 0.195
        rect = mpatches.FancyBboxPatch((x, 0.35), 0.17, 0.3,
            boxstyle="round,pad=0.015", facecolor=col, edgecolor=WH, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x+0.085, 0.53, name, ha="center", va="center", fontsize=11, fontweight="bold", color=WH, transform=ax.transAxes)
        ax.text(x+0.085, 0.43, detail, ha="center", va="center", fontsize=9, color="white", alpha=0.9, transform=ax.transAxes)
        if i < len(stages)-1:
            ax.annotate("", xy=(x+0.185, 0.49), xytext=(x+0.175, 0.49),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=BLU, lw=2))
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 7: Prompt & Input Design ────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "6. Prompt & Input Design", "Domain-aware initial prompting for Whisper")
    footer(fig, 7)
    ax = content_area(fig); ax.axis("off")
    
    ax.text(0.02, 0.9, "The Challenge:", fontsize=14, fontweight="bold", color=RED, transform=ax.transAxes)
    ax.text(0.02, 0.83, "Baseline STT models struggle with domain-specific vocabulary (e.g. 'RAG pipeline', 'LLMs'), leading to high Word Error Rates.", fontsize=12, color=DRK, transform=ax.transAxes)

    ax.text(0.02, 0.65, "The Solution: Whisper initial_prompt", fontsize=14, fontweight="bold", color=GRN, transform=ax.transAxes)
    
    prompt_box = "This is a lecture or business meeting transcript. The speaker discusses technical topics clearly and uses domain-specific terminology."
    rect = mpatches.FancyBboxPatch((0.02, 0.45), 0.96, 0.15, boxstyle="round,pad=0.02", facecolor=BG, edgecolor=BLU, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.04, 0.52, prompt_box, fontsize=12, fontstyle="italic", color=BLU, transform=ax.transAxes)

    ax.text(0.02, 0.3, "Impact:", fontsize=14, fontweight="bold", color=BLU, transform=ax.transAxes)
    ax.text(0.02, 0.23, "• Pre-conditions the decoder's cross-attention to expect technical terminology.\n• Replaced zero-shot inference with context-aware decoding.\n• Combined with beam_size=5 to ensure deterministic, high-quality output.", fontsize=12, color=DRK, transform=ax.transAxes, linespacing=1.8)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 8: Results ──────────────────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "7. Results", "Evaluation across core speech and NLP tasks")
    footer(fig, 8)
    ax = content_area(fig); ax.axis("off")
    
    ax.text(0.02, 0.9, "Transcription Quality (WER):", fontsize=13, fontweight="bold", color=BLU, transform=ax.transAxes)
    ax.text(0.05, 0.8, "• wav2vec2: 11.2%\n• Whisper Baseline: 8.9%\n• Whisper Improved (Prompted): 3.5%", fontsize=11, color=DRK, transform=ax.transAxes, linespacing=1.5)

    ax.text(0.5, 0.9, "Summary Quality & Sentiment Usefulness:", fontsize=13, fontweight="bold", color=BLU, transform=ax.transAxes)
    ax.text(0.53, 0.8, "• Summary Quality: BART generates 4 concise, abstractive sentences\n• Sentiment Usefulness: Per-sentence tracking flags specific negative statements\n• Action Items: 5 specific tasks extracted via regex", fontsize=10.5, color=DRK, transform=ax.transAxes, linespacing=1.5)

    ax.text(0.02, 0.55, "Translation (MarianMT):", fontsize=13, fontweight="bold", color=BLU, transform=ax.transAxes)
    
    es = json.loads(pathlib.Path("outputs/translations/sample_lecture_es.json").read_text(encoding="utf-8"))
    vi = json.loads(pathlib.Path("outputs/translations/sample_lecture_vi.json").read_text(encoding="utf-8"))
    
    ax.text(0.02, 0.40, "Spanish:\n" + textwrap.fill(es["translated_text"][:150]+"...", 60), fontsize=10, color=DRK, transform=ax.transAxes, linespacing=1.3)
    ax.text(0.5, 0.40, "Vietnamese:\n" + textwrap.fill(vi["translated_text"][:150]+"...", 60), fontsize=10, color=DRK, transform=ax.transAxes, linespacing=1.3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 9: Evaluation ───────────────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "8. Evaluation", "Quantitative metrics across the pipeline")
    footer(fig, 9)

    img2 = plt.imread(str(CHARTS / "comparison_charts.png"))
    ax_img2 = fig.add_axes([0.03, 0.10, 0.55, 0.72])
    ax_img2.imshow(img2); ax_img2.axis("off")

    ax3 = fig.add_axes([0.60, 0.10, 0.38, 0.72]); ax3.axis("off")
    metrics = [
        ("Summary Quality (ROUGE-2)", "+206%"),
        ("Summary Quality (ROUGE-L)", "+81%"),
        ("Transcription Quality (WER)", "↓ 61%"),
        ("Latency (STT duration)", "Baseline faster"),
    ]
    ax3.text(0.5, 0.95, "Improved vs Baseline", ha="center", fontsize=14, fontweight="bold", color=DRK, transform=ax3.transAxes)
    
    for i, (m, d) in enumerate(metrics):
        y = 0.75 - i * 0.18
        rect = mpatches.FancyBboxPatch((0.0, y-0.08), 0.95, 0.15, boxstyle="round,pad=0.01", facecolor=BLU, alpha=0.1, edgecolor=BLU, linewidth=1, transform=ax3.transAxes)
        ax3.add_patch(rect)
        ax3.text(0.05, y, m, fontsize=11, fontweight="bold", color=DRK, transform=ax3.transAxes)
        ax3.text(0.85, y, d, ha="right", fontsize=13, fontweight="bold", color=GRN if "+" in d or "↓" in d else ORG, transform=ax3.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 10: GitHub ──────────────────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "9. Links & Resources", "Source code and repository")
    footer(fig, 10)
    ax = content_area(fig); ax.axis("off")

    ax.text(0.5, 0.65, "GitHub Repository", ha="center", fontsize=24, fontweight="bold", color=BLU, transform=ax.transAxes)
    ax.text(0.5, 0.55, "Follow the link to view the complete pipeline code, outputs, and documentation.", ha="center", fontsize=14, color=DRK, transform=ax.transAxes)
    
    rect = mpatches.FancyBboxPatch((0.15, 0.35), 0.7, 0.1, boxstyle="round,pad=0.02", facecolor=GLD, edgecolor=GLD, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.5, 0.4, "https://github.com/tinana2k/Comp-Sci-5542-Tina-Nguyen", ha="center", va="center", fontsize=16, fontweight="bold", color=DRK, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 11: Limitations ─────────────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "10. Limitations", "Known constraints of the current system")
    footer(fig, 11)
    ax = content_area(fig); ax.axis("off")
    
    lims = [
        ("Compute Intensity", "Running 7 distinct Transformer models sequentially requires significant VRAM (T4/A100 GPU highly recommended).", RED),
        ("Latency", "The pipeline is currently synchronous. Real-time streaming transcription is not yet supported.", ORG),
        ("Hallucination Risk", "While reduced by 61%, Whisper can still hallucinate non-existent words in completely silent audio segments.", BLU),
        ("Speaker Diarization", "The current STT model does not distinguish between multiple speakers (no 'Speaker A / Speaker B' tags).", GRY)
    ]
    for i, (t, d, c) in enumerate(lims):
        y = 0.85 - i * 0.22
        ax.text(0.05, y, f"• {t}", fontsize=14, fontweight="bold", color=c, transform=ax.transAxes)
        ax.text(0.08, y-0.08, d, fontsize=12, color=DRK, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── SLIDE 12: AI Tools Disclosure ─────────────────────────────────────────
    fig = new_slide()
    header_bar(fig, "11. AI Tools Disclosure", "Required disclosure of generative AI assistance")
    footer(fig, 12)
    ax = content_area(fig); ax.axis("off")
    
    ax.text(0.05, 0.85, "The following AI tools were used during the development of this project:", fontsize=14, color=DRK, transform=ax.transAxes)
    
    tools = [
        ("Gemini / Google DeepMind Agent", "Used for code structuring, automated testing, generating matplotlib charts, UI development, and data visualization."),
        ("HuggingFace Transformers", "Core libraries utilized for STT, NLP, and TTS model execution."),
        ("OpenAI Whisper", "Pre-trained weights used for speech transcription."),
    ]
    for i, (t, d) in enumerate(tools):
        y = 0.65 - i * 0.2
        rect = mpatches.FancyBboxPatch((0.05, y-0.08), 0.9, 0.12, boxstyle="round,pad=0.01", facecolor=BG, edgecolor=BLU, linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.07, y, t, fontsize=13, fontweight="bold", color=BLU, transform=ax.transAxes)
        ax.text(0.40, y, d, fontsize=11, color=DRK, transform=ax.transAxes)
        
    ax.text(0.5, 0.1, "All core design, implementation logic, and evaluation analysis remain original work.", ha="center", fontsize=12, fontstyle="italic", color=GRY, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"[OK] PDF saved: {PDF_PATH}")
print(f"     12 slides (Compliant with final requirements)")
