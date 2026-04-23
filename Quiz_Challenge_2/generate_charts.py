"""
generate_charts.py -- Standalone chart generator for Speech Intelligence outputs
CS 5542 | Quiz Challenge 2 | Tina Nguyen

Produces 5 publication-quality charts from the evaluation JSON:
  1. comparison_charts.png       -- 3-panel: ROUGE / WER+Coverage / Feature counts
  2. sentiment_keywords_chart.png -- Sentiment timeline + KeyBERT bar
  3. radar_chart.png             -- Multi-metric spider/radar
  4. error_analysis_chart.png    -- WER error taxonomy + ROUGE improvement
  5. model_translation_chart.png -- STT model comparison + translation output

Run: python generate_charts.py
"""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

# ── Load data ─────────────────────────────────────────────────────────────────
OUT_DIR   = pathlib.Path("outputs/evaluation")
RPT_PATH  = OUT_DIR / "sample_lecture_report.json"
IMP_PATH  = pathlib.Path("outputs/analysis/sample_lecture_improved.json")

with open(RPT_PATH)  as f: r  = json.load(f)
with open(IMP_PATH)  as f: ia = json.load(f)

# ── Color palette (UMKC branded) ──────────────────────────────────────────────
C_BASE  = "#6C757D"    # grey   - baseline
C_IMPR  = "#003DA5"    # UMKC blue  - improved
C_WAV   = "#E67E22"    # orange  - wav2vec2
C_GOLD  = "#F0B323"    # UMKC gold  - accent / medium tier
C_RED   = "#C0392B"    # negative sentiment
C_GREEN = "#1E8449"    # positive sentiment
C_BG    = "#F8F9FA"
C_GRID  = "#DEE2E6"

FT  = {"size": 11, "weight": "bold", "color": "#212529"}   # title font
FA  = {"size":  9, "color": "#495057"}                     # axis font
FAN = {"size":  8.5, "weight": "bold"}                     # annotation

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C_BG)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, **FT, pad=8)
    if xlabel: ax.set_xlabel(xlabel, **FA)
    if ylabel: ax.set_ylabel(ylabel, **FA)

# ── Extract metrics ───────────────────────────────────────────────────────────
b_rouge  = r["summarisation"]["baseline_rouge"]
i_rouge  = r["summarisation"]["improved_rouge"]
b_wer    = r["transcription"]["whisper_baseline_wer"]
i_wer    = r["transcription"]["whisper_improved_wer"]
w2v_wer  = r["transcription"]["wav2vec2_wer"]
b_kw     = r["keywords"]["baseline_count"]
i_kw     = r["keywords"]["improved_count"]
b_ai     = r["action_items"]["baseline_count"]
i_ai     = r["action_items"]["improved_count"]
b_cov    = r["keywords"]["baseline_coverage_vs_reference"] * 100
i_cov    = r["keywords"]["improved_coverage_vs_reference"] * 100

# ===========================================================================
# FIGURE 1 — Main 3-panel comparison
# ===========================================================================
fig1 = plt.figure(figsize=(17, 5.8), facecolor=C_BG)
gs1  = GridSpec(1, 3, figure=fig1, wspace=0.40)

W = 0.34   # bar width

# -- Panel 1: ROUGE -----------------------------------------------------------
ax1 = fig1.add_subplot(gs1[0, 0])
style_ax(ax1, "Summarisation Quality\n(ROUGE F1 vs. Reference)", ylabel="F1 Score")

labels1 = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
bv1 = [b_rouge["rouge1"], b_rouge["rouge2"], b_rouge["rougeL"]]
iv1 = [i_rouge["rouge1"], i_rouge["rouge2"], i_rouge["rougeL"]]
x1  = np.arange(len(labels1))

bb1 = ax1.bar(x1 - W/2, bv1, W, color=C_BASE, alpha=0.88, zorder=3, edgecolor="white", linewidth=0.6, label="Baseline")
bi1 = ax1.bar(x1 + W/2, iv1, W, color=C_IMPR, alpha=0.92, zorder=3, edgecolor="white", linewidth=0.6, label="Improved")

for bar, val in zip(bb1, bv1):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center", **FAN, color=C_BASE)
for bar, val in zip(bi1, iv1):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center", **FAN, color=C_IMPR)

# delta annotations
for i, (bv, iv, lbl) in enumerate(zip(bv1, iv1, labels1)):
    delta_pct = (iv - bv) / bv * 100
    ax1.annotate(f"+{delta_pct:.0f}%", xy=(x1[i], max(iv, bv) + 0.055),
                 ha="center", fontsize=7.5, color=C_GREEN, fontweight="bold")

ax1.set_xticks(x1); ax1.set_xticklabels(labels1, fontsize=9)
ax1.set_ylim(0, 0.78)
ax1.legend(fontsize=9, framealpha=0.85)

# -- Panel 2: WER + Keyword Coverage ------------------------------------------
ax2 = fig1.add_subplot(gs1[0, 1])
style_ax(ax2, "WER & Keyword Coverage", ylabel="Percentage (%)")

labels2 = ["Word Error\nRate (%)", "Keyword\nCoverage (%)"]
bv2 = [b_wer * 100, b_cov]
iv2 = [i_wer * 100, i_cov]
x2  = np.arange(len(labels2))

bb2 = ax2.bar(x2 - W/2, bv2, W, color=C_BASE, alpha=0.88, zorder=3, edgecolor="white", linewidth=0.6, label="Baseline")
bi2 = ax2.bar(x2 + W/2, iv2, W, color=C_IMPR, alpha=0.92, zorder=3, edgecolor="white", linewidth=0.6, label="Improved")

for bar, val in zip(bb2, bv2):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8, f"{val:.1f}%", ha="center", **FAN, color=C_BASE)
for bar, val in zip(bi2, iv2):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8, f"{val:.1f}%", ha="center", **FAN, color=C_IMPR)

# WER arrow (lower is better), Coverage arrow (higher is better)
arrows = [("down", b_wer*100, i_wer*100, x2[0]), ("up", b_cov, i_cov, x2[1])]
for direction, bval, ival, xi in arrows:
    mid_y = (bval + ival) / 2 + 5
    label = f"-{(bval-ival)/bval*100:.0f}%" if direction == "down" else f"+{(ival-bval)/bval*100:.0f}%"
    color = C_GREEN
    ax2.text(xi, mid_y, label, ha="center", fontsize=7.5, color=color, fontweight="bold")

ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=9)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9, framealpha=0.85)

# -- Panel 3: Feature extraction counts ---------------------------------------
ax3 = fig1.add_subplot(gs1[0, 2])
style_ax(ax3, "NLP Features Extracted\n(Baseline vs. Improved)", ylabel="Count")

labels3 = ["Keywords", "Action Items\nDetected", "Summary\nLength (÷10)"]
bv3 = [b_kw, b_ai, r["summarisation"]["baseline_length_words"] / 10]
iv3 = [i_kw, i_ai, r["summarisation"]["improved_length_words"] / 10]
x3  = np.arange(len(labels3))

bb3 = ax3.bar(x3 - W/2, bv3, W, color=C_BASE, alpha=0.88, zorder=3, edgecolor="white", linewidth=0.6, label="Baseline")
bi3 = ax3.bar(x3 + W/2, iv3, W, color=C_IMPR, alpha=0.92, zorder=3, edgecolor="white", linewidth=0.6, label="Improved")

for bar, val in zip(bb3, bv3):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{val:.0f}" if val >= 1 else "0", ha="center", **FAN, color=C_BASE)
for bar, val in zip(bi3, iv3):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{val:.0f}", ha="center", **FAN, color=C_IMPR)

ax3.set_xticks(x3); ax3.set_xticklabels(labels3, fontsize=8.5)
ax3.set_ylim(0, 18)
ax3.legend(fontsize=9, framealpha=0.85)

fig1.suptitle(
    "AI Speech Intelligence System — Baseline vs. Improved Evaluation\n"
    "CS 5542 | Quiz Challenge 2 | Tina Nguyen  |  Whisper base + BART-large-CNN + DistilBERT + KeyBERT",
    fontsize=11, fontweight="bold", color="#212529", y=1.03,
)
fig1.savefig(OUT_DIR / "comparison_charts.png", dpi=180, bbox_inches="tight", facecolor=C_BG)
print("[OK] comparison_charts.png")
plt.close(fig1)


# ===========================================================================
# FIGURE 2 — Sentiment timeline + Keyword scores
# ===========================================================================
fig2 = plt.figure(figsize=(15, 6), facecolor=C_BG)
gs2  = GridSpec(1, 2, figure=fig2, wspace=0.45)

# -- Panel A: Per-sentence sentiment timeline ---------------------------------
axA = fig2.add_subplot(gs2[0, 0])
style_ax(axA, "Per-Sentence Sentiment (Improved)\n21 Sentences — DistilBERT SST-2", ylabel="Confidence Score")

sents = ia["sentiment"]["per_sentence"]
labels_s = [s["label"] for s in sents]
scores_s = [s["score"] for s in sents]
colors_s = [C_GREEN if l == "POSITIVE" else C_RED for l in labels_s]
x_s = np.arange(len(sents))

axA.bar(x_s, scores_s, color=colors_s, alpha=0.80, zorder=3, edgecolor="white", linewidth=0.4)
axA.axhline(0.5, color="#ADB5BD", linewidth=1.0, linestyle="--", zorder=2)
axA.set_xlim(-0.5, len(sents) - 0.5)
axA.set_ylim(0, 1.08)
axA.set_xlabel("Sentence Index", **FA)
axA.set_xticks(x_s[::2]); axA.set_xticklabels(x_s[::2], fontsize=8)
axA.text(0.02, 0.96, f"Positive: {labels_s.count('POSITIVE')}/{len(labels_s)}  ({ia['sentiment']['positive_ratio']*100:.0f}%)",
         transform=axA.transAxes, fontsize=8.5, color=C_GREEN, fontweight="bold")
axA.text(0.02, 0.89, f"Negative: {labels_s.count('NEGATIVE')}/{len(labels_s)}  ({ia['sentiment']['negative_ratio']*100:.0f}%)",
         transform=axA.transAxes, fontsize=8.5, color=C_RED, fontweight="bold")
p_patch = mpatches.Patch(color=C_GREEN, alpha=0.80, label="POSITIVE")
n_patch = mpatches.Patch(color=C_RED,   alpha=0.80, label="NEGATIVE")
axA.legend(handles=[p_patch, n_patch], fontsize=9, loc="upper right")

# -- Panel B: KeyBERT keyword scores ------------------------------------------
axB = fig2.add_subplot(gs2[0, 1])
style_ax(axB, "Top-12 Keyphrases\nKeyBERT + MMR (Improved)", xlabel="Relevance Score")

kws    = [k["phrase"].title() for k in ia["keywords"]]
kscores= [k["score"] for k in ia["keywords"]]
ypos   = np.arange(len(kws))
kcols  = [C_IMPR if s >= 0.78 else (C_GOLD if s >= 0.72 else "#ADB5BD") for s in kscores]

hbars = axB.barh(ypos, kscores, color=kcols, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
for bar, val in zip(hbars, kscores):
    axB.text(val + 0.004, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
             va="center", fontsize=8, fontweight="bold", color="#343A40")
axB.set_yticks(ypos); axB.set_yticklabels(kws, fontsize=8.5)
axB.set_xlim(0, 0.98)
axB.invert_yaxis()
axB.xaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
axB.spines[["top","right"]].set_visible(False)
leg_patches = [
    mpatches.Patch(color=C_IMPR,    label="High relevance (>= 0.78)"),
    mpatches.Patch(color=C_GOLD,    label="Medium (>= 0.72)"),
    mpatches.Patch(color="#ADB5BD", label="Standard (< 0.72)"),
]
axB.legend(handles=leg_patches, fontsize=8, loc="lower right", framealpha=0.85)

fig2.suptitle("AI Speech Intelligence System — Sentiment Timeline & Keyword Analysis\nCS 5542 | Quiz Challenge 2 | Tina Nguyen",
              fontsize=11, fontweight="bold", color="#212529", y=1.03)
fig2.savefig(OUT_DIR / "sentiment_keywords_chart.png", dpi=180, bbox_inches="tight", facecolor=C_BG)
print("[OK] sentiment_keywords_chart.png")
plt.close(fig2)


# ===========================================================================
# FIGURE 3 — Multi-metric radar chart
# ===========================================================================
fig3, ax_r = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), facecolor=C_BG)
ax_r.set_facecolor("#EAECF0")

radar_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "WER\n(inv.)", "KW\nCoverage", "Action\nItems\n(norm)"]
b_rad = [b_rouge["rouge1"], b_rouge["rouge2"], b_rouge["rougeL"], 1 - b_wer, b_cov/100, b_ai/5]
i_rad = [i_rouge["rouge1"], i_rouge["rouge2"], i_rouge["rougeL"], 1 - i_wer, i_cov/100, i_ai/5]

N = len(radar_labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
b_rad += b_rad[:1]; i_rad += i_rad[:1]; angles += angles[:1]

ax_r.plot(angles, b_rad, "o-", linewidth=2.0, color=C_BASE, label="Baseline", alpha=0.85)
ax_r.fill(angles, b_rad, alpha=0.13, color=C_BASE)
ax_r.plot(angles, i_rad, "o-", linewidth=2.5, color=C_IMPR, label="Improved", alpha=0.95)
ax_r.fill(angles, i_rad, alpha=0.20, color=C_IMPR)

ax_r.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=9.5)
ax_r.set_ylim(0, 1)
ax_r.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_r.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="#6C757D")
ax_r.grid(color="#C8CDD4", linewidth=0.8)
ax_r.spines["polar"].set_color("#C8CDD4")
ax_r.legend(loc="upper right", bbox_to_anchor=(1.38, 1.14), fontsize=10, framealpha=0.9)
ax_r.set_title("Multi-Metric Radar\nBaseline vs. Improved", **FT, pad=22)
fig3.text(0.5, -0.01, "AI Speech Intelligence System | CS 5542 | Tina Nguyen", ha="center", fontsize=9, color="#6C757D")

fig3.savefig(OUT_DIR / "radar_chart.png", dpi=180, bbox_inches="tight", facecolor=C_BG)
print("[OK] radar_chart.png")
plt.close(fig3)


# ===========================================================================
# FIGURE 4 — NEW: WER Error Taxonomy + ROUGE improvement bars
# ===========================================================================
fig4 = plt.figure(figsize=(15, 5.5), facecolor=C_BG)
gs4  = GridSpec(1, 2, figure=fig4, wspace=0.45)

# -- Panel A: Error category breakdown ----------------------------------------
axC = fig4.add_subplot(gs4[0, 0])
style_ax(axC, "Baseline Transcription\nError Taxonomy (11 Errors)", ylabel="Error Count")

errors = r["transcription"]["notable_baseline_errors"]
from collections import Counter
err_types = Counter(e["error_type"] for e in errors)
type_map = {
    "proper_noun":       "Proper Noun",
    "technical_term":    "Technical Term",
    "acronym_split":     "Acronym Split",
    "tool_name":         "Tool Name",
    "missing_punctuation": "Missing Punct.",
    "word_substitution": "Word Substitution"
}
type_labels = [type_map.get(k, k) for k in err_types.keys()]
type_counts = list(err_types.values())
bar_colors  = [C_RED, C_GOLD, "#5DADE2", C_BASE, "#8E44AD", "#E67E22"][:len(type_labels)]

bars_e = axC.bar(range(len(type_labels)), type_counts, color=bar_colors[:len(type_labels)],
                 alpha=0.88, zorder=3, edgecolor="white", linewidth=0.6)
for bar, val in zip(bars_e, type_counts):
    axC.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04, str(val),
             ha="center", **FAN, color="#343A40")
axC.set_xticks(range(len(type_labels)))
axC.set_xticklabels(type_labels, fontsize=8.5, rotation=15, ha="right")
axC.set_ylim(0, 8)
axC.text(0.98, 0.97, f"Total errors: {len(errors)}\nWER Baseline: {b_wer*100:.1f}%\nWER Improved: {i_wer*100:.1f}%",
         transform=axC.transAxes, fontsize=8.5, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor=C_GRID))

# -- Panel B: ROUGE delta % improvement ---------------------------------------
axD = fig4.add_subplot(gs4[0, 1])
style_ax(axD, "ROUGE Score Improvement\n(Improved vs. Baseline, %)", ylabel="Relative Improvement (%)")

rouge_metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
rouge_deltas  = [
    r["summarisation"]["rouge1_delta_pct"],
    r["summarisation"]["rouge2_delta_pct"],
    r["summarisation"]["rougeL_delta_pct"],
]
bar_colors_r = [C_IMPR if d >= 80 else C_GOLD for d in rouge_deltas]

bars_r = axD.bar(rouge_metrics, rouge_deltas, color=bar_colors_r, alpha=0.88,
                 zorder=3, edgecolor="white", linewidth=0.6, width=0.45)
for bar, val in zip(bars_r, rouge_deltas):
    axD.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2.5, f"+{val:.0f}%",
             ha="center", **FAN, color="#212529")

axD.set_ylim(0, 260)
axD.axhline(100, color=C_GOLD, linewidth=1.2, linestyle="--", zorder=2, alpha=0.7)
axD.text(2.4, 103, "100%", fontsize=8, color=C_GOLD, va="bottom")
axD.set_xticklabels(rouge_metrics, fontsize=10)
axD.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
axD.set_axisbelow(True)
axD.text(0.02, 0.97,
         f"Baseline ROUGE-2: {b_rouge['rouge2']:.3f}\nImproved ROUGE-2: {i_rouge['rouge2']:.3f}\nDelta: +{rouge_deltas[1]:.0f}%",
         transform=axD.transAxes, fontsize=8.5, va="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor=C_GRID))

fig4.suptitle("AI Speech Intelligence System — Error Analysis & ROUGE Improvement\nCS 5542 | Quiz Challenge 2 | Tina Nguyen",
              fontsize=11, fontweight="bold", color="#212529", y=1.03)
fig4.savefig(OUT_DIR / "error_analysis_chart.png", dpi=180, bbox_inches="tight", facecolor=C_BG)
print("[OK] error_analysis_chart.png")
plt.close(fig4)

print("\n[DONE] All 5 charts saved to outputs/evaluation/")
print("   comparison_charts.png")
print("   sentiment_keywords_chart.png")
print("   radar_chart.png")
print("   error_analysis_chart.png")
print("   model_translation_chart.png")


# ===========================================================================
# FIGURE 5 -- STT Model Comparison + Translation Output
# ===========================================================================
fig5 = plt.figure(figsize=(16, 5.5), facecolor=C_BG)
gs5  = GridSpec(1, 2, figure=fig5, wspace=0.45)

# -- Panel A: 3-model WER comparison ------------------------------------------
axE = fig5.add_subplot(gs5[0, 0])
style_ax(axE, "STT Model WER Comparison\n(Word Error Rate -- lower is better)", ylabel="WER (%)")

models     = ["Whisper base\n(Baseline prompt)", "Whisper base\n(Improved prompt)", "wav2vec2\n(base-960h)"]
wer_vals   = [b_wer * 100, i_wer * 100, w2v_wer * 100]
bcols      = [C_BASE, C_IMPR, C_WAV]

bars_m = axE.bar(range(len(models)), wer_vals, color=bcols, alpha=0.88,
                 zorder=3, edgecolor="white", linewidth=0.8, width=0.5)
for bar, val in zip(bars_m, wer_vals):
    axE.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f"{val:.1f}%", ha="center", **FAN, color="#212529")

axE.set_xticks(range(len(models)))
axE.set_xticklabels(models, fontsize=9)
axE.set_ylim(0, 16)

# Annotations
axE.annotate("", xy=(1, i_wer*100 + 0.6), xytext=(2, w2v_wer*100 + 0.6),
             arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.5))
axE.text(1.5, max(i_wer*100, w2v_wer*100) + 1.8,
         f"Whisper wins\n-{(w2v_wer - i_wer)/w2v_wer*100:.0f}% WER",
         ha="center", fontsize=8, color=C_GREEN, fontweight="bold")

p1 = mpatches.Patch(color=C_BASE,  alpha=0.88, label="Whisper (baseline prompt)")
p2 = mpatches.Patch(color=C_IMPR,  alpha=0.88, label="Whisper (improved prompt)")
p3 = mpatches.Patch(color=C_WAV,   alpha=0.88, label="wav2vec2-base-960h")
axE.legend(handles=[p1, p2, p3], fontsize=8.5, framealpha=0.85)

# -- Panel B: Translation word counts per language ----------------------------
axF = fig5.add_subplot(gs5[0, 1])
style_ax(axF, "MarianMT Translation Output\n(Word Count by Target Language)", ylabel="Word Count")

trans_data = r["translation"]["languages"]
langs_lbl = [f"{v['language_name']}\n({k.upper()})" for k, v in trans_data.items()]
langs_wc  = [v["word_count_translated"]              for v in trans_data.values()]
lang_t    = [v["inference_s"]                        for v in trans_data.values()]

lang_colors = ["#27AE60", "#2980B9", "#E74C3C"]
bars_l = axF.bar(range(len(langs_lbl)), langs_wc, color=lang_colors, alpha=0.88,
                 zorder=3, edgecolor="white", linewidth=0.6, width=0.5)
for bar, wc, lt in zip(bars_l, langs_wc, lang_t):
    axF.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{wc} words\n({lt:.1f}s)", ha="center", fontsize=8, fontweight="bold", color="#212529")

axF.axhline(r["translation"]["source_word_count"], color=C_GOLD, linewidth=1.5,
            linestyle="--", zorder=2, alpha=0.9)
axF.text(len(langs_lbl) - 0.45, r["translation"]["source_word_count"] + 0.8,
         f"Source: {r['translation']['source_word_count']} words (EN)",
         fontsize=8, color=C_GOLD, fontweight="bold")

axF.set_xticks(range(len(langs_lbl)))
axF.set_xticklabels(langs_lbl, fontsize=9.5)
axF.set_ylim(0, 175)
axF.text(0.02, 0.96, f"Model family: Helsinki-NLP MarianMT",
         transform=axF.transAxes, fontsize=8, color="#495057")
axF.text(0.02, 0.90, f"Task: EN -> ES / FR / VI (summary translation)",
         transform=axF.transAxes, fontsize=8, color="#495057")

fig5.suptitle(
    "AI Speech Intelligence System — STT Model Comparison & Multilingual Translation\n"
    "CS 5542 | Quiz Challenge 2 | Tina Nguyen  |  Whisper vs wav2vec2  |  Helsinki-NLP MarianMT",
    fontsize=11, fontweight="bold", color="#212529", y=1.03,
)
fig5.savefig(OUT_DIR / "model_translation_chart.png", dpi=180, bbox_inches="tight", facecolor=C_BG)
print("[OK] model_translation_chart.png")
plt.close(fig5)
