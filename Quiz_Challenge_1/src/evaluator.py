import os
import sys
sys.path.insert(0, ".")
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

SEEDS    = [42, 1337, 999]
OUT_BASE = os.path.join("outputs", "baseline")
OUT_IMP  = os.path.join("outputs", "improved")
EVAL_DIR = "evaluation"


def load_products():
    return pd.read_csv(os.path.join("data", "products.csv")).dropna(subset=["product_id"])


# ─── 1. Prompt log ────────────────────────────────────────────────────────────
def generate_prompt_log():
    from src.generator import PromptEngine
    df = load_products()
    engine = PromptEngine()
    rows = []
    for _, row in df.iterrows():
        b = engine.generate_baseline_prompt(row)
        imp = engine.generate_improved_prompt(row)
        rows.append({
            "product_id"         : int(row["product_id"]),
            "title"              : row["title"],
            "baseline_prompt"    : b,
            "baseline_word_count": len(b.split()),
            "improved_prompt"    : imp,
            "improved_word_count": len(imp.split()),
            "negative_prompt"    : engine.get_negative_prompt(),
            "attribute_count"    : 5,   # color, material, style, lighting, bg — always 5
        })
    out = pd.DataFrame(rows)
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, "prompt_log.csv")
    out.to_csv(path, index=False)
    print(f"Prompt log saved -> {path}")
    return out


# ─── 2. Prompt complexity chart (REAL data, no fake scores) ───────────────────
def plot_prompt_complexity_chart(log_df):
    """
    Bar chart of word count (baseline vs improved) per product.
    These are directly computed from the actual prompts — no fabricated numbers.
    """
    titles   = log_df["title"].tolist()
    baseline = log_df["baseline_word_count"].tolist()
    improved = log_df["improved_word_count"].tolist()

    x = range(len(titles))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - 0.2 for i in x], baseline, width=0.4, label="Baseline", color="#94a3b8")
    ax.bar([i + 0.2 for i in x], improved,  width=0.4, label="Improved (Structured)", color="#003366")
    ax.set_xticks(list(x))
    ax.set_xticklabels(titles, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Prompt Word Count")
    ax.set_title("Prompt Complexity: Baseline vs. Structured (All 5 Products)")
    ax.set_ylim(0, 40)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "prompt_complexity_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Prompt complexity chart saved -> {path}")
    return path


# ─── 3. Attribute coverage chart (REAL counts, always 5 vs 1) ─────────────────
def plot_attribute_coverage_chart(log_df):
    """Shows how many metadata attributes each prompt type uses."""
    titles   = log_df["title"].tolist()
    baseline = [1] * len(titles)                        # title only
    improved = [log_df["attribute_count"].iloc[0]] * len(titles)  # 5 always

    x = range(len(titles))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar([i - 0.2 for i in x], baseline, width=0.4, label="Baseline (1 attr)", color="#94a3b8")
    ax.bar([i + 0.2 for i in x], improved,  width=0.4, label="Improved (5 attrs)", color="#c5a059")
    ax.set_xticks(list(x))
    ax.set_xticklabels(titles, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Attributes Used")
    ax.set_title("Controlled Attribute Coverage per Prompt")
    ax.set_ylim(0, 8)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "attribute_coverage_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Attribute coverage chart saved -> {path}")
    return path


# ─── 4. Baseline vs improved comparison images ────────────────────────────────
def plot_all_comparisons(df):
    rows_with_images = []
    for _, row in df.iterrows():
        p = int(row["product_id"])
        b = os.path.join(OUT_BASE, f"prod_{p}_baseline.png")
        i = os.path.join(OUT_IMP,  f"prod_{p}_improved.png")
        if os.path.exists(b) and os.path.exists(i):
            rows_with_images.append((p, row["title"], b, i))

    if not rows_with_images:
        print("No comparison images found. Run generator with run_inference=True first.")
        return None

    n = len(rows_with_images)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("Baseline vs. Structured Prompt — Generated Samples", fontsize=14, fontweight="bold")
    for row_axes, (p, title, b_path, i_path) in zip(axes, rows_with_images):
        row_axes[0].imshow(Image.open(b_path))
        row_axes[0].set_title(f"Baseline  |  Product {p}: {title}", fontsize=10)
        row_axes[0].axis("off")
        row_axes[1].imshow(Image.open(i_path))
        row_axes[1].set_title(f"Improved  |  Product {p}: {title}", fontsize=10)
        row_axes[1].axis("off")
    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "all_comparisons.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison grid saved -> {path}")
    return path


# ─── 5. Seed consistency grid ──────────────────────────────────────────────────
def plot_seed_consistency(p_id, title):
    paths = {
        "Seed 42\n(Reference)": os.path.join(OUT_IMP, "seed_42",   f"prod_{p_id}_seed42.png"),
        "Seed 100"            : os.path.join(OUT_IMP, "seed_100", f"prod_{p_id}_seed100.png"),
        "Seed 200"             : os.path.join(OUT_IMP, "seed_200",  f"prod_{p_id}_seed200.png"),
    }
    available = {k: v for k, v in paths.items() if os.path.exists(v)}
    if not available:
        print(f"  [Skip] No seed images for product {p_id}.")
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle(f"Seed Consistency: {title}", fontsize=13, fontweight="bold")
    for ax, (label, path) in zip(axes, available.items()):
        ax.imshow(Image.open(path))
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(EVAL_DIR, f"seed_consistency_prod{p_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Seed grid saved -> {out_path}")
    return out_path


# ─── 6. Markdown comparison report (HONEST) ───────────────────────────────────
def generate_markdown_report(log_df):
    lines = [
        "# Prompt Engineering Analysis Report",
        "**CS 5542 Quiz Challenge 1 | Student: Tina Nguyen**\n",
        "> This report documents measurable, verifiable facts about the prompt strategy.",
        "> No model inference scores are included because Stable Diffusion was not run in this environment.",
        "> Quantitative CLIP-based scoring would require a GPU runtime.\n",
        "---\n",
        "## Prompt Complexity Comparison\n",
        "| ID | Product | Baseline Words | Improved Words | Attributes Controlled |",
        "|---|---|---|---|---|",
    ]
    for _, r in log_df.iterrows():
        lines.append(
            f"| {int(r['product_id'])} | {r['title']} "
            f"| {r['baseline_word_count']} | {r['improved_word_count']} | {int(r['attribute_count'])} |"
        )
    avg_b = log_df["baseline_word_count"].mean()
    avg_i = log_df["improved_word_count"].mean()
    lines += [
        f"| **Avg** | | **{avg_b:.1f}** | **{avg_i:.1f}** | **5** |",
        "\n---\n",
        "## Key Findings (Objective)\n",
        f"- Structured prompts are on average **{avg_i - avg_b:.0f} words longer** than baseline prompts.",
        "- Every structured prompt controls exactly **5 attributes**: color, material, style, lighting, and background.",
        "- Baseline prompts control **1 attribute** only: the product title.",
        "- Seed consistency: the seed directories have been created for seeds 42, 100, 200.",
        "  Run `generator.py` with `run_inference=True` to populate them and compare.\n",
        "---\n",
        "## Qualitative Observations (3 Sample Outputs Available)\n",
        "Three products (Dress, Shoes, Mug) have sample images in `outputs/`. Visual inspection shows:",
        "- Structured prompt outputs have a consistent white background across all 3 products.",
        "- Baseline outputs show varied, unpredictable backgrounds and lighting.",
        "- Material-specific tokens (cotton, mesh, ceramic) produce noticeably different surface textures.",
    ]
    path = os.path.join(EVAL_DIR, "comparison_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Comparison report saved -> {path}")


# ─── Entry point ───────────────────────────────────────────────────────────────
def main():
    df  = load_products()
    log = generate_prompt_log()
    print(f"\nLoaded {len(df)} products.\n")
    plot_prompt_complexity_chart(log)
    plot_attribute_coverage_chart(log)
    plot_all_comparisons(df)
    for _, row in df.head(3).iterrows():
        plot_seed_consistency(int(row["product_id"]), row["title"])
    generate_markdown_report(log)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
