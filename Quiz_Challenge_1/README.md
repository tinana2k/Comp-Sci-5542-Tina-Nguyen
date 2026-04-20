# Controlled E-Commerce Product Image Generation (Stable Diffusion)
**CS 5542 — Quiz Challenge 1 | Author: Tina Nguyen**

## 1. Overview
This project builds a **metadata-driven image generation pipeline** using Stable Diffusion v1.5. It converts structured e-commerce product metadata into professional studio-style image prompts and demonstrates how prompt control affects output quality and consistency.

> **Note:** Stable Diffusion inference requires a GPU. The pipeline code is fully implemented and ready to run. Sample outputs (including placeholders) are included for demonstration; full generation of all 30 images (5 products × 2 modes × 3 seeds) requires running `generator.py` with `run_inference=True` on a CUDA-enabled machine.

---

## 2. Prompt Engineering Results (Real Metrics)

The core contribution is the `PromptEngine` class in `src/generator.py`, which translates product metadata into structured prompts. The table shows real, computed values from running the engine on the dataset:

| Product | Baseline Words | Improved Words | Attributes Controlled |
|---|---|---|---|
| Women Summer Dress | 3 | 28 | 5 |
| Running Shoes      | 2 | 27 | 5 |
| Coffee Mug         | 2 | 27 | 5 |
| Casual T-Shirt     | 2 | 27 | 5 |
| Ceramic Bowl       | 2 | 27 | 5 |
| **Average**        | **2.2** | **27.2** | **5** |

- Structured prompts are on average **25 words longer**, controlling color, material, style, lighting, and background.
- Baseline prompts control **1 attribute only** — the product title.

See `evaluation/prompt_complexity_chart.png` and `evaluation/attribute_coverage_chart.png` for visual charts.

---

## 3. Qualitative Visual Observations

For the products with generated sample images (`outputs/`), visual inspection shows:

| Aspect | Baseline | Structured |
|---|---|---|
| Background | Unpredictable | Consistent white |
| Lighting | Ambient/random | Soft cinematic studio |
| Material | Generic | Token-specific (cotton, mesh, ceramic) |

See `evaluation/all_comparisons.png` for the side-by-side comparison.

---

## 4. Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.x | Pipeline scripting |
| PyTorch | Deep learning runtime |
| Hugging Face `diffusers` | Stable Diffusion inference |
| `transformers`, `accelerate`, `safetensors` | Model loading |
| Pandas | Dataset loading & prompt log |
| Matplotlib / Pillow | Charts & image processing |

---

## 5. Use of AI Tools (Disclosure)
Per assignment transparency requirements:
- **Development**: Antigravity AI assisted in building the pipeline, evaluator, and documentation.
- All analytical conclusions and design decisions were reviewed and approved by the student.

---

## 6. Repository Layout
```
Quiz_Challenge_1/
├── data/               products.csv + dataset_description.md
├── src/                generator.py, evaluator.py
├── notebooks/          ecommerce_generation.ipynb
├── outputs/
│   ├── baseline/       prod_N_baseline.png + seed_*/
│   └── improved/       prod_N_improved.png + seed_42/ seed_100/ seed_200/
├── evaluation/         results.md, comparison_report.md,
│                       prompt_complexity_chart.png, attribute_coverage_chart.png,
│                       all_comparisons.png, prompt_log.csv, report.pdf
├── slides/             presentation.pdf, presentation.html
├── demo/               video_script.md
└── tests/              test_pipeline.py (Unit tests)
```

---

## 7. How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate all prompts and inspect (no GPU needed)
python src/generator.py

# 3. Run full inference (requires CUDA GPU)
#    Open src/generator.py and set: run_pipeline(run_inference=True)

# 4. Regenerate evaluation charts and reports
python src/evaluator.py

# 5. Run unit tests
python -m pytest tests/test_pipeline.py

# 6. Explore interactively
jupyter notebook notebooks/ecommerce_generation.ipynb
```
