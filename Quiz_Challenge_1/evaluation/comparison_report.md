<<<<<<< HEAD
# Prompt Engineering Analysis Report
**CS 5542 Quiz Challenge 1 | Student: Tina Nguyen**

> This report documents measurable, verifiable facts about the prompt strategy.
> No model inference scores are included because Stable Diffusion was not run in this environment.
> Quantitative CLIP-based scoring would require a GPU runtime.

---

## Prompt Complexity Comparison

| ID | Product | Baseline Words | Improved Words | Attributes Controlled |
|---|---|---|---|---|
| 1 | Women Summer Dress | 3 | 28 | 5 |
| 2 | Running Shoes | 2 | 27 | 5 |
| 3 | Coffee Mug | 2 | 27 | 5 |
| 4 | Casual T-Shirt | 2 | 27 | 5 |
| 5 | Ceramic Bowl | 2 | 27 | 5 |
| **Avg** | | **2.2** | **27.2** | **5** |

---

## Key Findings (Objective)

- Structured prompts are on average **25 words longer** than baseline prompts.
- Every structured prompt controls exactly **5 attributes**: color, material, style, lighting, and background.
- Baseline prompts control **1 attribute** only: the product title.
- Seed consistency: the seed directories have been created for seeds 42, 100, 200.
  Run `generator.py` with `run_inference=True` to populate them and compare.

---

## Qualitative Observations (3 Sample Outputs Available)

Three products (Dress, Shoes, Mug) have sample images in `outputs/`. Visual inspection shows:
- Structured prompt outputs have a consistent white background across all 3 products.
- Baseline outputs show varied, unpredictable backgrounds and lighting.
=======
# Prompt Engineering Analysis Report
**CS 5542 Quiz Challenge 1 | Student: Tina Nguyen**

> This report documents measurable, verifiable facts about the prompt strategy.
> No model inference scores are included because Stable Diffusion was not run in this environment.
> Quantitative CLIP-based scoring would require a GPU runtime.

---

## Prompt Complexity Comparison

| ID | Product | Baseline Words | Improved Words | Attributes Controlled |
|---|---|---|---|---|
| 1 | Women Summer Dress | 3 | 28 | 5 |
| 2 | Running Shoes | 2 | 27 | 5 |
| 3 | Coffee Mug | 2 | 27 | 5 |
| 4 | Casual T-Shirt | 2 | 27 | 5 |
| 5 | Ceramic Bowl | 2 | 27 | 5 |
| **Avg** | | **2.2** | **27.2** | **5** |

---

## Key Findings (Objective)

- Structured prompts are on average **25 words longer** than baseline prompts.
- Every structured prompt controls exactly **5 attributes**: color, material, style, lighting, and background.
- Baseline prompts control **1 attribute** only: the product title.
- Seed consistency: the seed directories have been created for seeds 42, 100, 200.
  Run `generator.py` with `run_inference=True` to populate them and compare.

---

## Qualitative Observations (3 Sample Outputs Available)

Three products (Dress, Shoes, Mug) have sample images in `outputs/`. Visual inspection shows:
- Structured prompt outputs have a consistent white background across all 3 products.
- Baseline outputs show varied, unpredictable backgrounds and lighting.
>>>>>>> 6708d2e (Finalize Quiz Challenge 1: 5-product consistency experiment with UMKC-themed PDF report and slides)
- Material-specific tokens (cotton, mesh, ceramic) produce noticeably different surface textures.