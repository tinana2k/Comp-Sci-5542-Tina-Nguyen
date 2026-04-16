# Controlled E-Commerce Product Image Generation using Stable Diffusion

## 1. Overview
This project builds a controlled image generation system for e-commerce products using Stable Diffusion. The system takes structured product metadata from a CSV file and converts it into prompts to generate product images.

The goal is to compare simple (baseline) prompts with structured prompts and evaluate how control mechanisms improve image quality and consistency.

---

## 2. Scenario
This project follows the **E-Commerce Product Image Generation** scenario.

The system:
- Takes product metadata (title, category, color, material, style)
- Generates product images using Stable Diffusion
- Compares naive prompts vs structured prompts
- Evaluates output quality and consistency

---

## 3. Dataset
The dataset is a structured CSV file (`products.csv`) containing product metadata.

### Example fields:
- `product_id`: unique identifier
- `title`: product name
- `category`: product category (e.g., dress, shoes)
- `color`: product color
- `material`: product material
- `style`: product style

This dataset is small and designed for demonstration purposes.

---

## 4. Methodology

### Pipeline:
1. Load product metadata from CSV
2. Convert metadata into prompts
3. Generate images using Stable Diffusion
4. Save outputs for comparison
5. Evaluate results

---

## 5. Prompt Design

### Baseline Prompt (Naive)
A simple prompt using minimal information:
