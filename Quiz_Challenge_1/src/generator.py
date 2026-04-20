import os
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────
# Three seeds used to prove output consistency
# ─────────────────────────────────────────────
SEEDS = [42, 100, 200]


class PromptEngine:
    """Converts product metadata into baseline and structured prompts."""

    @staticmethod
    def generate_baseline_prompt(row):
        """Naive prompt: title only. Low control."""
        return str(row["title"])

    @staticmethod
    def generate_improved_prompt(row):
        """
        Structured prompt template:
        [Lighting + Format] + [Color] + [Material] + [Title] + [Style] + [Quality tags]
        """
        return (
            f"Professional studio photography of a {row['color']} {row['material']} {row['title']}, "
            f"{row['style']} aesthetic, centered composition, "
            f"soft cinematic lighting, pure white solid background, "
            f"8k ultra-detailed, photorealistic, sharp focus, no shadows."
        )

    @staticmethod
    def get_negative_prompt():
        """Universal negative prompt to filter artifacts."""
        return (
            "blurry, low quality, distorted, disfigured, extra objects, "
            "text, watermark, logo, cluttered background, low resolution, "
            "grain, noise, oversaturated, cartoon, painting."
        )


class ECommerceGenerator:
    """Stable Diffusion inference engine with seed-controlled generation."""

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.pipe = None

    def load_model(self):
        import torch
        from diffusers import StableDiffusionPipeline
        print(f"Loading {self.model_id} on {self.device} ...")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)
        print("Model loaded.\n")

    def generate(self, prompt, neg_prompt, seed, output_path):
        """Generate one image with a fixed seed for reproducibility."""
        import torch
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt,
            negative_prompt=neg_prompt,
            generator=generator,
            num_inference_steps=40,
        )
        img = result.images[0]
        img.save(output_path)
        print(f"  Saved -> {output_path}")
        return img


def run_pipeline(run_inference=False):
    data_path = os.path.join("data", "products.csv")
    df = pd.read_csv(data_path).dropna(subset=["product_id"])

    engine = PromptEngine()
    neg_prompt = engine.get_negative_prompt()

    print("=" * 60)
    print("CS 5542 Quiz Challenge 1 — Seed Consistency Experiment")
    print(f"Products: {len(df)}  |  Seeds: {SEEDS}")
    print("=" * 60)

    prompt_log = []

    for _, row in df.iterrows():
        p_id = int(row["product_id"])
        baseline = engine.generate_baseline_prompt(row)
        improved = engine.generate_improved_prompt(row)

        print(f"\nProduct {p_id}: {row['title']}")
        print(f"  [Baseline]  {baseline}")
        print(f"  [Improved]  {improved[:80]}...")

        prompt_log.append(
            {
                "product_id": p_id,
                "title": row["title"],
                "baseline_prompt": baseline,
                "improved_prompt": improved,
            }
        )

        if run_inference:
            gen = ECommerceGenerator()
            gen.load_model()

            for seed in SEEDS:
                # Baseline image
                gen.generate(
                    baseline,
                    neg_prompt,
                    seed,
                    os.path.join("outputs", "baseline", f"prod_{p_id}_seed{seed}.png"),
                )
                # Improved image
                gen.generate(
                    improved,
                    neg_prompt,
                    seed,
                    os.path.join("outputs", "improved", f"prod_{p_id}_seed{seed}.png"),
                )

    # Save prompt log for reference
    log_df = pd.DataFrame(prompt_log)
    log_path = os.path.join("evaluation", "prompt_log.csv")
    os.makedirs("evaluation", exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nPrompt log saved -> {log_path}")

    print("\nDone. Set run_inference=True to generate actual images.")


if __name__ == "__main__":
    run_pipeline(run_inference=False)
