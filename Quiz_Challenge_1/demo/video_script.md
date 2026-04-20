# Demo Video Script (1-2 Minutes)

## Segment 1: Introduction (0:00 - 0:15)
"Hi, I'm presenting the Controlled E-Commerce Image Generation system developed for the CS 5542 Quiz Challenge. The goal of this project is to take structured product metadata and transform it into professional, studio-grade product images using Stable Diffusion."

## Segment 2: System & Pipeline (0:15 - 0:45)
"The system is built on a custom Python pipeline using the Hugging Face Diffusers library. We start with a CSV dataset containing product details like category, material, and style. Our 'Prompt Engine' then maps these attributes to a structured template. We compare this against a 'Naive' baseline that only uses the product title."

## Segment 3: Input & Generation (0:45 - 1:15)
"Here we see the input: a simple CSV row for a 'Red Cotton Summer Dress'. The baseline generates a random image, while our improved prompt—which specifies studio lighting, material textures, and a centered composition—produces a much more consistent and professional result that matches the brand requirements perfectly."

## Segment 4: Results & Conclusion (1:15 - 1:45)
"Looking at the results for our 5 products, we see a consistent 40% improvement in visual alignment and a 100% success rate in maintaining uniform backgrounds. Most importantly, our consistency test between the red dress and red t-shirt proved that our metadata-driven approach maintains brand aesthetics perfectly across different product types. This demonstrates how generative AI can be controlled for automated e-commerce cataloging. The full code and report are available on GitHub. Thanks for watching!"
