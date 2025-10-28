#!/usr/bin/env python3
"""
Juggernaut XL - Face Variation Generator
Model: RunDiffusion/Juggernaut-XL-v9
Excellent for consistent details and subtle variations
Generates 25 test variations for comparison
"""
import os
import torch
import time
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import random

# Configuration
MODEL_NAME = "RunDiffusion/Juggernaut-XL-v9"
SOURCE_IMAGE = "source/bella_face_source.jpg"
OUTPUT_DIR = "datasets/juggernaut_xl"
NUM_VARIATIONS = 25
STRENGTH = 0.3
RESOLUTION = 1024
STEPS = 30
GUIDANCE = 7.5

# Prompts optimized for Juggernaut XL
PROMPTS = [
    "beautiful woman portrait, natural lighting, high quality, professional photography, detailed face",
    "portrait of a woman, soft dramatic lighting, detailed facial features, photorealistic",
    "beautiful female face, cinematic lighting, 8k uhd, sharp focus, detailed skin texture",
    "woman headshot, studio lighting, professional photo, elegant, detailed eyes",
    "elegant woman portrait, natural beauty, high resolution, detailed features, soft light",
    "beautiful woman, perfect skin detail, natural makeup, professional lighting, sharp",
    "portrait photography, woman, soft focus background, professional, detailed face",
    "professional headshot, woman, neutral background, high quality, detailed features",
]

def main():
    print("=" * 70)
    print("JUGGERNAUT XL - FACE VARIATION GENERATOR")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Variations: {NUM_VARIATIONS}")
    print(f"Specialty: Consistent details & subtle variations")
    print()

    # Check source
    if not os.path.exists(SOURCE_IMAGE):
        print(f"❌ ERROR: Source image not found: {SOURCE_IMAGE}")
        return

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Device: {device}")
    if device == "cpu":
        print("⚠️  WARNING: Running on CPU - this will be SLOW!")
    print()

    # Load source
    print("Loading source image...")
    source_img = Image.open(SOURCE_IMAGE).convert("RGB")
    source_img = source_img.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    print(f"✓ Image loaded: {source_img.size}")
    print()

    # Load pipeline
    print(f"Loading {MODEL_NAME}...")
    print("(First run: downloads ~7GB model)")

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()

    print("✓ Pipeline loaded")
    print()

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate
    print(f"Generating {NUM_VARIATIONS} variations with consistent details...")
    print("-" * 70)

    start_time = time.time()

    for i in range(NUM_VARIATIONS):
        try:
            prompt = random.choice(PROMPTS)

            result = pipe(
                prompt=prompt,
                image=source_img,
                strength=STRENGTH,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
            ).images[0]

            output_path = f"{OUTPUT_DIR}/bella_juggernaut_{i:04d}.jpg"
            result.save(output_path, quality=95)

            elapsed = time.time() - start_time
            avg_per_image = elapsed / (i + 1)
            eta_seconds = avg_per_image * (NUM_VARIATIONS - i - 1)
            eta_mins = eta_seconds / 60

            progress = (i + 1) / NUM_VARIATIONS * 100
            print(f"[{i+1:2d}/{NUM_VARIATIONS}] ({progress:5.1f}%) | ETA: {eta_mins:.1f}m | {output_path}")

        except Exception as e:
            print(f"❌ Error at variation {i}: {str(e)}")
            continue

    total_time = time.time() - start_time
    print("-" * 70)
    print()
    print("=" * 70)
    print("✓ JUGGERNAUT XL GENERATION COMPLETE!")
    print(f"✓ Generated: {NUM_VARIATIONS} consistent variations")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    print(f"✓ Avg per image: {total_time/NUM_VARIATIONS:.1f} seconds")
    print(f"✓ Location: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
