#!/usr/bin/env python3
"""
Realistic Vision V5.1 - Face Variation Generator
Model: SG161222/Realistic_Vision_V5.1_noVAE
Specialized for photorealistic faces
Generates 25 test variations for comparison
"""
import os
import torch
import time
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import random

# Configuration
MODEL_NAME = "SG161222/Realistic_Vision_V5.1_noVAE"
SOURCE_IMAGE = "source/bella_face_source.jpg"
OUTPUT_DIR = "datasets/realistic_vision"
NUM_VARIATIONS = 25
STRENGTH = 0.3
RESOLUTION = 512  # Realistic Vision is SD 1.5 based
STEPS = 30
GUIDANCE = 7.5

# Prompts optimized for Realistic Vision
PROMPTS = [
    "RAW photo, beautiful woman portrait, 8k uhd, high quality, film grain, natural lighting",
    "portrait photo of a woman, photorealistic, detailed face, soft lighting, professional photography",
    "beautiful female face, photoreal, natural skin texture, cinematic lighting, high resolution",
    "woman headshot, ultra realistic, studio lighting, detailed, sharp focus",
    "elegant woman portrait, photorealistic, natural beauty, professional photo, detailed",
    "photo of beautiful woman, realistic skin, natural makeup, soft light, 8k",
    "portrait photography, realistic woman, professional lighting, detailed face",
    "professional photorealistic headshot, woman, neutral background, high detail",
]

def main():
    print("=" * 70)
    print("REALISTIC VISION V5.1 - FACE VARIATION GENERATOR")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Variations: {NUM_VARIATIONS}")
    print(f"Specialty: Photorealistic faces")
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
    print("(First run: downloads ~2GB model)")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    if device == "cuda":
        pipe = pipe.to(device)

    print("✓ Pipeline loaded")
    print()

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate
    print(f"Generating {NUM_VARIATIONS} photorealistic variations...")
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

            output_path = f"{OUTPUT_DIR}/bella_realistic_{i:04d}.jpg"
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
    print("✓ REALISTIC VISION GENERATION COMPLETE!")
    print(f"✓ Generated: {NUM_VARIATIONS} photorealistic variations")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    print(f"✓ Avg per image: {total_time/NUM_VARIATIONS:.1f} seconds")
    print(f"✓ Location: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
