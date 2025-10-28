#!/usr/bin/env python3
"""
SDXL Base - Face Variation Generator
Model: stabilityai/stable-diffusion-xl-base-1.0
TEST MODE: 5 images with strength=0.8 for high variation testing
"""
import os
import torch
import time
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import random

# Configuration
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
SOURCE_IMAGE = "source/bella_face_source.jpg"
OUTPUT_DIR = "datasets/sdxl_base"
NUM_VARIATIONS = 3  # TEST: Quick 3-image run
STRENGTH = 0.5  # FIXED: Preserve face identity (was 0.8)
RESOLUTION = 1024
STEPS = 30
GUIDANCE = 7.5  # FIXED: Follow prompts better (was 6.0)

# Negative prompt to preserve face identity
NEGATIVE_PROMPT = "ugly, deformed, distorted face, blurry, low quality, mutated, disfigured, bad anatomy, extra limbs"

# Prompts for REAL variations - diverse scenarios and contexts
PROMPTS = [
    "woman at outdoor cafe, golden hour sunlight, candid shot, bokeh background",
    "professional headshot, white background, studio lighting, business attire",
    "woman laughing, colorful urban street background, natural daylight, casual style",
    "close-up portrait, dramatic side lighting, dark moody background, film noir aesthetic",
    "woman in nature, soft morning light, trees in background, peaceful expression",
    "beach portrait, sunset lighting, ocean in background, windswept hair, relaxed",
    "indoor home setting, window light, cozy atmosphere, soft smile, warm tones",
    "nighttime city lights background, bokeh, cool blue tones, elegant evening look",
    "gym/fitness setting, energetic pose, bright lighting, athletic wear, determined expression",
    "art gallery background, sophisticated look, neutral tones, cultural setting",
    "coffee shop interior, casual candid moment, warm lighting, relaxed posture",
    "park bench, autumn leaves background, natural light, contemplative mood",
    "modern office background, professional setting, natural window light, confident pose",
    "library/bookstore background, intellectual vibe, soft lighting, thoughtful expression",
    "rooftop terrace, cityscape background, golden hour, stylish casual outfit",
]

def main():
    print("=" * 70)
    print("SDXL BASE - FACE VARIATION GENERATOR")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Variations: {NUM_VARIATIONS}")
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
        print("   On GPU: ~2-3 minutes | On CPU: ~30+ minutes")
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
    print(f"Generating {NUM_VARIATIONS} variations...")
    print("-" * 70)

    start_time = time.time()

    for i in range(NUM_VARIATIONS):
        try:
            prompt = random.choice(PROMPTS)

            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=source_img,
                strength=STRENGTH,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
            ).images[0]

            output_path = f"{OUTPUT_DIR}/bella_sdxl_{i:04d}.jpg"
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
    print("✓ TEST COMPLETE - FACE-LOCKED VARIATIONS")
    print(f"✓ Generated: {NUM_VARIATIONS} test variations")
    print(f"✓ Settings: strength={STRENGTH}, guidance={GUIDANCE}")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    print(f"✓ Avg per image: {total_time/NUM_VARIATIONS:.1f} seconds")
    print(f"✓ Location: {OUTPUT_DIR}/")
    print()
    print("WHAT TO EXPECT:")
    print("  ✓ Same face maintained (lower strength + negative prompt)")
    print("  ✓ Different backgrounds/settings (diverse prompts)")
    print("  ✓ Varied lighting and moods")
    print()
    print("NEXT: Review images, then run full batch or train LoRA")
    print("=" * 70)

if __name__ == "__main__":
    main()
