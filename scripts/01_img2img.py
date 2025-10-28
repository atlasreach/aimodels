#!/usr/bin/env python3
"""
Generate 100 face variations using SDXL img2img
"""
import os
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import random

# Configuration
SOURCE_IMAGE = "source/bella_face_source.jpg"
OUTPUT_DIR = "datasets/bella_face_aug"
NUM_VARIATIONS = 100
STRENGTH = 0.3  # How much to transform (0.0-1.0)

# Prompts for variations
PROMPTS = [
    "beautiful woman portrait, natural lighting, high quality, professional photography",
    "portrait of a woman, soft lighting, detailed face",
    "beautiful female face, cinematic lighting, 8k uhd",
    "woman headshot, studio lighting, professional photo",
    "elegant woman portrait, natural beauty, high resolution",
    "beautiful woman, perfect skin, natural makeup",
    "portrait photography, woman, soft focus background",
    "professional headshot, woman, neutral background",
]

def main():
    print("=" * 60)
    print("BELLA FACE AUGMENTATION - IMG2IMG VARIATION GENERATOR")
    print("=" * 60)

    # Check source image exists
    if not os.path.exists(SOURCE_IMAGE):
        print(f"❌ ERROR: Source image not found at {SOURCE_IMAGE}")
        return

    print(f"✓ Source image: {SOURCE_IMAGE}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"✓ Target variations: {NUM_VARIATIONS}")
    print(f"✓ Transformation strength: {STRENGTH}")
    print()

    # Load source image
    print("Loading source image...")
    source_img = Image.open(SOURCE_IMAGE).convert("RGB")
    # Resize to optimal SDXL size (1024x1024)
    source_img = source_img.resize((1024, 1024), Image.Resampling.LANCZOS)
    print(f"✓ Image loaded and resized to {source_img.size}")
    print()

    # Load SDXL pipeline
    print("Loading Stable Diffusion XL img2img pipeline...")
    print("(This may take a few minutes on first run)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()

    print("✓ Pipeline loaded successfully")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate variations
    print(f"Generating {NUM_VARIATIONS} variations...")
    print("-" * 60)

    for i in range(NUM_VARIATIONS):
        try:
            # Select random prompt
            prompt = random.choice(PROMPTS)

            # Generate variation
            result = pipe(
                prompt=prompt,
                image=source_img,
                strength=STRENGTH,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # Save image
            output_path = f"{OUTPUT_DIR}/bella_face_{i:04d}.jpg"
            result.save(output_path, quality=95)

            # Print progress
            progress = (i + 1) / NUM_VARIATIONS * 100
            print(f"[{i+1:3d}/{NUM_VARIATIONS}] ({progress:5.1f}%) - Saved: {output_path}")

        except Exception as e:
            print(f"❌ Error generating variation {i}: {str(e)}")
            continue

    print("-" * 60)
    print()
    print("=" * 60)
    print("✓ AUGMENTATION COMPLETE!")
    print(f"✓ Generated {NUM_VARIATIONS} face variations")
    print(f"✓ Output location: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
