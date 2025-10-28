# Bella AI Influencer - Face Model Training

AI model training pipeline for creating a consistent AI influencer character using Stable Diffusion XL and LoRA fine-tuning.

## Project Goal

Create "Bella", an AI influencer with a consistent face across all generated content. This is accomplished through:

1. **Source Image**: Starting with a single high-quality face image
2. **Data Augmentation**: Generating 100+ variations using SDXL img2img
3. **LoRA Training**: Fine-tuning a lightweight model adapter
4. **Face Swapping**: Future integration with body images for full-body content

## Directory Structure

```
aimodels/
├── source/
│   └── bella_face_source.jpg          # Original source face image
├── datasets/
│   ├── bella_face_aug/                # 100+ augmented face variations (original)
│   ├── sdxl_base/                     # SDXL Base test outputs (25 images)
│   ├── realistic_vision/              # Realistic Vision V5.1 test outputs
│   ├── juggernaut_xl/                 # Juggernaut XL test outputs
│   ├── sd15/                          # SD 1.5 baseline test outputs
│   └── bella_body_cropped/            # Future: body images for face swap
├── loras/
│   └── bella_face.safetensors         # Trained LoRA model
├── generated/
│   ├── test_swaps/                    # Test outputs
│   └── verification_report.txt        # Dataset quality report
├── reports/
│   └── model_comparison.txt           # Model comparison analysis
├── scripts/
│   ├── 01_img2img.py                  # Original: Generate 100 variations (SDXL Refiner)
│   ├── 01_img2img_sdxl.py             # NEW: SDXL Base test (25 variations)
│   ├── 02_img2img_realistic.py        # NEW: Realistic Vision test
│   ├── 03_img2img_juggernaut.py       # NEW: Juggernaut XL test
│   ├── 04_img2img_sd15.py             # NEW: SD 1.5 baseline test
│   ├── 05_verify_compare.py           # NEW: Compare all models
│   ├── 02_train_lora.sh               # Train LoRA model
│   └── 03_verify_images.py            # Verify dataset integrity
└── README.md
```

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install torch torchvision
pip install diffusers transformers accelerate
pip install Pillow

# For LoRA training (optional - Kohya_ss)
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss && pip install -r requirements.txt
```

### Step 1: Test Models & Choose Best (RECOMMENDED START HERE!)

**⚠️ IMPORTANT**: The original `01_img2img.py` uses SDXL Refiner (meant for upscaling, not variations). **Skip it** and use the model comparison scripts instead!

**Test 4 models with 25 images each to find the best:**

```bash
# On RunPod GPU - Run all 4 models (~10-15 mins total)
cd /workspace/aimodels

python scripts/01_img2img_sdxl.py          # SDXL Base - balanced
python scripts/02_img2img_realistic.py     # Realistic Vision - photoreal
python scripts/03_img2img_juggernaut.py    # Juggernaut XL - consistent
python scripts/04_img2img_sd15.py          # SD 1.5 - fast baseline
```

**Expected output**: 25 varied images per model with different backgrounds, lighting, and expressions

**Want to run them in parallel? See "Run Multiple Models at Once" section below.**

### Step 2: Verify Dataset

Check dataset quality and integrity:

```bash
python scripts/03_verify_images.py
```

This will:
- Scan all images in `datasets/bella_face_aug/`
- Check for corrupted files
- Report resolutions and formats
- Save detailed report to `generated/verification_report.txt`

**Expected output**: Verification report showing 100 valid images

### Step 3: Train LoRA Model

Train a custom LoRA model on your dataset:

```bash
./scripts/02_train_lora.sh
```

This will:
- Use Kohya_ss with SDXL base model
- Train with Rank 64, Prodigy optimizer
- Run for 15 epochs
- Output to `loras/bella_face.safetensors`

**Expected output**: LoRA model file (~150MB, 2-4 hours training on GPU)

### Step 4: Use Your LoRA

Copy the trained LoRA to your Stable Diffusion WebUI:

```bash
# Copy to SD WebUI LoRA folder
cp loras/bella_face.safetensors /path/to/stable-diffusion-webui/models/Lora/

# Use in prompts
# "beautiful woman portrait <lora:bella_face:0.8>"
```

Adjust the weight (0.0-1.0) to control how strongly the LoRA affects generation.

---

## Model Comparison Testing (NEW!)

Before committing to 100+ images with one model, test multiple models with 25 variations each to find the best fit for Bella's face.

### Available Models for Testing

| Model | Script | Resolution | Specialty | Speed |
|-------|--------|------------|-----------|-------|
| **SDXL Base** | `01_img2img_sdxl.py` | 1024x1024 | Balanced quality/consistency | Medium |
| **Realistic Vision V5.1** | `02_img2img_realistic.py` | 512x512 | Photorealistic faces | Fast |
| **Juggernaut XL** | `03_img2img_juggernaut.py` | 1024x1024 | Consistent details | Medium |
| **SD 1.5 Baseline** | `04_img2img_sd15.py` | 512x512 | Fast baseline | Fastest |

### Run Model Comparison Tests

**Option A: Run All Models in Parallel (FASTEST - 3-5 mins!)**

Your RTX 4090 has 24GB VRAM - run multiple models at once:

```bash
# On RunPod - Open 4 terminal tabs (or use tmux/screen)
cd /workspace/aimodels

# Terminal 1:
python scripts/01_img2img_sdxl.py &

# Terminal 2:
python scripts/02_img2img_realistic.py &

# Terminal 3:
python scripts/03_img2img_juggernaut.py &

# Terminal 4:
python scripts/04_img2img_sd15.py &

# Wait for all to finish (monitor with: jobs)
wait

# Verify and compare results
python scripts/05_verify_compare.py
```

**Or run 2 at once (safer for VRAM):**
```bash
# Run the two XL models first (heavier)
python scripts/01_img2img_sdxl.py & python scripts/03_img2img_juggernaut.py & wait

# Then run the two 1.5 models (lighter)
python scripts/02_img2img_realistic.py & python scripts/04_img2img_sd15.py & wait

# Compare
python scripts/05_verify_compare.py
```

**Option B: Run All Models Sequentially (SAFEST - 10-15 mins)**
```bash
# On RunPod
cd /workspace/aimodels

# One after another
python scripts/01_img2img_sdxl.py
python scripts/02_img2img_realistic.py
python scripts/03_img2img_juggernaut.py
python scripts/04_img2img_sd15.py

# Verify and compare results
python scripts/05_verify_compare.py
```

**Option C: Run One at a Time**
```bash
# Test SDXL Base first
python scripts/01_img2img_sdxl.py

# Check output
ls datasets/sdxl_base/

# If good, continue with others...
```

### Compare Results

After running tests:

```bash
# View comparison report
cat reports/model_comparison.txt

# Manually review images in each folder
ls datasets/sdxl_base/
ls datasets/realistic_vision/
ls datasets/juggernaut_xl/
ls datasets/sd15/
```

**The report includes:**
- Image counts per model
- File sizes and resolutions
- Corruption checks
- Side-by-side statistics
- Recommendations

### Choose Your Model

Based on results:

1. **Best visual quality?** → Use that model for full 100+ generation
2. **Most consistent?** → Prioritize for LoRA training
3. **Fastest with good results?** → Good for iteration

**Example: Once you choose Juggernaut XL as best:**
```bash
# Modify the original script to use Juggernaut
# Or regenerate 100 images with:
# python scripts/03_img2img_juggernaut.py (edit NUM_VARIATIONS=100)
```

---

## Training Configuration

**LoRA Training Settings** (in `scripts/02_train_lora.sh`):

- **Base Model**: stabilityai/stable-diffusion-xl-base-1.0
- **Rank**: 64 (model complexity)
- **Network Alpha**: 32
- **Optimizer**: Prodigy (adaptive learning rate)
- **Learning Rate**: 1.0 (initial)
- **Epochs**: 15
- **Resolution**: 1024x1024
- **Batch Size**: 1
- **Precision**: FP16 (mixed precision)

**Img2img Settings** (in `scripts/01_img2img.py`):

- **Strength**: 0.3 (transformation amount)
- **Steps**: 30 (inference steps)
- **Guidance Scale**: 7.5
- **Variations**: 100 images

## Next Steps

### Phase 1: Face Model (Current)
- [x] Generate 100+ face variations
- [x] Train face LoRA model
- [x] Verify dataset quality

### Phase 2: Body Integration (Planned)
- [ ] Download/generate body pose images
- [ ] Crop and prepare body dataset
- [ ] Implement face swap pipeline
- [ ] Generate full-body influencer content

### Phase 3: Content Generation (Future)
- [ ] Create Instagram-ready posts
- [ ] Generate TikTok-style content
- [ ] Batch generation scripts
- [ ] Style consistency tools

## Troubleshooting

**Out of Memory Error**
```bash
# Reduce batch size in 02_train_lora.sh
BATCH_SIZE=1  # Already minimum

# Or reduce resolution
RESOLUTION=768
```

**Corrupted Images**
```bash
# Remove corrupted files listed in verification report
python scripts/03_verify_images.py
# Delete files marked as corrupted
# Re-run img2img to regenerate
```

**LoRA Not Working**
```bash
# Check LoRA weight (try 0.5-1.0)
# Ensure LoRA is in correct folder
# Try different prompts
# Verify SDXL base model compatibility
```

## Resources

- [Stable Diffusion XL](https://github.com/Stability-AI/generative-models)
- [Kohya_ss LoRA Training](https://github.com/bmaltais/kohya_ss)
- [Diffusers Library](https://github.com/huggingface/diffusers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

MIT License - Feel free to use for your own AI influencer projects!

---

**Generated**: 2025-10-28
**Status**: Phase 1 - Face Model Training
**Version**: v2.0