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

### Step 1: Generate Face Variations

Generate 100 augmented face images from the source:

```bash
python scripts/01_img2img.py
```

This will:
- Load `source/bella_face_source.jpg`
- Use SDXL img2img to create variations
- Save 100 images to `datasets/bella_face_aug/`
- Show progress during generation

**Expected output**: 100 face variations (~1-2 hours on GPU)

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

**Option A: Run All Models Sequentially**
```bash
# On RunPod (after git pull)
cd /workspace/aimodels

# Run all 4 models (takes ~10-15 mins on RTX 4090)
python scripts/01_img2img_sdxl.py
python scripts/02_img2img_realistic.py
python scripts/03_img2img_juggernaut.py
python scripts/04_img2img_sd15.py

# Verify and compare results
python scripts/05_verify_compare.py
```

**Option B: Run One at a Time**
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