#!/bin/bash
# LoRA Training Script for Bella Face
# Using Kohya_ss with SDXL

set -e  # Exit on error

echo "======================================================================"
echo "BELLA FACE LORA TRAINING - KOHYA_SS"
echo "======================================================================"

# Configuration
DATASET_DIR="datasets/bella_face_aug"
OUTPUT_DIR="loras"
OUTPUT_NAME="bella_face"
RANK=64
NETWORK_ALPHA=32
LEARNING_RATE=1.0
OPTIMIZER="prodigy"
EPOCHS=15
RESOLUTION=1024
BATCH_SIZE=1

echo "Configuration:"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR/$OUTPUT_NAME.safetensors"
echo "  Rank: $RANK"
echo "  Optimizer: $OPTIMIZER"
echo "  Epochs: $EPOCHS"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ ERROR: Dataset directory not found: $DATASET_DIR"
    echo "   Please run scripts/01_img2img.py first"
    exit 1
fi

# Count images in dataset
NUM_IMAGES=$(find "$DATASET_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
echo "✓ Found $NUM_IMAGES images in dataset"

if [ "$NUM_IMAGES" -lt 10 ]; then
    echo "⚠️  WARNING: Dataset has fewer than 10 images. Recommended: 100+"
fi

echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Kohya_ss is available
if ! command -v accelerate &> /dev/null; then
    echo "❌ ERROR: Kohya_ss (accelerate) not found in PATH"
    echo "   Please install Kohya_ss:"
    echo "   git clone https://github.com/bmaltais/kohya_ss.git"
    echo "   cd kohya_ss && pip install -r requirements.txt"
    exit 1
fi

echo "✓ Kohya_ss detected"
echo ""

# Create a simple dataset config
echo "Creating dataset configuration..."
cat > /tmp/bella_dataset_config.toml <<EOF
[general]
enable_bucket = true
resolution = $RESOLUTION
min_bucket_reso = 256
max_bucket_reso = 1024

[[datasets]]
[[datasets.subsets]]
image_dir = '$DATASET_DIR'
num_repeats = 1
EOF

echo "✓ Dataset config created"
echo ""

# Training command
echo "======================================================================"
echo "STARTING LORA TRAINING"
echo "======================================================================"
echo ""

accelerate launch --num_cpu_threads_per_process=2 \
  sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="/tmp/bella_dataset_config.toml" \
  --output_dir="$OUTPUT_DIR" \
  --output_name="$OUTPUT_NAME" \
  --save_model_as=safetensors \
  --prior_loss_weight=1.0 \
  --max_train_steps=$((NUM_IMAGES * EPOCHS / BATCH_SIZE)) \
  --learning_rate="$LEARNING_RATE" \
  --optimizer_type="$OPTIMIZER" \
  --text_encoder_lr=1.0 \
  --unet_lr=1.0 \
  --network_dim="$RANK" \
  --network_alpha="$NETWORK_ALPHA" \
  --network_module=networks.lora \
  --train_batch_size="$BATCH_SIZE" \
  --resolution="$RESOLUTION,$RESOLUTION" \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --max_data_loader_n_workers=1 \
  --save_every_n_epochs=5 \
  --logging_dir="logs" \
  --log_prefix="bella_face" \
  --noise_offset=0.0357 \
  --adaptive_noise_scale=0.00357 \
  --caption_extension=".txt" \
  --keep_tokens=1

echo ""
echo "======================================================================"
echo "✓ TRAINING COMPLETE!"
echo "======================================================================"
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME.safetensors"
echo ""
echo "To use this LoRA:"
echo "  1. Copy to your SD WebUI lora folder"
echo "  2. Use in prompt: <lora:bella_face:0.8>"
echo "  3. Adjust weight (0.0-1.0) as needed"
echo "======================================================================"
