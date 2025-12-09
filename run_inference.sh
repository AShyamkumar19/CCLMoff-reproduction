#!/bin/bash

# Set paths
# MODEL_PATH="checkpoints/CCLMoff_V1.ckpt"
TEST_DATA="data/standardized/benchmark.tsv"
OUTPUT_DIR="outputs/inference_results"

# Create output directory if not exists
mkdir -p ${OUTPUT_DIR}

# Run inference
echo "🔍 Running inference using model: ${MODEL_PATH}"
python inference.py \
    --input ${TEST_DATA} \
    --output_dir ${OUTPUT_DIR} \
    # --use_gpu

echo "Inference completed. Results saved in ${OUTPUT_DIR}"
