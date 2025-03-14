#!/bin/bash

# Define variables
DATASET="data/semantics/semantic_data.csv"
MODEL_DIR="models/batch_size_8"
OUTPUT_DIR="results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run evaluation script
python src/eval.py \
    --dataset $DATASET \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \

# Print completion message
echo "Evaluation completed. Results are saved in $OUTPUT_DIR."