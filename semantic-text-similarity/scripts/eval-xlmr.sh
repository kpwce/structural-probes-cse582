#!/bin/bash

# Define variables
DATASET="data/semantics/cs_en_hi.csv"
MODEL_DIR="models/batch_size_8"
OUTPUT_DIR="results-xlmr"
ENCODER="xlm-roberta-base"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run evaluation script
python src/eval-en_hi.py \
    --dataset $DATASET \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --encoder $ENCODER

# Print completion message
echo "Evaluation completed. Results are saved in $OUTPUT_DIR."