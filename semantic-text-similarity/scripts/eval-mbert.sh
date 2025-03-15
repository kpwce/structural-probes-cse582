#!/bin/bash

# Define variables
DATASET="data/semantics/semantic_data.csv"
MODEL_DIR="models/batch_size_32_epochs_5"
OUTPUT_DIR="results-mbert"
ENCODER="bert-base-multilingual-uncased"

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