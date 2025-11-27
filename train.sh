#!/bin/bash
################################################################################
# W2V-BERT 2.0 Training Script
#
# Usage:
#   ./train.sh isizulu              # Train on isiZulu
#   ./train.sh all                  # Train on all 6 languages
#   ./train.sh isizulu --resume     # Resume from checkpoint
################################################################################

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
OFFICIAL_SCRIPT="${SCRIPT_DIR}/run_speech_recognition_ctc_adapter.py"

# --- AUTHENTICATION ---
# Load from .env file if it exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    export $(grep -v '^#' "${SCRIPT_DIR}/.env" | xargs)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Pushing to Hub will fail."
    echo "Please create a .env file with your tokens (see .env.example)"
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not set. Logging to Weights & Biases will be disabled."
fi
# ----------------------

# Ensure official script exists
if [ ! -f "$OFFICIAL_SCRIPT" ]; then
    echo "Error: $OFFICIAL_SCRIPT not found!"
    exit 1
fi

# Function to train a language
train_language() {
    local lang=$1
    local config_file="${CONFIGS_DIR}/${lang}_config.json"
    
    echo "Starting training for $lang..."
    
    # Single GPU for streaming datasets with dynamic padding
    # Multi-GPU + IterableDataset + dynamic padding = incompatible in this accelerate version
    # But single A100 with batch_size=32 + bf16 is still very fast!
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/train_streaming.py" "$config_file"
}

# Main logic
if [ "$1" == "all" ]; then
    for lang in isizulu isixhosa sesotho setswana tshivenda xitsonga; do
        train_language $lang
    done
else
    if [ -z "$1" ]; then
        echo "Usage: ./train.sh <language_name> (e.g., isizulu)"
        exit 1
    fi
    train_language $1
fi
