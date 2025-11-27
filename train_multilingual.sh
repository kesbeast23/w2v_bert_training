#!/bin/bash
################################################################################
# W2V-BERT 2.0 Multilingual Training Script
#
# Usage:
#   ./train_multilingual.sh
################################################################################

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/configs/multilingual_config.json"

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

echo "Starting multilingual training..."
echo "Config: $CONFIG_FILE"

# Run training
# NOTE: The original script recommended single GPU for streaming datasets due to accelerate compatibility.
# If you want to use both A100s, you might need to use 'accelerate launch' or ensure your environment supports it.
# We are removing the hardcoded CUDA_VISIBLE_DEVICES=0 to allow you to control this via the environment.
python3 "${SCRIPT_DIR}/train_streaming.py" "$CONFIG_FILE"
