# W2V-BERT 2.0 Fine-tuning for African Languages

This project contains scripts to fine-tune Meta's **Wav2Vec2-BERT 2.0** model on low-resource African languages using the **DSFSI-ANV** dataset. It supports both single-language and multilingual training using streaming datasets to minimize disk usage.

## Features
- **Streaming Data Loading**: efficient handling of large audio datasets without full download.
- **Multilingual Support**: interleave datasets from multiple languages (isiZulu, isiXhosa, Sesotho, Setswana, Tshivenda, Xitsonga).
- **W2V-BERT 2.0**: utilizes the latest self-supervised model with adapter fine-tuning.
- **SpecAugment**: includes time and feature masking for robust training.
- **Experiment Tracking**: integrated with Weights & Biases (WandB).

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory (see `.env.example`):
   ```bash
   HF_TOKEN=your_huggingface_token
   WANDB_API_KEY=your_wandb_key
   WANDB_ENTITY=dsfsi
   WANDB_PROJECT=dsfsi-asr
   ```

## Usage

### 1. Multilingual Training (Recommended)
To train a single model on all 6 languages simultaneously:

```bash
./train_multilingual.sh
```
*   **Config**: `configs/multilingual_config.json`
*   **Note**: This script is configured to use all available GPUs (e.g., 2x A100). If you need to restrict it, use `CUDA_VISIBLE_DEVICES`.

### 2. Single Language Training
To train on a specific language (e.g., isiZulu):

```bash
./train.sh isizulu
```
*   **Available Languages**: `isizulu`, `isixhosa`, `sesotho`, `setswana`, `tshivenda`, `xitsonga`.
*   **Config**: `configs/<language>_config.json`

### 3. Dry Run
To verify the setup with a small subset of data:

```bash
python3 train_streaming.py configs/dry_run_multilingual.json
```

## Configuration
The training is controlled by JSON configuration files in the `configs/` directory. Key parameters include:

- `dataset_config_name`: Language code(s) to load. Can be a list for multilingual.
- `max_steps`: Total training steps.
- `learning_rate`: Learning rate (default: 5e-5).
- `add_adapter`: Whether to train the adapter layer (default: true).
- `mask_time_prob` / `mask_feature_prob`: SpecAugment masking probabilities.

## VM Deployment Notes
- The scripts are optimized for **A100 GPUs**.
- For multi-GPU training, the scripts use `Trainer` which handles data parallelism.
- Ensure `soundfile` backend is used for audio loading (handled in script).
