# W2V-BERT 2.0 & MMS Fine-tuning for African Languages

This project contains scripts to fine-tune **Wav2Vec2-BERT 2.0** and **MMS (Massively Multilingual Speech)** models on low-resource African languages using the **DSFSI-ANV** dataset. It supports both single-language and multilingual training using streaming datasets to minimize disk usage.

## Models Supported

| Model | Script | Trainable Params | Best For |
|-------|--------|-----------------|----------|
| **W2V-BERT 2.0** | `train_streaming.py` | ~580M (full) | High-resource, best accuracy |
| **MMS Adapters** | `train_streaming_mms.py` | ~96K (0.01%) | Low-resource, fast training |

## Features
- **Streaming Data Loading**: efficient handling of large audio datasets without full download.
- **Multilingual Support**: interleave datasets from multiple languages (isiZulu, isiXhosa, Sesotho, Setswana, Tshivenda, Xitsonga).
- **W2V-BERT 2.0**: utilizes the latest self-supervised model with adapter fine-tuning.
- **MMS Adapters**: memory-efficient adapter training for 1000+ languages.
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

### Option 1: MMS Adapter Training (Recommended for Low-Resource)

MMS adapter training is **much faster** and **more memory efficient** - only 0.01% of parameters are trained!

**Single Language (isiZulu):**
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_streaming_mms.py configs/mms_zul_config.json
```

**Multilingual (6 languages):**
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_streaming_mms.py configs/mms_multilingual_config.json
```

### Option 2: W2V-BERT 2.0 Full Fine-tuning

For maximum accuracy when you have more data and compute:

**Multilingual Training:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_streaming.py configs/multilingual_config.json
```

**Single Language Training:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_streaming.py configs/isizulu_config.json
```

### Option 3: Using Shell Scripts

```bash
./train_multilingual.sh
```
*   **Config**: `configs/multilingual_config.json`
*   **Note**: Uses all available GPUs by default. Use `CUDA_VISIBLE_DEVICES` to restrict.

## Configuration Files

| Config | Model | Languages | Steps |
|--------|-------|-----------|-------|
| `configs/mms_zul_config.json` | MMS | isiZulu | 1000 |
| `configs/mms_multilingual_config.json` | MMS | 6 languages | 2000 |
| `configs/isizulu_config.json` | W2V-BERT | isiZulu | 2000 |
| `configs/multilingual_config.json` | W2V-BERT | 6 languages | 2000 |

### Key Parameters

| Parameter | W2V-BERT | MMS | Description |
|-----------|----------|-----|-------------|
| `learning_rate` | 5e-5 | 1e-3 | Higher LR for adapters |
| `per_device_train_batch_size` | 8 | 8 | Samples per GPU |
| `gradient_accumulation_steps` | 4 | 4 | Effective batch = 32 |
| `max_steps` | 2000 | 2000 | Training iterations |
| `bf16` | true | true | Use bfloat16 precision |

## VM Deployment Notes
- The scripts are optimized for **A100 GPUs**.
- **Single GPU recommended** for streaming datasets (use `CUDA_VISIBLE_DEVICES=0`).
- Ensure `soundfile` backend is used for audio loading (handled in script).

## Supported Languages

| Language | Dataset Config | MMS Code |
|----------|---------------|----------|
| isiZulu | isizulu | zul |
| isiXhosa | isixhosa | xho |
| Sesotho | sesotho | sot |
| Setswana | setswana | tsn |
| Tshivenda | tshivenda | ven |
| Xitsonga | xitsonga | tso |
