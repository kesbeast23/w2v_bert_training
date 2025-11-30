#!/usr/bin/env python3
"""
Streaming-compatible NVIDIA Parakeet CTC fine-tuning script
Adapted for dsfsi-anv/za-african-next-voices with minimal disk usage.

Parakeet-CTC-1.1B is NVIDIA's state-of-the-art English ASR model based on
FastConformer architecture with CTC decoder. It achieves excellent WER
on various benchmarks and can be fine-tuned for other languages.

Usage:
  python train_parakeet_streaming.py configs/parakeet_config.json
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import numpy as np
import regex as re

# Avoid torchaudio/torchcodec conflicts inside datasets
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
# Note: Set CUDA_LAUNCH_BLOCKING=1 for debugging CUDA errors if needed

from dotenv import load_dotenv
import wandb
import evaluate
from datasets import load_dataset, Audio, interleave_datasets
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ------------------------------
# Text cleaning
# ------------------------------

def clean_text(text: str) -> str:
    """Clean and normalize transcript text (language-agnostic)."""
    if text is None:
        return ""
    text = str(text).lower()

    # Remove filler/noise tags like [?], [cs], [um], [pause]
    tags_to_remove = [r"\[\?\]", r"\[cs\]", r"\[um\]", r"\[pause\]"]
    combined_pattern = "|".join(tags_to_remove)
    text = re.sub(combined_pattern, " ", text, flags=re.IGNORECASE)

    # Remove punctuation and special chars
    text = re.sub(r'[,?.!;\:"\'\u00bb\u00ab\[\]\(\)%-]', "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------
# Data collator for Parakeet CTC
# ------------------------------

@dataclass
class DataCollatorParakeetCTC:
    """
    Data collator for Parakeet CTC training.
    Parakeet processor returns input_features (log-mel) and labels.
    """
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract audio arrays - handle both dict format and direct array
        audios = []
        texts = []
        
        for f in features:
            audio = f["audio"]
            text = f["text"]
            
            # Handle different audio formats
            if isinstance(audio, dict):
                arr = audio["array"]
            elif hasattr(audio, "array"):
                arr = audio.array
            else:
                arr = audio
            
            # Ensure float32
            if hasattr(arr, 'astype'):
                arr = arr.astype(np.float32)
            
            audios.append(arr)
            texts.append(text)

        # Process with Parakeet processor
        batch = self.processor(
            audio=audios,
            text=texts,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Replace padding token id with -100 for CTC loss
        if "labels" in batch:
            labels = batch["labels"].clone()
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            batch["labels"] = labels

        return batch


# ------------------------------
# Main
# ------------------------------

def main():
    load_dotenv()

    if len(sys.argv) != 2 or not sys.argv[1].endswith(".json"):
        print("Usage: python train_parakeet_streaming.py <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    set_seed(cfg.get("seed", 42))

    hf_token = os.getenv("HF_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")

    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            entity=os.getenv("WANDB_ENTITY", "dsfsi"),
            project=os.getenv("WANDB_PROJECT", "dsfsi-asr"),
            name=cfg.get("run_name", "parakeet-ctc"),
        )

    # --------------------------
    # Load processor & model FIRST
    # (Need processor for preprocessing)
    # --------------------------
    model_name = cfg.get("model_name_or_path", "nvidia/parakeet-ctc-1.1b")
    print(f"Loading processor: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    
    # Get the expected sampling rate from processor
    target_sr = processor.feature_extractor.sampling_rate
    print(f"Model expects {target_sr} Hz audio")

    # --------------------------
    # Load streaming datasets
    # --------------------------
    dataset_name = cfg["dataset_name"]
    dataset_config_names = cfg["dataset_config_name"]
    if isinstance(dataset_config_names, str):
        dataset_config_names = [dataset_config_names]

    train_split = cfg.get("train_split_name", "train")
    eval_split = cfg.get("eval_split_name", "dev_test")

    train_datasets = []
    for config_name in dataset_config_names:
        print(f"Loading training dataset: {dataset_name} ({config_name})")
        ds = load_dataset(
            dataset_name,
            config_name,
            split=train_split,
            streaming=True,
            token=hf_token,
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        train_datasets.append(ds)

    if len(train_datasets) > 1:
        print(f"Interleaving {len(train_datasets)} training datasets...")
        raw_train = interleave_datasets(train_datasets, seed=cfg.get("seed", 42))
    else:
        raw_train = train_datasets[0]
    
    # NOTE: Do NOT use .shuffle() with streaming + Trainer - it causes hangs

    eval_dataset_name = cfg.get("eval_dataset_name", dataset_name)
    eval_config_names = cfg.get("eval_dataset_config_name", dataset_config_names)
    if isinstance(eval_config_names, str):
        eval_config_names = [eval_config_names]

    eval_datasets = []
    for config_name in eval_config_names:
        print(f"Loading eval dataset: {eval_dataset_name} ({config_name})")
        ds = load_dataset(
            eval_dataset_name,
            config_name,
            split=eval_split,
            streaming=True,
            token=hf_token,
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        eval_datasets.append(ds)

    if len(eval_datasets) > 1:
        print(f"Interleaving {len(eval_datasets)} eval datasets...")
        raw_eval = interleave_datasets(eval_datasets, seed=cfg.get("seed", 42))
    else:
        raw_eval = eval_datasets[0]

    # --------------------------
    # Audio casting
    # --------------------------
    print(f"Casting audio to {target_sr} Hz...")
    raw_train = raw_train.cast_column("audio", Audio(sampling_rate=target_sr))
    raw_eval = raw_eval.cast_column("audio", Audio(sampling_rate=target_sr))

    text_col = cfg.get("text_column_name", "transcript")
    max_dur = cfg.get("max_duration_in_seconds", 20.0)
    min_dur = cfg.get("min_duration_in_seconds", 0.5)

    # --------------------------
    # Preprocessing
    # Keep raw audio + cleaned text for data collator
    # (Parakeet processor handles feature extraction in collator)
    # --------------------------
    def prepare_batch(batch):
        # Validate text
        text = batch.get(text_col)
        if text is None or str(text).strip() == "":
            return {"audio": None, "text": "", "valid": False}

        text = clean_text(text)
        if not text:
            return {"audio": None, "text": "", "valid": False}

        audio = batch["audio"]
        
        # Handle different audio formats (dict vs AudioDecoder)
        if isinstance(audio, dict):
            sr = audio["sampling_rate"]
            arr = audio["array"]
        elif hasattr(audio, "sampling_rate") and hasattr(audio, "array"):
            sr = audio.sampling_rate
            arr = audio.array
        else:
            return {"audio": None, "text": "", "valid": False}

        # Duration filter
        duration = len(arr) / sr
        if duration < min_dur or duration > max_dur:
            return {"audio": None, "text": "", "valid": False}

        # Store audio as dict with array for data collator
        return {
            "audio": {"array": arr, "sampling_rate": sr},
            "text": text,
            "valid": True,
        }

    def filter_valid(batch):
        return batch.get("valid", False)

    print("Preprocessing train dataset...")
    train_dataset = (
        raw_train
        .map(prepare_batch, remove_columns=raw_train.column_names)
        .filter(filter_valid)
        .remove_columns(["valid"])
    )

    print("Preprocessing eval dataset...")
    eval_dataset = (
        raw_eval
        .map(prepare_batch, remove_columns=raw_eval.column_names)
        .filter(filter_valid)
        .remove_columns(["valid"])
    )

    # --------------------------
    # Load model
    # --------------------------
    print(f"Loading model: {model_name}")
    model = AutoModelForCTC.from_pretrained(
        model_name,
        token=hf_token,
        ctc_loss_reduction=cfg.get("ctc_loss_reduction", "mean"),
        ctc_zero_infinity=cfg.get("ctc_zero_infinity", True),
    )

    # NOTE: Gradient checkpointing is DISABLED for Parakeet
    # Parakeet uses NeMo's FastConformer with custom CUDA kernels that are
    # incompatible with PyTorch's gradient checkpointing (causes "GET was 
    # unable to find an engine" error). Use smaller batch sizes instead.
    print("Note: Gradient checkpointing disabled for Parakeet (use smaller batches for memory)")

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # --------------------------
    # Metrics
    # --------------------------
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        logits = pred.predictions
        pred_ids = np.argmax(logits, axis=-1)

        # Replace -100 with pad token id for decoding
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    # --------------------------
    # Training arguments
    # --------------------------
    output_dir = cfg.get("output_dir", "./outputs/parakeet-ctc-multilingual")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=cfg.get("overwrite_output_dir", True),
        # Batch settings - smaller batch due to no gradient checkpointing
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        # Learning rate
        learning_rate=cfg.get("learning_rate", 5e-5),
        warmup_steps=cfg.get("warmup_steps", 500),
        max_steps=cfg.get("max_steps", 10000),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # Precision - use fp32 for Parakeet stability initially
        bf16=False,
        fp16=False,
        # NO gradient checkpointing for Parakeet (incompatible with FastConformer)
        gradient_checkpointing=False,
        # Evaluation (newer transformers uses eval_strategy, not evaluation_strategy)
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 1000),
        save_steps=cfg.get("save_steps", 1000),
        logging_steps=cfg.get("logging_steps", 25),
        # Streaming-friendly - CRITICAL: no multiprocessing with IterableDataset
        group_by_length=False,
        dataloader_num_workers=0,  # Must be 0 for streaming datasets
        remove_unused_columns=False,  # Keep audio column for data collator
        # Saving
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "wer"),
        greater_is_better=cfg.get("greater_is_better", False),
        # Hub & logging
        push_to_hub=cfg.get("push_to_hub", True),
        hub_model_id=cfg.get("hub_model_id", "kesbeast23/multilingual-parakeet-ctc-1.1b"),
        hub_strategy=cfg.get("hub_strategy", "end"),
        report_to=cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("run_name", "parakeet-ctc-multilingual"),
        # Reproducibility
        seed=cfg.get("seed", 42),
        adam_beta2=cfg.get("adam_beta2", 0.98),
    )

    # Data collator
    data_collator = DataCollatorParakeetCTC(processor=processor)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # --------------------------
    # Train
    # --------------------------
    print("Starting Parakeet CTC training...")
    trainer.train()

    # --------------------------
    # Save
    # --------------------------
    print("Saving model and processor...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
