#!/usr/bin/env python3
"""
Streaming-compatible Whisper v3 Turbo fine-tuning script
Based on HuggingFace blog: https://huggingface.co/blog/fine-tune-whisper
Adapted for dsfsi-anv/za-african-next-voices with minimal disk usage.
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import numpy as np
import regex as re

# Avoid torchaudio/torchcodec conflicts inside datasets
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

from dotenv import load_dotenv
import wandb
import evaluate
from datasets import load_dataset, Audio, interleave_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

# ------------------------------
# Data collator (from HF blog)
# ------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Based on official HuggingFace Whisper fine-tuning blog.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut bos token if present (it's appended later)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


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
# Main
# ------------------------------

def main():
    load_dotenv()

    if len(sys.argv) != 2 or not sys.argv[1].endswith(".json"):
        print("Usage: python train_whisper_turbo_streaming.py <config.json>")
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
            name=cfg.get("run_name", "whisper-v3-turbo"),
        )

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
    # Processor & model
    # --------------------------
    model_name = cfg.get("model_name_or_path", "openai/whisper-large-v3-turbo")

    print(f"Loading processor: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name, token=hf_token)
    
    print(f"Loading model: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        token=hf_token,
    )

    # Set generation config for multilingual transcription
    # Per HF blog: set language explicitly for fine-tuning
    language = cfg.get("language", None)  # e.g. "zu", "xh", or None for auto
    task = cfg.get("task", "transcribe")  # "transcribe" or "translate"

    if language is not None:
        model.generation_config.language = language
        print(f"Set language to: {language}")
    model.generation_config.task = task
    
    # Clear legacy forced_decoder_ids (per HF blog recommendation)
    model.generation_config.forced_decoder_ids = None

    # Gradient checkpointing for memory efficiency (per HF blog)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("Enabled gradient checkpointing")

    # --------------------------
    # Audio casting
    # --------------------------
    target_sr = 16000
    print(f"Casting audio to {target_sr} Hz...")
    raw_train = raw_train.cast_column("audio", Audio(sampling_rate=target_sr))
    raw_eval = raw_eval.cast_column("audio", Audio(sampling_rate=target_sr))

    text_col = cfg.get("text_column_name", "transcript")
    max_dur = cfg.get("max_duration_in_seconds", 30.0)  # Whisper max is 30s
    min_dur = cfg.get("min_duration_in_seconds", 0.5)

    # --------------------------
    # Preprocessing
    # --------------------------
    def prepare_batch(batch):
        # Drop missing / empty text
        text = batch.get(text_col)
        if text is None or str(text).strip() == "":
            return {"input_features": [], "labels": [], "valid": False}

        text = clean_text(text)
        if not text:
            return {"input_features": [], "labels": [], "valid": False}

        audio = batch["audio"]
        sr = audio["sampling_rate"]
        arr = audio["array"]

        duration = len(arr) / sr
        if duration < min_dur or duration > max_dur:
            return {"input_features": [], "labels": [], "valid": False}

        # Whisper feature extractor: pad/truncate to 30s, compute log-Mel spectrogram
        inputs = processor.feature_extractor(
            arr,
            sampling_rate=sr,
            return_tensors="np",
        )
        input_features = inputs.input_features[0]

        # Tokenize text for decoder (per HF blog)
        labels = processor.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
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
    # Metrics (per HF blog)
    # --------------------------
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    # --------------------------
    # Training args (per HF blog)
    # --------------------------
    output_dir = cfg.get("output_dir", "./outputs/whisper-v3-turbo-multilingual")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # Batch settings
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        # Learning rate (HF blog uses 1e-5 for Whisper)
        learning_rate=cfg.get("learning_rate", 1e-5),
        warmup_steps=cfg.get("warmup_steps", 500),
        max_steps=cfg.get("max_steps", 5000),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # Precision
        fp16=cfg.get("fp16", False) and not cfg.get("bf16", False),
        bf16=cfg.get("bf16", True),
        # Checkpointing
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        # Evaluation
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 1000),
        save_steps=cfg.get("save_steps", 1000),
        logging_steps=cfg.get("logging_steps", 25),
        # Generation settings for evaluation
        predict_with_generate=True,
        generation_max_length=cfg.get("generation_max_length", 225),
        # Streaming-friendly settings
        dataloader_num_workers=0,
        # Saving
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "wer"),
        greater_is_better=cfg.get("greater_is_better", False),
        # Logging & Hub
        report_to=cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("run_name", "whisper-v3-turbo-multilingual"),
        push_to_hub=cfg.get("push_to_hub", True),
        hub_model_id=cfg.get("hub_model_id", "kesbeast23/multilingual-whisper-v3-turbo"),
        hub_strategy=cfg.get("hub_strategy", "end"),
        # Reproducibility
        seed=cfg.get("seed", 42),
        adam_beta2=cfg.get("adam_beta2", 0.98),
    )

    # Data collator (per HF blog)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # --------------------------
    # Train
    # --------------------------
    print("Starting Whisper v3 Turbo training...")
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
