#!/usr/bin/env python3
"""
Streaming-compatible MMS Adapter Training Script
Fine-tunes MMS adapter layers for low-resource South African languages
Based on: https://huggingface.co/blog/mms_adapters

MMS Adapter training is:
- More memory efficient (~2.5M trainable params vs 1B total)
- More robust for low-resource languages
- Faster to train (10-20 minutes for small datasets)
"""
import torch
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import json
import sys

# CRITICAL: Disable torchcodec audio backend  
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

from dotenv import load_dotenv
import wandb
import evaluate
from datasets import load_dataset, Audio, interleave_datasets
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoFeatureExtractor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
import regex as re

# Load environment variables
load_dotenv()


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for CTC with padding"""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input_values and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Replace padding with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        
        return batch


def clean_text(text: str) -> str:
    """Clean and normalize text for MMS"""
    if text is None:
        return ""
    text = str(text).lower()
    # Remove filler/noise tags like [?], [cs], [um], [pause]
    tags_to_remove = [r'\[\?\]', r'\[cs\]', r'\[um\]', r'\[pause\]']
    combined_pattern = '|'.join(tags_to_remove)
    text = re.sub(combined_pattern, ' ', text, flags=re.IGNORECASE)
    # Remove punctuation and special characters
    text = re.sub(r'[,?.!;\:"\'\u00bb\u00ab\[\]\(\)%-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Language code mapping for MMS
MMS_LANGUAGE_CODES = {
    "isizulu": "zul",
    "isixhosa": "xho",
    "sesotho": "sot",
    "setswana": "tsn",
    "tshivenda": "ven",
    "xitsonga": "tso",
    "sepedi": "nso",
    "siswati": "ssw",
    "isindebele": "nbl",
    "afrikaans": "afr",
    "english": "eng",
    # Add more as needed
}


def main():
    # Load config
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.json'):
        print("Usage: python train_streaming_mms.py <config.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    
    # Set seed
    set_seed(cfg.get("seed", 42))
    
    # Auth & WandB
    hf_token = os.getenv("HF_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")
    
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            entity=os.getenv("WANDB_ENTITY", "dsfsi"),
            project=os.getenv("WANDB_PROJECT", "dsfsi-asr"),
            name=cfg.get("run_name", "mms-adapter"),
        )
    
    # Get target language for adapter
    target_language = cfg.get("target_language", "zul")  # MMS language code
    print(f"Training MMS adapter for language: {target_language}")
    
    # Load MMS model and processor
    model_name = cfg.get("model_name_or_path", "facebook/mms-1b-all")
    print(f"Loading MMS model: {model_name}")
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(model_name, token=hf_token)
    
    # Load model with adapter for target language
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        token=hf_token,
        target_lang=target_language,
        ignore_mismatched_sizes=True,
    )
    
    # Set target language to load correct adapter
    model.load_adapter(target_language)
    processor.tokenizer.set_target_lang(target_language)
    
    # Freeze base model, only train adapter
    model.freeze_base_model()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load streaming datasets
    dataset_name = cfg['dataset_name']
    dataset_config_names = cfg['dataset_config_name']
    
    if isinstance(dataset_config_names, str):
        dataset_config_names = [dataset_config_names]
        
    train_datasets = []
    for config_name in dataset_config_names:
        print(f"Loading training dataset: {dataset_name} ({config_name})")
        ds = load_dataset(
            dataset_name,
            config_name,
            split=cfg.get("train_split_name", "train"),
            streaming=True,
            token=hf_token,
            trust_remote_code=cfg.get("trust_remote_code", False)
        )
        train_datasets.append(ds)
        
    if len(train_datasets) > 1:
        print(f"Interleaving {len(train_datasets)} training datasets...")
        raw_train = interleave_datasets(train_datasets, seed=cfg.get("seed", 42))
    else:
        raw_train = train_datasets[0]
    
    # Load eval dataset
    eval_dataset_name = cfg.get("eval_dataset_name", cfg["dataset_name"])
    eval_config_names = cfg.get("eval_dataset_config_name", cfg["dataset_config_name"])
    
    if isinstance(eval_config_names, str):
        eval_config_names = [eval_config_names]
        
    eval_datasets = []
    for config_name in eval_config_names:
        print(f"Loading eval dataset: {eval_dataset_name} ({config_name})")
        ds = load_dataset(
            eval_dataset_name,
            config_name,
            split=cfg.get("eval_split_name", "dev_test"),
            streaming=True,
            token=hf_token,
            trust_remote_code=cfg.get("trust_remote_code", False)
        )
        eval_datasets.append(ds)
        
    if len(eval_datasets) > 1:
        print(f"Interleaving {len(eval_datasets)} eval datasets...")
        raw_eval = interleave_datasets(eval_datasets, seed=cfg.get("seed", 42))
    else:
        raw_eval = eval_datasets[0]
    
    # Cast audio to 16kHz
    print("Casting audio to 16kHz...")
    raw_train = raw_train.cast_column("audio", Audio(sampling_rate=16000))
    raw_eval = raw_eval.cast_column("audio", Audio(sampling_rate=16000))
    
    # Preprocessing function
    def prepare_dataset(batch):
        # Check for valid text first
        text_col = cfg.get("text_column_name", "transcript")
        text = batch.get(text_col)
        if text is None or str(text).strip() == "":
            return {"input_values": [], "labels": [], "valid": False}
        
        text = clean_text(text)
        if not text:
            return {"input_values": [], "labels": [], "valid": False}
        
        audio = batch["audio"]
        
        # Check duration
        duration = len(audio["array"]) / 16000
        if duration < cfg.get("min_duration_in_seconds", 0.5) or duration > cfg.get("max_duration_in_seconds", 30.0):
            return {"input_values": [], "labels": [], "valid": False}
        
        # Process audio - MMS uses input_values (not input_features like W2V-BERT)
        input_values = processor(
            audio["array"], 
            sampling_rate=16000,
            return_tensors="np"
        ).input_values[0]
        
        # Tokenize text using the new API (avoids deprecation warning)
        labels = processor.tokenizer(text).input_ids
        
        return {"input_values": input_values, "labels": labels, "valid": True}
    
    # Filter function
    def filter_valid(batch):
        return batch.get("valid", False)
    
    # Apply preprocessing
    print("Preprocessing datasets...")
    train_dataset = raw_train.map(prepare_dataset, remove_columns=raw_train.column_names).filter(filter_valid).remove_columns(["valid"])
    eval_dataset = raw_eval.map(prepare_dataset, remove_columns=raw_eval.column_names).filter(filter_valid).remove_columns(["valid"])
    
    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        # Replace -100 with pad token id
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}
    
    # Training arguments - MMS adapter training is much faster
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 200),
        save_steps=cfg.get("save_steps", 200),
        logging_steps=cfg.get("logging_steps", 25),
        learning_rate=cfg.get("learning_rate", 1e-3),  # Higher LR for adapters
        warmup_steps=cfg.get("warmup_steps", 100),
        max_steps=cfg.get("max_steps", 1000),  # Adapters need fewer steps
        fp16=cfg.get("fp16", False) and not cfg.get("bf16", False),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        group_by_length=False,
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "wer"),
        greater_is_better=cfg.get("greater_is_better", False),
        push_to_hub=cfg.get("push_to_hub", False),
        hub_model_id=cfg.get("hub_model_id"),
        hub_strategy=cfg.get("hub_strategy", "end"),
        report_to=cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 42),
        dataloader_num_workers=0,
    )
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
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
    
    # Train
    print("Starting MMS adapter training...")
    trainer.train()
    
    # Save model and adapter
    print("Saving model...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    
    # Save adapter weights separately for easy sharing
    adapter_path = os.path.join(training_args.output_dir, f"adapter.{target_language}.safetensors")
    model.save_adapter(training_args.output_dir, target_language)
    print(f"Adapter saved to: {adapter_path}")
    
    print(f"Training complete! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
