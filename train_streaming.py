#!/usr/bin/env python3
"""
Streaming-compatible W2V-BERT 2.0 CTC training script
Adapted for dsfsi-anv datasets with minimal disk usage
"""
import torch
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, List, Union
import json
import sys

# CRITICAL: Disable torchcodec audio backend  
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

from dotenv import load_dotenv
import wandb
import evaluate
from datasets import load_dataset, Audio, IterableDataset, interleave_datasets
from transformers import (
    Wav2Vec2BertProcessor,
    Wav2Vec2BertForCTC,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
import regex as re

# Load environment variables
load_dotenv()

@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for CTC"""
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
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
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        
        return batch


def clean_text(text: str) -> str:
    """Clean and normalize text"""
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


def main():
    # Load config
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.json'):
        print("Usage: python train_streaming.py <config.json>")
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
            name=cfg.get("run_name", "w2v-bert"),
        )
    
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
    
    # ============================================
    # CREATE VOCABULARY FROM DATASET  
    # ============================================
    print("Creating vocabulary from dataset...")
    
    # DON'T cast audio yet - we only need text for vocab creation
    import functools
    
    def extract_all_chars(batch):
        text = batch[cfg.get("text_column_name", "transcript")]
        if text:
            text = clean_text(text)
            return {"vocab": list(set(text)), "all_text": [text]}
        return {"vocab": [], "all_text": [""]}
    
    # Sample examples to build vocab
    # For multilingual, take more samples to capture all language characters
    num_train_vocab_samples = cfg.get("vocab_train_samples", 2000)
    num_eval_vocab_samples = cfg.get("vocab_eval_samples", 1000)
    
    # REMOVE audio column to avoid torchcodec issues
    text_only_train = raw_train.remove_columns(["audio"])
    text_only_eval = raw_eval.remove_columns(["audio"])
    
    vocab_train = text_only_train.take(num_train_vocab_samples)
    vocab_eval = text_only_eval.take(num_eval_vocab_samples)
    
    print(f"Sampling {num_train_vocab_samples} train + {num_eval_vocab_samples} eval examples for vocabulary...")
    
    # Extract characters
    vocab_list = []
    for batch in vocab_train:
        result = extract_all_chars(batch)
        vocab_list.extend(result["vocab"])
    for batch in vocab_eval:
        result = extract_all_chars(batch)
        vocab_list.extend(result["vocab"])
    
    vocab_set = set(vocab_list)
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}
    
    # Add special tokens
    word_delimiter = cfg.get("word_delimiter_token", "|")
    unk_token = cfg.get("unk_token", "[UNK]")
    pad_token = cfg.get("pad_token", "[PAD]")
    
    if word_delimiter and " " in vocab_dict:
        vocab_dict[word_delimiter] = vocab_dict[" "]
        del vocab_dict[" "]
    
    if unk_token:
        vocab_dict[unk_token] = len(vocab_dict)
    if pad_token:
        vocab_dict[pad_token] = len(vocab_dict)
    
    # Save vocab to output dir
    os.makedirs(cfg["output_dir"], exist_ok=True)
    vocab_file = os.path.join(cfg["output_dir"], "vocab.json")
    
    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary created with {len(vocab_dict)} tokens")
    
    # Load processor with created vocab
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token=cfg.get("unk_token", "[UNK]"),
        pad_token=cfg.get("pad_token", "[PAD]"),
        word_delimiter_token=cfg.get("word_delimiter_token", "|")
    )
    
    # W2V-BERT uses SeamlessM4TFeatureExtractor
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(cfg["model_name_or_path"])
    
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # NOW cast audio to correct sampling rate (after vocab is created)
    print("Casting audio to 16kHz...")
    raw_train = raw_train.cast_column("audio", Audio(sampling_rate=16000))
    raw_eval = raw_eval.cast_column("audio", Audio(sampling_rate=16000))
    
    # Combined preprocessing function that handles filtering internally
    # Returns a dict with a "valid" field to mark samples for filtering
    def prepare_dataset(batch):
        # Check for valid text first
        text_col = cfg.get("text_column_name", "transcript")
        text = batch.get(text_col)
        if text is None or str(text).strip() == "":
            # Return minimal valid structure but mark as invalid
            return {"input_features": [], "labels": [], "valid": False}
        
        text = clean_text(text)
        if not text:  # If text becomes empty after cleaning
            return {"input_features": [], "labels": [], "valid": False}
        
        audio = batch["audio"]
        
        # Check duration
        duration = len(audio["array"]) / 16000
        if duration < cfg.get("min_duration_in_seconds", 0.5) or duration > cfg.get("max_duration_in_seconds", 30.0):
            return {"input_features": [], "labels": [], "valid": False}
        
        # W2V-BERT processor returns input_features, not input_values
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        labels = processor.tokenizer(text).input_ids
        return {"input_features": input_features, "labels": labels, "valid": True}
    
    # Filter function to keep only valid samples
    def filter_valid(batch):
        return batch.get("valid", False)
    
    # Apply preprocessing with map, then filter out invalid samples
    print("Preprocessing datasets...")
    train_dataset = raw_train.map(prepare_dataset, remove_columns=raw_train.column_names).filter(filter_valid).remove_columns(["valid"])
    eval_dataset = raw_eval.map(prepare_dataset, remove_columns=raw_eval.column_names).filter(filter_valid).remove_columns(["valid"])
    
    # Load model directly with all config parameters
    # Based on HuggingFace blog: https://huggingface.co/blog/fine-tune-w2v2-bert
    # Do NOT load config separately - pass parameters directly to from_pretrained
    print(f"Loading model: {cfg['model_name_or_path']}")
    
    model = Wav2Vec2BertForCTC.from_pretrained(
        cfg["model_name_or_path"],
        token=hf_token,
        # Vocabulary settings
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        # Dropout settings (disable for adapter training to prevent overfitting issues)
        attention_dropout=cfg.get("attention_dropout", 0.0),
        hidden_dropout=cfg.get("hidden_dropout", 0.0),
        feat_proj_dropout=cfg.get("feat_proj_dropout", 0.0),
        layerdrop=cfg.get("layerdrop", 0.0),
        # SpecAugment settings
        mask_time_prob=cfg.get("mask_time_prob", 0.0),  # Set to 0 initially, can increase
        mask_time_length=cfg.get("mask_time_length", 10),
        mask_feature_prob=cfg.get("mask_feature_prob", 0.0),
        mask_feature_length=cfg.get("mask_feature_length", 10),
        # CTC settings
        ctc_loss_reduction=cfg.get("ctc_loss_reduction", "mean"),
        # Adapter settings - CRITICAL for efficient fine-tuning
        add_adapter=cfg.get("add_adapter", True),
    )
    
    # W2V-BERT doesn't use freeze_feature_encoder like regular Wav2Vec2
    # The model trains the adapter layer by default when add_adapter=True
    
    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        # Replace -100 with pad token id for decoding
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 1000),
        save_steps=cfg.get("save_steps", 1000),
        logging_steps=cfg.get("logging_steps", 50),
        learning_rate=cfg.get("learning_rate", 5e-5),
        warmup_steps=cfg.get("warmup_steps", 500),
        max_steps=cfg.get("max_steps", 10000),
        fp16=cfg.get("fp16", False) and not cfg.get("bf16", False),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        group_by_length=False,  # Not supported with streaming
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
        dataloader_num_workers=0,  # Streaming doesn't support multiprocessing
        adam_beta2=cfg.get("adam_beta2", 0.98),  # Smoother loss curves (HF blog recommends 0.95-0.98)
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
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    
    print(f"Training complete! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
