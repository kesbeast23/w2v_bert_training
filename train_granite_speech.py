#!/usr/bin/env python3
"""
Fine-tuning IBM Granite Speech 2B for South African Languages
Based on the HuggingFace example by Avihu Dekel.

Granite Speech 2B is a powerful speech model that excels in ASR.
We fine-tune only the LoRA adapters and projector layers for efficiency.

This script is optimized to achieve the BEST possible WER on South African languages:
- isiZulu (zul), isiXhosa (xho), Sesotho (sot), Setswana (tsn), Tshivenda (ven), Xitsonga (tso)

Usage:
  python train_granite_speech.py configs/granite_config.json
"""

import os
import sys
import json
from typing import Dict, List, Any

import torch
import numpy as np
import regex as re
import tqdm

# Avoid torchaudio/torchcodec conflicts
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

from dotenv import load_dotenv
import wandb
import evaluate
from datasets import load_dataset, Audio, interleave_datasets
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.models.granite_speech import (
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechProcessor,
)
from transformers.feature_extraction_utils import BatchFeature
from torch.utils.data import DataLoader

# ------------------------------
# Text cleaning for South African languages
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

    # Remove punctuation and special chars (keep basic punctuation for Granite)
    text = re.sub(r'["\'\u00bb\u00ab\[\]\(\)%-]', "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------
# Data Collator for Granite Speech
# ------------------------------

class GraniteSpeechCollator:
    """
    Data collator for Granite Speech training.
    Handles prompt formatting and audio processing.
    Follows the official HuggingFace example exactly.
    """
    def __init__(self, processor, inference_mode=False):
        self.processor = processor
        self.inference_mode = inference_mode

    def __call__(self, examples: List[Dict]) -> BatchFeature:
        prompts = [example["prompt"] for example in examples]
        audios = [example["audio"] for example in examples]
        
        # Handle different audio formats - extract array
        processed_audios = []
        for audio in audios:
            if isinstance(audio, dict):
                processed_audios.append(audio["array"])
            elif hasattr(audio, "array"):
                processed_audios.append(audio.array)
            else:
                processed_audios.append(audio)

        # Process with Granite processor (left padding for decoder)
        processed = self.processor(
            prompts, 
            processed_audios, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left"
        )
        
        input_ids = processed.input_ids
        attention_mask = processed.attention_mask
        labels = None
        
        # For training: tokenize targets and combine with prompts
        if not self.inference_mode:
            targets = [
                example["text"] + self.processor.tokenizer.eos_token 
                for example in examples
            ]
            targets = self.processor.tokenizer(
                targets, 
                return_tensors="pt", 
                padding=True, 
                padding_side="right"
            )
            
            # Combine prompt + targets
            input_ids = torch.cat([input_ids, targets.input_ids], dim=1)
            attention_mask = torch.cat([attention_mask, targets.attention_mask], dim=1)
            
            # Create labels (only compute loss on target tokens, not prompts)
            labels = targets.input_ids.clone()
            # Mask padding tokens
            labels[~(targets.attention_mask.bool())] = -100
            # Mask prompt tokens (we only want loss on the transcription)
            labels = torch.cat([torch.full_like(processed.input_ids, -100), labels], dim=1)

        return BatchFeature(data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": processed.input_features,
            "input_features_mask": processed.input_features_mask,
        })


# ------------------------------
# WER Computation (for evaluation during and after training)
# ------------------------------

def compute_wer_on_dataset(model, processor, dataset, batch_size=8, num_beams=4):
    """
    Compute WER on a dataset using beam search decoding.
    This matches the official evaluation setup.
    """
    if len(dataset) == 0:
        print("⚠️  Empty dataset, cannot compute WER")
        return {"wer": 1.0, "cer": 1.0}, [], []
    
    collator = GraniteSpeechCollator(processor, inference_mode=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    model = model.eval().cuda()
    
    all_outputs = []
    all_references = []
    
    print(f"Running inference with beam_size={num_beams} on {len(dataset)} samples...")
    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        batch = batch.to("cuda")
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **batch, 
                max_new_tokens=256, 
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,  # Greedy/beam search for best WER
            )
        
        input_length = batch.input_ids.shape[1]
        outputs = outputs[:, input_length:].cpu()
        
        for x in outputs:
            decoded = processor.tokenizer.decode(x, skip_special_tokens=True)
            all_outputs.append(decoded.lower().strip())
    
    # Get ground truth from dataset
    for example in dataset:
        all_references.append(example["text"].lower().strip())
    
    if len(all_outputs) == 0 or len(all_references) == 0:
        print("⚠️  No outputs or references, cannot compute WER")
        return {"wer": 1.0, "cer": 1.0}, [], []
    
    wer = wer_metric.compute(references=all_references, predictions=all_outputs)
    cer = cer_metric.compute(references=all_references, predictions=all_outputs)
    
    return {"wer": wer, "cer": cer}, all_outputs, all_references


# ------------------------------
# Custom Trainer with WER evaluation
# ------------------------------

class GraniteSpeechTrainer(Trainer):
    """Custom trainer that computes WER/CER during evaluation and logs to wandb."""
    
    def __init__(self, *args, processor=None, eval_dataset_for_wer=None, wer_batch_size=8, num_beams=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.granite_processor = processor
        self.eval_dataset_for_wer = eval_dataset_for_wer
        self.wer_batch_size = wer_batch_size
        self.num_beams = num_beams
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute WER/CER with proper generation.
        This runs every eval_steps and logs to wandb.
        """
        # First run standard evaluation to get loss
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Now compute WER/CER with generation
        if self.eval_dataset_for_wer is not None:
            print(f"\n📊 Computing WER/CER at step {self.state.global_step}...")
            wer, cer = self._compute_wer_cer()
            
            # Add to output metrics
            output[f"{metric_key_prefix}_wer"] = wer
            output[f"{metric_key_prefix}_cer"] = cer
            
            print(f"   WER: {wer*100:.2f}% | CER: {cer*100:.2f}%")
            
            # Log to wandb explicitly
            if wandb.run is not None:
                wandb.log({
                    "eval/wer": wer,
                    "eval/cer": cer,
                    "step": self.state.global_step,
                })
        
        return output
    
    def _compute_wer_cer(self):
        """Compute WER and CER using generation (beam search)."""
        collator = GraniteSpeechCollator(self.granite_processor, inference_mode=True)
        dataloader = DataLoader(
            self.eval_dataset_for_wer, 
            batch_size=self.wer_batch_size, 
            collate_fn=collator, 
            num_workers=0
        )
        
        self.model.eval()
        all_outputs = []
        all_references = []
        
        for batch in dataloader:
            batch = batch.to(self.model.device)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **batch,
                    max_new_tokens=256,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    do_sample=False,
                )
            
            input_length = batch.input_ids.shape[1]
            outputs = outputs[:, input_length:].cpu()
            
            for x in outputs:
                decoded = self.granite_processor.tokenizer.decode(x, skip_special_tokens=True)
                all_outputs.append(decoded.lower().strip())
        
        # Get references
        for example in self.eval_dataset_for_wer:
            all_references.append(example["text"].lower().strip())
        
        # Compute metrics
        wer = self.wer_metric.compute(references=all_references, predictions=all_outputs)
        cer = self.cer_metric.compute(references=all_references, predictions=all_outputs)
        
        return wer, cer


# ------------------------------
# Main
# ------------------------------

def main():
    load_dotenv()

    if len(sys.argv) != 2 or not sys.argv[1].endswith(".json"):
        print("Usage: python train_granite_speech.py <config.json>")
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
            name=cfg.get("run_name", "granite-speech-2b"),
        )

    # --------------------------
    # Load processor & model
    # --------------------------
    model_name = cfg.get("model_name_or_path", "ibm-granite/granite-speech-3.3-2b")
    print(f"Loading processor: {model_name}")
    
    processor = GraniteSpeechProcessor.from_pretrained(model_name, token=hf_token)
    
    print(f"Loading model: {model_name}")
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    
    # Get the expected sampling rate
    target_sr = processor.audio_processor.sampling_rate
    print(f"Model expects {target_sr} Hz audio")

    # --------------------------
    # Freeze most parameters, train only LoRA + projector
    # This is the key to efficient fine-tuning without overfitting
    # --------------------------
    trainable_params = 0
    total_params = 0
    trainable_names = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Only train projector and LoRA layers (as per official example)
        if "projector" in name or "lora" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            trainable_names.append(name)
        else:
            param.requires_grad = False
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Training {len(trainable_names)} parameter groups (projector + LoRA)")
    
    # Move model to GPU
    model = model.cuda()

    # --------------------------
    # Load datasets (non-streaming for Granite - it needs full dataset access)
    # --------------------------
    dataset_name = cfg["dataset_name"]
    dataset_config_names = cfg["dataset_config_name"]
    if isinstance(dataset_config_names, str):
        dataset_config_names = [dataset_config_names]

    train_split = cfg.get("train_split_name", "train")
    eval_split = cfg.get("eval_split_name", "dev_test")
    
    # For Granite, we'll use streaming but materialize small portions
    streaming = cfg.get("streaming", True)

    train_datasets = []
    for config_name in dataset_config_names:
        print(f"Loading training dataset: {dataset_name} ({config_name})")
        ds = load_dataset(
            dataset_name,
            config_name,
            split=train_split,
            streaming=streaming,
            token=hf_token,
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        train_datasets.append(ds)

    if len(train_datasets) > 1:
        print(f"Interleaving {len(train_datasets)} training datasets...")
        raw_train = interleave_datasets(train_datasets, seed=cfg.get("seed", 42))
    else:
        raw_train = train_datasets[0]

    eval_datasets = []
    eval_config_names = cfg.get("eval_dataset_config_name", dataset_config_names)
    if isinstance(eval_config_names, str):
        eval_config_names = [eval_config_names]

    for config_name in eval_config_names:
        print(f"Loading eval dataset: {dataset_name} ({config_name})")
        ds = load_dataset(
            dataset_name,
            config_name,
            split=eval_split,
            streaming=streaming,
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
    max_dur = cfg.get("max_duration_in_seconds", 30.0)
    min_dur = cfg.get("min_duration_in_seconds", 0.5)

    # --------------------------
    # Preprocessing
    # --------------------------
    instruction = cfg.get(
        "instruction_prompt",
        "Please transcribe the following audio to text<|audio|>"
    )
    
    def prepare_example(batch):
        """Prepare a single example for Granite Speech."""
        # Validate text
        text = batch.get(text_col)
        if text is None or str(text).strip() == "":
            return {"audio": None, "text": "", "prompt": "", "valid": False}

        text = clean_text(text)
        if not text:
            return {"audio": None, "text": "", "prompt": "", "valid": False}

        audio = batch["audio"]
        
        # Handle different audio formats
        try:
            if isinstance(audio, dict):
                # Standard dict format from datasets
                sr = audio.get("sampling_rate", target_sr)
                arr = audio.get("array")
            elif hasattr(audio, "__call__"):
                # AudioDecoder - needs to be called to decode
                decoded = audio()
                if isinstance(decoded, dict):
                    sr = decoded.get("sampling_rate", target_sr)
                    arr = decoded.get("array")
                else:
                    sr = target_sr
                    arr = decoded
            elif hasattr(audio, "sampling_rate") and hasattr(audio, "array"):
                # Object with attributes
                sr = audio.sampling_rate
                arr = audio.array
            else:
                # Try to decode if it's an AudioDecoder-like object
                # Access the underlying data
                if hasattr(audio, '_decode'):
                    decoded = audio._decode()
                    sr = decoded.get("sampling_rate", target_sr)
                    arr = decoded.get("array")
                else:
                    print(f"Unknown audio format: {type(audio)}")
                    return {"audio": None, "text": "", "prompt": "", "valid": False}
        except Exception as e:
            print(f"Error processing audio: {e}")
            return {"audio": None, "text": "", "prompt": "", "valid": False}
        
        if arr is None:
            return {"audio": None, "text": "", "prompt": "", "valid": False}
        
        # Convert to numpy if needed
        if hasattr(arr, 'numpy'):
            arr = arr.numpy()
        elif hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        arr = np.array(arr, dtype=np.float32)

        # Duration filter
        duration = len(arr) / sr
        if duration < min_dur or duration > max_dur:
            return {"audio": None, "text": "", "prompt": "", "valid": False}

        # Create prompt using chat template
        chat = [dict(role="user", content=instruction)]
        prompt = processor.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        return {
            "audio": {"array": arr, "sampling_rate": sr},
            "text": text,
            "prompt": prompt,
            "valid": True,
        }

    def filter_valid(batch):
        return batch.get("valid", False)

    print("Preprocessing train dataset...")
    train_dataset = (
        raw_train
        .map(prepare_example, remove_columns=raw_train.column_names)
        .filter(filter_valid)
        .remove_columns(["valid"])
    )

    print("Preprocessing eval dataset...")
    eval_dataset = (
        raw_eval
        .map(prepare_example, remove_columns=raw_eval.column_names)
        .filter(filter_valid)
        .remove_columns(["valid"])
    )

    # --------------------------
    # Training arguments - OPTIMIZED FOR BEST WER
    # Based on official HuggingFace Granite Speech example
    # --------------------------
    output_dir = cfg.get("output_dir", "./outputs/granite-speech-multilingual")

    # Calculate effective batch size
    per_device_batch = cfg.get("per_device_train_batch_size", 16)
    grad_accum = cfg.get("gradient_accumulation_steps", 2)
    effective_batch = per_device_batch * grad_accum
    print(f"Effective batch size: {effective_batch} (per_device={per_device_batch} x grad_accum={grad_accum})")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=cfg.get("overwrite_output_dir", True),
        remove_unused_columns=False,  # Keep audio/prompt columns
        
        # Batch settings - larger batches help convergence
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=grad_accum,
        
        # Learning rate - 3e-5 is optimal for adapter fine-tuning
        learning_rate=cfg.get("learning_rate", 3e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        
        # Warmup - 20% warmup ratio helps stabilize training
        warmup_ratio=cfg.get("warmup_ratio", 0.2),
        
        # Training duration
        num_train_epochs=cfg.get("num_train_epochs", 3),
        max_steps=cfg.get("max_steps", -1),  # -1 means use num_train_epochs
        
        # Precision - bf16 is optimal for A100
        bf16=cfg.get("bf16", True),
        fp16=False,
        
        # Evaluation strategy - evaluate frequently to track WER
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 0.1),  # Evaluate every 10% of epoch
        
        # Logging
        logging_strategy="steps",
        logging_steps=cfg.get("logging_steps", 0.1),  # Log every 10% of epoch
        logging_first_step=True,
        
        # Saving - save based on eval loss
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 0.1),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Dataloader settings - 0 workers for streaming datasets
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        dataloader_pin_memory=True,
        
        # No length grouping for speech (variable length is fine)
        group_by_length=False,
        
        # Hub & logging
        push_to_hub=cfg.get("push_to_hub", True),
        hub_model_id=cfg.get("hub_model_id", "kesbeast23/granite-speech-2b-multilingual"),
        hub_strategy=cfg.get("hub_strategy", "end"),
        report_to=cfg.get("report_to", ["wandb"]),
        run_name=cfg.get("run_name", "granite-speech-2b-multilingual"),
        
        # Reproducibility
        seed=cfg.get("seed", 42),
        data_seed=cfg.get("seed", 42),
        
        # Gradient clipping for stability
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
    )

    # Data collator
    data_collator = GraniteSpeechCollator(processor=processor, inference_mode=False)

    # Early stopping callback to prevent overfitting
    callbacks = []
    if cfg.get("early_stopping", True):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.get("early_stopping_patience", 3),
                early_stopping_threshold=cfg.get("early_stopping_threshold", 0.001),
            )
        )

    # Prepare a fixed eval sample for WER computation (materialized list)
    print("Preparing eval samples for WER computation...")
    wer_eval_samples = cfg.get("wer_eval_samples", 300)
    
    # Materialize eval samples - iterate through the streaming dataset
    eval_sample_for_wer = []
    print(f"Collecting up to {wer_eval_samples} eval samples...")
    for i, example in enumerate(eval_dataset):
        eval_sample_for_wer.append(example)
        if len(eval_sample_for_wer) >= wer_eval_samples:
            break
        if (i + 1) % 50 == 0:
            print(f"  Collected {len(eval_sample_for_wer)} samples...")
    
    print(f"Using {len(eval_sample_for_wer)} samples for WER evaluation every {cfg.get('eval_steps', 1000)} steps")
    
    if len(eval_sample_for_wer) == 0:
        print("⚠️  WARNING: No eval samples collected! Check your dataset filtering.")
        print("   Disabling baseline WER computation...")
        cfg["compute_baseline_wer"] = False

    # Custom Trainer with WER tracking
    trainer = GraniteSpeechTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        processor=processor,
        eval_dataset_for_wer=eval_sample_for_wer,
        wer_batch_size=cfg.get("per_device_eval_batch_size", 16),
        num_beams=cfg.get("num_beams", 4),
        callbacks=callbacks,
    )

    # --------------------------
    # Compute WER BEFORE training (baseline)
    # --------------------------
    if cfg.get("compute_baseline_wer", True) and len(eval_sample_for_wer) > 0:
        print("\n" + "="*60)
        print("Computing BASELINE WER before training...")
        print("="*60)
        
        baseline_metrics, _, _ = compute_wer_on_dataset(
            model, processor, eval_sample_for_wer, 
            batch_size=cfg.get("per_device_eval_batch_size", 16),
            num_beams=cfg.get("num_beams", 4)
        )
        
        print(f"\n📊 BASELINE WER: {baseline_metrics['wer']*100:.2f}%")
        print(f"📊 BASELINE CER: {baseline_metrics['cer']*100:.2f}%")
        
        if wandb_key:
            wandb.log({
                "baseline_wer": baseline_metrics['wer'],
                "baseline_cer": baseline_metrics['cer'],
            })
    else:
        baseline_metrics = None

    # --------------------------
    # Train
    # --------------------------
    print("\n" + "="*60)
    print("Starting Granite Speech 2B training...")
    print("Training only projector + LoRA layers for best WER")
    print("="*60 + "\n")
    
    trainer.train()

    # --------------------------
    # Save
    # --------------------------
    print("\nSaving model and processor...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    print(f"Training complete! Model saved to {output_dir}")

    # --------------------------
    # Compute FINAL WER (after training)
    # --------------------------
    if len(eval_sample_for_wer) > 0:
        print("\n" + "="*60)
        print("Computing FINAL WER after training...")
        print("="*60)
        
        final_metrics, predictions, references = compute_wer_on_dataset(
            model, processor, eval_sample_for_wer,
            batch_size=cfg.get("per_device_eval_batch_size", 16),
            num_beams=cfg.get("num_beams", 4)
        )
        
        print(f"\n📊 FINAL WER: {final_metrics['wer']*100:.2f}%")
        print(f"📊 FINAL CER: {final_metrics['cer']*100:.2f}%")
        
        if baseline_metrics is not None:
            wer_improvement = baseline_metrics['wer'] - final_metrics['wer']
            print(f"\n🎯 WER IMPROVEMENT: {wer_improvement*100:.2f}% absolute")
            if baseline_metrics['wer'] > 0:
                print(f"🎯 Relative improvement: {(wer_improvement/baseline_metrics['wer'])*100:.1f}%")
            
            if wandb_key:
                wandb.log({
                    "final_wer": final_metrics['wer'],
                    "final_cer": final_metrics['cer'],
                    "wer_improvement": wer_improvement,
                })
        
        # Save some example predictions
        print("\n" + "="*60)
        print("Sample predictions:")
        print("="*60)
        for i in range(min(5, len(predictions))):
            print(f"\nReference:  {references[i]}")
            print(f"Prediction: {predictions[i]}")
    
    # Push to hub if requested
    if cfg.get("push_to_hub", True):
        print("\nPushing to HuggingFace Hub...")
        trainer.push_to_hub()
        print(f"Model pushed to: {cfg.get('hub_model_id')}")


if __name__ == "__main__":
    main()
