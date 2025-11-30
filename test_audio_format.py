#!/usr/bin/env python3
"""Test script to debug audio format from datasets."""

# Set env vars BEFORE any imports
import os
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

from datasets import load_dataset, Audio

print("Loading dataset with streaming=True...")
ds = load_dataset(
    "dsfsi-anv/za-african-next-voices",
    "zul",
    split="train",
    streaming=True,
    trust_remote_code=False,
)

print(f"Dataset type: {type(ds)}")
print(f"Features: {ds.features}")

# Cast audio
ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))
print(f"After cast - Features: {ds.features}")

# Get first example
print("\nGetting first example...")
for i, example in enumerate(ds):
    audio = example["audio"]
    print(f"Audio type: {type(audio)}")
    print(f"Audio repr: {repr(audio)[:200]}")
    
    if isinstance(audio, dict):
        print("  -> Dict format!")
        print(f"  Keys: {audio.keys()}")
        print(f"  sampling_rate: {audio.get('sampling_rate')}")
        arr = audio.get('array')
        print(f"  array type: {type(arr)}")
        print(f"  array shape: {arr.shape if hasattr(arr, 'shape') else 'N/A'}")
    else:
        print(f"  -> NOT dict: {type(audio)}")
        print(f"  dir(audio): {[x for x in dir(audio) if not x.startswith('_')]}")
        
        # Try various methods
        if hasattr(audio, 'array'):
            print(f"  audio.array type: {type(audio.array)}")
        if hasattr(audio, 'sampling_rate'):
            print(f"  audio.sampling_rate: {audio.sampling_rate}")
        if callable(audio):
            try:
                decoded = audio()
                print(f"  audio() returned: {type(decoded)}")
            except Exception as e:
                print(f"  audio() error: {e}")
    
    if i >= 2:
        break

print("\nDone!")
