#!/usr/bin/env python3
import os
from datasets import load_dataset
from dotenv import load_dotenv

# Load .env file
load_dotenv()
print("✓ Loaded .env file")

# Get token from environment
token = os.getenv("HF_TOKEN")

if not token:
    print("ERROR: HF_TOKEN not set!")
    exit(1)

print(f"Token found: {token[:10]}...")

# Test 1: Try loading the training dataset
print("\n=== Testing dsfsi-anv/anv_train_sample ===")
try:
    dataset = load_dataset(
        "dsfsi-anv/anv_train_sample",
        "isizulu",
        split="train",
        streaming=True,
        token=token
    )
    print("✓ SUCCESS: Can access anv_train_sample")
    # Try to get first item
    first_item = next(iter(dataset))
    print(f"  First item keys: {list(first_item.keys())}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: Try loading the eval dataset
print("\n=== Testing dsfsi-anv/za-african-next-voices ===")
try:
    dataset = load_dataset(
        "dsfsi-anv/za-african-next-voices",
        "zul",
        split="dev_test",
        streaming=True,
        token=token
    )
    print("✓ SUCCESS: Can access za-african-next-voices")
    first_item = next(iter(dataset))
    print(f"  First item keys: {list(first_item.keys())}")
except Exception as e:
    print(f"✗ FAILED: {e}")
