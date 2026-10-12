"""
One-time script to train a custom openWakeWord model for the word "jarvis".

Requirements:
    pip install openwakeword[training]

Usage:
    python scripts/train_wake_word.py

Output:
    models/jarvis.onnx   -- copy this path into wake_word.py WAKE_WORD_MODEL
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from openwakeword.train import train
except ImportError:
    print("ERROR: training extras not installed. Run: pip install openwakeword[training]")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent.parent / "home-agent-sk" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WORD = "jarvis"

print(f"Training wake word model for '{TARGET_WORD}'...")
print("This will generate synthetic audio samples and train a small neural network.")
print("Expected time: ~10 minutes on a modern CPU.\n")

train(
    target_word=TARGET_WORD,
    output_dir=str(OUTPUT_DIR),
    n_samples=500,
    steps=5_000,
)

model_path = OUTPUT_DIR / f"{TARGET_WORD}.onnx"
print(f"\nModel saved to: {model_path}")
print(f"\nTo use it, open src/home_agent/voice/wake_word.py and change:")
print(f'    WAKE_WORD_MODEL = "hey_jarvis"')
print(f'to:')
print(f'    WAKE_WORD_MODEL = r"{model_path}"')
