"""
Head-to-Head Model Comparison Module.
Compares Baseline Seq2Seq vs. Bahdanau Attention Seq2Seq side-by-side
across short, medium, and long test sentences.
"""

import os
import time
import torch
from typing import List, Dict, Any

from inference import (
    load_trained_models,
    translate_baseline,
    translate_attention,
    plot_attention_heatmap
)


def run_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running comparison on device: {device}")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

    # 1. Load models
    baseline_model, attention_model, eng_vocab, uzb_vocab = load_trained_models(
        checkpoint_dir=CHECKPOINT_DIR,
        device=device
    )

    if baseline_model is None or attention_model is None:
        print("\n[!] Warning: One or both model checkpoints were not found in checkpoints/")
        print("    Please train both models first using:")
        print("      python3 src/train_baseline.py")
        print("      python3 src/train_attention.py")
        return

    # 2. Benchmark Test Suite (Short, Medium, Long)
    test_cases = [
        # Short
        ("Good morning.", "Short (2 words)"),
        ("I am happy.", "Short (3 words)"),
        ("She likes reading books.", "Short (4 words)"),
        
        # Medium
        ("He wanted to buy a new house in the city.", "Medium (10 words)"),
        ("They visited the beautiful park with their friends yesterday.", "Medium (9 words)"),
        ("The young girl was sitting quietly in the corner.", "Medium (9 words)"),
        
        # Long (Stress-testing the Information Bottleneck)
        ("He loved reading books every afternoon because he wanted to learn new things about the world.", "Long (16 words)"),
        ("After living in the big city for several years they decided to find a smaller house.", "Long (16 words)")
    ]

    print("\n" + "=" * 80)
    print("      HEAD-TO-HEAD BENCHMARK: BASELINE vs. BAHDANAU ATTENTION")
    print("=" * 80)

    for i, (sentence, category) in enumerate(test_cases, start=1):
        print(f"\n[Test #{i} | {category}]")
        print(f"  English Source: \"{sentence}\"")

        # Baseline Translation
        t0 = time.time()
        base_tokens = translate_baseline(sentence, baseline_model, eng_vocab, uzb_vocab, device=device)
        base_time = (time.time() - t0) * 1000
        base_translation = " ".join(base_tokens)

        # Attention Translation
        t0 = time.time()
        attn_tokens, attn_matrix = translate_attention(sentence, attention_model, eng_vocab, uzb_vocab, device=device)
        attn_time = (time.time() - t0) * 1000
        attn_translation = " ".join(attn_tokens)

        print(f"  -> Baseline (No Attn): \"{base_translation}\" ({base_time:.1f}ms)")
        print(f"  -> Attention Model:   \"{attn_translation}\" ({attn_time:.1f}ms)")

    print("\n" + "=" * 80)
    print("Benchmark complete! To see the attention heatmap for any sentence, use inference.plot_attention_heatmap()")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
