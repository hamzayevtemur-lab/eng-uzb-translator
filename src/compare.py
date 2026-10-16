"""
Head-to-Head Model Comparison Module.
Compares Baseline Seq2Seq vs. Bahdanau Attention Seq2Seq side-by-side
across short, medium, and long test sentences, with optional beam search decoding.
For held-out SacreBLEU evaluation, run: python3 src/evaluate.py
"""

import os
import time
from typing import List, Dict, Any
import torch

from inference import (
    load_trained_models,
    translate_baseline,
    translate_attention,
    beam_search_attention,
    plot_attention_heatmap
)


def run_comparison():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device for inference: {device}")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

    baseline_model, attention_model, eng_vocab, uzb_vocab = load_trained_models(
        checkpoint_dir=CHECKPOINT_DIR,
        device=device
    )

    if baseline_model is None or attention_model is None:
        print("\n[!] Warning: One or both model checkpoints were not found in checkpoints/")
        print("    Please train both models first using: python3 src/train.py --model all")
        return

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

    for sentence, category in test_cases:
        print(f"\n[{category}] Input: \"{sentence}\"")

        # 1. Baseline
        t0 = time.time()
        base_tokens = translate_baseline(sentence, baseline_model, eng_vocab, uzb_vocab, device=device)
        base_time = (time.time() - t0) * 1000

        # 2. Attention (Greedy)
        t0 = time.time()
        attn_tokens, attn_matrix = translate_attention(sentence, attention_model, eng_vocab, uzb_vocab, device=device)
        attn_time = (time.time() - t0) * 1000

        # 3. Attention (Beam Search)
        t0 = time.time()
        beam_tokens, _ = beam_search_attention(sentence, attention_model, eng_vocab, uzb_vocab, beam_width=3, device=device)
        beam_time = (time.time() - t0) * 1000

        print(f"  ⚡ Baseline (No Attention): {' '.join(base_tokens)}  [{base_time:.1f}ms]")
        print(f"  🌟 Bahdanau (Greedy):       {' '.join(attn_tokens)}  [{attn_time:.1f}ms]")
        print(f"  🔍 Bahdanau (Beam Width 3): {' '.join(beam_tokens)}  [{beam_time:.1f}ms]")

    print("\n" + "=" * 80)
    print("For statistical SacreBLEU and chrF++ scores across sentence length buckets, run:")
    print("  python3 src/evaluate.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comparison()
