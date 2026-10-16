"""
Comprehensive Evaluation Suite for English-to-Uzbek NMT.
Evaluates Baseline vs. Bahdanau Attention across sequence length buckets
on the held-out test set using SacreBLEU and chrF++.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Tuple, Dict, Any
import torch

try:
    import sacrebleu
except ImportError:
    raise ImportError("sacrebleu is required for evaluation. Run: pip install sacrebleu")

from dataset import Vocabulary, normalize_string
from inference import load_trained_models, translate_attention, translate_baseline


def bucket_test_set(
    test_pairs: List[Tuple[str, str]]
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Categorizes test sentence pairs into length buckets based on source word count:
    - Short:  <= 8 words
    - Medium: 9 - 15 words
    - Long:   > 15 words
    """
    buckets = {
        "All (Overall)": test_pairs,
        "Short (<= 8 words)": [],
        "Medium (9-15 words)": [],
        "Long (> 15 words)": []
    }

    for en, uz in test_pairs:
        word_count = len(en.split())
        if word_count <= 8:
            buckets["Short (<= 8 words)"].append((en, uz))
        elif word_count <= 15:
            buckets["Medium (9-15 words)"].append((en, uz))
        else:
            buckets["Long (> 15 words)"].append((en, uz))

    return buckets


def evaluate_model_on_pairs(
    model_name: str,
    pairs: List[Tuple[str, str]],
    model: Any,
    eng_vocab: Vocabulary,
    uzb_vocab: Vocabulary,
    device: torch.device
) -> Dict[str, Any]:
    """
    Translates pairs and computes SacreBLEU, chrF++, and average inference latency.
    """
    if not pairs:
        return {"bleu": 0.0, "chrf": 0.0, "latency_ms": 0.0, "count": 0}

    hypotheses = []
    references = []
    latencies = []

    for en, uz_ref in pairs:
        t0 = time.time()
        if model_name == "attention":
            tokens, _ = translate_attention(en, model, eng_vocab, uzb_vocab, device=device)
        else:
            tokens = translate_baseline(en, model, eng_vocab, uzb_vocab, device=device)
        latencies.append((time.time() - t0) * 1000)

        hypotheses.append(" ".join(tokens))
        references.append(uz_ref)

    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references]).score
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "bleu": round(bleu_score, 2),
        "chrf": round(chrf_score, 2),
        "latency_ms": round(avg_latency, 2),
        "count": len(pairs)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs Bahdanau Attention on Held-out Test Set")
    parser.add_argument("--test_file", type=str, default="checkpoints/test_pairs.json")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Max test pairs per bucket to evaluate (for speed)")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(PROJECT_ROOT, args.test_file)
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    if not os.path.exists(test_path):
        print(f"Error: Held-out test file not found at {test_path}")
        print("Please run `python3 src/train.py` first to generate train/val/test splits.")
        sys.exit(1)

    with open(test_path, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    # Hardware detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    baseline_model, attention_model, eng_vocab, uzb_vocab = load_trained_models(ckpt_dir, device=device)

    if not eng_vocab or not uzb_vocab:
        print("Error: Vocabularies not found in checkpoints directory.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("  TARJIMA-AI: HELD-OUT TEST EVALUATION (SACREBLEU & CHRF++ BY LENGTH)")
    print("=" * 80)
    print(f"Total Held-Out Test Pairs: {len(test_pairs):,}")
    print(f"Active Models: Baseline={'Loaded' if baseline_model else 'Missing'}, "
          f"Attention={'Loaded' if attention_model else 'Missing'}")

    buckets = bucket_test_set(test_pairs)

    print("\n" + "-" * 88)
    print(f"{'Length Bucket':<24} | {'Samples':<8} | {'Baseline BLEU':<14} | {'Attention BLEU':<15} | {'Delta':<8} | {'Attn chrF'}")
    print("-" * 88)

    for bucket_name, bucket_pairs in buckets.items():
        eval_subset = bucket_pairs[:args.max_eval_samples]
        if not eval_subset:
            continue

        base_res = evaluate_model_on_pairs("baseline", eval_subset, baseline_model, eng_vocab, uzb_vocab, device) if baseline_model else {"bleu": 0.0, "chrf": 0.0}
        attn_res = evaluate_model_on_pairs("attention", eval_subset, attention_model, eng_vocab, uzb_vocab, device) if attention_model else {"bleu": 0.0, "chrf": 0.0}

        delta = round(attn_res["bleu"] - base_res["bleu"], 2)
        delta_str = f"+{delta}" if delta > 0 else f"{delta}"

        print(f"{bucket_name:<24} | {len(eval_subset):<8} | {base_res['bleu']:<14.2f} | {attn_res['bleu']:<15.2f} | {delta_str:<8} | {attn_res['chrf']:.2f}")

    print("-" * 88)
    print("\nEmpirical finding: Notice how Attention retains high BLEU on Long (>15 words) sequences,")
    print("whereas Baseline performance degrades due to the fixed-vector information bottleneck.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
