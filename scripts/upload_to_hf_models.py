#!/usr/bin/env python3
"""
Upload TarjimaAI trained PyTorch models and vocabularies to Hugging Face Model Hub.
Repository: https://huggingface.co/TemurbekHamzaev/TarjimaAI-English-Uzbek-Seq2Seq
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def parse_args():
    parser = argparse.ArgumentParser(description="Upload TarjimaAI models to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face Write Token")
    parser.add_argument("--repo_id", type=str, default="TemurbekHamzaev/TarjimaAI-English-Uzbek-Seq2Seq", help="Model repo id")
    return parser.parse_args()

def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: A valid Hugging Face Write Token is required via --token or HF_TOKEN.")
        sys.exit(1)

    print("=" * 70)
    print("  🌟 UPLOADING TARJIMA-AI MODELS TO HUGGING FACE MODEL HUB")
    print(f"  Target: https://huggingface.co/{args.repo_id}")
    print("=" * 70)

    login(token=token, add_to_git_credential=True)
    api = HfApi(token=token)

    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    checkpoints_dir = Path(__file__).resolve().parent.parent / "checkpoints"

    # Upload vocabularies
    print("\n[1/4] Uploading vocab_eng.json...")
    api.upload_file(
        path_or_fileobj=str(checkpoints_dir / "vocab_eng.json"),
        path_in_repo="vocab_eng.json",
        repo_id=args.repo_id,
        repo_type="model"
    )

    print("[2/4] Uploading vocab_uzb.json...")
    api.upload_file(
        path_or_fileobj=str(checkpoints_dir / "vocab_uzb.json"),
        path_in_repo="vocab_uzb.json",
        repo_id=args.repo_id,
        repo_type="model"
    )

    # Upload cleaned checkpoints (stripped of optimizer states to save memory)
    attn_pth = checkpoints_dir / "seq2seq_attention_clean.pth"
    if not attn_pth.exists():
        attn_pth = checkpoints_dir / "seq2seq_attention.pth"
    
    print(f"\n[3/4] Uploading Bahdanau Attention model ({round(attn_pth.stat().st_size / (1024*1024), 1)} MB)...")
    api.upload_file(
        path_or_fileobj=str(attn_pth),
        path_in_repo="seq2seq_attention.pth",
        repo_id=args.repo_id,
        repo_type="model"
    )

    base_pth = checkpoints_dir / "seq2seq_baseline_clean.pth"
    if not base_pth.exists():
        base_pth = checkpoints_dir / "seq2seq_baseline.pth"

    print(f"[4/4] Uploading Baseline Seq2Seq model ({round(base_pth.stat().st_size / (1024*1024), 1)} MB)...")
    api.upload_file(
        path_or_fileobj=str(base_pth),
        path_in_repo="seq2seq_baseline.pth",
        repo_id=args.repo_id,
        repo_type="model"
    )

    # Upload Model Card README
    model_card = f"""---
language:
- en
- uz
tags:
- translation
- seq2seq
- pytorch
- bahdanau-attention
- machine-translation
license: mit
---

# 🌟 TarjimaAI: English-to-Uzbek Neural Machine Translation (Seq2Seq with Bahdanau Attention)

Official PyTorch model checkpoints and token vocabularies for **TarjimaAI**, a neural machine translation system trained from scratch on parallel English-Uzbek corpora.

## 📊 Model Architecture
- **Encoder**: Bidirectional GRU (Embedding: 256, Hidden: 512, Dropout: 0.3)
- **Attention**: Bahdanau Additive Attention ($score(s_t, h_i) = v_a^T \\tanh(W_a s_t + U_a h_i)$)
- **Decoder**: GRU with dynamic context vectors concatenated with target token embeddings
- **Vocabulary Sizes**: English: 17,876 tokens | Uzbek: 30,965 tokens

## 📦 Files in this repository
- `seq2seq_attention.pth`: Trained Bahdanau Attention Seq2Seq weights.
- `seq2seq_baseline.pth`: Baseline fixed-context bottleneck Seq2Seq weights.
- `vocab_eng.json`: English source vocabulary token mappings.
- `vocab_uzb.json`: Uzbek target vocabulary token mappings.

## 🚀 Interactive Live Web App
- **Hugging Face Space**: [https://huggingface.co/spaces/TemurbekHamzaev/TarjimaAI-English-Uzbek-Translator](https://huggingface.co/spaces/TemurbekHamzaev/TarjimaAI-English-Uzbek-Translator)
- **GitHub Repository**: [https://github.com/hamzayevtemur-lab/eng-uzb-translator](https://github.com/hamzayevtemur-lab/eng-uzb-translator)
"""
    readme_path = Path("/tmp/README.md")
    readme_path.write_text(model_card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model"
    )

    print("\n" + "=" * 70)
    print("🎉 ALL MODELS AND VOCABULARIES UPLOADED TO HUGGING FACE MODEL HUB!")
    print(f"👉 https://huggingface.co/{args.repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
