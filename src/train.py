"""
Unified Training Pipeline for English-to-Uzbek Neural Machine Translation.
Supports:
- Shared Bidirectional Packed Encoder (eliminates PAD pollution)
- Train / Validation / Held-Out Test (80 / 10 / 10) split
- Decoupled single-step vocabulary builder
- Model selection via CLI: --model attention | baseline | all
- Checkpoint hyperparameter serialization
"""

import os
import time
import math
import json
import argparse
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    load_dataset_splits,
    Vocabulary,
    TranslationDataset,
    collate_fn,
    PAD_TOKEN
)
from model_attention import Encoder, Decoder as AttentionDecoder, Seq2SeqAttention
from model_baseline import Decoder as BaselineDecoder, Seq2SeqBaseline
from inference import translate_attention, translate_baseline


# ---------------------------------------------------------------------------
# 1. GENERIC TRAINING AND EVALUATION LOOPS
# ---------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip_norm: float = 1.0,
    device: torch.device = torch.device("cpu")
) -> float:
    """Trains the model for one full epoch with packed sequence lengths."""
    model.train()
    total_loss = 0.0

    for src_batch, tgt_batch, src_lengths in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        # Both Seq2SeqAttention and Seq2SeqBaseline receive src_lengths for packed GRU
        if isinstance(model, Seq2SeqAttention):
            outputs, _ = model(src_batch, tgt_batch, src_lengths=src_lengths)
        else:
            outputs = model(src_batch, tgt_batch, src_lengths=src_lengths)

        output_dim = outputs.shape[-1]
        loss_outputs = outputs.contiguous().view(-1, output_dim)
        loss_targets = tgt_batch[:, 1:].contiguous().view(-1)

        loss = criterion(loss_outputs, loss_targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device = torch.device("cpu")
) -> float:
    """Evaluates validation sequence perplexity with ground-truth sequence conditioning."""
    model.eval()
    total_loss = 0.0

    original_tf_ratio = model.teacher_forcing_ratio
    model.teacher_forcing_ratio = 1.0

    for src_batch, tgt_batch, src_lengths in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        if isinstance(model, Seq2SeqAttention):
            outputs, _ = model(src_batch, tgt_batch, src_lengths=src_lengths)
        else:
            outputs = model(src_batch, tgt_batch, src_lengths=src_lengths)

        output_dim = outputs.shape[-1]
        loss_outputs = outputs.contiguous().view(-1, output_dim)
        loss_targets = tgt_batch[:, 1:].contiguous().view(-1)

        loss = criterion(loss_outputs, loss_targets)
        total_loss += loss.item()

    model.teacher_forcing_ratio = original_tf_ratio
    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# 2. MODEL BUILDERS
# ---------------------------------------------------------------------------
def build_attention_model(
    eng_vocab_size: int,
    uzb_vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    teacher_forcing_ratio: float,
    device: torch.device
) -> Seq2SeqAttention:
    encoder = Encoder(
        input_size=eng_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )
    decoder = AttentionDecoder(
        output_size=uzb_vocab_size,
        embedding_dim=embedding_dim,
        enc_hidden_size=hidden_dim * 2,  # Bidirectional encoder
        dec_hidden_size=hidden_dim,
        dropout=dropout
    )
    return Seq2SeqAttention(encoder, decoder, device, teacher_forcing_ratio).to(device)


def build_baseline_model(
    eng_vocab_size: int,
    uzb_vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    teacher_forcing_ratio: float,
    device: torch.device
) -> Seq2SeqBaseline:
    encoder = Encoder(
        input_size=eng_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )
    decoder = BaselineDecoder(
        output_size=uzb_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )
    return Seq2SeqBaseline(encoder, decoder, device, teacher_forcing_ratio).to(device)


# ---------------------------------------------------------------------------
# 3. MASTER TRAINER
# ---------------------------------------------------------------------------
def run_training(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eng_vocab: Vocabulary,
    uzb_vocab: Vocabulary,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: str
):
    print("\n" + "=" * 70)
    print(f"  STARTING TRAINING: {model_name.upper()} SEQ2SEQ")
    print("=" * 70)

    if model_name == "attention":
        model = build_attention_model(
            eng_vocab_size=eng_vocab.n_words,
            uzb_vocab_size=uzb_vocab.n_words,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            device=device
        )
        ckpt_filename = "seq2seq_attention.pth"
    else:
        model = build_baseline_model(
            eng_vocab_size=eng_vocab.n_words,
            uzb_vocab_size=uzb_vocab.n_words,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            device=device
        )
        ckpt_filename = "seq2seq_baseline.pth"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, ckpt_filename)
    test_sample = "I love reading books in the library."

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device=device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl = math.exp(min(val_loss, 100))

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")

        # Live sample translation
        if model_name == "attention":
            sample_tokens, _ = translate_attention(test_sample, model, eng_vocab, uzb_vocab, device=device)
        else:
            sample_tokens = translate_baseline(test_sample, model, eng_vocab, uzb_vocab, device=device)
        print(f"  [Sample] '{test_sample}' -> '{' '.join(sample_tokens)}'")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "enc_hidden_dim": args.hidden_dim * 2,
                "dec_hidden_dim": args.hidden_dim,
                "vocab_size_eng": eng_vocab.n_words,
                "vocab_size_uzb": uzb_vocab.n_words
            }, best_path)
            print(f"  --> Saved new best checkpoint to {best_path}")

    print(f"\n{model_name.capitalize()} Training Complete! Best Validation Loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train English-to-Uzbek Seq2Seq Neural Translator")
    parser.add_argument("--model", type=str, choices=["attention", "baseline", "all"], default="attention")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=45000)
    args = parser.parse_args()

    # Hardware detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware Accelerator: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Hardware Accelerator: NVIDIA CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Hardware Accelerator: CPU")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Translations.jsonl")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Stratified Splits: 80% Train, 10% Validation, 10% Held-Out Test
    print(f"\n[1/4] Loading and partitioning dataset from {DATA_PATH}...")
    train_pairs, val_pairs, test_pairs = load_dataset_splits(
        filepath=DATA_PATH,
        max_samples=args.max_samples,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    # Persist held-out test split for sacrebleu evaluation
    test_path = os.path.join(CHECKPOINT_DIR, "test_pairs.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(test_pairs):,} held-out test pairs to {test_path}")

    # 2. Build and persist vocabularies ONCE
    print("\n[2/4] Building vocabularies from training split...")
    eng_vocab = Vocabulary("English")
    uzb_vocab = Vocabulary("Uzbek")

    for en, uz in train_pairs:
        eng_vocab.add_sentence(en)
        uzb_vocab.add_sentence(uz)

    eng_vocab_path = os.path.join(CHECKPOINT_DIR, "vocab_eng.json")
    uzb_vocab_path = os.path.join(CHECKPOINT_DIR, "vocab_uzb.json")
    eng_vocab.save(eng_vocab_path)
    uzb_vocab.save(uzb_vocab_path)
    print(f"Final Vocabularies -> English: {eng_vocab.n_words:,} | Uzbek: {uzb_vocab.n_words:,}")
    print(f"Saved vocabularies to {CHECKPOINT_DIR}/")

    # 3. DataLoaders with length tracking
    print("\n[3/4] Preparing batched DataLoaders with sequence length collation...")
    train_dataset = TranslationDataset(train_pairs, eng_vocab, uzb_vocab)
    val_dataset = TranslationDataset(val_pairs, eng_vocab, uzb_vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Train Selected Model(s)
    if args.model in ("baseline", "all"):
        run_training("baseline", train_loader, val_loader, eng_vocab, uzb_vocab, args, device, CHECKPOINT_DIR)

    if args.model in ("attention", "all"):
        run_training("attention", train_loader, val_loader, eng_vocab, uzb_vocab, args, device, CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
