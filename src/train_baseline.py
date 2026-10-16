"""
Training Pipeline for Baseline Seq2Seq (WITHOUT Attention).
Trains on the exact same English-Uzbek parallel split to establish a fair benchmark.
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    load_pairs,
    Vocabulary,
    TranslationDataset,
    collate_fn,
    PAD_TOKEN
)
from model_baseline import Encoder, Decoder, Seq2SeqBaseline


# ---------------------------------------------------------------------------
# 1. ONE TRAINING EPOCH
# ---------------------------------------------------------------------------
def train_epoch(
    model: Seq2SeqBaseline,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip_norm: float = 1.0,
    device: torch.device = torch.device("cpu")
) -> float:
    """Trains the baseline model for one pass through the dataloader."""
    model.train()
    total_loss = 0.0

    for step, (src_batch, tgt_batch, _) in enumerate(dataloader):
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        # Forward pass: outputs is [batch_size, target_len - 1, output_vocab_size]
        outputs = model(src_batch, tgt_batch)

        output_dim = outputs.shape[-1]
        loss_outputs = outputs.contiguous().view(-1, output_dim)
        loss_targets = tgt_batch[:, 1:].contiguous().view(-1)

        loss = criterion(loss_outputs, loss_targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# 2. EVALUATION FUNCTION
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_epoch(
    model: Seq2SeqBaseline,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device = torch.device("cpu")
) -> float:
    """Evaluates validation loss with zero teacher forcing."""
    model.eval()
    total_loss = 0.0

    original_tf_ratio = model.teacher_forcing_ratio
    model.teacher_forcing_ratio = 0.0

    for src_batch, tgt_batch, _ in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        outputs = model(src_batch, tgt_batch)

        output_dim = outputs.shape[-1]
        loss_outputs = outputs.contiguous().view(-1, output_dim)
        loss_targets = tgt_batch[:, 1:].contiguous().view(-1)

        loss = criterion(loss_outputs, loss_targets)
        total_loss += loss.item()

    model.teacher_forcing_ratio = original_tf_ratio
    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# 3. MAIN TRAINING CONTROLLER
# ---------------------------------------------------------------------------
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA acceleration!")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Translations.jsonl")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    MAX_SAMPLES = 15000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    TEACHER_FORCING_RATIO = 0.5

    # 1. Load pairs
    print(f"\n[1/5] Loading clean English-Uzbek pairs from {DATA_PATH}...")
    pairs = load_pairs(DATA_PATH, max_samples=MAX_SAMPLES, min_length=2, max_length=20)
    print(f"Loaded {len(pairs)} filtered sentence pairs.")

    # 2. Split (90% Train / 10% Val)
    train_size = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:train_size], pairs[train_size:]

    # 3. Vocabularies
    print("\n[2/5] Building English and Uzbek vocabularies...")
    eng_vocab = Vocabulary("English")
    uzb_vocab = Vocabulary("Uzbek")

    for en, uz in train_pairs:
        eng_vocab.add_sentence(en)
        uzb_vocab.add_sentence(uz)

    print(f"Vocabulary sizes -> English: {eng_vocab.n_words} | Uzbek: {uzb_vocab.n_words}")

    # 4. DataLoaders
    print("\n[3/5] Initializing DataLoaders...")
    train_dataset = TranslationDataset(train_pairs, eng_vocab, uzb_vocab)
    val_dataset = TranslationDataset(val_pairs, eng_vocab, uzb_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 5. Initialize Baseline Model
    print("\n[4/5] Initializing Baseline Seq2Seq (without attention)...")
    encoder = Encoder(
        input_size=eng_vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_DIM,
        num_layers=1,
        dropout=0.1
    )
    decoder = Decoder(
        output_size=uzb_vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_DIM,
        num_layers=1,
        dropout=0.1
    )
    model = Seq2SeqBaseline(
        encoder=encoder,
        decoder=decoder,
        device=device,
        teacher_forcing_ratio=TEACHER_FORCING_RATIO
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # 6. Training Loop
    print("\n[5/5] Starting Baseline Training...")
    best_val_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, "seq2seq_baseline.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device=device)

        elapsed = time.time() - start_time
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl = math.exp(min(val_loss, 100))

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM
            }, best_model_path)
            print(f"  --> Saved new best baseline checkpoint to {best_model_path}")

    print(f"\nBaseline Training Complete! Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
