"""
Training Pipeline for Seq2Seq with Bahdanau Attention.
Trains on English-to-Uzbek parallel translation pairs, evaluates on validation split,
and saves vocabulary mapping and best model checkpoint.
"""

import os
import time
import math
import random
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
from model_attention import Encoder, Decoder, Seq2SeqAttention
from inference import translate_attention


# ---------------------------------------------------------------------------
# 1. ONE TRAINING EPOCH
# ---------------------------------------------------------------------------
def train_epoch(
    model: Seq2SeqAttention,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip_norm: float = 1.0,
    device: torch.device = torch.device("cpu")
) -> float:
    """Trains the model for one full pass through the dataloader."""
    model.train()
    total_loss = 0.0

    for step, (src_batch, tgt_batch, _) in enumerate(dataloader):
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        # Forward pass: outputs is [batch_size, target_len - 1, output_vocab_size]
        outputs, _ = model(src_batch, tgt_batch)

        # Flatten outputs and targets for CrossEntropyLoss
        output_dim = outputs.shape[-1]
        loss_outputs = outputs.contiguous().view(-1, output_dim)
        loss_targets = tgt_batch[:, 1:].contiguous().view(-1)

        loss = criterion(loss_outputs, loss_targets)
        loss.backward()

        # Gradient clipping prevents exploding gradients in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# 2. EVALUATION FUNCTION
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_epoch(
    model: Seq2SeqAttention,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device = torch.device("cpu")
) -> float:
    """
    Evaluates validation cross-entropy measuring sequence perplexity.
    Uses teacher forcing = 1.0 to prevent positional cascade desynchronization.
    """
    model.eval()
    total_loss = 0.0

    original_tf_ratio = model.teacher_forcing_ratio
    model.teacher_forcing_ratio = 1.0  # Ground-truth sequence likelihood

    for src_batch, tgt_batch, _ in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        outputs, _ = model(src_batch, tgt_batch)

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
    # Hardware acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA acceleration!")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Paths & Hyperparameters
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Translations.jsonl")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    MAX_SAMPLES = 45000       # Use large dataset (out of 48k pairs)
    MIN_FREQ = 1              # Keep all words from dataset to prevent UNK inflation
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5       # L2 regularization
    DROPOUT = 0.3             # Dropout regularization
    NUM_EPOCHS = 12
    TEACHER_FORCING_RATIO = 0.5

    # 1. Load and normalize parallel pairs
    print(f"\n[1/5] Loading clean English-Uzbek pairs from {DATA_PATH}...")
    pairs = load_pairs(DATA_PATH, max_samples=MAX_SAMPLES, min_length=2, max_length=25)
    print(f"Loaded {len(pairs)} filtered sentence pairs.")

    # Shuffle before splitting
    random.seed(42)
    random.shuffle(pairs)

    # 2. Split into Train (90%) and Val (10%)
    train_size = int(0.9 * len(pairs))
    val_size = len(pairs) - train_size
    train_pairs, val_pairs = pairs[:train_size], pairs[train_size:]
    print(f"Split: {len(train_pairs):,} training pairs | {len(val_pairs):,} validation pairs.")

    # 3. Build Vocabularies
    print("\n[2/5] Building English and Uzbek vocabularies...")
    eng_vocab = Vocabulary("English")
    uzb_vocab = Vocabulary("Uzbek")

    for en, uz in train_pairs:
        eng_vocab.add_sentence(en)
        uzb_vocab.add_sentence(uz)

    # Filter rare one-off words
    eng_vocab.trim(min_freq=MIN_FREQ)
    uzb_vocab.trim(min_freq=MIN_FREQ)

    print(f"Final Vocabulary Sizes -> English: {eng_vocab.n_words:,} | Uzbek: {uzb_vocab.n_words:,}")

    # Save vocabularies for inference
    eng_vocab.save(os.path.join(CHECKPOINT_DIR, "vocab_eng.json"))
    uzb_vocab.save(os.path.join(CHECKPOINT_DIR, "vocab_uzb.json"))
    print(f"Saved vocabularies to {CHECKPOINT_DIR}/")

    # 4. Create PyTorch DataLoaders
    print("\n[3/5] Initializing batched DataLoaders...")
    train_dataset = TranslationDataset(train_pairs, eng_vocab, uzb_vocab)
    val_dataset = TranslationDataset(val_pairs, eng_vocab, uzb_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 5. Initialize Model, Optimizer, Criterion
    print("\n[4/5] Initializing Seq2Seq with Bahdanau Attention model...")
    encoder = Encoder(
        input_size=eng_vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_DIM,
        num_layers=1,
        dropout=DROPOUT
    )
    decoder = Decoder(
        output_size=uzb_vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_DIM,
        num_layers=1,
        dropout=DROPOUT
    )
    model = Seq2SeqAttention(
        encoder=encoder,
        decoder=decoder,
        device=device,
        teacher_forcing_ratio=TEACHER_FORCING_RATIO
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # 6. Training Loop
    print("\n[5/5] Starting Training...")
    best_val_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, "seq2seq_attention.pth")
    test_sample = "I love reading books in the library."

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device=device)

        scheduler.step(val_loss)
        elapsed = time.time() - start_time
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl = math.exp(min(val_loss, 100))

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")

        # Live sample translation check
        sample_tokens, _ = translate_attention(test_sample, model, eng_vocab, uzb_vocab, device=device)
        print(f"  [Sample] '{test_sample}' -> '{' '.join(sample_tokens)}'")

        # Save checkpoint if validation loss improves
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
            print(f"  --> Saved new best checkpoint to {best_model_path}")

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
