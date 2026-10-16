"""
Inference & Visualization Module.
Performs greedy decoding for both Baseline and Attention Seq2Seq models,
extracts attention matrices, and visualizes word alignment heatmaps.
"""

import os
import torch
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import (
    Vocabulary,
    normalize_string,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN
)
from model_baseline import Encoder as BaselineEncoder, Decoder as BaselineDecoder, Seq2SeqBaseline
from model_attention import Encoder as AttentionEncoder, Decoder as AttentionDecoder, Seq2SeqAttention


# ---------------------------------------------------------------------------
# 1. TRANSLATION WITH ATTENTION (WITH HEATMAP EXTRACTION)
# ---------------------------------------------------------------------------
@torch.no_grad()
def translate_attention(
    sentence: str,
    model: Seq2SeqAttention,
    eng_vocab: Vocabulary,
    uzb_vocab: Vocabulary,
    max_length: int = 25,
    device: torch.device = torch.device("cpu")
) -> Tuple[List[str], torch.Tensor]:
    """
    Translates an English sentence to Uzbek using Bahdanau Attention.
    
    Returns:
        translated_tokens: List of translated Uzbek words
        attention_matrix:  Tensor of shape [len(uzb_words), len(eng_words)]
    """
    model.eval()

    # 1. Clean and tokenize input
    clean_sent = normalize_string(sentence, is_uzbek=False)
    input_indices = [SOS_TOKEN] + eng_vocab.sentence_to_indices(clean_sent) + [EOS_TOKEN]
    src_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)  # [1, src_len]

    # 2. Encode
    encoder_outputs, hidden = model.encoder(src_tensor)
    source_mask = (src_tensor != PAD_TOKEN)

    # 3. Decode step-by-step
    input_token = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
    translated_tokens = []
    attention_weights_list = []

    for _ in range(max_length):
        logits, hidden, attn_weights = model.decoder(
            input_token, hidden, encoder_outputs, source_mask
        )
        # Record attention weights for this word: [src_len]
        attention_weights_list.append(attn_weights.squeeze(0).cpu())

        top1 = logits.argmax(dim=-1).item()
        if top1 == EOS_TOKEN:
            break

        translated_tokens.append(uzb_vocab.index2word.get(top1, "<UNK>"))
        input_token = torch.tensor([[top1]], dtype=torch.long, device=device)

    if not attention_weights_list:
        attention_matrix = torch.zeros(1, len(input_indices))
    else:
        attention_matrix = torch.stack(attention_weights_list, dim=0)  # [tgt_len, src_len]

    return translated_tokens, attention_matrix


# ---------------------------------------------------------------------------
# 2. TRANSLATION WITHOUT ATTENTION (BASELINE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def translate_baseline(
    sentence: str,
    model: Seq2SeqBaseline,
    eng_vocab: Vocabulary,
    uzb_vocab: Vocabulary,
    max_length: int = 25,
    device: torch.device = torch.device("cpu")
) -> List[str]:
    """
    Translates an English sentence using the Baseline (no attention) model.
    """
    model.eval()

    clean_sent = normalize_string(sentence, is_uzbek=False)
    input_indices = [SOS_TOKEN] + eng_vocab.sentence_to_indices(clean_sent) + [EOS_TOKEN]
    src_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)

    # Compress into single context vector
    _, hidden = model.encoder(src_tensor)

    input_token = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
    translated_tokens = []

    for _ in range(max_length):
        logits, hidden = model.decoder(input_token, hidden)
        top1 = logits.argmax(dim=-1).item()
        if top1 == EOS_TOKEN:
            break

        translated_tokens.append(uzb_vocab.index2word.get(top1, "<UNK>"))
        input_token = torch.tensor([[top1]], dtype=torch.long, device=device)

    return translated_tokens


# ---------------------------------------------------------------------------
# 3. ATTENTION HEATMAP VISUALIZATION
# ---------------------------------------------------------------------------
def plot_attention_heatmap(
    sentence: str,
    translated_tokens: List[str],
    attention_matrix: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plots an interactive 2D alignment matrix of English words vs. Uzbek words.
    """
    clean_sent = normalize_string(sentence, is_uzbek=False)
    src_tokens = ["<SOS>"] + clean_sent.split() + ["<EOS>"]
    tgt_tokens = translated_tokens

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(attention_matrix.numpy(), cmap="viridis")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="left", fontsize=11)
    ax.set_yticklabels(tgt_tokens, fontsize=11)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(f"Bahdanau Attention Alignment: '{sentence}'", pad=30, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved attention heatmap to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 4. MODEL LOADER UTILITY
# ---------------------------------------------------------------------------
def load_trained_models(
    checkpoint_dir: str = "checkpoints",
    device: torch.device = torch.device("cpu")
) -> Tuple[Optional[Seq2SeqBaseline], Optional[Seq2SeqAttention], Vocabulary, Vocabulary]:
    """Loads vocabulary mappings and available model weights from disk."""
    eng_vocab = Vocabulary.load(os.path.join(checkpoint_dir, "vocab_eng.json"))
    uzb_vocab = Vocabulary.load(os.path.join(checkpoint_dir, "vocab_uzb.json"))

    baseline_model = None
    attention_model = None

    # Load Baseline if checkpoint exists
    base_ckpt = os.path.join(checkpoint_dir, "seq2seq_baseline.pth")
    if os.path.exists(base_ckpt):
        state = torch.load(base_ckpt, map_location=device)
        enc = BaselineEncoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
        dec = BaselineDecoder(uzb_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
        baseline_model = Seq2SeqBaseline(enc, dec, device).to(device)
        baseline_model.load_state_dict(state["model_state_dict"])
        baseline_model.eval()

    # Load Attention if checkpoint exists
    attn_ckpt = os.path.join(checkpoint_dir, "seq2seq_attention.pth")
    if os.path.exists(attn_ckpt):
        state = torch.load(attn_ckpt, map_location=device)
        enc = AttentionEncoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
        dec = AttentionDecoder(uzb_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
        attention_model = Seq2SeqAttention(enc, dec, device).to(device)
        attention_model.load_state_dict(state["model_state_dict"])
        attention_model.eval()

    return baseline_model, attention_model, eng_vocab, uzb_vocab


if __name__ == "__main__":
    print("Inference module loaded successfully. Run after training checkpoints are created.")
