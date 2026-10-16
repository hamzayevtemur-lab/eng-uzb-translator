"""
Inference & Visualization Module.
Performs greedy decoding and beam search decoding for both Baseline and Attention Seq2Seq models,
extracts attention matrices, and visualizes word alignment heatmaps.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F

from dataset import (
    Vocabulary,
    normalize_string,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN
)
from model_baseline import Decoder as BaselineDecoder, Seq2SeqBaseline
from model_attention import Encoder, Decoder as AttentionDecoder, Seq2SeqAttention


# ---------------------------------------------------------------------------
# 1. GREEDY TRANSLATION WITH ATTENTION (WITH HEATMAP EXTRACTION)
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
    Translates an English sentence to Uzbek using Bahdanau Attention (Greedy decoding).
    
    Returns:
        translated_tokens: List of translated Uzbek words
        attention_matrix:  Tensor of shape [len(uzb_words), len(eng_words)]
    """
    model.eval()

    # 1. Clean and tokenize input
    clean_sent = normalize_string(sentence, is_uzbek=False)
    input_indices = [SOS_TOKEN] + eng_vocab.sentence_to_indices(clean_sent) + [EOS_TOKEN]
    src_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)  # [1, src_len]
    src_lengths = torch.tensor([len(input_indices)], dtype=torch.long)

    # 2. Encode with packed sequence support
    encoder_outputs, hidden = model.encoder(src_tensor, src_lengths=src_lengths)
    source_mask = (src_tensor != PAD_TOKEN)

    # 3. Decode step-by-step
    input_token = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
    translated_tokens = []
    attention_weights_list = []
    generated_indices = []

    for _ in range(max_length):
        logits, hidden, attn_weights = model.decoder(
            input_token, hidden, encoder_outputs, source_mask
        )
        attention_weights_list.append(attn_weights.squeeze(0).cpu())

        # Mask out special non-word tokens
        logits[:, PAD_TOKEN] = -float("inf")
        logits[:, SOS_TOKEN] = -float("inf")
        logits[:, UNK_TOKEN] = -float("inf")

        # Repetition penalty on recently generated tokens
        for prev_idx in generated_indices[-3:]:
            logits[:, prev_idx] -= 2.0

        top1 = logits.argmax(dim=-1).item()
        if top1 == EOS_TOKEN:
            break

        word = uzb_vocab.index2word.get(top1, "<UNK>")
        translated_tokens.append(word)
        generated_indices.append(top1)

        if word in (".", "!", "?") and len(translated_tokens) > 2:
            break

        input_token = torch.tensor([[top1]], dtype=torch.long, device=device)

    if not attention_weights_list:
        attention_matrix = torch.zeros(1, len(input_indices))
    else:
        attention_matrix = torch.stack(attention_weights_list, dim=0)  # [tgt_len, src_len]

    return translated_tokens, attention_matrix


# ---------------------------------------------------------------------------
# 2. BEAM SEARCH TRANSLATION WITH ATTENTION
# ---------------------------------------------------------------------------
@torch.no_grad()
def beam_search_attention(
    sentence: str,
    model: Seq2SeqAttention,
    eng_vocab: Vocabulary,
    uzb_vocab: Vocabulary,
    beam_width: int = 3,
    max_length: int = 25,
    device: torch.device = torch.device("cpu")
) -> Tuple[List[str], torch.Tensor]:
    """
    Translates an English sentence using Beam Search decoding with Bahdanau Attention.
    Maintains top-K candidate paths, yielding noticeable fluency gains over greedy decoding.
    """
    model.eval()

    clean_sent = normalize_string(sentence, is_uzbek=False)
    input_indices = [SOS_TOKEN] + eng_vocab.sentence_to_indices(clean_sent) + [EOS_TOKEN]
    src_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
    src_lengths = torch.tensor([len(input_indices)], dtype=torch.long)

    encoder_outputs, hidden = model.encoder(src_tensor, src_lengths=src_lengths)
    source_mask = (src_tensor != PAD_TOKEN)

    # Beam item: (cumulative_log_prob, [token_ids], hidden_state, [attn_weights])
    beams = [(0.0, [SOS_TOKEN], hidden, [])]
    completed_beams = []

    for step in range(max_length):
        all_candidates = []

        for score, tokens, h, attn_hist in beams:
            last_token = tokens[-1]
            if last_token == EOS_TOKEN:
                completed_beams.append((score / (len(tokens) ** 0.7), tokens, attn_hist))
                continue

            input_tok = torch.tensor([[last_token]], dtype=torch.long, device=device)
            logits, next_h, attn_w = model.decoder(input_tok, h, encoder_outputs, source_mask)

            # Mask non-word tokens
            logits[:, PAD_TOKEN] = -float("inf")
            logits[:, SOS_TOKEN] = -float("inf")
            logits[:, UNK_TOKEN] = -float("inf")

            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab_size]
            topk_probs, topk_ids = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                tok_id = topk_ids[i].item()
                tok_score = topk_probs[i].item()
                new_tokens = tokens + [tok_id]
                new_attn = attn_hist + [attn_w.squeeze(0).cpu()]
                all_candidates.append((score + tok_score, new_tokens, next_h, new_attn))

        if not all_candidates:
            break

        # Prune down to top beam_width
        all_candidates.sort(key=lambda x: x[0] / (len(x[1]) ** 0.7), reverse=True)
        beams = all_candidates[:beam_width]

        if len(completed_beams) >= beam_width:
            break

    if completed_beams:
        completed_beams.sort(key=lambda x: x[0], reverse=True)
        best_score, best_tokens, best_attn = completed_beams[0]
    else:
        beams.sort(key=lambda x: x[0], reverse=True)
        best_score, best_tokens, _, best_attn = beams[0]

    # Filter SOS and EOS
    translated_tokens = [
        uzb_vocab.index2word.get(t, "<UNK>") 
        for t in best_tokens 
        if t not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)
    ]

    if not best_attn:
        attention_matrix = torch.zeros(1, len(input_indices))
    else:
        attention_matrix = torch.stack(best_attn, dim=0)[:len(translated_tokens)]

    return translated_tokens, attention_matrix


# ---------------------------------------------------------------------------
# 3. TRANSLATION WITHOUT ATTENTION (BASELINE)
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
    src_lengths = torch.tensor([len(input_indices)], dtype=torch.long)

    # Baseline uses only the bridge-projected initial hidden state
    _, hidden = model.encoder(src_tensor, src_lengths=src_lengths)

    input_token = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
    translated_tokens = []
    generated_indices = []

    for _ in range(max_length):
        logits, hidden = model.decoder(input_token, hidden)

        logits[:, PAD_TOKEN] = -float("inf")
        logits[:, SOS_TOKEN] = -float("inf")
        logits[:, UNK_TOKEN] = -float("inf")

        for prev_idx in generated_indices[-3:]:
            logits[:, prev_idx] -= 2.0

        top1 = logits.argmax(dim=-1).item()
        if top1 == EOS_TOKEN:
            break

        word = uzb_vocab.index2word.get(top1, "<UNK>")
        translated_tokens.append(word)
        generated_indices.append(top1)

        if word in (".", "!", "?") and len(translated_tokens) > 2:
            break

        input_token = torch.tensor([[top1]], dtype=torch.long, device=device)

    return translated_tokens


# ---------------------------------------------------------------------------
# 4. ATTENTION HEATMAP VISUALIZATION
# ---------------------------------------------------------------------------
def plot_attention_heatmap(
    sentence: str,
    translated_tokens: List[str],
    attention_matrix: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plots a 2D alignment matrix of English words vs. Uzbek words.
    Imports matplotlib lazily to keep server startup fast and lightweight.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    clean_sent = normalize_string(sentence, is_uzbek=False)
    src_tokens = ["<SOS>"] + clean_sent.split() + ["<EOS>"]
    tgt_tokens = translated_tokens

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(attention_matrix.numpy(), cmap="viridis")
    fig.colorbar(cax)

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
# 5. MODEL LOADER UTILITY
# ---------------------------------------------------------------------------
def load_trained_models(
    checkpoint_dir: str = "checkpoints",
    device: torch.device = torch.device("cpu")
) -> Tuple[Optional[Seq2SeqBaseline], Optional[Seq2SeqAttention], Optional[Vocabulary], Optional[Vocabulary]]:
    """Loads vocabulary mappings and available model weights from disk safely."""
    vocab_eng_path = os.path.join(checkpoint_dir, "vocab_eng.json")
    vocab_uzb_path = os.path.join(checkpoint_dir, "vocab_uzb.json")

    if not os.path.exists(vocab_eng_path) or not os.path.exists(vocab_uzb_path):
        return None, None, None, None

    eng_vocab = Vocabulary.load(vocab_eng_path)
    uzb_vocab = Vocabulary.load(vocab_uzb_path)

    baseline_model = None
    attention_model = None

    base_ckpt = os.path.join(checkpoint_dir, "seq2seq_baseline.pth")
    if os.path.exists(base_ckpt):
        try:
            state = torch.load(base_ckpt, map_location=device, weights_only=True)
            enc = Encoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            dec = BaselineDecoder(uzb_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            baseline_model = Seq2SeqBaseline(enc, dec, device).to(device)
            baseline_model.load_state_dict(state["model_state_dict"])
            baseline_model.eval()
        except Exception as e:
            print(f"[ModelLoader] Error loading baseline: {e}")

    attn_ckpt = os.path.join(checkpoint_dir, "seq2seq_attention.pth")
    if os.path.exists(attn_ckpt):
        try:
            state = torch.load(attn_ckpt, map_location=device, weights_only=True)
            enc = Encoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            dec = AttentionDecoder(
                uzb_vocab.n_words, 
                state["embedding_dim"], 
                state["hidden_dim"] * 2,  # enc_hidden_size
                state["hidden_dim"]       # dec_hidden_size
            )
            attention_model = Seq2SeqAttention(enc, dec, device).to(device)
            attention_model.load_state_dict(state["model_state_dict"])
            attention_model.eval()
        except Exception as e:
            print(f"[ModelLoader] Error loading attention model: {e}")

    return baseline_model, attention_model, eng_vocab, uzb_vocab
