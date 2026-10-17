"""
Attention-Augmented Seq2Seq Model (Bahdanau et al., 2015).
Features:
- Bidirectional GRU Encoder with pack_padded_sequence to eliminate the <PAD> processing bug
- Bridge projection layer mapping [h_forward; h_backward] -> s_0
- Bahdanau Additive Attention with U_a in R^{d x 2d} matching the theoretical formulation
- Dynamic context vector routing at every decoding timestep
"""

import random
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dataset import PAD_TOKEN
except ImportError:
    from dataset import PAD_TOKEN


# ---------------------------------------------------------------------------
# 1. BIDIRECTIONAL ENCODER WITH PACKED SEQUENCE SUPPORT
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder that processes variable-length sequences.
    Uses pack_padded_sequence so the final hidden state strictly reflects the
    last non-pad token, eliminating padding degradation.
    """
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Bridge layer: s_0 = tanh(W_bridge * [h_forward; h_backward])
        # Projects concatenated bidirectional states (2 * hidden_size) to decoder state (hidden_size)
        self.fc_bridge = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self, 
        source_tokens: torch.Tensor, 
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_tokens: [batch_size, source_length] — token IDs
            src_lengths:   [batch_size] — actual sequence lengths (excluding <PAD>)
        Returns:
            outputs: [batch_size, source_length, hidden_size * 2] — bidirectional encoder annotations h_i
            s0:      [1, batch_size, hidden_size] — initial decoder hidden state
        """
        embedded = self.dropout(self.embedding(source_tokens))

        if src_lengths is not None:
            # Move lengths to CPU for packing (PyTorch requirement)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_outputs, hidden = self.gru(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, hidden = self.gru(embedded)

        # hidden shape: [2 * num_layers, batch_size, hidden_size]
        # Extract the last layer's forward and backward hidden states
        h_forward = hidden[-2, :, :]
        h_backward = hidden[-1, :, :]
        bridge_input = torch.cat([h_forward, h_backward], dim=1)  # [batch, hidden_size * 2]
        s0 = torch.tanh(self.fc_bridge(bridge_input)).unsqueeze(0)  # [1, batch, hidden_size]

        return outputs, s0


# ---------------------------------------------------------------------------
# 2. BAHDANAU (ADDITIVE) ATTENTION MODULE
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    """
    Bahdanau Additive Attention:
    score(s_{t-1}, h_i) = v_a^T * tanh(W_a * s_{t-1} + U_a * h_i)
    
    Where:
        h_i in R^{2d} (bidirectional encoder outputs)
        s_{t-1} in R^{d} (unidirectional decoder hidden state)
        U_a in R^{d x 2d}
        W_a in R^{d x d}
        v_a in R^{d}
    """
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int):
        super().__init__()
        # Projects encoder annotations h_i: [batch, src_len, enc_hidden_size] -> [batch, src_len, dec_hidden_size]
        self.encoder_projection = nn.Linear(enc_hidden_size, dec_hidden_size, bias=False)

        # Projects decoder state s_{t-1}: [batch, dec_hidden_size] -> [batch, dec_hidden_size]
        self.decoder_projection = nn.Linear(dec_hidden_size, dec_hidden_size, bias=False)

        # Alignment energy vector v_a
        self.energy = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden:  [batch_size, dec_hidden_size] — s_{t-1} query
            encoder_outputs: [batch_size, src_len, enc_hidden_size] — h_i keys/values
            source_mask:     [batch_size, src_len] — True for non-pad tokens
        Returns:
            context:           [batch_size, 1, enc_hidden_size] — dynamic context vector c_t
            attention_weights: [batch_size, src_len] — alpha_{t, i} softmax distribution
        """
        # [batch, src_len, dec_hidden_size]
        encoder_features = self.encoder_projection(encoder_outputs)

        # [batch, 1, dec_hidden_size]
        decoder_features = self.decoder_projection(decoder_hidden).unsqueeze(1)

        # Additive combination: tanh(U_a * h_i + W_a * s_{t-1})
        combined = torch.tanh(encoder_features + decoder_features)

        # Alignment scores: [batch, src_len]
        scores = self.energy(combined).squeeze(-1)

        # Mask padding tokens so their attention weight is 0.0
        if source_mask is not None:
            scores = scores.masked_fill(~source_mask, float("-inf"))

        # Softmax across source sequence tokens
        attention_weights = F.softmax(scores, dim=-1)

        # Context vector c_t: [batch, 1, src_len] x [batch, src_len, enc_hidden_size] -> [batch, 1, enc_hidden_size]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)

        return context, attention_weights


# ---------------------------------------------------------------------------
# 3. ATTENTION DECODER
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Decodes target Uzbek tokens conditioned on previous token embedding
    and the dynamic Bahdanau context vector.
    """
    def __init__(
        self,
        output_size: int,
        embedding_dim: int,
        enc_hidden_size: int,
        dec_hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(enc_hidden_size, dec_hidden_size)

        # GRU input: [y_{t-1}; c_t]
        self.gru = nn.GRU(
            input_size=embedding_dim + enc_hidden_size,
            hidden_size=dec_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Vocabulary projection: [s_t; c_t] -> vocab_size
        self.fc_out = nn.Linear(dec_hidden_size + enc_hidden_size, output_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, emb_dim]
        last_hidden = hidden[-1]  # [batch, dec_hidden_size]

        context, attention_weights = self.attention(last_hidden, encoder_outputs, source_mask)

        # Concat [y_{t-1}; c_t]
        gru_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.gru(gru_input, hidden)

        # Concat [s_t; c_t]
        combined = torch.cat([output.squeeze(1), context.squeeze(1)], dim=1)
        logits = self.fc_out(combined)

        return logits, hidden, attention_weights


# ---------------------------------------------------------------------------
# 4. UNIFIED SEQ2SEQ ATTENTION WRAPPER
# ---------------------------------------------------------------------------
class Seq2SeqAttention(nn.Module):
    """
    Coordinates Bidirectional Encoder and Attention-Augmented Decoder.
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: torch.device,
        teacher_forcing_ratio: float = 0.5
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = source.size(0)
        target_len = target.size(1)
        source_len = source.size(1)
        output_vocab_size = self.decoder.fc_out.out_features

        if source_mask is None:
            source_mask = (source != PAD_TOKEN)

        outputs = torch.zeros(batch_size, target_len - 1, output_vocab_size, device=self.device)
        attention_history = torch.zeros(batch_size, target_len - 1, source_len, device=self.device)

        # 1. Encode with packing to eliminate pad corruption
        encoder_outputs, hidden = self.encoder(source, src_lengths=src_lengths)

        # 2. First decoder input is target <SOS>
        input_token = target[:, 0].unsqueeze(1)

        # 3. Decode step-by-step
        for t in range(1, target_len):
            logits, hidden, attn_weights = self.decoder(
                input_token, hidden, encoder_outputs, source_mask
            )
            outputs[:, t - 1] = logits
            attention_history[:, t - 1] = attn_weights

            is_teacher_forcing = random.random() < self.teacher_forcing_ratio
            top1_predicted = logits.argmax(dim=-1).unsqueeze(1)
            input_token = target[:, t].unsqueeze(1) if is_teacher_forcing else top1_predicted

        return outputs, attention_history
