"""
Baseline Seq2Seq Model (Sutskever et al., 2014) Without Attention.
Features:
- Bidirectional GRU Encoder with pack_padded_sequence to eliminate <PAD> pollution
- Bridge projection layer mapping [h_forward; h_backward] -> s_0
- Unidirectional Decoder decoding strictly from the fixed s_0 context bottleneck
"""

import random
from typing import Tuple, Optional
import torch
import torch.nn as nn

try:
    from .dataset import PAD_TOKEN
    from .model_attention import Encoder
except ImportError:
    from dataset import PAD_TOKEN
    from model_attention import Encoder


# ---------------------------------------------------------------------------
# 1. BASELINE DECODER (FIXED CONTEXT BOTTLENECK)
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Decodes target Uzbek tokens using only the previous token embedding
    and the recurring hidden state. Has NO attention access to encoder outputs.
    """
    def __init__(
        self,
        output_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs one step of baseline decoding.
        """
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden


# ---------------------------------------------------------------------------
# 2. UNIFIED BASELINE SEQ2SEQ WRAPPER
# ---------------------------------------------------------------------------
class Seq2SeqBaseline(nn.Module):
    """
    Coordinates Bidirectional Encoder and Fixed-Bottleneck Decoder.
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
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = source.size(0)
        target_len = target.size(1)
        output_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, target_len - 1, output_vocab_size, device=self.device)

        # 1. Encode with packing to eliminate pad corruption
        # Baseline uses only the bridge-projected final hidden state s0
        _, hidden = self.encoder(source, src_lengths=src_lengths)

        # 2. First decoder input is target <SOS>
        input_token = target[:, 0].unsqueeze(1)

        # 3. Decode step-by-step from fixed bottleneck
        for t in range(1, target_len):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t - 1] = logits

            is_teacher_forcing = random.random() < self.teacher_forcing_ratio
            top1_predicted = logits.argmax(dim=-1).unsqueeze(1)
            input_token = target[:, t].unsqueeze(1) if is_teacher_forcing else top1_predicted

        return outputs
