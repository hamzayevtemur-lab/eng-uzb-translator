"""
Baseline Seq2Seq Model (WITHOUT Attention)
Based on Sutskever et al., 2014.
Compresses the entire source sentence into a single final context vector.
"""

import random
import torch
import torch.nn as nn
from typing import Tuple, Optional

# Pad token index from dataset module
PAD_TOKEN = 0


# ---------------------------------------------------------------------------
# 1. BASELINE ENCODER
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Reads the English source sentence and compresses it into a final hidden state.
    
    Args:
        input_size: English vocabulary size
        embedding_dim: Word vector size (e.g., 256)
        hidden_size: GRU hidden dimension = context vector dimension (e.g., 512)
        num_layers: Number of GRU layers
        dropout: Dropout probability
    """
    def __init__(
        self, 
        input_size: int, 
        embedding_dim: int, 
        hidden_size: int, 
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, source_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_tokens: [batch_size, source_length] — integer token IDs
        Returns:
            outputs: [batch_size, source_length, hidden_size] — intermediate hidden states
            hidden:  [num_layers, batch_size, hidden_size] — final context vector
        """
        # 1. Embed source tokens
        embedded = self.dropout(self.embedding(source_tokens))  # [batch, src_len, emb_dim]

        # 2. Process sequentially through GRU
        outputs, hidden = self.gru(embedded)  # outputs: [batch, src_len, hidden], hidden: [layers, batch, hidden]

        return outputs, hidden


# ---------------------------------------------------------------------------
# 2. BASELINE DECODER
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Generates Uzbek tokens one at a time using only the GRU hidden state.
    
    Args:
        output_size: Uzbek vocabulary size
        embedding_dim: Word vector size
        hidden_size: GRU hidden dimension (must match Encoder)
        num_layers: Number of GRU layers
        dropout: Dropout probability
    """
    def __init__(
        self, 
        output_size: int, 
        embedding_dim: int, 
        hidden_size: int, 
        num_layers: int = 1,
        dropout: float = 0.1
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

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a single decoding timestep.
        
        Args:
            input_token: [batch_size, 1] — current target token ID
            hidden:      [num_layers, batch_size, hidden_size] — hidden state from previous step
        Returns:
            logits: [batch_size, output_size] — raw unnormalized vocabulary scores
            hidden: [num_layers, batch_size, hidden_size] — updated hidden state
        """
        # 1. Embed single token: [batch, 1, emb_dim]
        embedded = self.dropout(self.embedding(input_token))

        # 2. Step GRU for 1 step: output is [batch, 1, hidden_size]
        output, hidden = self.gru(embedded, hidden)

        # 3. Project to vocabulary logits: [batch, output_size]
        logits = self.fc_out(output.squeeze(1))

        return logits, hidden


# ---------------------------------------------------------------------------
# 3. UNIFIED SEQ2SEQ BASELINE WRAPPER
# ---------------------------------------------------------------------------
class Seq2SeqBaseline(nn.Module):
    """
    Coordinates the Encoder and Decoder with Teacher Forcing during training.
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

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.
        
        Args:
            source: [batch_size, source_length] — English sentence tokens
            target: [batch_size, target_length] — Ground truth Uzbek tokens
        Returns:
            outputs: [batch_size, target_length - 1, output_vocab_size]
        """
        batch_size = source.size(0)
        target_len = target.size(1)
        output_vocab_size = self.decoder.fc_out.out_features

        # Pre-allocate tensor to store predictions for every timestep
        outputs = torch.zeros(batch_size, target_len - 1, output_vocab_size, device=self.device)

        # 1. ENCODE: Entire English sentence compressed into context vector `hidden`
        _, hidden = self.encoder(source)

        # 2. First decoder input is the <SOS> token from the target: [batch_size, 1]
        input_token = target[:, 0].unsqueeze(1)

        # 3. DECODE loop: generate target words one by one
        for t in range(1, target_len):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t - 1] = logits

            # Teacher Forcing: randomly decide whether to use ground truth or prediction
            is_teacher_forcing = random.random() < self.teacher_forcing_ratio
            top1_predicted = logits.argmax(dim=-1).unsqueeze(1)

            input_token = target[:, t].unsqueeze(1) if is_teacher_forcing else top1_predicted

        return outputs


# ---------------------------------------------------------------------------
# 4. STANDALONE SHAPE VERIFICATION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cpu")
    SRC_VOCAB = 500
    TGT_VOCAB = 600
    EMB_DIM = 64
    HIDDEN_DIM = 128
    BATCH = 4
    SRC_LEN = 10
    TGT_LEN = 12

    enc = Encoder(SRC_VOCAB, EMB_DIM, HIDDEN_DIM).to(device)
    dec = Decoder(TGT_VOCAB, EMB_DIM, HIDDEN_DIM).to(device)
    model = Seq2SeqBaseline(enc, dec, device).to(device)

    dummy_src = torch.randint(0, SRC_VOCAB, (BATCH, SRC_LEN)).to(device)
    dummy_tgt = torch.randint(0, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)

    out = model(dummy_src, dummy_tgt)
    print("Baseline Model Forward Pass Success!")
    print(f"Input Source: {dummy_src.shape}")
    print(f"Target:       {dummy_tgt.shape}")
    print(f"Model Output: {out.shape} -> Expected [batch={BATCH}, timesteps={TGT_LEN-1}, vocab={TGT_VOCAB}]")
