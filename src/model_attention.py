"""
Attention-Augmented Seq2Seq Model (Bahdanau et al., 2015)
Computes dynamic context vectors at every decoding timestep
and records attention weights for alignment visualization.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Special token index from dataset module
PAD_TOKEN = 0


# 1. ATTENTION ENCODER

class Encoder(nn.Module):
    """
    Encodes English sentences and preserves ALL hidden states at every timestep.
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
            source_tokens: [batch_size, source_length]
        Returns:
            outputs: [batch_size, source_length, hidden_size] — all encoder states (Keys/Values)
            hidden:  [num_layers, batch_size, hidden_size] — final state
        """
        embedded = self.dropout(self.embedding(source_tokens))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


# 2. BAHDANAU (ADDITIVE) ATTENTION MODULE

class BahdanauAttention(nn.Module):
    """
    Additive Attention:
    score(s_{t-1}, h_i) = v_a^T * tanh(W_a * s_{t-1} + U_a * h_i)
    """
    def __init__(self, hidden_size: int):
        super().__init__()

        # Projects encoder states h_i (Keys)
        self.encoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        # Projects decoder state s_{t-1} (Query)
        self.decoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        # Alignment energy vector v_a
        self.energy = nn.Linear(hidden_size, 1, bias=False)


    def forward(
        self, 
        decoder_hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor, 
        source_mask: Optional[torch.Tensor] = None
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden:  [batch_size, hidden_size] — Query
            encoder_outputs: [batch_size, source_length, hidden_size] — Keys/Values
            source_mask:     [batch_size, source_length] — True for real tokens, False for PAD
        Returns:
            context:           [batch_size, 1, hidden_size] — dynamic weighted representation
            attention_weights: [batch_size, source_length] — softmax distribution
        """
        # 1. Project keys and query
        encoder_features = self.encoder_projection(encoder_outputs)  # [batch, src_len, hidden]
        decoder_features = self.decoder_projection(decoder_hidden).unsqueeze(1)  # [batch, 1, hidden]

        # 2. Additive combination: tanh(W * s + U * h)
        combined = torch.tanh(encoder_features + decoder_features)  # [batch, src_len, hidden]

        # 3. Alignment energy scores: [batch, src_len]
        scores = self.energy(combined).squeeze(-1)

        # 4. Mask out <PAD> positions with -infinity so softmax gives them 0.0 weight
        if source_mask is not None:
            scores = scores.masked_fill(~source_mask, float("-inf"))

        # 5. Softmax across source sequence positions
        attention_weights = F.softmax(scores, dim=1)  # [batch, src_len]

        # 6. Weighted sum: [batch, 1, src_len] x [batch, src_len, hidden] -> [batch, 1, hidden]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)

        return context, attention_weights


# 3. ATTENTION DECODER
class Decoder(nn.Module):
    """
    Decodes target Uzbek tokens using both the previous token embedding
    and the dynamic context vector computed by Bahdanau attention.
    """
    def __init__(
        self, output_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int=1,
        dropout: float=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_size)

        # GRU input receives [embedding; context_vector]
        self.gru = nn.GRU(
            input_size=embedding_dim + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Linear classifier receives [gru_output; context_vector]
        self.fc_out = nn.Linear(hidden_size * 2, output_size)

    def forward(
        self, 
        input_token: torch.Tensor, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor, 
        source_mask: Optional[torch.Tensor] = None
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs one decoding step with attention.
        
        Args:
            input_token:     [batch_size, 1] — current target token ID
            hidden:          [num_layers, batch_size, hidden_size] — previous GRU state
            encoder_outputs: [batch_size, source_length, hidden_size]
            source_mask:     [batch_size, source_length]
        Returns:
            logits:            [batch_size, output_size]
            hidden:            [num_layers, batch_size, hidden_size]
            attention_weights: [batch_size, source_length]
        """
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, emb_dim]
        last_hidden_layer = hidden[-1]  # [batch, hidden_size]

        # 1. Compute dynamic attention context vector and weights
        context, attention_weights = self.attention(last_hidden_layer, encoder_outputs, source_mask)

        # 2. Concatenate input embedding with context vector
        gru_input = torch.cat([embedded, context], dim=2)  # [batch, 1, emb_dim + hidden_size]

        # 3. Step GRU
        output, hidden = self.gru(gru_input, hidden)  # output: [batch, 1, hidden_size]

        # 4. Concatenate GRU output and context for vocabulary projection
        output_flat = output.squeeze(1)
        context_flat = context.squeeze(1)
        combined = torch.cat([output_flat, context_flat], dim=1)  # [batch, hidden_size * 2]

        logits = self.fc_out(combined)  # [batch, output_size]

        return logits, hidden, attention_weights



# 4. UNIFIED SEQ2SEQ ATTENTION WRAPPER
class Seq2SeqAttention(nn.Module):
    """
    Coordinates Encoder and Attention-Augmented Decoder with Teacher Forcing.
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
        source_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            source:      [batch_size, source_length]
            target:      [batch_size, target_length]
            source_mask: [batch_size, source_length]
        Returns:
            outputs:           [batch_size, target_length - 1, output_vocab_size]
            attention_history: [batch_size, target_length - 1, source_length]
        """
        batch_size = source.size(0)
        target_len = target.size(1)
        source_len = source.size(1)
        output_vocab_size = self.decoder.fc_out.out_features

        # Auto-compute mask if not provided (True for real tokens, False for PAD)
        if source_mask is None:
            source_mask = (source != PAD_TOKEN)

        outputs = torch.zeros(batch_size, target_len - 1, output_vocab_size, device=self.device)
        attention_history = torch.zeros(batch_size, target_len - 1, source_len, device=self.device)

        # 1. ENCODE: Keep all encoder hidden states
        encoder_outputs, hidden = self.encoder(source)

        # 2. First decoder input is target <SOS>
        input_token = target[:, 0].unsqueeze(1)

        # 3. DECODE with attention step-by-step
        for t in range(1, target_len):
            logits, hidden, attn_weights = self.decoder(
                input_token, hidden, encoder_outputs, source_mask
            )
            outputs[:, t - 1] = logits
            attention_history[:, t - 1] = attn_weights

            # Teacher Forcing
            is_teacher_forcing = random.random() < self.teacher_forcing_ratio
            top1_predicted = logits.argmax(dim=-1).unsqueeze(1)
            input_token = target[:, t].unsqueeze(1) if is_teacher_forcing else top1_predicted

        return outputs, attention_history

# 5. STANDALONE SHAPE VERIFICATION
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

    model = Seq2SeqAttention(enc, dec, device).to(device)

    dummy_src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN)).to(device)
    dummy_tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)

    out, attns = model(dummy_src, dummy_tgt)
    print("Attention Model Forward Pass Success!")
    print(f"Input Source:      {dummy_src.shape}")
    print(f"Target:            {dummy_tgt.shape}")
    print(f"Model Logits:      {out.shape}   -> Expected [batch={BATCH}, timesteps={TGT_LEN-1}, vocab={TGT_VOCAB}]")
    print(f"Attention Weights: {attns.shape} -> Expected [batch={BATCH}, timesteps={TGT_LEN-1}, src_len={SRC_LEN}]")



