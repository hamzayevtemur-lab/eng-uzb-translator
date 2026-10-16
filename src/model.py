"""
Unified Model Interface.
Re-exports both Baseline (without attention) and Bahdanau (with attention) architectures.
"""

from model_baseline import (
    Encoder as BaselineEncoder,
    Decoder as BaselineDecoder,
    Seq2SeqBaseline
)
from model_attention import (
    Encoder as AttentionEncoder,
    Decoder as AttentionDecoder,
    BahdanauAttention,
    Seq2SeqAttention
)

# Default aliases point to the flagship Attention architecture
Encoder = AttentionEncoder
Decoder = AttentionDecoder
Seq2Seq = Seq2SeqAttention

__all__ = [
    # Defaults (Attention)
    "Encoder",
    "Decoder",
    "Seq2Seq",
    "BahdanauAttention",
    "Seq2SeqAttention",
    
    # Baseline
    "BaselineEncoder",
    "BaselineDecoder",
    "Seq2SeqBaseline"
]
