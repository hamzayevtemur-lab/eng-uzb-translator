"""
Unified Model Architecture Factory.
Provides shared bidirectional packed Encoder and exports both:
1. Baseline Seq2Seq (Sutskever et al., 2014)
2. Bahdanau Additive Attention Seq2Seq (Bahdanau et al., 2015)
"""

try:
    from .model_attention import (
        Encoder,
        BahdanauAttention,
        Decoder as AttentionDecoder,
        Seq2SeqAttention
    )
    from .model_baseline import (
        Decoder as BaselineDecoder,
        Seq2SeqBaseline
    )
except ImportError:
    from model_attention import (
        Encoder,
        BahdanauAttention,
        Decoder as AttentionDecoder,
        Seq2SeqAttention
    )
    from model_baseline import (
        Decoder as BaselineDecoder,
        Seq2SeqBaseline
    )

__all__ = [
    "Encoder",
    "BahdanauAttention",
    "AttentionDecoder",
    "Seq2SeqAttention",
    "BaselineDecoder",
    "Seq2SeqBaseline"
]
