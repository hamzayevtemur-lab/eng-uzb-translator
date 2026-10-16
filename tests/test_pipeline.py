"""
Automated Unit and Integration Tests for TarjimaAI.
Compatible with standard python3 -m unittest and pytest.
"""

import sys
import unittest
from pathlib import Path
import torch

# Add src to path
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import (
    normalize_string, 
    Vocabulary, 
    PAD_TOKEN, 
    SOS_TOKEN, 
    EOS_TOKEN, 
    UNK_TOKEN
)
from model_attention import Encoder, BahdanauAttention, Decoder as AttentionDecoder, Seq2SeqAttention
from model_baseline import Decoder as BaselineDecoder, Seq2SeqBaseline


class TestTarjimaAIPipeline(unittest.TestCase):

    def test_normalizer_apostrophes_and_cyrillic(self):
        """Verifies Uzbek apostrophe normalization and Cyrillic character preservation."""
        variations = ["o'quvchi", "o`quvchi", "o‘quvchi", "oʻquvchi"]
        for v in variations:
            cleaned = normalize_string(v, is_uzbek=True)
            self.assertEqual(cleaned, "o'quvchi")

        cyrillic_uz = "Ўзбекистон мустақил давлат."
        cleaned_cyr = normalize_string(cyrillic_uz, is_uzbek=True)
        self.assertIn("ўзбекистон", cleaned_cyr)

    def test_vocabulary_unk_and_indexing(self):
        """Verifies that Vocabulary maps known words and routes unseen words to UNK."""
        vocab = Vocabulary("English")
        vocab.add_sentence("the quick brown fox")
        
        indices = vocab.sentence_to_indices("the fox")
        self.assertEqual(indices, [vocab.word2index["the"], vocab.word2index["fox"]])

        unknown_indices = vocab.sentence_to_indices("the dinosaur")
        self.assertEqual(unknown_indices, [vocab.word2index["the"], UNK_TOKEN])

    def test_encoder_packed_padding_invariance(self):
        """Verifies that bidirectional Encoder correctly handles variable length sequences."""
        batch_size = 4
        max_len = 8
        vocab_size = 100
        emb_dim = 64
        hidden_dim = 128

        encoder = Encoder(vocab_size, emb_dim, hidden_dim)

        tokens = torch.randint(4, vocab_size, (batch_size, max_len))
        lengths = torch.tensor([8, 6, 4, 3], dtype=torch.long)

        for i, l in enumerate(lengths):
            tokens[i, l:] = PAD_TOKEN

        outputs, s0 = encoder(tokens, src_lengths=lengths)

        self.assertEqual(outputs.shape, (batch_size, max_len, hidden_dim * 2))
        self.assertEqual(s0.shape, (1, batch_size, hidden_dim))

    def test_bahdanau_attention_distribution_sums_to_one(self):
        """Verifies that Bahdanau attention weights sum to 1.0 and PAD weights are 0.0."""
        batch_size = 3
        src_len = 6
        enc_hidden = 256
        dec_hidden = 128

        attention = BahdanauAttention(enc_hidden_size=enc_hidden, dec_hidden_size=dec_hidden)

        decoder_hidden = torch.randn(batch_size, dec_hidden)
        encoder_outputs = torch.randn(batch_size, src_len, enc_hidden)

        mask = torch.ones(batch_size, src_len, dtype=torch.bool)
        mask[0, 4:] = False

        context, attn_weights = attention(decoder_hidden, encoder_outputs, source_mask=mask)

        # 1. Attention rows must sum to 1.0
        sums = attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(batch_size), atol=1e-5))

        # 2. Padded positions must have exact 0.0 weight
        self.assertEqual(attn_weights[0, 4].item(), 0.0)
        self.assertEqual(attn_weights[0, 5].item(), 0.0)

        # 3. Context shape: [batch, 1, enc_hidden]
        self.assertEqual(context.shape, (batch_size, 1, enc_hidden))

    def test_full_model_forward_passes(self):
        """Verifies end-to-end forward step for both Attention and Baseline models."""
        device = torch.device("cpu")
        batch_size = 2
        src_len = 7
        tgt_len = 5
        src_vocab = 50
        tgt_vocab = 60
        emb_dim = 32
        hidden_dim = 64

        # 1. Attention Model
        enc = Encoder(src_vocab, emb_dim, hidden_dim)
        dec = AttentionDecoder(tgt_vocab, emb_dim, enc_hidden_size=hidden_dim * 2, dec_hidden_size=hidden_dim)
        attn_model = Seq2SeqAttention(enc, dec, device, teacher_forcing_ratio=0.5)

        src = torch.randint(4, src_vocab, (batch_size, src_len))
        tgt = torch.randint(4, tgt_vocab, (batch_size, tgt_len))
        src_lengths = torch.tensor([7, 5], dtype=torch.long)

        outputs, attn_hist = attn_model(src, tgt, src_lengths=src_lengths)
        self.assertEqual(outputs.shape, (batch_size, tgt_len - 1, tgt_vocab))
        self.assertEqual(attn_hist.shape, (batch_size, tgt_len - 1, src_len))

        # 2. Baseline Model
        base_enc = Encoder(src_vocab, emb_dim, hidden_dim)
        base_dec = BaselineDecoder(tgt_vocab, emb_dim, hidden_size=hidden_dim)
        base_model = Seq2SeqBaseline(base_enc, base_dec, device, teacher_forcing_ratio=0.5)

        base_outputs = base_model(src, tgt, src_lengths=src_lengths)
        self.assertEqual(base_outputs.shape, (batch_size, tgt_len - 1, tgt_vocab))


if __name__ == "__main__":
    unittest.main()
