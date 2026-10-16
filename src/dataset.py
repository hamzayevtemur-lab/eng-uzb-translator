"""
Dataset & Preprocessing Module for English-to-Uzbek Translation.
Handles text normalization (Latin & Cyrillic Uzbek), apostrophe standardization,
vocabulary indexing, train/val/test splits, and PyTorch batched tensor collation.
"""

import json
import re
import random
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ---------------------------------------------------------------------------
# 1. SPECIAL TOKENS
# ---------------------------------------------------------------------------
PAD_TOKEN = 0  # <PAD>: Equalizes batch sequence lengths
SOS_TOKEN = 1  # <SOS>: Start of Sequence marker for Decoder
EOS_TOKEN = 2  # <EOS>: End of Sequence marker (stopping criterion)
UNK_TOKEN = 3  # <UNK>: Out-of-Vocabulary fallback


# ---------------------------------------------------------------------------
# 2. TEXT NORMALIZATION & CLEANING
# ---------------------------------------------------------------------------
def unicode_to_ascii(text: str) -> str:
    """Decomposes unicode accents while keeping standard characters intact."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_string(text: str, is_uzbek: bool = False) -> str:
    """
    Cleans and standardizes text:
    - Standardizes various apostrophe styles (' ` ’ ʻ \u02bb) to standard '
    - Converts to lowercase
    - Separates punctuation from words
    - Preserves both Latin and Cyrillic Uzbek alphabets
    - Collapses multiple whitespace
    """
    text = text.strip()

    # 1. Standardize all apostrophe variants to standard single quote '
    text = re.sub(r"[`’‘ʻ\u02bb\u2018\u2019]", "'", text)

    # 2. Lowercase
    text = text.lower()

    if not is_uzbek:
        text = unicode_to_ascii(text)

    # 3. Add space around common punctuation (. , ! ? - : ;)
    text = re.sub(r"([.,!?:;\-\"()])", r" \1 ", text)

    # 4. Remove unwanted symbols, keeping Latin, Cyrillic, numbers, apostrophe, and punctuation
    text = re.sub(r"[^a-zA-Z0-9\u0400-\u04FF'.,!?:;\- ]+", " ", text)

    # 5. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# 3. VOCABULARY CLASS
# ---------------------------------------------------------------------------
class Vocabulary:
    """
    Bidirectional mapping between words and numerical indices (w2i and i2w).
    Tracks token occurrence frequency for filtering rare words.
    """
    def __init__(self, name: str):
        self.name = name

        self.word2index: Dict[str, int] = {
            "<PAD>": PAD_TOKEN,
            "<SOS>": SOS_TOKEN,
            "<EOS>": EOS_TOKEN,
            "<UNK>": UNK_TOKEN
        }

        self.index2word: Dict[int, str] = {
            PAD_TOKEN: "<PAD>",
            SOS_TOKEN: "<SOS>",
            EOS_TOKEN: "<EOS>",
            UNK_TOKEN: "<UNK>"
        }

        self.word2count: Dict[str, int] = {}
        self.n_words: int = 4

    def add_sentence(self, sentence: str):
        """Tokens in sentence are registered and counted."""
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word: str):
        """Adds a single word to the vocabulary."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_freq: int = 2):
        """
        Filters out rare words with frequency less than min_freq.
        Rare and unseen words will automatically map to <UNK>.
        """
        keep_words = [w for w, c in self.word2count.items() if c >= min_freq]
        
        # Reset mappings
        self.word2index = {
            "<PAD>": PAD_TOKEN,
            "<SOS>": SOS_TOKEN,
            "<EOS>": EOS_TOKEN,
            "<UNK>": UNK_TOKEN
        }
        self.index2word = {
            PAD_TOKEN: "<PAD>",
            SOS_TOKEN: "<SOS>",
            EOS_TOKEN: "<EOS>",
            UNK_TOKEN: "<UNK>"
        }
        old_count = self.n_words
        self.n_words = 4

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

        print(f"[{self.name} Vocab] Trimmed rare words (min_freq={min_freq}): {old_count:,} -> {self.n_words:,} words")

    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Converts text sentence to a list of integer IDs (defaults to UNK)."""
        return [self.word2index.get(word, UNK_TOKEN) for word in sentence.split()]

    def indices_to_sentence(self, indices: List[int]) -> str:
        """Converts integer IDs back into readable words."""
        words = []
        for idx in indices:
            if idx == EOS_TOKEN:
                break
            if idx not in (PAD_TOKEN, SOS_TOKEN):
                words.append(self.index2word.get(idx, "<UNK>"))
        return " ".join(words)

    def save(self, filepath: str):
        """Saves vocabulary mapping to JSON file."""
        data = {
            "name": self.name,
            "word2index": self.word2index,
            "index2word": {str(k): v for k, v in self.index2word.items()},
            "n_words": self.n_words
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Loads vocabulary mapping from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(data["name"])
        vocab.word2index = data["word2index"]
        vocab.index2word = {int(k): v for k, v in data["index2word"].items()}
        vocab.n_words = data["n_words"]

        return vocab


# ---------------------------------------------------------------------------
# 4. DATASET LOADER & TRAIN / VAL / TEST SPLITS
# ---------------------------------------------------------------------------
def load_pairs(
    filepath: str, 
    max_samples: Optional[int] = None, 
    min_length: int = 2, 
    max_length: int = 25
) -> List[Tuple[str, str]]:
    """
    Parses Translations.jsonl, normalizes text, and filters by sentence length.
    Returns list of clean (english, uzbek) tuples.
    """
    pairs: List[Tuple[str, str]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                en_raw = row.get("en", "")
                uz_raw = row.get("uz", "")

                if not en_raw or not uz_raw:
                    continue

                en_clean = normalize_string(en_raw, is_uzbek=False)
                uz_clean = normalize_string(uz_raw, is_uzbek=True)

                en_len = len(en_clean.split())
                uz_len = len(uz_clean.split())

                if min_length <= en_len <= max_length and min_length <= uz_len <= max_length:
                    pairs.append((en_clean, uz_clean))

                if max_samples and len(pairs) >= max_samples:
                    break
            except Exception:
                continue

    return pairs


def load_dataset_splits(
    filepath: str,
    max_samples: Optional[int] = 45000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_length: int = 2,
    max_length: int = 25,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Loads parallel pairs and creates stratified Train / Val / Held-out Test splits.
    Default: 80% Train / 10% Validation / 10% Held-Out Test.
    """
    pairs = load_pairs(filepath, max_samples=max_samples, min_length=min_length, max_length=max_length)
    
    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"[Dataset Split] Total: {n_total:,} | Train: {len(train_pairs):,} (80%) | "
          f"Val: {len(val_pairs):,} (10%) | Held-Out Test: {len(test_pairs):,} (10%)")

    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# 5. PYTORCH TRANSLATION DATASET
# ---------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """
    Wraps parallel pairs into numerical tensors with <SOS> and <EOS> framing.
    """
    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        en_sent, uz_sent = self.pairs[idx]

        src_ids = [SOS_TOKEN] + self.src_vocab.sentence_to_indices(en_sent) + [EOS_TOKEN]
        tgt_ids = [SOS_TOKEN] + self.tgt_vocab.sentence_to_indices(uz_sent) + [EOS_TOKEN]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# 6. BATCH COLLATOR (PADDING & LENGTH TRACKING)
# ---------------------------------------------------------------------------
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads variable-length sequences to the max length in the current batch.
    
    Returns:
        padded_src:  [batch_size, max_src_len]
        padded_tgt:  [batch_size, max_tgt_len]
        src_lengths: [batch_size] length of each source sequence (excluding pad)
    """
    src_list, tgt_list = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_list], dtype=torch.long)

    padded_src = pad_sequence(src_list, batch_first=True, padding_value=PAD_TOKEN)
    padded_tgt = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_TOKEN)

    return padded_src, padded_tgt, src_lengths


# ---------------------------------------------------------------------------
# 7. STANDALONE VERIFICATION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(project_root, "data", "Translations.jsonl")
    
    if os.path.exists(test_path):
        train_p, val_p, test_p = load_dataset_splits(test_path, max_samples=100)
        print("Sample pair:", train_p[0])
    else:
        print("Data file not found for standalone test, dataset module syntax verified.")