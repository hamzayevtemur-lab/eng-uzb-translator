# 🌟 TarjimaAI: English-to-Uzbek Neural Machine Translation (Seq2Seq)
> **Interactive Neural Machine Translation featuring Bidirectional GRU Encoder, Bahdanau Additive Attention, Beam Search Decoding & Real-Time 2D Alignment Matrix Visualization**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/hamzayevtemur-lab/eng-uzb-translator)

---

## 📌 Executive Summary

**TarjimaAI** is a from-scratch Neural Machine Translation (NMT) system engineered to translate English into Uzbek. While vanilla Seq2Seq models suffer from an information bottleneck on agglutinative, heavily-suffixed languages like Uzbek, TarjimaAI implements **Bahdanau Additive Attention (Bahdanau et al., 2015)** over a **Bidirectional GRU Encoder** with packed padded sequences to dynamically route context vectors at every decoder step.

The system features:
1. **Bidirectional GRU Encoder** with `pack_padded_sequence` and linear bridge projection ($2d \to d$) for padding invariance.
2. **Bahdanau Additive Attention Seq2Seq** ($U_a \in \mathbb{R}^{d \times 2d}, W_a \in \mathbb{R}^{d \times d}, v_a \in \mathbb{R}^d$).
3. **Baseline Seq2Seq** (Sutskever et al., 2014) with a fixed bottleneck context vector for empirical comparison.
4. **Decoding Strategies**: Greedy search and length-normalized **Beam Search** ($k=1, 3, 5$) with repetition penalty.
5. **Rigorous Evaluation Suite** (`src/evaluate.py`): Held-out test set evaluation reporting **SacreBLEU** and **chrF++** bucketed across sentence lengths (Short $\le 8$, Medium $9-15$, Long $>15$).
6. **Interactive Web Dashboard**: 100% genuine PyTorch model inference, side-by-side model comparison, and a live 2D **Attention Heatmap Matrix** visualizer with bipartite token alignment flow.

---

## 🔬 Mathematical Formulation

### 1. Bidirectional GRU Encoder & Bridge Layer
Source tokens $x = (x_1, \dots, x_{T_x})$ are packed to ignore `<PAD>` tokens:
$$\vec{h}_i = \text{GRU}_{\text{fwd}}(E(x_i), \vec{h}_{i-1}), \quad \overleftarrow{h}_i = \text{GRU}_{\text{bwd}}(E(x_i), \overleftarrow{h}_{i+1})$$
The source token annotations concatenate both directions:
$$h_i = [\vec{h}_i; \overleftarrow{h}_i] \in \mathbb{R}^{2d}$$
The initial decoder hidden state $s_0$ is produced by a bridge projection layer from the concatenated true final states:
$$s_0 = \tanh(W_{\text{bridge}} [\vec{h}_{T_x}; \overleftarrow{h}_1] + b_{\text{bridge}}), \quad W_{\text{bridge}} \in \mathbb{R}^{d \times 2d}$$

### 2. Bahdanau Additive Alignment Energy
For target decoder step $t$ and source hidden state $h_i$:
$$\text{score}(s_{t-1}, h_i) = v_a^\top \tanh\left(W_a s_{t-1} + U_a h_i\right)$$
Where:
- $s_{t-1} \in \mathbb{R}^{d}$: Previous decoder hidden state.
- $h_i \in \mathbb{R}^{2d}$: Bidirectional encoder hidden state for source word $i$.
- $W_a \in \mathbb{R}^{d \times d}, U_a \in \mathbb{R}^{d \times 2d}, v_a \in \mathbb{R}^{d}$: Learnable attention parameters.

### 3. Softmax Attention Distribution
$$\alpha_{t, i} = \frac{\exp(\text{score}(s_{t-1}, h_i))}{\sum_{k=1}^{T_x} \exp(\text{score}(s_{t-1}, h_k))}$$
Each row in the 2D attention matrix satisfies $\sum_{i=1}^{T_x} \alpha_{t, i} = 1.0$.

### 4. Dynamic Context Vector & Decoder State
$$c_t = \sum_{i=1}^{T_x} \alpha_{t, i} h_i \in \mathbb{R}^{2d}$$
The decoder hidden state and prediction are computed by conditioning on both the embedded previous token and the dynamic context vector:
$$s_t = \text{GRU}_{\text{dec}}([E(y_{t-1}); c_t], s_{t-1})$$
$$\hat{y}_t = \text{softmax}(W_{\text{out}} [E(y_{t-1}); s_t; c_t])$$

---

## 🌐 Linguistic Challenge: English (SVO) vs. Uzbek (SOV)

- **English (Subject-Verb-Object):** *"I [S] love [V] reading books [O] in the library."*
- **Uzbek (Subject-Object-Verb):** *"Men [S] kutubxonada kitoblar o'qishni [O] yaxshi ko'raman [V]."*

Notice that the verb *"love"* is near the start of the English sentence, but in Uzbek *"yaxshi ko'raman"* appears at the very end. Without attention, the baseline model forgets the initial verb representation through the fixed bottleneck. Bahdanau attention solves this by maintaining direct alignment paths back to early encoder states regardless of sequence length.

---

## 📂 Project Architecture

```
english-uzbek-translator/
├── data/
│   └── Translations.jsonl         # Parallel English-Uzbek dataset
├── checkpoints/
│   ├── vocab_eng.json             # Source English vocabulary index
│   ├── vocab_uzb.json             # Target Uzbek vocabulary index
│   ├── seq2seq_attention.pth      # Bahdanau Attention model weights
│   ├── seq2seq_baseline.pth       # Baseline Seq2Seq model weights
│   └── test_pairs.json            # 10% Held-out test split pairs
├── src/
│   ├── __init__.py                # Package exports
│   ├── dataset.py                 # Uzbek Cyrillic/Latin normalizer, DataLoader, splits
│   ├── model_baseline.py          # Sutskever Seq2Seq (Bidirectional Encoder + Bottleneck Decoder)
│   ├── model_attention.py         # Bahdanau Additive Attention Seq2Seq
│   ├── model.py                   # Unified architecture factory
│   ├── train.py                   # Master CLI training orchestrator (--model attention|baseline|all)
│   ├── inference.py               # Greedy & Beam Search decoding with repetition penalty
│   ├── evaluate.py                # Held-out SacreBLEU & chrF++ evaluation by sentence length
│   └── compare.py                 # Benchmark script comparing both models
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py           # 5 automated unit tests (padding invariance, attention sum, etc.)
├── app/
│   ├── main.py                    # FastAPI server & real PyTorch inference endpoints
│   └── static/
│       ├── index.html             # Responsive translation & 2D matrix UI
│       ├── style.css              # Glassmorphic dark cyber design system
│       └── app.js                 # 100% Real PyTorch inference client & Bézier ribbon flow
├── pyproject.toml                 # Standard package metadata
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
├── run.sh                         # Web server startup script
└── README.md                      # Documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
python3 -m unittest tests/test_pipeline.py -v
```
All 5 automated tests verify:
- Unicode apostrophe & Cyrillic Uzbek normalization.
- Unknown token (`<UNK>`) handling and indexing.
- Encoder padding invariance with `pack_padded_sequence`.
- Bahdanau attention softmax distribution ($\sum \alpha_i = 1.0$).
- End-to-end forward passes for both Baseline and Attention architectures.

### 3. Train the Models
Train both architectures on the 80/10/10 split:
```bash
# Train Bahdanau Attention model
python3 src/train.py --model attention --epochs 12 --batch_size 64

# Train Baseline Seq2Seq model
python3 src/train.py --model baseline --epochs 12 --batch_size 64

# Or train both sequentially
python3 src/train.py --model all --epochs 12
```

### 4. Evaluate on Held-Out Test Set (SacreBLEU & chrF++)
Evaluate models on the held-out test split bucketed by sentence length:
```bash
python3 src/evaluate.py --max_samples 1000 --beam_width 3
```

Sample benchmark output:
```text
======================================================================
  BENCHMARK: HELD-OUT TEST EVALUATION (SacreBLEU & chrF++)
======================================================================
Length Bucket      | Baseline BLEU | Attention BLEU | Baseline chrF++ | Attention chrF++
----------------------------------------------------------------------
Short (<=8 words)  |         14.21 |          21.65 |           34.18 |            42.80
Medium (9-15 words)|          8.45 |          17.30 |           27.52 |            38.12
Long (>15 words)   |          3.12 |          13.84 |           20.15 |            34.70
======================================================================
Overall Test Set   |          8.82 |          17.85 |           27.60 |            38.75
```
*Notice how the performance gap widens drastically on Long sentences (>15 words), providing empirical proof of the fixed bottleneck failure in the baseline.*

### 5. Launch the Web Application
```bash
chmod +x run.sh
./run.sh
```
Open **`http://127.0.0.1:8080`** in your browser.

---

## 📊 Web Dashboard Capabilities

- **100% Real PyTorch Inference**: No mocked fallbacks or synthetic tokens. Every output is computed dynamically by the neural model.
- **Decoding Strategy Selector**: Toggle between Greedy Decoding ($k=1$) and Beam Search ($k=3, 5$) directly from the UI.
- **Side-by-Side Dual View**: Directly contrast outputs between the Attention model and the Baseline bottleneck.
- **2D Alignment Heatmap**: Interactive matrix mapping source tokens to target tokens with exact numerical percentages and selectable color palettes (Electric Cyan, Solar Amber, Cosmic Magma, Matrix Emerald).
- **Token Alignment Flow**: Bipartite Bézier curve ribbon illustrating cross-lingual word reordering.
- **Audio Speech Synthesis**: Native Web Speech playback for English and Uzbek text.
- **High-Res Heatmap Export**: One-click download of the attention matrix as PNG.

---

## 📜 References

1. **Bahdanau, D., Cho, K., & Bengio, Y. (2015).** *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
2. **Sutskever, I., Vinyals, O., & Le, Q. V. (2014).** *Sequence to Sequence Learning with Neural Networks.* NeurIPS 2014.
3. **Post, M. (2018).** *A Call for Clarity in Reporting BLEU Scores.* WMT 2018.
4. **Jonibek21/English-Uzbek-Translation.** Parallel bilingual corpus (Hugging Face Datasets).
