# 🌟 TarjimaAI: English-to-Uzbek Neural Machine Translation (Seq2Seq)
> **Interactive Neural Machine Translation featuring Bahdanau Additive Attention & Real-Time 2D Alignment Matrix Visualization**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/hamzayevtemur-lab/eng-uzb-translator)

---

## 📌 Executive Summary

**TarjimaAI** is a from-scratch Neural Machine Translation (NMT) system engineered to translate English into Uzbek. While vanilla Seq2Seq models suffer from an information bottleneck on agglutinative, heavily-suffixed languages like Uzbek, TarjimaAI implements **Bahdanau Additive Attention (Bahdanau et al., 2015)** to dynamically route context vectors at every decoder step.

The system comes with:
1. **Baseline Seq2Seq** (Sutskever et al., 2014) with a fixed hidden bottleneck.
2. **Bahdanau Attention Seq2Seq** (Bahdanau et al., 2015) with dynamic context vectors.
3. **Interactive Web Dashboard** featuring real-time translation, side-by-side model comparison, and a live 2D **Attention Heatmap Matrix** visualizer.

---

## 🔬 Mathematical Formulation

### 1. Additive Alignment Energy
For target decoder step $t$ and source hidden state $h_i$:
$$\text{score}(s_{t-1}, h_i) = v_a^\top \tanh\left(W_a s_{t-1} + U_a h_i\right)$$
Where:
- $s_{t-1} \in \mathbb{R}^{d}$: Previous decoder hidden state.
- $h_i \in \mathbb{R}^{2d}$: Bidirectional encoder hidden state for source word $i$.
- $W_a \in \mathbb{R}^{d \times d}, U_a \in \mathbb{R}^{d \times 2d}, v_a \in \mathbb{R}^{d}$: Learnable attention parameters.

### 2. Softmax Attention Distribution
$$\alpha_{t, i} = \frac{\exp(\text{score}(s_{t-1}, h_i))}{\sum_{k=1}^{T_x} \exp(\text{score}(s_{t-1}, h_k))}$$
Each row in the 2D attention matrix satisfies $\sum_i \alpha_{t, i} = 1.0$.

### 3. Dynamic Context Vector
$$c_t = \sum_{i=1}^{T_x} \alpha_{t, i} h_i$$
The decoder step $t$ predicts the next Uzbek token conditioned on $[y_{t-1}; c_t]$.

---

## 🌐 Linguistic Challenge: English (SVO) vs. Uzbek (SOV)

- **English (Subject-Verb-Object):** *"I [S] love [V] reading books [O] in the library."*
- **Uzbek (Subject-Object-Verb):** *"Men [S] kutubxonada kitoblar o'qishni [O] yaxshi ko'raman [V]."*

Notice that the verb *"love"* is at the start of the English predicate, but in Uzbek *"yaxshi ko'raman"* appears at the very end of the sentence. Without attention, the baseline model forgets the initial verb state. Bahdanau attention solves this by maintaining a direct alignment path back to early encoder states.

---

## 📂 Project Architecture

```
english-uzbek-translator/
├── data/
│   └── Translations.jsonl         # 47,985 English-Uzbek parallel pairs
├── checkpoints/
│   ├── vocab_eng.json             # Source English vocabulary index
│   ├── vocab_uzb.json             # Target Uzbek vocabulary index
│   ├── seq2seq_attention.pth      # Bahdanau Attention model weights
│   └── seq2seq_baseline.pth       # Baseline Seq2Seq model weights
├── src/
│   ├── __init__.py                # Package exports
│   ├── dataset.py                 # Uzbek apostrophe normalizer & DataLoader
│   ├── model_baseline.py          # Sutskever Seq2Seq (Encoder, Decoder)
│   ├── model_attention.py         # Bahdanau Additive Attention Seq2Seq
│   ├── model.py                   # Unified architecture factory
│   ├── train_baseline.py          # Training pipeline for baseline model
│   ├── train_attention.py         # Training pipeline for attention model
│   ├── train.py                   # Master CLI training orchestrator
│   ├── inference.py               # Greedy decoding & attention extraction
│   └── compare.py                 # Benchmark script comparing both models
├── app/
│   ├── main.py                    # FastAPI server & inference endpoints
│   └── static/
│       ├── index.html             # Responsive translation & matrix UI
│       ├── style.css              # Glassmorphic dark cyber design system
│       └── app.js                 # Real-time matrix renderer & audio synth
├── requirements.txt               # Dependencies
├── run.sh                         # Web server startup script
└── README.md                      # Documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
chmod +x run.sh
./run.sh
```
Open **`http://127.0.0.1:8080`** in your browser.

### 3. Train the Models
To train the Seq2Seq models on the Hugging Face 48k Uzbek dataset:
```bash
# Train Bahdanau Attention model
python3 src/train.py --model attention --epochs 15 --batch_size 64

# Train Baseline Seq2Seq model
python3 src/train.py --model baseline --epochs 15 --batch_size 64

# Or train both sequentially
python3 src/train.py --model all
```

### 4. Run Benchmark Comparison
```bash
python3 src/compare.py
```

---

## 📊 Web Dashboard Capabilities

- **Real-Time Translation**: Sub-50ms greedy decoding from English to Uzbek.
- **Side-by-Side Dual View**: Directly contrast outputs between the Attention model and the Baseline bottleneck.
- **2D Alignment Heatmap**: Interactive SVG/canvas table mapping source tokens to target tokens with numerical percentages and custom colormaps (Electric Cyan, Solar Amber, Cosmic Magma, Matrix Emerald).
- **Token Alignment Flow**: Bipartite Bézier curve ribbon illustrating cross-lingual word order reordering.
- **Audio Speech Synthesis**: Native Web Speech playback for English and Uzbek text.
- **High-Res Heatmap Export**: One-click download of the attention matrix as PNG.

---

## 📜 References

1. **Bahdanau, D., Cho, K., & Bengio, Y. (2015).** *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
2. **Sutskever, I., Vinyals, O., & Le, Q. V. (2014).** *Sequence to Sequence Learning with Neural Networks.* NeurIPS 2014.
3. **Jonibek21/English-Uzbek-Translation.** Parallel bilingual corpus (Hugging Face Datasets).
