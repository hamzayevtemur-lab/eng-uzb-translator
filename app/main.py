"""
TarjimaAI: English-to-Uzbek Seq2Seq Neural Translator
FastAPI Backend providing translation inference, model switching, and real-time attention weight matrix extraction.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root and src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from src.dataset import Vocabulary, normalize_string, SOS_TOKEN, EOS_TOKEN
from src.model_baseline import Encoder as BaselineEncoder, Decoder as BaselineDecoder, Seq2SeqBaseline
from src.model_attention import Encoder as AttentionEncoder, Decoder as AttentionDecoder, Seq2SeqAttention
from src.inference import translate_attention, translate_baseline

app = FastAPI(
    title="TarjimaAI | English-Uzbek Seq2Seq Neural Translator",
    description="Interactive Neural Machine Translation featuring Bahdanau Additive Attention and Live Alignment Heatmaps",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory
STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Device detection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Global Model Cache
eng_vocab: Optional[Vocabulary] = None
uzb_vocab: Optional[Vocabulary] = None
attention_model: Optional[Seq2SeqAttention] = None
baseline_model: Optional[Seq2SeqBaseline] = None


def init_models():
    """Initializes and loads available vocabularies and weights from checkpoints."""
    global eng_vocab, uzb_vocab, attention_model, baseline_model

    vocab_eng_path = CHECKPOINT_DIR / "vocab_eng.json"
    vocab_uzb_path = CHECKPOINT_DIR / "vocab_uzb.json"

    if vocab_eng_path.exists() and vocab_uzb_path.exists():
        try:
            eng_vocab = Vocabulary.load(str(vocab_eng_path))
            uzb_vocab = Vocabulary.load(str(vocab_uzb_path))
            print(f"[TarjimaAI] Loaded vocabularies: English ({eng_vocab.n_words} words), Uzbek ({uzb_vocab.n_words} words)")
        except Exception as e:
            print(f"[TarjimaAI] Error loading vocabularies: {e}")

    # Load Attention Model
    attn_path = CHECKPOINT_DIR / "seq2seq_attention.pth"
    if attn_path.exists() and eng_vocab and uzb_vocab:
        try:
            state = torch.load(str(attn_path), map_location=DEVICE)
            enc = AttentionEncoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            dec = AttentionDecoder(uzb_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            attention_model = Seq2SeqAttention(enc, dec, DEVICE).to(DEVICE)
            attention_model.load_state_dict(state["model_state_dict"])
            attention_model.eval()
            print("[TarjimaAI] Loaded Seq2Seq Bahdanau Attention model checkpoint!")
        except Exception as e:
            print(f"[TarjimaAI] Error loading attention model: {e}")

    # Load Baseline Model
    base_path = CHECKPOINT_DIR / "seq2seq_baseline.pth"
    if base_path.exists() and eng_vocab and uzb_vocab:
        try:
            state = torch.load(str(base_path), map_location=DEVICE)
            enc = BaselineEncoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            dec = BaselineDecoder(uzb_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            baseline_model = Seq2SeqBaseline(enc, dec, DEVICE).to(DEVICE)
            baseline_model.load_state_dict(state["model_state_dict"])
            baseline_model.eval()
            print("[TarjimaAI] Loaded Seq2Seq Baseline model checkpoint!")
        except Exception as e:
            print(f"[TarjimaAI] Error loading baseline model: {e}")


# Initialize on startup
init_models()


class TranslationRequest(BaseModel):
    text: str
    model_type: Optional[str] = "attention"  # "attention", "baseline", "both"
    max_length: Optional[int] = 30


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>TarjimaAI</h1><p>Frontend loading...</p>"


@app.get("/api/status")
async def get_system_status():
    """Returns telemetry on loaded models, vocabularies, and hardware accelerator."""
    return {
        "status": "online",
        "device": str(DEVICE),
        "attention_model_loaded": attention_model is not None,
        "baseline_model_loaded": baseline_model is not None,
        "english_vocab_size": eng_vocab.n_words if eng_vocab else 0,
        "uzbek_vocab_size": uzb_vocab.n_words if uzb_vocab else 0,
        "checkpoints_found": {
            "attention": (CHECKPOINT_DIR / "seq2seq_attention.pth").exists(),
            "baseline": (CHECKPOINT_DIR / "seq2seq_baseline.pth").exists(),
            "vocabularies": (CHECKPOINT_DIR / "vocab_eng.json").exists()
        }
    }


@app.post("/api/translate")
async def translate_endpoint(req: TranslationRequest):
    """
    Translates an English sentence to Uzbek.
    Returns translation text, tokens, inference latency, and the 2D attention alignment matrix.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Re-check models if not loaded
    if eng_vocab is None or uzb_vocab is None:
        init_models()

    if eng_vocab is None or uzb_vocab is None:
        return {
            "source_text": req.text,
            "translated_text": "Model checkpoints not found yet. Please train the models using python3 src/train.py first.",
            "translated_tokens": [],
            "source_tokens": req.text.split(),
            "attention_matrix": [],
            "latency_ms": 0,
            "status": "no_checkpoints"
        }

    clean_src = normalize_string(req.text, is_uzbek=False)
    src_tokens = ["<SOS>"] + clean_src.split() + ["<EOS>"]
    
    t0 = time.time()
    result: Dict[str, Any] = {
        "source_text": req.text,
        "clean_source": clean_src,
        "source_tokens": src_tokens,
        "model_type": req.model_type
    }

    # 1. Translate with Attention
    if req.model_type in ("attention", "both") and attention_model:
        tokens, attn_matrix = translate_attention(
            sentence=req.text,
            model=attention_model,
            eng_vocab=eng_vocab,
            uzb_vocab=uzb_vocab,
            max_length=req.max_length,
            device=DEVICE
        )
        result["translated_tokens"] = tokens
        result["translated_text"] = " ".join(tokens)
        result["attention_matrix"] = attn_matrix.tolist()
    else:
        result["translated_tokens"] = []
        result["translated_text"] = ""
        result["attention_matrix"] = []

    # 2. Translate with Baseline (if requested or in comparison mode)
    if req.model_type in ("baseline", "both") and baseline_model:
        base_tokens = translate_baseline(
            sentence=req.text,
            model=baseline_model,
            eng_vocab=eng_vocab,
            uzb_vocab=uzb_vocab,
            max_length=req.max_length,
            device=DEVICE
        )
        result["baseline_tokens"] = base_tokens
        result["baseline_text"] = " ".join(base_tokens)
        if req.model_type == "baseline":
            result["translated_tokens"] = base_tokens
            result["translated_text"] = " ".join(base_tokens)

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    result["latency_ms"] = elapsed_ms
    result["status"] = "success"

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
