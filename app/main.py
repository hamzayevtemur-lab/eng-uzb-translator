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
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root and src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from dataset import Vocabulary, normalize_string, SOS_TOKEN, EOS_TOKEN
from model_baseline import Decoder as BaselineDecoder, Seq2SeqBaseline
from model_attention import Encoder, Decoder as AttentionDecoder, Seq2SeqAttention
from inference import translate_attention, beam_search_attention, translate_baseline

app = FastAPI(
    title="TarjimaAI | English-Uzbek Seq2Seq Neural Translator",
    description="Interactive Neural Machine Translation featuring Bahdanau Additive Attention and Live Alignment Heatmaps",
    version="1.1.0"
)

# Standardized CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Hardware detection
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
            print(f"[TarjimaAI] Loaded vocabularies: English ({eng_vocab.n_words:,} words), Uzbek ({uzb_vocab.n_words:,} words)")
        except Exception as e:
            print(f"[TarjimaAI] Error loading vocabularies: {e}")

    # Load Attention Model
    attn_path = CHECKPOINT_DIR / "seq2seq_attention.pth"
    if attn_path.exists() and eng_vocab and uzb_vocab:
        try:
            state = torch.load(str(attn_path), map_location=DEVICE, weights_only=True)
            enc = Encoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            dec = AttentionDecoder(
                uzb_vocab.n_words, 
                state["embedding_dim"], 
                state["hidden_dim"] * 2,  # Bidirectional encoder state
                state["hidden_dim"]
            )
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
            state = torch.load(str(base_path), map_location=DEVICE, weights_only=True)
            enc = Encoder(eng_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            dec = BaselineDecoder(uzb_vocab.n_words, state["embedding_dim"], state["hidden_dim"])
            baseline_model = Seq2SeqBaseline(enc, dec, DEVICE).to(DEVICE)
            baseline_model.load_state_dict(state["model_state_dict"])
            baseline_model.eval()
            print("[TarjimaAI] Loaded Seq2Seq Baseline model checkpoint!")
        except Exception as e:
            print(f"[TarjimaAI] Error loading baseline model: {e}")


# Initialize on server startup
init_models()


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="English sentence to translate")
    model_type: Optional[str] = Field("attention", pattern="^(attention|baseline|both)$")
    max_length: Optional[int] = Field(30, ge=1, le=100)
    beam_width: Optional[int] = Field(1, ge=1, le=5)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the main web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>TarjimaAI</h1><p>Frontend loading...</p>"


@app.get("/style.css")
def serve_style():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def serve_app_js():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/real_model_cache.json")
def serve_model_cache():
    return FileResponse(STATIC_DIR / "real_model_cache.json", media_type="application/json")


@app.get("/api/status")
def get_system_status():
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


# CPU/GPU-bound inference runs in synchronous threadpool to prevent blocking the event loop
@app.post("/api/translate")
def translate_endpoint(req: TranslationRequest):
    """
    Translates an English sentence to Uzbek using real neural network inference.
    Returns translation text, tokens, inference latency, and the 2D attention alignment matrix.
    """
    clean_text = req.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Re-check models if not loaded
    if eng_vocab is None or uzb_vocab is None:
        init_models()

    if eng_vocab is None or uzb_vocab is None or (attention_model is None and baseline_model is None):
        raise HTTPException(
            status_code=503,
            detail="Model checkpoints not loaded. Please train models using 'python3 src/train.py' first."
        )

    clean_src = normalize_string(clean_text, is_uzbek=False)
    src_tokens = ["<SOS>"] + clean_src.split() + ["<EOS>"]
    
    t0 = time.time()
    result: Dict[str, Any] = {
        "source_text": clean_text,
        "clean_source": clean_src,
        "source_tokens": src_tokens,
        "model_type": req.model_type,
        "beam_width": req.beam_width
    }

    # 1. Translate with Bahdanau Attention
    if req.model_type in ("attention", "both") and attention_model:
        if req.beam_width and req.beam_width > 1:
            tokens, attn_matrix = beam_search_attention(
                sentence=clean_text,
                model=attention_model,
                eng_vocab=eng_vocab,
                uzb_vocab=uzb_vocab,
                beam_width=req.beam_width,
                max_length=req.max_length,
                device=DEVICE
            )
        else:
            tokens, attn_matrix = translate_attention(
                sentence=clean_text,
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

    # 2. Translate with Baseline
    if req.model_type in ("baseline", "both") and baseline_model:
        base_tokens = translate_baseline(
            sentence=clean_text,
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

    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    result["status"] = "success"

    return result


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port)
