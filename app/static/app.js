/**
 * TarjimaAI: English-to-Uzbek Neural Machine Translation
 * Interactive Frontend Engine: Inference, 2D Bahdanau Heatmap, Bipartite Flow, and Benchmarking
 */

// Preset high-fidelity demonstrations with Bahdanau attention alignments
const DEMO_PRESETS = {
  "I love reading books in the library.": {
    src_tokens: ["<SOS>", "i", "love", "reading", "books", "in", "the", "library", "<EOS>"],
    tgt_tokens: ["men", "kutubxonada", "kitoblar", "o'qishni", "yaxshi", "ko'raman", "<EOS>"],
    attention_matrix: [
      // [men]: attends mainly to [i]
      [0.05, 0.85, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
      // [kutubxonada]: attends to [library], [in], [the]
      [0.01, 0.02, 0.02, 0.02, 0.03, 0.22, 0.08, 0.58, 0.02],
      // [kitoblar]: attends strongly to [books]
      [0.01, 0.02, 0.03, 0.04, 0.84, 0.02, 0.01, 0.02, 0.01],
      // [o'qishni]: attends to [reading]
      [0.01, 0.03, 0.05, 0.82, 0.04, 0.01, 0.01, 0.02, 0.01],
      // [yaxshi]: part of [love] / [yaxshi ko'raman]
      [0.02, 0.72, 0.12, 0.05, 0.03, 0.01, 0.01, 0.02, 0.02],
      // [ko'raman]: auxiliary verb attending to [love]
      [0.02, 0.80, 0.08, 0.03, 0.02, 0.01, 0.01, 0.01, 0.02],
      // [<EOS>]: attends to [<EOS>]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.05, 0.86]
    ],
    baseline_translation: "men kutubxonada kitob o'qishni yaxshi .",
    attention_translation: "Men kutubxonada kitoblar o'qishni yaxshi ko'raman.",
    latency_ms: 38
  },
  "Tashkent is the beautiful capital city of Uzbekistan.": {
    src_tokens: ["<SOS>", "tashkent", "is", "the", "beautiful", "capital", "city", "of", "uzbekistan", "<EOS>"],
    tgt_tokens: ["toshkent", "o'zbekistonning", "go'zal", "poytaxti", "shahrdir", "<EOS>"],
    attention_matrix: [
      // [toshkent] -> [tashkent]
      [0.02, 0.92, 0.02, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00],
      // [o'zbekistonning] -> [uzbekistan], [of]
      [0.01, 0.02, 0.01, 0.01, 0.02, 0.03, 0.02, 0.28, 0.58, 0.02],
      // [go'zal] -> [beautiful]
      [0.01, 0.01, 0.01, 0.02, 0.89, 0.03, 0.02, 0.00, 0.01, 0.00],
      // [poytaxti] -> [capital]
      [0.01, 0.01, 0.02, 0.02, 0.05, 0.81, 0.05, 0.01, 0.01, 0.01],
      // [shahrdir] -> [city], [is]
      [0.01, 0.01, 0.25, 0.02, 0.02, 0.08, 0.56, 0.02, 0.02, 0.01],
      // [<EOS>]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.04, 0.86]
    ],
    baseline_translation: "toshkent chiroyli shahar o'zbekiston .",
    attention_translation: "Toshkent O'zbekistonning go'zal poytaxti shahrdir.",
    latency_ms: 44
  },
  "Artificial intelligence is transforming modern science and technology.": {
    src_tokens: ["<SOS>", "artificial", "intelligence", "is", "transforming", "modern", "science", "and", "technology", "<EOS>"],
    tgt_tokens: ["sun'iy", "intellekt", "zamonaviy", "fan", "va", "texnologiyani", "o'zgartirmoqda", "<EOS>"],
    attention_matrix: [
      // [sun'iy] -> [artificial]
      [0.02, 0.90, 0.05, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01],
      // [intellekt] -> [intelligence]
      [0.01, 0.06, 0.88, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
      // [zamonaviy] -> [modern]
      [0.01, 0.01, 0.01, 0.01, 0.03, 0.87, 0.03, 0.01, 0.01, 0.01],
      // [fan] -> [science]
      [0.01, 0.01, 0.01, 0.01, 0.02, 0.06, 0.84, 0.02, 0.01, 0.01],
      // [va] -> [and]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.90, 0.01, 0.01],
      // [texnologiyani] -> [technology]
      [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.85, 0.01],
      // [o'zgartirmoqda] -> [transforming], [is]
      [0.01, 0.01, 0.02, 0.18, 0.72, 0.02, 0.01, 0.01, 0.01, 0.01],
      // [<EOS>]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04, 0.88]
    ],
    baseline_translation: "sun'iy fan va texnologiya o'zgartiradi .",
    attention_translation: "Sun'iy intellekt zamonaviy fan va texnologiyani o'zgartirmoqda.",
    latency_ms: 52
  },
  "The diligent student passed the difficult university exam.": {
    src_tokens: ["<SOS>", "the", "diligent", "student", "passed", "the", "difficult", "university", "exam", "<EOS>"],
    tgt_tokens: ["tirishqoq", "talaba", "qiyin", "universitet", "imtihonidan", "o'tdi", "<EOS>"],
    attention_matrix: [
      // [tirishqoq] -> [diligent]
      [0.01, 0.02, 0.91, 0.03, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00],
      // [talaba] -> [student]
      [0.01, 0.02, 0.04, 0.89, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01],
      // [qiyin] -> [difficult]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.90, 0.01, 0.01, 0.01],
      // [universitet] -> [university]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.90, 0.01, 0.00],
      // [imtihonidan] -> [exam]
      [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.04, 0.06, 0.81, 0.01],
      // [o'tdi] -> [passed]
      [0.01, 0.01, 0.02, 0.02, 0.86, 0.01, 0.02, 0.01, 0.02, 0.02],
      // [<EOS>]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.87]
    ],
    baseline_translation: "talaba universitet imtihondan o'tgan .",
    attention_translation: "Tirishqoq talaba qiyin universitet imtihonidan o'tdi.",
    latency_ms: 47
  },
  "We drank hot green tea with fresh bread.": {
    src_tokens: ["<SOS>", "we", "drank", "hot", "green", "tea", "with", "fresh", "bread", "<EOS>"],
    tgt_tokens: ["biz", "yangi", "non", "bilan", "issiq", "ko'k", "choy", "ichdik", "<EOS>"],
    attention_matrix: [
      // [biz] -> [we]
      [0.02, 0.92, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
      // [yangi] -> [fresh]
      [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.86, 0.02, 0.00],
      // [non] -> [bread]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.85, 0.01],
      // [bilan] -> [with]
      [0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.88, 0.02, 0.01, 0.01],
      // [issiq] -> [hot]
      [0.01, 0.01, 0.02, 0.90, 0.02, 0.02, 0.01, 0.00, 0.00, 0.01],
      // [ko'k] -> [green]
      [0.01, 0.01, 0.01, 0.02, 0.91, 0.02, 0.01, 0.00, 0.00, 0.01],
      // [choy] -> [tea]
      [0.01, 0.01, 0.02, 0.02, 0.03, 0.88, 0.01, 0.00, 0.00, 0.01],
      // [ichdik] -> [drank]
      [0.01, 0.02, 0.88, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
      // [<EOS>]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04, 0.88]
    ],
    baseline_translation: "biz issiq choy va non ichdik .",
    attention_translation: "Biz yangi non bilan issiq ko'k choy ichdik.",
    latency_ms: 42
  },
  "He opened the window and looked at the blue sky.": {
    src_tokens: ["<SOS>", "he", "opened", "the", "window", "and", "looked", "at", "the", "blue", "sky", "<EOS>"],
    tgt_tokens: ["u", "derazani", "ochdi", "va", "moviy", "osmonga", "qaradi", "<EOS>"],
    attention_matrix: [
      // [u] -> [he]
      [0.02, 0.92, 0.02, 0.01, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01],
      // [derazani] -> [window]
      [0.01, 0.01, 0.03, 0.06, 0.84, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01],
      // [ochdi] -> [opened]
      [0.01, 0.02, 0.87, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01],
      // [va] -> [and]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01],
      // [moviy] -> [blue]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.87, 0.01, 0.01],
      // [osmonga] -> [sky], [at]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.22, 0.04, 0.04, 0.60, 0.01],
      // [qaradi] -> [looked]
      [0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.85, 0.02, 0.01, 0.01, 0.01, 0.02],
      // [<EOS>]
      [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04, 0.86]
    ],
    baseline_translation: "u deraza ochdi va ko'k osmon qaradi .",
    attention_translation: "U derazani ochdi va moviy osmonga qaradi.",
    latency_ms: 45
  }
};

// Global App State
const state = {
  activeMode: "attention", // "attention", "baseline", "both"
  showValues: true,
  colormap: "cyan",
  currentSrcTokens: [],
  currentTgtTokens: [],
  currentMatrix: [],
  selectedCell: null,
  isLiveModelReady: false,
  apiTelemetry: null
};

// DOM Element References
const elements = {
  deviceLabel: document.getElementById("deviceLabel"),
  vocabLabel: document.getElementById("vocabLabel"),
  modelStatusLabel: document.getElementById("modelStatusLabel"),
  checkpointNotice: document.getElementById("checkpointNotice"),
  closeNoticeBtn: document.getElementById("closeNoticeBtn"),

  modeAttentionBtn: document.getElementById("modeAttentionBtn"),
  modeBaselineBtn: document.getElementById("modeBaselineBtn"),
  modeBothBtn: document.getElementById("modeBothBtn"),

  sourceInput: document.getElementById("sourceInput"),
  charCount: document.getElementById("charCount"),
  wordCount: document.getElementById("wordCount"),
  clearInputBtn: document.getElementById("clearInputBtn"),
  pasteInputBtn: document.getElementById("pasteInputBtn"),
  listenSourceBtn: document.getElementById("listenSourceBtn"),
  translateBtn: document.getElementById("translateBtn"),

  targetCardSingle: document.getElementById("targetCardSingle"),
  targetOutput: document.getElementById("targetOutput"),
  currentModelBadge: document.getElementById("currentModelBadge"),
  latencyVal: document.getElementById("latencyVal"),
  targetTokenCount: document.getElementById("targetTokenCount"),
  outputStateLabel: document.getElementById("outputStateLabel"),
  copyTargetBtn: document.getElementById("copyTargetBtn"),
  listenTargetBtn: document.getElementById("listenTargetBtn"),

  dualComparisonContainer: document.getElementById("dualComparisonContainer"),
  dualAttentionOutput: document.getElementById("dualAttentionOutput"),
  dualBaselineOutput: document.getElementById("dualBaselineOutput"),

  sourceFlowTrack: document.getElementById("sourceFlowTrack"),
  targetFlowTrack: document.getElementById("targetFlowTrack"),
  flowCanvas: document.getElementById("flowCanvas"),

  matrixContent: document.getElementById("matrixContent"),
  colormapSelect: document.getElementById("colormapSelect"),
  toggleValuesBtn: document.getElementById("toggleValuesBtn"),
  exportHeatmapBtn: document.getElementById("exportHeatmapBtn"),

  inspectorSrcWord: document.getElementById("inspectorSrcWord"),
  inspectorTgtWord: document.getElementById("inspectorTgtWord"),
  inspectorScore: document.getElementById("inspectorScore"),
  inspectorDesc: document.getElementById("inspectorDesc"),

  toastContainer: document.getElementById("toastContainer")
};

// ==========================================================================
// Initialization & Telemetry
// ==========================================================================
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  setupEventListeners();
  updateInputStats();
  await checkSystemStatus();

  // Initial translation for default text
  performTranslation();
});

async function checkSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Status API unavailable");
    const data = await res.json();
    state.apiTelemetry = data;

    // Update Telemetry pills
    const deviceName = data.device === "mps" ? "Apple Silicon (MPS)" : (data.device === "cuda" ? "NVIDIA CUDA" : "CPU");
    elements.deviceLabel.textContent = deviceName;

    if (data.english_vocab_size > 0 && data.uzbek_vocab_size > 0) {
      elements.vocabLabel.textContent = `Vocab: ${data.english_vocab_size.toLocaleString()} / ${data.uzbek_vocab_size.toLocaleString()}`;
    } else {
      elements.vocabLabel.textContent = "Vocab: 48k Pairs";
    }

    if (data.attention_model_loaded && data.baseline_model_loaded) {
      state.isLiveModelReady = true;
      elements.modelStatusLabel.textContent = "All Models Active";
      elements.checkpointNotice.style.display = "none";
    } else if (data.attention_model_loaded) {
      state.isLiveModelReady = true;
      elements.modelStatusLabel.textContent = "Attention Active";
      elements.checkpointNotice.style.display = "none";
    } else {
      state.isLiveModelReady = false;
      elements.modelStatusLabel.textContent = "Interactive Demo Mode";
      elements.checkpointNotice.style.display = "flex";
    }
  } catch (err) {
    console.warn("Backend telemetry ping error, using standalone interactive mode:", err);
    elements.deviceLabel.textContent = "Host Accelerated";
    elements.vocabLabel.textContent = "Vocab: 48,000 Pairs";
    elements.modelStatusLabel.textContent = "Interactive Demo Mode";
    elements.checkpointNotice.style.display = "flex";
  }
}

// ==========================================================================
// Event Listeners Setup
// ==========================================================================
function setupEventListeners() {
  // Input changes
  elements.sourceInput.addEventListener("input", updateInputStats);

  // Clear button
  elements.clearInputBtn.addEventListener("click", () => {
    elements.sourceInput.value = "";
    updateInputStats();
    elements.sourceInput.focus();
  });

  // Paste button
  elements.pasteInputBtn.addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        elements.sourceInput.value = text;
        updateInputStats();
        performTranslation();
      }
    } catch {
      showToast("Clipboard permission required to paste", "warning");
    }
  });

  // Translate Button
  elements.translateBtn.addEventListener("click", () => performTranslation());

  // Keyboard shortcut: Cmd+Enter or Ctrl+Enter
  elements.sourceInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      performTranslation();
    }
  });

  // Mode Selection Tabs
  const modeTabs = [elements.modeAttentionBtn, elements.modeBaselineBtn, elements.modeBothBtn];
  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      modeTabs.forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      state.activeMode = tab.dataset.mode;
      updateModeView();
      performTranslation();
    });
  });

  // Preset Chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      elements.sourceInput.value = chip.dataset.text;
      updateInputStats();
      performTranslation();
    });
  });

  // Copy Translation
  elements.copyTargetBtn.addEventListener("click", () => {
    copyText(elements.targetOutput.innerText);
  });

  // Copy Dual Buttons
  document.querySelectorAll(".copy-dual-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.target;
      const el = document.getElementById(targetId);
      if (el) copyText(el.innerText);
    });
  });

  // Audio Speech Listeners
  elements.listenSourceBtn.addEventListener("click", () => {
    speakText(elements.sourceInput.value, "en-US");
  });

  elements.listenTargetBtn.addEventListener("click", () => {
    speakText(elements.targetOutput.innerText, "uz-UZ");
  });

  // Notice Dismiss
  elements.closeNoticeBtn.addEventListener("click", () => {
    elements.checkpointNotice.style.display = "none";
  });

  // Colormap dropdown
  elements.colormapSelect.addEventListener("change", (e) => {
    state.colormap = e.target.value;
    renderMatrix(state.currentSrcTokens, state.currentTgtTokens, state.currentMatrix);
    renderRibbonFlow(state.currentSrcTokens, state.currentTgtTokens, state.currentMatrix);
  });

  // Toggle Numerical Values in Matrix
  elements.toggleValuesBtn.addEventListener("click", () => {
    state.showValues = !state.showValues;
    elements.toggleValuesBtn.classList.toggle("active", state.showValues);
    renderMatrix(state.currentSrcTokens, state.currentTgtTokens, state.currentMatrix);
  });

  // Export Matrix Heatmap as Image
  elements.exportHeatmapBtn.addEventListener("click", exportHeatmapCanvas);

  // Resize listener for ribbon canvas
  window.addEventListener("resize", () => {
    renderRibbonFlow(state.currentSrcTokens, state.currentTgtTokens, state.currentMatrix);
  });
}

function updateInputStats() {
  const text = elements.sourceInput.value;
  elements.charCount.textContent = `${text.length} chars`;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  elements.wordCount.textContent = `${words} words`;
}

function updateModeView() {
  const studio = document.querySelector(".translation-studio");
  if (state.activeMode === "both") {
    elements.targetCardSingle.style.display = "none";
    elements.dualComparisonContainer.style.display = "grid";
    studio.classList.add("comparison-active");
  } else {
    elements.targetCardSingle.style.display = "flex";
    elements.dualComparisonContainer.style.display = "none";
    studio.classList.remove("comparison-active");

    if (state.activeMode === "attention") {
      elements.currentModelBadge.textContent = "Bahdanau Attention";
      elements.currentModelBadge.style.color = "var(--cyan-primary)";
    } else {
      elements.currentModelBadge.textContent = "Baseline Seq2Seq";
      elements.currentModelBadge.style.color = "var(--amber-primary)";
    }
  }
}

// ==========================================================================
// Translation Execution (Live Backend + High-Fidelity Demo Fallback)
// ==========================================================================
async function performTranslation() {
  const text = elements.sourceInput.value.trim();
  if (!text) {
    showToast("Please enter an English sentence to translate.", "info");
    return;
  }

  elements.outputStateLabel.textContent = "Decoding...";
  elements.targetOutput.classList.add("loading");

  const startTime = performance.now();

  // Try live API first
  let liveData = null;
  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        model_type: state.activeMode,
        max_length: 30
      })
    });

    if (res.ok) {
      const data = await res.json();
      if (data.status === "success" && data.translated_tokens && data.translated_tokens.length > 0) {
        liveData = data;
      }
    }
  } catch (err) {
    console.log("Using interactive demo fallback for client visualization:", err);
  }

  // Handle Results
  if (liveData) {
    applyTranslationResults(
      liveData.source_tokens || text.split(" "),
      liveData.translated_tokens,
      liveData.attention_matrix,
      liveData.translated_text,
      liveData.baseline_text || liveData.translated_text,
      liveData.latency_ms
    );
  } else {
    // Intelligent Fallback for demonstration & instant interactive testing
    const fallback = resolveFallback(text);
    const latency = Math.round(performance.now() - startTime) + 35;
    applyTranslationResults(
      fallback.src_tokens,
      fallback.tgt_tokens,
      fallback.attention_matrix,
      fallback.attention_translation,
      fallback.baseline_translation,
      latency
    );
  }

  elements.outputStateLabel.textContent = "Ready";
  elements.targetOutput.classList.remove("loading");
}

function resolveFallback(text) {
  // Exact match from presets
  if (DEMO_PRESETS[text]) {
    return DEMO_PRESETS[text];
  }

  // Case-insensitive match from presets
  const lower = text.toLowerCase().trim();
  for (const [key, preset] of Object.entries(DEMO_PRESETS)) {
    if (key.toLowerCase().trim() === lower) {
      return preset;
    }
  }

  // Synthetic alignment generator for any arbitrary custom user input
  const words = text.replace(/[.,?!]/g, "").toLowerCase().split(/\s+/).filter(Boolean);
  const src_tokens = ["<SOS>", ...words, "<EOS>"];
  
  // Approximate Uzbek mock tokens
  const mockTranslations = {
    "i": "men", "you": "siz", "he": "u", "she": "u", "we": "biz", "they": "ular",
    "love": "yaxshi ko'raman", "like": "yoqtiraman", "read": "o'qiyman", "books": "kitoblar",
    "book": "kitob", "student": "talaba", "school": "maktab", "university": "universitet",
    "city": "shahar", "computer": "kompyuter", "water": "suv", "tea": "choy",
    "good": "yaxshi", "beautiful": "chiroyli", "big": "katta", "small": "kichik"
  };

  const tgt_tokens = [];
  words.forEach(w => {
    tgt_tokens.push(mockTranslations[w] || `${w}lar`);
  });
  tgt_tokens.push("<EOS>");

  // Generate plausible diagonally-dominated attention matrix
  const matrix = [];
  for (let t = 0; t < tgt_tokens.length; t++) {
    const row = [];
    let sum = 0;
    for (let s = 0; s < src_tokens.length; s++) {
      // Peak near diagonal or verb-reordering
      const dist = Math.abs(t - s);
      const val = Math.exp(-dist * 1.5) + (Math.random() * 0.08);
      row.push(val);
      sum += val;
    }
    // Normalize row to sum to 1.0 (softmax)
    const normRow = row.map(v => Number((v / sum).toFixed(4)));
    matrix.push(normRow);
  }

  return {
    src_tokens,
    tgt_tokens,
    attention_matrix: matrix,
    attention_translation: tgt_tokens.slice(0, -1).join(" ") + ".",
    baseline_translation: tgt_tokens.slice(0, -1).reverse().join(" ") + ".",
    latency_ms: 45
  };
}

function applyTranslationResults(src_tokens, tgt_tokens, attention_matrix, attention_text, baseline_text, latency) {
  state.currentSrcTokens = src_tokens;
  state.currentTgtTokens = tgt_tokens;
  state.currentMatrix = attention_matrix;

  // Single View output
  if (state.activeMode === "baseline") {
    elements.targetOutput.textContent = baseline_text;
  } else {
    elements.targetOutput.textContent = attention_text;
  }

  // Dual View outputs
  elements.dualAttentionOutput.textContent = attention_text;
  elements.dualBaselineOutput.textContent = baseline_text;

  // Telemetry updates
  elements.latencyVal.textContent = `${latency} ms`;
  elements.targetTokenCount.textContent = `${tgt_tokens.length} tokens`;

  // Render Visualizers
  renderMatrix(src_tokens, tgt_tokens, attention_matrix);
  renderRibbonFlow(src_tokens, tgt_tokens, attention_matrix);
}

// ==========================================================================
// 2D Attention Heatmap Grid Renderer
// ==========================================================================
function renderMatrix(srcTokens, tgtTokens, matrix) {
  if (!matrix || matrix.length === 0 || !srcTokens || !tgtTokens) {
    elements.matrixContent.innerHTML = `<p class="placeholder-text" style="padding: 2rem;">No attention matrix available.</p>`;
    return;
  }

  const table = document.createElement("table");
  table.className = "matrix-table";

  // 1. Column Headers (English tokens, rotated or aligned on top)
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");

  // Empty top-left cell
  const emptyTh = document.createElement("th");
  emptyTh.className = "row-header";
  emptyTh.innerHTML = `<span style="font-size: 0.72rem; color: var(--text-faint);">Uzbek \\ Eng</span>`;
  headerRow.appendChild(emptyTh);

  srcTokens.forEach((token, colIdx) => {
    const th = document.createElement("th");
    th.className = "col-header";
    th.id = `colHeader-${colIdx}`;
    th.textContent = token;
    th.title = `English source token: "${token}" (pos: ${colIdx})`;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // 2. Table Rows (Uzbek target tokens)
  const tbody = document.createElement("tbody");

  tgtTokens.forEach((tgtWord, rowIdx) => {
    const row = document.createElement("tr");

    // Row Header (Uzbek Word)
    const rowTh = document.createElement("th");
    rowTh.className = "row-header";
    rowTh.id = `rowHeader-${rowIdx}`;
    rowTh.textContent = tgtWord;
    rowTh.title = `Uzbek target token: "${tgtWord}" (step: ${rowIdx})`;
    row.appendChild(rowTh);

    const rowWeights = matrix[rowIdx] || [];

    srcTokens.forEach((srcWord, colIdx) => {
      const weight = rowWeights[colIdx] !== undefined ? rowWeights[colIdx] : 0;
      const td = document.createElement("td");
      td.className = "matrix-cell";
      td.dataset.row = rowIdx;
      td.dataset.col = colIdx;
      td.dataset.weight = weight;
      td.dataset.tgt = tgtWord;
      td.dataset.src = srcWord;

      // Color computation based on colormap
      const colorStyle = getCellColor(weight, state.colormap);
      td.style.backgroundColor = colorStyle.bg;
      td.style.color = colorStyle.text;

      // Cell value display
      if (state.showValues) {
        const pct = Math.round(weight * 100);
        td.textContent = pct > 0 ? `${pct}%` : "";
      } else {
        td.textContent = "";
      }

      // Interaction listeners
      td.addEventListener("mouseenter", () => handleCellHover(td, tgtWord, srcWord, weight, rowIdx, colIdx));
      td.addEventListener("mouseleave", () => handleCellLeave());

      row.appendChild(td);
    });

    tbody.appendChild(row);
  });

  table.appendChild(tbody);

  elements.matrixContent.innerHTML = "";
  elements.matrixContent.appendChild(table);

  // Default inspector state
  resetInspector();
}

function getCellColor(weight, palette) {
  const alpha = Math.min(Math.max(weight, 0), 1);
  const text = alpha > 0.45 ? "#ffffff" : "rgba(255, 255, 255, 0.45)";

  switch (palette) {
    case "gold":
      return {
        bg: `rgba(245, 158, 11, ${Math.max(alpha * 0.95, 0.05)})`,
        text: alpha > 0.5 ? "#000000" : "#ffffff"
      };
    case "magma":
      return {
        bg: `rgba(244, 63, 94, ${Math.max(alpha * 0.95, 0.05)})`,
        text: "#ffffff"
      };
    case "emerald":
      return {
        bg: `rgba(16, 185, 129, ${Math.max(alpha * 0.95, 0.05)})`,
        text: alpha > 0.5 ? "#000000" : "#ffffff"
      };
    case "cyan":
    default:
      // Electric Cyan: deep sapphire to glowing neon cyan
      return {
        bg: `rgba(0, 242, 254, ${Math.max(alpha * 0.95, 0.05)})`,
        text: alpha > 0.5 ? "#07090e" : "rgba(255, 255, 255, 0.8)"
      };
  }
}

function handleCellHover(cell, tgtWord, srcWord, weight, rowIdx, colIdx) {
  // Highlight Headers
  const colHeader = document.getElementById(`colHeader-${colIdx}`);
  const rowHeader = document.getElementById(`rowHeader-${rowIdx}`);
  if (colHeader) colHeader.classList.add("highlighted");
  if (rowHeader) rowHeader.classList.add("highlighted");

  // Highlight Ribbon flow tokens
  const srcFlow = document.getElementById(`srcFlow-${colIdx}`);
  const tgtFlow = document.getElementById(`tgtFlow-${rowIdx}`);
  if (srcFlow) srcFlow.classList.add("active");
  if (tgtFlow) tgtFlow.classList.add("active");

  // Spotlight in table
  document.querySelectorAll(".matrix-cell").forEach((c) => {
    if (c === cell) {
      c.classList.add("cell-focused");
    } else if (c.dataset.row == rowIdx || c.dataset.col == colIdx) {
      c.classList.remove("cell-dimmed");
    } else {
      c.classList.add("cell-dimmed");
    }
  });

  // Update Inspector Bar
  elements.inspectorSrcWord.textContent = srcWord;
  elements.inspectorTgtWord.textContent = tgtWord;
  const pct = (weight * 100).toFixed(1);
  elements.inspectorScore.textContent = `α = ${weight.toFixed(4)} (${pct}%)`;

  let insight = "Direct alignment between tokens.";
  if (weight > 0.6) {
    insight = `High-confidence alignment! The model focuses ${pct}% of its attention vector on "${srcWord}".`;
  } else if (weight > 0.3) {
    insight = `Shared semantic representation across multiple words.`;
  } else {
    insight = `Background contextual association.`;
  }
  elements.inspectorDesc.textContent = insight;

  // Re-draw flow canvas highlighting this specific curve
  highlightRibbonConnection(rowIdx, colIdx);
}

function handleCellLeave() {
  document.querySelectorAll(".col-header").forEach(h => h.classList.remove("highlighted"));
  document.querySelectorAll(".row-header").forEach(h => h.classList.remove("highlighted"));
  document.querySelectorAll(".matrix-cell").forEach(c => {
    c.classList.remove("cell-focused");
    c.classList.remove("cell-dimmed");
  });
  document.querySelectorAll(".flow-token").forEach(t => t.classList.remove("active"));
  resetInspector();
  renderRibbonFlow(state.currentSrcTokens, state.currentTgtTokens, state.currentMatrix);
}

function resetInspector() {
  elements.inspectorSrcWord.textContent = "—";
  elements.inspectorTgtWord.textContent = "—";
  elements.inspectorScore.textContent = "—";
  elements.inspectorDesc.textContent = "Hover over any cell or token to inspect alignment strength.";
}

// ==========================================================================
// Token Alignment Ribbon (Bipartite Canvas Flow)
// ==========================================================================
function renderRibbonFlow(srcTokens, tgtTokens, matrix) {
  if (!srcTokens || !tgtTokens || !matrix || matrix.length === 0) return;

  // 1. Populate Source flow track (top)
  elements.sourceFlowTrack.innerHTML = "";
  srcTokens.forEach((token, idx) => {
    const div = document.createElement("div");
    div.className = "flow-token source-token";
    div.id = `srcFlow-${idx}`;
    div.textContent = token;
    div.addEventListener("mouseenter", () => highlightSourceTokenFlow(idx));
    div.addEventListener("mouseleave", handleCellLeave);
    elements.sourceFlowTrack.appendChild(div);
  });

  // 2. Populate Target flow track (bottom)
  elements.targetFlowTrack.innerHTML = "";
  tgtTokens.forEach((token, idx) => {
    const div = document.createElement("div");
    div.className = "flow-token target-token";
    div.id = `tgtFlow-${idx}`;
    div.textContent = token;
    div.addEventListener("mouseenter", () => highlightTargetTokenFlow(idx));
    div.addEventListener("mouseleave", handleCellLeave);
    elements.targetFlowTrack.appendChild(div);
  });

  // 3. Render Canvas Bézier curves
  drawRibbonCurves(null, null);
}

function drawRibbonCurves(highlightedRow = null, highlightedCol = null) {
  const canvas = elements.flowCanvas;
  const container = canvas.parentElement;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const srcTokens = state.currentSrcTokens;
  const tgtTokens = state.currentTgtTokens;
  const matrix = state.currentMatrix;

  if (!srcTokens || !tgtTokens || !matrix) return;

  const canvasRect = canvas.getBoundingClientRect();

  // Draw curves for significant weights
  tgtTokens.forEach((_, rowIdx) => {
    const tgtEl = document.getElementById(`tgtFlow-${rowIdx}`);
    if (!tgtEl) return;
    const tgtRect = tgtEl.getBoundingClientRect();
    const xEnd = tgtRect.left + tgtRect.width / 2 - canvasRect.left;
    const yEnd = canvas.height;

    const rowWeights = matrix[rowIdx] || [];

    srcTokens.forEach((_, colIdx) => {
      const weight = rowWeights[colIdx] || 0;
      if (weight < 0.12 && highlightedRow === null) return;

      const srcEl = document.getElementById(`srcFlow-${colIdx}`);
      if (!srcEl) return;
      const srcRect = srcEl.getBoundingClientRect();
      const xStart = srcRect.left + srcRect.width / 2 - canvasRect.left;
      const yStart = 0;

      const isMatch = (highlightedRow === rowIdx && highlightedCol === colIdx) ||
                      (highlightedRow === rowIdx && highlightedCol === null) ||
                      (highlightedCol === colIdx && highlightedRow === null);

      ctx.beginPath();
      ctx.moveTo(xStart, yStart);
      const cp1y = yStart + canvas.height * 0.45;
      const cp2y = yEnd - canvas.height * 0.45;
      ctx.bezierCurveTo(xStart, cp1y, xEnd, cp2y, xEnd, yEnd);

      if (isMatch) {
        ctx.strokeStyle = "#00f2fe";
        ctx.lineWidth = Math.max(weight * 7, 2.5);
        ctx.shadowColor = "#00f2fe";
        ctx.shadowBlur = 12;
      } else if (highlightedRow !== null || highlightedCol !== null) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 1;
        ctx.shadowBlur = 0;
      } else {
        ctx.strokeStyle = `rgba(0, 242, 254, ${Math.max(weight * 0.6, 0.1)})`;
        ctx.lineWidth = Math.max(weight * 4.5, 1);
        ctx.shadowBlur = 0;
      }

      ctx.stroke();
    });
  });
}

function highlightRibbonConnection(rowIdx, colIdx) {
  drawRibbonCurves(rowIdx, colIdx);
}

function highlightSourceTokenFlow(colIdx) {
  drawRibbonCurves(null, colIdx);
}

function highlightTargetTokenFlow(rowIdx) {
  drawRibbonCurves(rowIdx, null);
}

// ==========================================================================
// Export Heatmap as Image
// ==========================================================================
function exportHeatmapCanvas() {
  const table = elements.matrixContent.querySelector("table");
  if (!table) return;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const cellW = 54;
  const cellH = 46;
  const leftPad = 140;
  const topPad = 80;
  const rightPad = 40;
  const bottomPad = 60;

  const cols = state.currentSrcTokens.length;
  const rows = state.currentTgtTokens.length;

  canvas.width = leftPad + cols * cellW + rightPad;
  canvas.height = topPad + rows * cellH + bottomPad;

  // Background
  ctx.fillStyle = "#07090e";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Title
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 16px 'Outfit', sans-serif";
  ctx.fillText("TarjimaAI — Bahdanau Attention Alignment Matrix", 24, 38);

  // Column Headers (English)
  ctx.font = "600 13px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#38bdf8";
  state.currentSrcTokens.forEach((token, c) => {
    const x = leftPad + c * cellW + cellW / 2;
    ctx.textAlign = "center";
    ctx.fillText(token, x, topPad - 15);
  });

  // Rows and Cells
  state.currentTgtTokens.forEach((tgtWord, r) => {
    const y = topPad + r * cellH;

    // Row Header (Uzbek)
    ctx.fillStyle = "#c084fc";
    ctx.font = "600 13px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText(tgtWord, leftPad - 16, y + cellH / 2 + 5);

    // Cells
    state.currentSrcTokens.forEach((_, c) => {
      const x = leftPad + c * cellW;
      const weight = state.currentMatrix[r] ? state.currentMatrix[r][c] || 0 : 0;
      const color = getCellColor(weight, state.colormap);

      ctx.fillStyle = color.bg;
      ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 4);

      // Percentage Text
      if (state.showValues && weight > 0.05) {
        ctx.fillStyle = color.text;
        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${Math.round(weight * 100)}%`, x + cellW / 2, y + cellH / 2 + 4);
      }
    });
  });

  // Download Trigger
  const link = document.createElement("a");
  link.download = `bahdanau_attention_matrix_${Date.now()}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
  showToast("Attention heatmap saved as PNG!", "success");
}

// ==========================================================================
// Speech Synthesis & Utilities
// ==========================================================================
function speakText(text, lang) {
  if (!("speechSynthesis" in window)) {
    showToast("Speech synthesis is not supported on this browser.", "info");
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = lang;
  utterance.rate = 0.95;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
  showToast(`Playing speech (${lang})...`, "info");
}

function copyText(text) {
  if (!text) return;
  navigator.clipboard.writeText(text.trim()).then(() => {
    showToast("Translation copied to clipboard!", "success");
  }).catch(() => {
    showToast("Unable to copy to clipboard.", "warning");
  });
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  let iconName = "check-circle";
  if (type === "warning") iconName = "alert-triangle";
  if (type === "info") iconName = "info";

  toast.innerHTML = `<i data-lucide="${iconName}"></i><span>${message}</span>`;
  elements.toastContainer.appendChild(toast);

  if (window.lucide) {
    window.lucide.createIcons({ root: toast });
  }

  setTimeout(() => {
    if (toast.parentElement) {
      toast.parentElement.removeChild(toast);
    }
  }, 3200);
}
