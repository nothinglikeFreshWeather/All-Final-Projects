"""
Fake News Detection — Flask API & Frontend Server
==================================================
Runs on  : http://localhost:5000
Endpoints:
  GET  /          → Serves the web UI (templates/index.html)
  POST /predict   → JSON {"text": "..."} → prediction + confidence
  GET  /health    → Model status check

Model loading strategy:
  1. Try  ./fake_news_model/   (fine-tuned weights)
  2. Fall back to  distilroberta-base  from HuggingFace Hub
     (untrained — predictions won't be meaningful, but UI works)
"""

import os
import json
import logging

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
LOCAL_MODEL_DIR = "./fake_news_model"
HUB_MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 128

# ── Global model state ───────────────────────────────────────────────────────
tokenizer = None
model = None
device = None
model_source = None          # "local" | "hub"
model_ready = False


# ── Model loading ─────────────────────────────────────────────────────────────
def _is_valid_model_dir(path: str) -> bool:
    """Return True if the directory contains a saved HuggingFace model."""
    required = {"config.json"}
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    return bool(required & files)


def load_model():
    global tokenizer, model, device, model_source, model_ready

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if _is_valid_model_dir(LOCAL_MODEL_DIR):
        load_path = LOCAL_MODEL_DIR
        model_source = "local"
        logger.info(f"Loading fine-tuned model from  {LOCAL_MODEL_DIR}")
    else:
        load_path = HUB_MODEL_NAME
        model_source = "hub"
        logger.warning(
            f"No fine-tuned weights found in '{LOCAL_MODEL_DIR}'. "
            f"Falling back to base model '{HUB_MODEL_NAME}' from HuggingFace Hub. "
            "Predictions will NOT be meaningful until fine-tuned weights are provided."
        )

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=1,
    )
    model.to(device)
    model.eval()
    model_ready = True
    logger.info(f"Model ready  (source: {model_source})")


# Load model at startup
load_model()


# ── Helper ────────────────────────────────────────────────────────────────────
def predict_text(text: str) -> dict:
    """
    Run inference on a single text string.

    Returns
    -------
    dict with keys:
        label            : "FAKE" or "REAL"
        confidence       : float 0-1  (confidence in the predicted label)
        probability_fake : float 0-1
        probability_real : float 0-1
        logit            : raw model output
    """
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        output = model(**encoding)

    logit = output.logits.squeeze(-1).item()
    prob_fake = float(1.0 / (1.0 + np.exp(-logit)))   # sigmoid
    prob_real = 1.0 - prob_fake

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else prob_real

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probability_fake": round(prob_fake, 4),
        "probability_real": round(prob_real, 4),
        "logit": round(logit, 4),
        "model_source": model_source,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_source=model_source)


@app.route("/predict", methods=["POST"])
def predict():
    if not model_ready:
        return jsonify({"error": "Model is not ready yet."}), 503

    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide a non-empty 'text' field."}), 400

    if len(text) < 10:
        return jsonify({"error": "Text is too short. Please enter at least 10 characters."}), 400

    try:
        result = predict_text(text)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Inference error")
        return jsonify({"error": f"Inference failed: {str(exc)}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if model_ready else "loading",
        "model_source": model_source,
        "device": str(device) if device else None,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
