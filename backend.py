import re
import requests
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./bert_fineturn"
DEVICE = "cpu"

# Load once at startup, critical for speed
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def fetch_html(url: str, timeout: int = 10) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (PhishingDetector/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text

def html_to_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text: str, max_length: int = 256):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # [1, num_labels] or [1,1]

        if logits.shape[-1] == 1:
            phishing_prob = torch.sigmoid(logits).item()
            pred_label = 1 if phishing_prob >= 0.5 else 0
        else:
            probs = F.softmax(logits, dim=-1)[0]
            phishing_prob = probs[1].item()
            pred_label = int(torch.argmax(probs).item())

    return pred_label, phishing_prob

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "num_labels": int(getattr(model.config, "num_labels", -1)),
        "id2label": getattr(model.config, "id2label", None),
    })

@app.post("/predict_page")
def predict_page():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    html = data.get("html")
    max_chars = int(data.get("max_chars", 4000))
    max_length = int(data.get("max_length", 256))

    if not url and not html:
        return jsonify({"error": "Provide 'url' or 'html'."}), 400

    try:
        if url:
            html = fetch_html(url)
        text = html_to_visible_text(html)
        text = text[:max_chars]

        label, prob = predict_text(text, max_length=max_length)

        return jsonify({
            "label": int(label),
            "phishing_prob": float(prob),
            "text_len": len(text),
            "max_chars": max_chars,
            "max_length": max_length
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="XXX", port=5000, debug=True)
