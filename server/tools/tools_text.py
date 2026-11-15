import os
import re
from pathlib import Path

import joblib

MODEL_PATH = Path(os.environ.get(
    "CLASSIFIER_PATH",
    Path(__file__).resolve().parent.parent / "models" / "tfidf_svc.joblib"))

MIN_WORDS = 5
MIN_ARABIC_RATIO = 0.5
LOW_CONFIDENCE = 0.5  

_bundle = None


def _error(code: str, message: str, retryable: bool = False) -> dict:
    return {"error": code, "message": message, "retryable": retryable}


def normalize_arabic(text: str) -> str:
    """THE shared normalisation: training and inference must both call this."""
    try:
        from camel_tools.utils.dediac import dediac_ar
        from camel_tools.utils.normalize import (
            normalize_alef_ar, normalize_alef_maksura_ar, normalize_teh_marbuta_ar)
        text = dediac_ar(text)
        text = normalize_alef_ar(text)
        text = normalize_alef_maksura_ar(text)
        text = normalize_teh_marbuta_ar(text)
    except ImportError: 
        text = re.sub(r"[ً-ْٰ]", "", text)
        text = re.sub(r"[إأآٱ]", "ا", text)
        text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ـ", "")  # tatweel
    return re.sub(r"\s+", " ", text).strip()


def classify(text: str) -> dict:
    global _bundle
    if not isinstance(text, str) or not text.strip():
        return _error("invalid_input", "text is empty; pass the article body (fetch it with get_article).")

    letters = re.findall(r"[^\W\d_]", text)
    arabic = re.findall(r"[؀-ۿ]", text)
    if not letters or len(arabic) / len(letters) < MIN_ARABIC_RATIO:
        return _error("not_arabic", "text is not mainly Arabic; this classifier only supports Arabic news text.")

    norm = normalize_arabic(text)
    if len(norm.split()) < MIN_WORDS:
        return _error("text_too_short", f"text has fewer than {MIN_WORDS} words; pass the article body, not a title.")

    try:
        if _bundle is None:
            _bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        return _error("model_unavailable", f"classifier could not be loaded ({type(e).__name__}).")

    model = _bundle["model"]
    proba = model.predict_proba([norm])[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda x: -x[1])
    best_label, best_p = ranked[0]
    return {
        "category": str(best_label),
        "confidence": round(float(best_p), 4),
        "top3": [{"category": str(c), "confidence": round(float(p), 4)} for c, p in ranked[:3]],
        "low_confidence": bool(best_p < LOW_CONFIDENCE),
        "model": "tfidf_linearsvc_calibrated",
    }


def register(mcp):
    @mcp.tool()
    def preprocess_text(text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return _error("invalid_input", "text is empty.")
        return {"normalized_text": normalize_arabic(text)}

    @mcp.tool()
    def classify_text(text: str) -> dict:
        return classify(text)