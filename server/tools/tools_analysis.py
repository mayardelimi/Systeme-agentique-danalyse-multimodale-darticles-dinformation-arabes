"""analyze_text: keywords, named entities and extractive summary.

Deterministic code and local models only: no LLM inside the server.
"""
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from tools.tools_text import _error, normalize_arabic

IDF_PATH = Path(os.environ.get(
    "KEYWORD_IDF_PATH",
    Path(__file__).resolve().parent.parent / "models" / "keyword_idf.joblib"))

MIN_WORDS = 20            # below this, a summary/keywords are meaningless
MIN_ARABIC_RATIO = 0.5
MAX_SENTENCES = 60        # cap for TextRank (latency)
MAX_NER_TOKENS = 1500     # cap for NER (latency)
NER_CHUNK = 100           # tokens per NER call (BERT max length safety)

_AR_WORD = re.compile(r"[ء-ي]+")  # letters only: diacritics excluded

_RAW_STOP = """
في من على الى إلى عن أن ان إن هذا هذه ذلك تلك التي الذي الذين اللذان كان كانت
يكون تكون قد لا ما لم لن هو هي هم هن كما أو او ثم بين بعد قبل عند حتى كل بعض
مع أي اي إذا اذا لكن أيضا ايضا حيث قال قالت خلال أكثر اكثر غير تم يتم هناك
نحو منذ فقد به بها له لها فيه فيها عليه عليها منه منها أنه انه أنها انها إلى
هذان كلا أما اما إلا الا ولا وقد وفي ومن وعلى كذلك لذلك بعد اليوم أمس
بسبب عبر ضمن لدى لدي دون وفق وفقا نفس
""".split()

_STOP = {normalize_arabic(w) for w in _RAW_STOP}

_idf = None      
_ner = None
_ner_failed = False


@lru_cache(maxsize=50000)
def _norm_word(w: str) -> str:
    return normalize_arabic(w)


def _is_stop(n: str) -> bool:
    return n in _STOP or (len(n) > 3 and n.startswith("و") and n[1:] in _STOP)


def _validate(text):
    if not isinstance(text, str) or not text.strip():
        return _error("invalid_input", "text is empty; pass the article body (fetch it with get_article).")
    letters = re.findall(r"[^\W\d_]", text)
    arabic = re.findall(r"[؀-ۿ]", text)
    if not letters or len(arabic) / len(letters) < MIN_ARABIC_RATIO:
        return _error("not_arabic", "text is not mainly Arabic; this tool only supports Arabic news text.")
    if len(text.split()) < MIN_WORDS:
        return _error("text_too_short",
                      f"text has fewer than {MIN_WORDS} words; pass the article body, not a title.")
    return None


def _load_idf():
    global _idf
    if _idf is None:
        try:
            import joblib
            d = joblib.load(IDF_PATH)
            _idf = (d["idf"], float(d["default_idf"]))
        except Exception:
            _idf = ({}, 1.0)   
    return _idf


def _sentences(text: str):
    parts = re.split(r"(?<=[.!?؟])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]



def _keywords(text: str, k: int):
    counts, surface = Counter(), {}
    for w in _AR_WORD.findall(text):
        w = w.replace("ـ", "")
        n = _norm_word(w)
        if len(n) < 3 or _is_stop(n):
            continue
        counts[n] += 1
        surface.setdefault(n, Counter())[w] += 1
    idf, default = _load_idf()
    scored = sorted(((n, (1 + math.log(c)) * idf.get(n, default)) for n, c in counts.items()),
                    key=lambda x: -x[1])[:k]
    return [{"term": surface[n].most_common(1)[0][0], "score": round(s, 3)} for n, s in scored]



def _summarize(text: str, n: int) -> str:
    sents = _sentences(text)[:MAX_SENTENCES]
    sents = [s for s in sents if len(s.split()) >= 4] or sents
    if len(sents) <= n:
        return " ".join(sents)
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(w for w in normalize_arabic(s).split() if not _is_stop(w)) for s in sents]
    try:
        X = TfidfVectorizer().fit_transform(docs)
    except ValueError:
        return " ".join(sents[:n])
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0)
    rows = sim.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    M = sim / rows
    N = len(sents)
    r = np.full(N, 1.0 / N)
    prior = 1.0 / np.sqrt(np.arange(1, N + 1))  # news puts key facts early
    prior /= prior.sum()
    for _ in range(50):                       # PageRank power iteration
        nr = 0.15 * prior + 0.85 * (M.T @ r)
        done = np.abs(nr - r).sum() < 1e-6
        r = nr
        if done:
            break
    top = sorted(np.argsort(-r)[:n])           # keep original order
    return " ".join(sents[i] for i in top)



def _entities(text: str):
    """Returns a list, or None when the NER model is unavailable."""
    global _ner, _ner_failed
    if _ner_failed:
        return None
    try:
        if _ner is None:
            from camel_tools.ner import NERecognizer
            _ner = NERecognizer.pretrained()
        from camel_tools.tokenizers.word import simple_word_tokenize
    except Exception:
        _ner_failed = True
        return None

    spans, budget = [], MAX_NER_TOKENS
    for sent in _sentences(text):
        tokens = simple_word_tokenize(sent)
        for i in range(0, len(tokens), NER_CHUNK):
            if budget <= 0:
                break
            chunk = tokens[i:i + NER_CHUNK]
            budget -= len(chunk)
            labels = _ner.predict_sentence(chunk)
            cur = None
            for tok, lab in zip(chunk, labels):
                if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur[1] != lab[2:])):
                    if cur:
                        spans.append(cur)
                    cur = ([tok], lab[2:])
                elif lab.startswith("I-"):
                    cur[0].append(tok)
                else:
                    if cur:
                        spans.append(cur)
                    cur = None
            if cur:
                spans.append(cur)

    counts, order = Counter(), []
    for toks, typ in spans:
        key = (" ".join(toks), typ)
        if key not in counts:
            order.append(key)
        counts[key] += 1
    return [{"text": t, "type": ty, "count": counts[(t, ty)]} for t, ty in order]



def analyze(text: str, max_keywords: int = 8, summary_sentences: int = 2) -> dict:
    err = _validate(text)
    if err:
        return err
    max_keywords = max(1, min(int(max_keywords), 20))
    summary_sentences = max(1, min(int(summary_sentences), 5))

    warnings = []
    if not _load_idf()[0]:
        warnings.append("idf_unavailable: keywords ranked by term frequency only")
    entities = _entities(text)
    if entities is None:
        warnings.append("ner_unavailable: entities could not be computed; do not report entities")

    return {
        "keywords": _keywords(text, max_keywords),
        "entities": entities,          # None means "not computed", [] means "none found"
        "summary": _summarize(text, summary_sentences),
        "warnings": warnings,
        "methods": {"keywords": "tfidf", "entities": "camel_tools_ner", "summary": "textrank_extractive"},
    }


def register(mcp):
    @mcp.tool()
    def analyze_text(text: str, max_keywords: int = 8, summary_sentences: int = 2) -> dict:
        """Extract keywords, named entities and an extractive summary from an Arabic
        article body. entities=null means NER was not computed (do not report any
        entities); entities=[] means none were found. Always mention any warnings."""        
        return analyze(text, max_keywords, summary_sentences)