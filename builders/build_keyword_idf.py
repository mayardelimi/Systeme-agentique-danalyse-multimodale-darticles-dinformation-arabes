import csv
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools_text import normalize_arabic  # same normalisation as inference


def load_texts(src: Path, split: str = "train") -> list[str]:
    if src.suffix == ".json":
        raw = json.loads(src.read_text(encoding="utf-8"))
        arts = raw["articles"] if isinstance(raw, dict) else raw  # {"meta","articles"} or bare list
        return [a["text_raw"] for a in arts if a.get("split", "train") == split]
    if src.suffix == ".csv":
        csv.field_size_limit(10**9)
        with open(src, encoding="utf-8", newline="") as f:
            return [r["text"] for r in csv.DictReader(f)]
    return src.read_text(encoding="utf-8").splitlines()  # plain text, one article per line


src = Path("data/corpus.json" if len(sys.argv) < 2 else sys.argv[1]).resolve()
texts = load_texts(src)

docs = [normalize_arabic(t) for t in texts if t and t.strip()]
vec = TfidfVectorizer(token_pattern=r"[ء-ي]{3,}", min_df=3, sublinear_tf=True)
vec.fit(docs)

out = Path(__file__).resolve().parent / "models" / "keyword_idf.joblib"
out.parent.mkdir(exist_ok=True)
joblib.dump({"idf": dict(zip(vec.get_feature_names_out(), map(float, vec.idf_))),
             "default_idf": float(vec.idf_.max())}, out)
print(f"{len(docs)} docs, {len(vec.vocabulary_)} terms -> {out}")