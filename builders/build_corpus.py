"""Build data/corpus.json and data/splits/*.csv from SANAD.

Usage
    python build_corpus.py                      # downloads via kagglehub
    python build_corpus.py --sanad /path/to/sanad_dir   # use a local copy
    python build_corpus.py --pairs data/pair_ids.txt    # reserve article-image pair articles

Layout detection: every .txt file under the root is an article. Its label comes from
any folder name matching a known category, its source from any folder name matching
khaleej / arabiya / akhbarona. If your copy is CSV instead, see load_csv().
"""
import argparse, csv, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
TEST_SIZE = 1000          # held-out, balanced across the 7 classes
VAL_FRACTION = 0.1        # of what remains
MIN_CHARS = 200

LABELS = {
    "culture": "ثقافة", "finance": "اقتصاد", "medical": "طب", "politics": "سياسة",
    "religion": "دين", "sports": "رياضة", "tech": "تكنولوجيا",
}

# folder-name variants -> label id (compared lowercase)
FOLDER_TO_LABEL = {
    "culture": "culture",
    "finance": "finance", "economy": "finance",
    "medical": "medical", "medicine": "medical", "health": "medical",
    "politics": "politics",
    "religion": "religion",
    "sports": "sports", "sport": "sports",
    "tech": "tech", "technology": "tech",
}
SOURCES = ("khaleej", "arabiya", "akhbarona")


def preprocess_text(text: str) -> str:
    """Replace with the shared function from server/tools so training and inference match."""
    try:
        from camel_tools.utils.dediac import dediac_ar
        from camel_tools.utils.normalize import (
            normalize_alef_ar, normalize_alef_maksura_ar, normalize_teh_marbuta_ar)
        text = dediac_ar(text)
        text = normalize_alef_ar(text)
        text = normalize_alef_maksura_ar(text)
        text = normalize_teh_marbuta_ar(text)
    except ImportError:  # regex fallback, same rules
        text = re.sub(r"[ً-ْٰ]", "", text)
        text = re.sub(r"[إأآٱ]", "ا", text)
        text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ـ", "")  # tatweel
    return re.sub(r"\s+", " ", text).strip()


def label_and_source(path: Path, root: Path):
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    label = next((FOLDER_TO_LABEL[p] for p in parts if p in FOLDER_TO_LABEL), None)
    source = next((p for p in parts if p in SOURCES), "unknown")
    return label, source


def load_txt(root: Path):
    rows = []
    for f in sorted(root.rglob("*.txt")):
        label, source = label_and_source(f, root)
        if label is None:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        rows.append({"source": source, "label": label, "text_raw": text,
                     "title": None, "date": None})
    return rows


def load_csv(root: Path):
    """Fallback for CSV copies: needs a text column and a category column."""
    rows = []
    text_cols = ("text", "body", "article", "content")
    label_cols = ("label", "category", "class")
    for f in sorted(root.rglob("*.csv")):
        with open(f, newline="", encoding="utf-8", errors="ignore") as fh:
            rd = csv.DictReader(fh)
            cols = {c.lower(): c for c in (rd.fieldnames or [])}
            tc = next((cols[c] for c in text_cols if c in cols), None)
            lc = next((cols[c] for c in label_cols if c in cols), None)
            if not tc or not lc:
                continue
            for r in rd:
                label = FOLDER_TO_LABEL.get((r[lc] or "").strip().lower())
                if label:
                    rows.append({"source": "unknown", "label": label,
                                 "text_raw": (r[tc] or "").strip(),
                                 "title": r.get("title"), "date": r.get("date")})
    return rows


def split_ids(clean, reserved, seed=SEED):
    """Balanced test set, stratified val, rest train. Returns dict id -> split."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in clean:
        if r["id"] not in reserved:
            by_label[r["label"]].append(r["id"])
    for ids in by_label.values():
        ids.sort(); rng.shuffle(ids)

    n = len(LABELS)
    per_class = [TEST_SIZE // n + (1 if i < TEST_SIZE % n else 0) for i in range(n)]
    assign = {}
    for lab, k in zip(sorted(LABELS), per_class):
        ids = by_label[lab]
        for i in ids[:k]:
            assign[i] = "test"
        rest = ids[k:]
        n_val = int(len(rest) * VAL_FRACTION)
        for i in rest[:n_val]:
            assign[i] = "val"
        for i in rest[n_val:]:
            assign[i] = "train"
    for i in reserved:
        assign[i] = "pair_corpus"
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanad", help="local SANAD folder; omit to download with kagglehub")
    ap.add_argument("--out", default="data")
    ap.add_argument("--pairs", help="text file, one article id per line, for the image-pair corpus")
    a = ap.parse_args()

    if a.sanad:
        root = Path(a.sanad)
    else:
        import kagglehub
        root = Path(kagglehub.dataset_download("haithemhermessi/sanad-dataset"))
    print("SANAD root:", root)

    rows = load_txt(root) or load_csv(root)
    if not rows:
        tree = sorted({str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir()})[:30]
        raise SystemExit(f"No articles found. Folders seen: {tree}\n"
                         "Adjust FOLDER_TO_LABEL or load_txt/load_csv to your layout.")

    # clean: drop short/empty and exact duplicates (on normalised text)
    seen, clean, dropped = set(), [], Counter()
    for r in rows:
        norm = preprocess_text(r["text_raw"])
        if len(norm) < MIN_CHARS:
            dropped["short"] += 1; continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen:
            dropped["duplicate"] += 1; continue
        seen.add(h)
        r["text_norm"] = norm
        clean.append(r)
    clean.sort(key=lambda r: (r["label"], r["source"], r["text_norm"][:80]))  # deterministic ids
    for i, r in enumerate(clean):
        r["id"] = f"sanad_{i:06d}"
        r.update(image_path=None, visual_category=None, coherent=None)
    print(f"{len(rows)} loaded, {len(clean)} kept, dropped {dict(dropped)}")
    print("class counts:", dict(Counter(r["label"] for r in clean)))

    reserved = set()
    if a.pairs:
        reserved = {l.strip() for l in Path(a.pairs).read_text().splitlines() if l.strip()}
        print(f"{len(reserved)} ids reserved for pair_corpus")

    assign = split_ids(clean, reserved)
    for r in clean:
        r["split"] = assign.get(r["id"], "train")

    out = Path(a.out); (out / "splits").mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test", "pair_corpus"):
        ids = [r["id"] for r in clean if r["split"] == name]
        if not ids and name == "pair_corpus":
            continue
        with open(out / "splits" / f"{name}_ids.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["id"]); w.writerows([[i] for i in ids])

    corpus = {
        "meta": {
            "name": "sanad_corpus", "version": "0.1", "seed": SEED, "encoding": "UTF-8",
            "normalisation": "preprocess_text(): dediac, tatweel, alef/ya/ta marbuta unified",
            "labels": LABELS,
            "splits": {k: sum(r["split"] == k for r in clean)
                       for k in ("train", "val", "test", "pair_corpus")},
            "fields": {
                "id": "stable unique ID", "title": "title or null", "text_raw": "original article",
                "text_norm": "preprocess_text(text_raw)", "label": "one of the 7 label IDs",
                "source": "khaleej | arabiya | akhbarona | unknown", "date": "ISO 8601 or null",
                "split": "train | val | test | pair_corpus",
                "image_path": "relative path under data/images/ or null",
                "visual_category": "hand label or null", "coherent": "hand label or null",
            },
        },
        "articles": clean,
    }
    (out / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print("splits:", corpus["meta"]["splits"], "->", out / "corpus.json")


if __name__ == "__main__":
    main()