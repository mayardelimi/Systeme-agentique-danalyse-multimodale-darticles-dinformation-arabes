import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from tools import tools_text, tools_image

mcp = FastMCP("arabic-news")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_index = None


@mcp.tool()
def ping() -> dict:
    """Health check. Call only to verify the server is alive.
    Takes no input, returns {"ok": true}. Never use it for article analysis."""
    return {"ok": True, "message": "pong"}


@mcp.tool()
def get_article(article_id: str) -> dict:
    global _index
    article_id = str(article_id).strip()
    if not article_id:
        return {"error": "invalid_input", "message": "article_id is empty.", "retryable": False}

    if _index is None:
        try:
            with open(DATA_DIR / "corpus.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            records = raw["articles"] if isinstance(raw, dict) else raw
            _index = {str(r["id"]): r for r in records}
        except FileNotFoundError:
            return {"error": "corpus_not_found", "message": f"Missing {DATA_DIR / 'corpus.json'}", "retryable": False}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {"error": "corpus_invalid", "message": str(e), "retryable": False}

    rec = _index.get(article_id)
    if rec is None:
        return {"error": "article_not_found", "message": f"No article with id '{article_id}'.", "retryable": False}

    image = rec.get("image_path") or rec.get("image")
    img = (DATA_DIR / image).resolve() if image else None
    return {
        "id": article_id,
        "title": rec.get("title"),
        "text": rec.get("text_raw") or rec.get("text"),
        "image_path": str(img) if img and img.is_file() else None,
        "source": rec.get("source"),
        "date": rec.get("date"),
    }


tools_text.register(mcp)
tools_image.register(mcp)


if __name__ == "__main__":
    mcp.run(transport="stdio")