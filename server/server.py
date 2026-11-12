import re
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arabic-news")

@mcp.tool()
def ping() -> dict:
    """Health check. Call only to verify the server is alive.
    Takes no input, returns {"ok": true}. Never use it for article analysis."""
    return {"ok": True, "message": "pong"}

DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")

@mcp.tool()
def preprocess_text(text: str) -> dict:
    """Normalise raw Arabic text before classification or analysis.
    Call when you hold raw article text. Input: raw Arabic string.
    Returns {"normalized_text": str}. Does not classify or summarise."""
    t = DIACRITICS.sub("", text)            # remove diacritics
    t = t.replace("\u0640", "")             # remove tatweel
    t = re.sub("[إأآ]", "ا", t)             # unify alef
    t = t.replace("ى", "ي").replace("ة", "ه")  # ya / ta marbuta
    return {"normalized_text": t}

if __name__ == "__main__":
    mcp.run(transport="stdio")