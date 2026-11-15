# Systeme-agentique-d-analyse-multimodale-d-articles-d-information-arabes

An end-to-end **Model Context Protocol (MCP)** server and LLM agent system engineered for retrieving, preprocessing, classifying, and analyzing Arabic news articles. 

The architecture separates deterministic NLP/ML execution (inside the MCP server) from high-level reasoning and tool orchestration (handled by an LLM client via `litellm`).

---

##  Repository Architecture

```text
├── agent/                  # LLM Client & Orchestration Layer
│   ├── agent.py            # Main agent logic
│   ├── client.py           # Stdio MCP client & LiteLLM tool-calling loop
│   ├── config.yaml         # Model selection & step limits
│   ├── llm.py              # LLM wrapper utilities
│   ├── schema.py           # Structured output/tool schemas
│   └── tracing.py          # Execution trace utilities
├── builders/               # Data & Feature Engineering Scripts
│   ├── build_corpus.py     # Builds data/corpus.json with train/val/test splits
│   └── build_keyword_idf.py# Computes corpus-wide IDF weights for keyword scoring
├── data/                   # Dataset Storage
│   └── corpus.json         # Indexed Arabic news articles & metadata
├── docs/                   # Architectural Documentation
│   └── master plan.md      # System design & roadmap
├── eval/                   # Evaluation & Telemetry
│   ├── classification mo.../ # Confusion matrix, reliability & training plots
├── notebooks/              # Model Training & Experimentation
│   └── train_tfidf_svc.ipynb # Baseline TF-IDF + Calibrated LinearSVC training
├── server/                 # Deterministic MCP Server
│   ├── models/             # Serialized Scikit-Learn/Joblib Artifacts
│   │   ├── keyword_idf.joblib # Precomputed IDF dictionary for TF-IDF keywords
│   │   └── tfidf_svc.joblib   # Calibrated LinearSVC text classifier
│   ├── tools/              # Modular MCP Tool Implementations
│   │   ├── tools_analysis.py  # Keywords, NER (CAMeL Tools), TextRank summary
│   │   ├── tools_image.py     # Multimodal/image tool hooks
│   │   └── tools_text.py      # Arabic normalization & topic classification
│   ├── server.py           # FastMCP server entry point (stdio transport)
│   └── stubs.py            # Fallback/mock tool stubs
├── .env                    # Environment variables & API keys
└── requirements.txt        # Python dependencies
```

---

##  1. Setup & Installation

### 1.1 Virtual Environment & Dependencies

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
   * **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```
   * **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate.bat
     ```

3. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

### 1.2 LLM Access Configuration

The agent uses `litellm` and reads its model configuration from `agent/config.yaml` (default: `gemini/gemini-2.5-flash`, max 8 steps).

1. **Configure `.env` in the project root:**
   ```env
   API_KEY=your_api_key_here    # model: "gemini/gemini-2.5-flash"
   GEMINI_API_KEY=your_api_key_here
   ```
   *(Tip: `litellm` automatically looks for `GEMINI_API_KEY` when using `gemini/...` models, so setting both `API_KEY` and `GEMINI_API_KEY` ensures compatibility across scripts.)*

2. **Export the key in your terminal session:**
   * **Linux / macOS:**
     ```bash
     export API_KEY="your_api_key_here"
     export GEMINI_API_KEY="your_api_key_here"
     ```
   * **Windows (PowerShell):**
     ```powershell
     $env:API_KEY="your_api_key_here"
     $env:GEMINI_API_KEY="your_api_key_here"
     ```

---

##  2. Verification & Running the System

### 2.1 Step 1: Test the MCP Server Standalone
Before connecting an LLM, verify that the MCP server and local ML models load properly using the official MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python server/server.py
```

Once the Inspector UI opens in your browser, connect to the server and test the tools in sequence:
1. **`ping`** — Verifies server health (`{"ok": true, "message": "pong"}`).
2. **`get_article`** — Pass a valid `article_id` (e.g., `"sanad_000006"`) to retrieve raw text and metadata from `data/corpus.json`.
3. **`preprocess_text`** — Pass raw Arabic text to verify diacritic removal and character normalization.
4. **`classify_text`** — Pass article body text to verify topic prediction and calibrated confidence scores.
5. **`analyze_text`** — Pass article body text to extract TF-IDF keywords, CAMeL Tools named entities, and a TextRank extractive summary.

### 2.2 Step 2: Run the LLM Agent
Ensure your working directory is the **project root** and pass your prompt to `agent/client.py`:

```bash
python agent/client.py "Fetch article sanad_000006, classify its category, and summarize its key entities."
```

Tool call arguments and execution latencies (in milliseconds) are automatically appended to `eval/trace.jsonl`.

---

##  3. MCP Server Tool Reference

All tools in `server/` are strictly deterministic and run locally without external LLM calls:

| Tool Name | Source Module | Description & Behavior |
| :--- | :--- | :--- |
| `ping` | `server/server.py` | Health check returning `{"ok": true, "message": "pong"}`. |
| `get_article` | `server/server.py` | Looks up an article by `article_id` in `data/corpus.json`, returning `id`, `title`, `text`, `image_path`, `source`, and `date`. |
| `preprocess_text` | `server/tools/tools_text.py` | Normalizes Arabic text using `camel_tools` (or regex fallback): strips diacritics (`tashkeel`) and `tatweel`, normalizes Alef variants (`إ أ آ ٱ` $\rightarrow$ `ا`), Alef Maksura (`ى` $\rightarrow$ `ي`), and Teh Marbuta (`ة` $\rightarrow$ `ه`). |
| `classify_text` | `server/tools/tools_text.py` | Validates Arabic character ratio ($\ge 50\%$) and word count ($\ge 5$), normalizes text, and predicts one of 7 categories (`culture`, `finance`, `medical`, `politics`, `religion`, `sports`, `tech`) with `top3` probabilities and a `low_confidence` flag ($< 0.5$). |
| `analyze_text` | `server/tools/tools_analysis.py` | Validates Arabic text ($\ge 20$ words) and returns:<br>• **Keywords**: Sublinear TF $\times$ precomputed IDF scores (`keyword_idf.joblib`).<br>• **Entities**: Chunked BIO Named Entity Recognition via `camel_tools.ner`.<br>• **Summary**: Position-biased TextRank extractive summary (PageRank over sentence TF-IDF cosine similarity). |

---

##  4. Data & Machine Learning Pipeline

If you need to rebuild the dataset or retrain the local models from scratch:

1. **Build Corpus & IDF Weights (`builders/`):**
   * Run `python builders/build_corpus.py` to generate `data/corpus.json` containing normalized text and stratified `train` (39,676), `val` (4,404), and `test` (1,000) splits.
   * Run `python builders/build_keyword_idf.py` to compute corpus-wide inverse document frequencies and save them to `server/models/keyword_idf.joblib`.

2. **Train & Evaluate Classifier (`notebooks/train_tfidf_svc.ipynb`):**
   * **Features:** `FeatureUnion` combining word $n$-grams $(1, 2)$ (`max_features=300,000`) and word-boundary character $n$-grams $(2, 5)$ (`char_wb`, `max_features=400,000`) to capture Arabic morphological prefixes/suffixes.
   * **Classifier:** `LinearSVC(C=0.5)` wrapped in `CalibratedClassifierCV(method="sigmoid", cv=5)` so the agent receives well-calibrated probability estimates (`ECE = 0.0102`, `macro-F1 = 0.9836` on validation).
   * **Artifact:** Exports `{"model": model, "labels": LABELS}` to `server/models/tfidf_svc.joblib`.