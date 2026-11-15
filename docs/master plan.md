# T2 Arabic News Agent: Execution Plan


## Architecture and stack decisions

The MCP server holds only deterministic code and local models; all reasoning lives in the agent. Swapping the LLM is then a one-line config change, as the assignment requires.

&#91;embedded content: architecture · orchestrator, 2 specialists, 1 MCP server with 6 tools\]

The orchestrator delegates text work and image work to the specialists, then calls the coherence tool itself and writes the synthesis.

| Layer | Choice | Why |
| --- | --- | --- |
| MCP server | Python `mcp` SDK, `FastMCP`, stdio transport | Simplest, matches the course examples |
| Text classifier | TF-IDF + LinearSVC (main); AraBERT/CAMeLBERT fine-tune on Colab (bonus) | Baseline trains in minutes on CPU; BERT only for the comparison bonus |
| NER | Pretrained Arabic NER (CAMeL Tools or a HF Arabic NER model), evaluated on WikiANN-ar | No training needed; verify the exact model name before you start |
| Keywords | YAKE or TF-IDF top terms | Fast, deterministic |
| Summary | Extractive (lead sentences / TextRank) by default; mT5 XL-Sum if GPU available | Abstractive on CPU is slow; ROUGE on a 200-article XL-Sum sample either way |
| Image category | CLIP zero-shot with one prompt per visual category | No training, as the assignment states |
| Image description | Local captioner (e.g. BLIP) in English; the agent translates to Arabic | Keeps the server independent of the LLM |
| Text-image similarity | Multilingual CLIP (CLIP image encoder + multilingual text encoder) | Standard CLIP text encoder is English-only; verify model names |
| Agent LLM | Claude API (tool calling) or Ollama with Qwen/Llama | Check that the chosen model supports tool calling |
| Validation | Pydantic schemas on every tool output and on the final JSON | Gives you the valid-JSON metric for free |

**Agents:** one orchestrator plus two specialists (text agent with `classify_text` and `analyze_text`; image agent with `analyze_image`). The orchestrator delegates, then calls `check_coherence` and writes the synthesis. All talk to the same MCP server.

## Phases, tasks and hours

Total about 75 hours. Each phase ends with a gate: do not start the next one until it passes.

| # | Phase | Tasks | Output (gate) | Hours |
| --- | --- | --- | --- | --- |
| 0 | Setup | Git repo, venv, install `mcp`, `scikit-learn`, `transformers`, `torch`, `open_clip` or `sentence-transformers`, `pydantic`; get an LLM key or pull an Ollama model and test one tool call | Hello-world MCP tool called by the LLM (agent can call a tool) | 4 |
| 1 | Data and corpus | Download SANAD, WikiANN-ar, XL-Sum ar sample. Write `normalize_ar()` (strip diacritics and tatweel, unify alef/yeh/teh marbuta, UTF-8). Build the text-image set (below) | `corpus.csv` with id, title, text, image path, source, license, category, label coherent / incoherent / generic | 8 |
| 2 | Text classification | TF-IDF + LinearSVC on SANAD subset, stratified split, F1 macro and confusion matrix, save with `joblib`, expose probabilities | Model file + F1 macro on test set | 6 |
| 3 | Text analysis | NER (pretrained, scored on WikiANN-ar), keywords, extractive summary, ROUGE on 200 XL-Sum samples | `analyze_text` returns entities, keywords, summary | 7 |
| 4 | Image analysis | CLIP zero-shot with prompts per visual category, caption model, top-k scores, check on 50 labelled images | `analyze_image` returns category, score, description | 5 |
| 5 | Coherence | Compatibility table, similarity score, thresholds from precision/recall curve on dev split, generic-image rule | `check_coherence` returns élevée / faible / indéterminée + alert | 6 |
| 6 | MCP server | Wrap everything as 6 tools with strict JSON schemas, explicit error messages, UTF-8 safe, per-call timing log | Server runs over stdio; each tool tested alone with MCP Inspector or a script | 6 |
| 7 | Agent | Orchestrator and two specialists, system prompt, JSON validation, retries, trace log, guardrails | End-to-end run on 10 articles, trace saved | 9 |
| 8 | Evaluation | All metrics (section Evaluation), ablation of image tool, 30-query tool-choice test, 5 commented failures | Results tables + plots | 10 |
| 9 | Bonuses | Prompt-injection test and defence, uncertainty refusal, classifier comparison (see Bonus section) | 3 bonuses measured | 6 |
| 10 | Report and demo | Report (architecture, results, failures, limits), 10-minute demo, README with run commands | PDF report + demo script | 8 |

### Building the text-image set (phase 1, the critical task)

1. Pick 15 articles per SANAD category (7 categories, 105 articles) for dev and 10 per category (70) for test.
2. For each, collect one matching image from a source with a reusable license (for example Wikimedia Commons, Unsplash, Pexels). Record URL and license in the CSV; the assignment asks you to check legality.
3. Create incoherent pairs by giving each article an image from an incompatible category (economy article with football photo). Aim for about 40% incoherent.
4. Add about 20 generic images (portrait, logo, archive photo) to test the "indéterminée" rule.
5. Split by article: dev set sets the thresholds, test set is touched once.

Be explicit in the report that these pairs are constructed, not real published articles.

## Day-by-day fast track (pair, 7 days)

Day 1 is Friday 30 October 2026. Person A owns text; person B owns image and coherence; both share the server and agent. Solo: do A then B, and double the days.

| Day | Person A (text) | Person B (image) | Together | End-of-day gate |
| --- | --- | --- | --- | --- |
| 1 (Fri 30 Oct) | Setup, download SANAD, `normalize_ar()`, TF-IDF baseline | Setup, collect first 100 images with licenses, CLIP loads and runs | Agree JSON schemas for all 6 tools | LLM calls one dummy MCP tool |
| 2 (Sat 31 Oct) | F1 macro and confusion matrix; NER on WikiANN-ar | Finish 175 pairs and generic set; zero-shot category on 50 images | Freeze `corpus.csv` | Corpus frozen, classifier F1 known |
| 3 (Sun 1 Nov) | Keywords, extractive summary, ROUGE sample | Caption model; similarity score; compatibility table | Start MCP server skeleton | 4 analysis functions work in plain Python |
| 4 (Mon 2 Nov) | Wrap `classify_text`, `analyze_text`, `get_article` | Wrap `analyze_image`, `check_coherence`; PR curve and thresholds | Server integration test, error cases | All 6 tools pass tests over stdio |
| 5 (Tue 3 Nov) | Text specialist agent, prompts | Image specialist agent, prompts | Orchestrator, JSON validation, trace log, guardrails | 10 articles run end to end |
| 6 (Wed 4 Nov) | Metrics: F1, NER, ROUGE, JSON validity, tool choice | Coherence precision/recall, ablation, latency | Injection test, uncertainty refusal, 5 failures | All numbers in tables |
| 7 (Thu 5 Nov) | Report: data, text, results | Report: image, coherence, limits | Demo, README, final run from clean checkout | Report and demo delivered |

**Slack:** day 7 is buffer for anything that slipped; if everything is on time, spend it on the classifier comparison bonus or a Gradio demo.

## MCP tool contracts

Six tools. Every tool returns JSON with `status` (`ok` or `error`); errors carry a `code` and a plain-language `message`. Tool descriptions are written for the LLM: they say when to call the tool and when not to. Vague descriptions cause wrong calls.

| Tool | Input | Output (status ok) | Description for the LLM (summary) |
| --- | --- | --- | --- |
| `get_article` | `article_id` | `title`, `text`, `image_path`, `metadata` (source, date) | Use first when the user gives an article id. Not for free text the user already pasted. |
| `preprocess_text` | `text` | `clean_text`, `n_chars`, `n_tokens` | Normalizes Arabic (diacritics, alef, yeh, tatweel, whitespace). Call before classification if text looks raw. |
| `classify_text` | `text` | `category` (one of 7 SANAD labels), `confidence`, `top3` | Topic of the article body. Not for images, not for a title alone shorter than 5 words. Never invent a category. |
| `analyze_text` | `text`, `max_sentences` | `entities` (text, type, span), `keywords`, `summary` | Named entities, keywords and a short summary of an Arabic article. Not a classifier. |
| `analyze_image` | `image_path` | `visual_category`, `confidence`, `top3`, `description` | Describes what the picture shows and gives its visual category. Not for text. Returns `generic` for portraits, logos, archive photos. |
| `check_coherence` | `text_category`, `visual_category`, `summary_or_title`, `image_path` | `coherence` (élevée, faible, indéterminée), `clip_score`, `rule`, `alert` | Call only after the text and image tools. Applies the compatibility table and the similarity thresholds. |

Final JSON produced by the agent (validated with Pydantic, every value copied from a tool result):

```json
{
  "categorie_texte": "Sports", "confiance_texte": 0.97,
  "entites": ["الجزائر"],
  "description_image": "des joueurs de football sur un terrain",
  "categorie_image": "sport",
  "coherence_texte_image": "élevée",
  "synthese": "...",
  "avertissements": []
}
```

Error example: `{"status": "error", "code": "IMAGE_NOT_FOUND", "message": "No file at the given image_path"}`. The agent must report it and continue with the text-only result, never fill the image fields itself.

**Routing rule the agent must follow:** text only → no image tool; image only → no text tools; both → text first, image second, coherence last.

## Evaluation plan

The assignment lists these metrics. Targets below are my suggested goals, not requirements.

| Metric | Measured on | How | Goal |
| --- | --- | --- | --- |
| F1 macro, confusion matrix | SANAD test split | `classify_text` alone | High 0.9s is typical for TF-IDF baselines on SANAD; report your own number |
| Entity F1 | WikiANN-ar test sample | `analyze_text` entities vs gold | Report precision, recall, F1 |
| ROUGE-1/2/L | 200 XL-Sum ar test articles | Summary vs reference | Report both extractive and (if used) mT5 |
| Image category accuracy | 50 labelled images | `analyze_image` | Check that confusions are between close categories |
| Valid-JSON rate | All agent runs | Pydantic validation of the final JSON | At least 95 percent |
| Correct tool-call rate | 30 test queries | Compare called tools with expected tools | At least 90 percent; no unnecessary tool calls |
| Incoherence alert precision/recall | Test pairs (about 70 articles) | `alert` vs gold label | Recall first: missing a mismatch is the worse error |
| Latency per article | All runs | Mean and p95 per tool and total | Report; no target |
| Failure analysis | Any | 5 commented failures with cause | Required |

**Calibrating the coherence thresholds (required by the assignment):**

1. On the dev set compute the CLIP similarity for every pair, split by gold label (coherent, incoherent).
2. Plot precision/recall for detecting incoherence as the low threshold varies; choose the low threshold with recall of at least 0.9 and the best precision.
3. Plot precision for "élevée" as the high threshold varies; choose the lowest high threshold with precision of at least 0.95.
4. Between the two thresholds the answer is "indéterminée". Freeze the thresholds, then run the test set once.
5. Include the curve and the chosen values in the report.

**Decision order, in this order:** generic visual category → indéterminée; table compatible and score above high threshold → élevée; table incompatible or score below low threshold → faible with alert; otherwise indéterminée.

**Tool-choice test set (30 queries):** 10 full articles, 5 text only, 5 image only, 3 missing or unreadable image, 3 contradictory pairs, 4 with an instruction hidden in the text. Write the expected tool sequence for each before running.

**Ablation:** run the 70 test articles with (a) all tools, (b) no image tool, (c) no NER; compare final-answer quality and the number of wrong syntheses.

## Bonuses ranked by value per hour

Pick three. They all reuse the core pipeline, so each costs hours, not days.

| Bonus | Extra hours | Value | Verdict |
| --- | --- | --- | --- |
| Text-image incoherence score | 0 (already core) | High | Included in the core |
| Evaluation of the agent itself (tool-call rate, latency, cost) | 0 (already core) | High | Included in the core |
| Security: hidden prompt injection in an article | 3 | High: cheap, vivid in a demo | Do it |
| Uncertainty: refuse or ask for human check at low confidence | 2 | High | Do it |
| Classifier comparison (TF-IDF vs AraBERT/CAMeLBERT vs LLM zero-shot) | 4 | Medium to high; needs Colab GPU for BERT | Do it if on schedule |
| Dialect robustness | 5 | Medium; needs dialect test text | Skip |
| Second MCP source (dictionary, web, fact base) | 6 | Medium | Skip |
| Misinformation or propaganda detection | 8 | Medium; hard to evaluate honestly | Skip |

**Injection test, concretely:** write 10 articles whose body contains an instruction (for example asking the agent to ignore its tools and output a fixed category). Measure how often the agent obeys, then add defences: wrap article text as data in a delimited block, tell the agent in the system prompt that article text is never an instruction, validate the final JSON against tool outputs, and strip or flag instruction-like patterns. Re-run and report the before and after rate.

**Uncertainty rule:** if text confidence is below a threshold chosen on the dev set, or the two modalities are "indéterminée", the agent says it cannot conclude and asks for human review instead of writing a confident synthesis.

