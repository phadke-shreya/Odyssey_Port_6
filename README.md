# SmartDoc — ask your PDFs, get cited answers

Ask a question in plain English about a library of PDFs and get an answer that
shows exactly which document, page, and section it came from. If the answer is
not in the documents, it says **"I don't know"** rather than making something up.

Built with FastAPI, LangChain, ChromaDB and Streamlit.

---

## What makes it different from "just chunk it and hope"

| | |
|---|---|
| **Parent/child chunking** | Small chunks are embedded for precise search; the whole surrounding section is what the model reads. Search small, answer big. |
| **Section-aware boundaries** | Parents come from the document's own headings (`5.2 Remote Work Policy`), so citations name a section, not just a page. Falls back safely on documents with no numbering. |
| **Tables are never split** | A table is one chunk, always, because rows are meaningless without their column headers. It also gets a caption so it is findable. |
| **Scanned pages are read, not skipped** | A page with no text layer is OCR'd and becomes searchable, with every citation marked `OCR - may contain errors`. Low-confidence output is discarded rather than indexed -- garbled text is worse than absent text, because it gets quoted as fact. |
| **Hybrid retrieval** | Vector search finds meaning; an exact-identifier pass finds `Table 2-2`, `Policy #37`, `03.05.03`. Embeddings are poor at codes -- this took identifier questions from 62% to 100%. |
| **"I don't know" is structural** | Two guards, both *before* the model is called: a distance threshold, and a check that any entity the question names (GDPR, HIPAA) is actually mentioned in what was retrieved. |
| **Answers cannot lose their sources** | Text and citations travel in one object; no code path returns text alone. |

## Documentation

| File | What it covers |
|---|---|
| [EXPLAINER.md](EXPLAINER.md) | What this does and why each tool was chosen, in plain language |
| [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) | Every file, line by line |
| [plan.md](plan.md) | The design and its tradeoffs, including known failure modes |
| [CLAUDE.md](CLAUDE.md) | Coding conventions and the invariants that must not break |

---

## Requirements

- **Python 3.11 or 3.12.** Not 3.14 — several dependencies have no wheels for it
  yet and the install fails. Check with `python3 --version`.
- An **OpenAI API key** for writing answers. Optional: without one, searching and
  citations still work fully, only the final answer sentence is missing.
- **Tesseract**, optional, for reading scanned pages: `brew install tesseract`
  (macOS) or `apt-get install tesseract-ocr` (Linux). Without it the app still
  runs and such pages are reported as unreadable.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env
cp .env.example .env
```

Then open `.env` and fill it in. **Embedding and chat are separate jobs with
separate credentials**, so you can pay for one and run the other free — or point
either at a company gateway — without the two interfering.

```ini
# --- Embeddings ---
#   openai = hosted, 8191-token window, needs a key
#   local  = ChromaDB's built-in model, no key, but only a ~256-token window
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-proj-...
EMBEDDING_BASE_URL=            # blank = api.openai.com

# --- Chat (writing the answer) ---
CHAT_MODEL=gpt-4o-mini
CHAT_API_KEY=sk-proj-...
CHAT_BASE_URL=                 # blank = api.openai.com
```

**To run entirely free and offline**, install [Ollama](https://ollama.com) and use:

```ini
EMBEDDING_PROVIDER=openai      # "openai" just means OpenAI-compatible
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://localhost:11434/v1

CHAT_MODEL=llama3.2
CHAT_API_KEY=ollama
CHAT_BASE_URL=http://localhost:11434/v1
```

Then `ollama pull nomic-embed-text llama3.2` and `python ingest.py --reset`.
Both configurations score 100% on the retrieval eval; the hosted one writes
better answers, the local one is free and needs no network.

A single shared `OPENAI_API_KEY` / `OPENAI_BASE_URL` pair still works as a
fallback for both jobs.

⚠️ **Changing `EMBEDDING_MODEL` requires `python ingest.py --reset`, and the
relevance threshold must be re-measured** — see "Measuring retrieval quality"
below. This is not optional: running one model at another's threshold made the
app answer "I don't know" to 10 of 23 answerable questions.

`.env` is gitignored. Never commit it.

---

## Load your documents

Put PDFs in `documents/`, then:

```bash
python ingest.py --reset
```

`--reset` wipes the database first. Use it whenever you change
`EMBEDDING_PROVIDER` or `EMBEDDING_MODEL` — vectors made by different models are
not comparable, and the app will refuse to query a mismatched database.

You should see a summary like:

```
Ingest finished in 21.3s
  chunks in database     : 1819
  documents in database  : 4
  embedding model        : local:all-MiniLM-L6-v2
```

---

## Run it

Two terminals, both with the venv activated.

**Terminal 1 — the API:**

```bash
./run_api.sh
```

**Terminal 2 — the UI:**

```bash
./run_ui.sh
```

Then open **http://localhost:8501**.

Type a question, press **Ask**, and expand any source to read the exact text the
answer came from.

---

## API

Interactive docs at http://127.0.0.1:8006/docs

| Endpoint | Does |
|---|---|
| `GET /health` | What is indexed, which embedding model, whether a key is set |
| `POST /ask` | `{"question": "..."}` → answer plus cited sources |
| `POST /upload` | Multipart PDF upload; parses, chunks, and indexes it |

```bash
curl -X POST http://127.0.0.1:8006/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the rules for family employees?"}'
```

---

## Tests

```bash
pytest -q                 # 126 tests
black --check app tests   # formatting
ruff check app tests      # linting
```

The two that matter most: a table must never be split, and every child chunk
must resolve to its parent. Many of the rest are regression tests pinning real
defects found in actual PDFs: page footers read as headings, sibling sections
wrongly nested, table rows separated from their column headers.

---

## Measuring retrieval quality

Retrieval quality is a number here, not an impression:

```bash
python eval/run_eval.py            # measure
python eval/run_eval.py --sweep    # also sweep the distance threshold
```

`eval/questions.json` holds 31 questions with ground truth, 10 out-of-scope
questions, and 4 deliberate near-misses. The harness **validates its own ground
truth first** -- if an expected phrase is not in the expected document, that is a
bug in the question set and it says so rather than scoring the app down.

Current numbers:

| Group | Metric | Score |
|---|---|---|
| In-scope (23) | Hit@1 / Recall@4 / MRR / Grounded | 100% / 100% / 1.00 / 100% |
| Hard identifiers (8) | Hit@1 / Grounded | 100% / 100% |
| Out-of-scope (10) | Refused | 100% |
| Near-miss (4) | Refused | 100% |

Run it after **any** change that could affect retrieval -- chunk sizes, embedding
model, threshold, search strategy. `--sweep` is how the threshold was chosen:
it prints in-scope recall against out-of-scope refusal across a range, and the
value in `config.py` sits in the middle of the window where both are 100%.

**The threshold belongs to the embedding model, not to the app.** Measured on
this corpus:

| Embedding model | In-scope tops out | Nearest excluded | Threshold |
|---|---|---|---|
| `nomic-embed-text` | 0.245 | 0.435 | **0.375** |
| `text-embedding-3-small` | 0.479 | 0.562 | **0.52** |

Both reach 100% on every metric *at their own threshold*. Using one model's
threshold with the other is what caused 10 of 23 answerable questions to be
refused, so `config.py` keys the value by model name.

## Project layout

```
app/
  config.py        every tunable setting, in one place
  pdf_parser.py    read -> clean -> classify (prose / table / OCR / unreadable)
  ocr.py           read text out of pages that have no text layer
  chunker.py       section detection -> parents -> children
  vector_store.py  embed, persist to ChromaDB, retrieve, expand to parents
  rag_chain.py     prompt -> model -> answer + citations
  main.py          FastAPI endpoints
streamlit_app.py   the UI (display only; calls the API over HTTP)
ingest.py          load every PDF in documents/
tests/             126 tests
eval/              questions.json + run_eval.py (retrieval metrics)
.github/workflows/ CI: lint, build the index, test, measure retrieval
documents/         your PDFs
chroma_db/         the vector database (created by ingest; gitignored)
```

---

## Troubleshooting

**`401 invalid_api_key`**
The key is not recognised by `api.openai.com`. If it came from a company
gateway, it will not work without `EMBEDDING_BASE_URL` (and/or `CHAT_BASE_URL`)
set to that gateway's address. Ask whoever issued it for the base URL and the
exact model names.

**Answerable questions get "I don't know"**
Almost always the relevance threshold does not match the embedding model. Run
`python eval/run_eval.py --sweep` and set the value for your model in
`config.py`. Measured examples: `nomic-embed-text` needs ~0.375,
`text-embedding-3-small` needs ~0.52 — a 40% difference on the same corpus.

**`EmbeddingModelMismatch`**
The database was built by a different embedding model. Run
`python ingest.py --reset`.

**The UI says it cannot reach the API**
Start it in another terminal: `./run_api.sh`.

**"There are no documents indexed yet"**
Put PDFs in `documents/` and run `python ingest.py`.

**`pip install` fails on numpy, chromadb, or onnxruntime**
You are probably on Python 3.14. Rebuild the venv with 3.11 or 3.12.

**Answers are poor, or cite the wrong thing**
Suspect retrieval before the model -- if the right section never came back, no
model can rescue the answer. Measure it rather than guessing:
`python eval/run_eval.py`. If in-scope recall is high but answers still read
badly, the model is the problem, not retrieval; try a larger `CHAT_MODEL`.

**A question that should be refused gets answered anyway**
Run `python eval/run_eval.py --sweep` and pick a threshold inside the window
where in-scope recall and out-of-scope refusal are both high. Note the two are in
tension: a lower threshold refuses more but starts dropping real answers.

---

## Known limitations

Stated plainly, because knowing where a system is weak is part of using it.

- **OCR is imperfect and labelled as such.** Scanned pages are read by Tesseract,
  but it misreads characters (`ACME CORP` came back as `ACME GORP`). Every OCR
  citation says so, and output below 55% confidence is discarded rather than
  indexed. Do not trust exact figures from an OCR source without checking the
  original.
- **Pages that are genuinely pictures stay unreadable.** OCR recovers *words*. A
  wiring diagram or a photo with no text yields nothing useful, and such pages are
  still reported as unreadable. Describing an image would need a vision language
  model, which is not implemented.
- **Counting and totalling questions are weak.** "How many X in total?" needs
  every chunk; retrieval fetches a handful. This is inherent to RAG.
- **Single turn only.** There is no conversation memory, so a follow-up like
  "and for managers?" is treated as a fresh question.
- **The named-entity guard only understands acronyms.** A near-miss phrased
  without one -- "what does the European privacy regulation require?" -- is not
  caught by it and relies on the distance threshold alone.
- **The distance threshold is corpus-specific.** It was measured on these five
  documents. A very different corpus should be re-measured with
  `python eval/run_eval.py --sweep`.
- **Answer wording depends on the model.** With a small local model such as
  `llama3.2`, the inline `(Source N)` pointer is occasionally missing or wrong.
  The Sources list is built by code and is always correct; only the model's
  inline reference can be off. A hosted model such as `gpt-4o-mini` does not have this problem.
- **Sparsely numbered documents get coarse sections.** With few headings,
  sections are large and get labelled `(part 3 of 16)`: accurate, but less
  useful than on a densely numbered document.
