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
| **Scanned pages are flagged, not dropped** | A page with no extractable text becomes a visible placeholder. A known gap beats a silent one. |
| **"I don't know" is structural** | Decided by retrieval distance *before* the model is called, so an out-of-scope question never reaches the LLM. |
| **Answers cannot lose their sources** | Text and citations travel in one object; no code path returns text alone. |

Full reasoning, including the tradeoffs, is in [plan.md](plan.md). Coding rules
are in [CLAUDE.md](CLAUDE.md).

---

## Requirements

- **Python 3.11 or 3.12.** Not 3.14 — several dependencies have no wheels for it
  yet and the install fails. Check with `python3 --version`.
- An **OpenAI API key** for writing answers. Optional: without one, searching and
  citations still work fully, only the final answer sentence is missing.

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

Then open `.env` and fill it in:

```ini
OPENAI_API_KEY=sk-...

# Leave blank to use api.openai.com directly.
# Set it only if you go through a company gateway / proxy.
OPENAI_BASE_URL=

# "local"  = embed on this machine, no key needed (default)
# "openai" = embed via OpenAI, better quality, needs a working key
EMBEDDING_PROVIDER=local
```

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
pytest -q                 # 85 tests
black --check app tests   # formatting
ruff check app tests      # linting
```

The two that matter most: a table must never be split, and every child chunk
must resolve to its parent. Many of the rest are regression tests pinning real
defects found in actual PDFs: page footers read as headings, sibling sections
wrongly nested, table rows separated from their column headers.

---

## Project layout

```
app/
  config.py        every tunable setting, in one place
  pdf_parser.py    read -> clean -> classify (prose / table / image-only)
  chunker.py       section detection -> parents -> children
  vector_store.py  embed, persist to ChromaDB, retrieve, expand to parents
  rag_chain.py     prompt -> model -> answer + citations
  main.py          FastAPI endpoints
streamlit_app.py   the UI (display only; calls the API over HTTP)
ingest.py          load every PDF in documents/
tests/             85 tests
documents/         your PDFs
chroma_db/         the vector database (created by ingest; gitignored)
```

---

## Troubleshooting

**`401 invalid_api_key`**
The key is not recognised by `api.openai.com`. If it came from a company
gateway, it will not work without `OPENAI_BASE_URL` set to that gateway's
address. Ask whoever issued it for the base URL and the exact model names.

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
Suspect retrieval before the model. Check which sources came back in the UI: if
the right section is not among them, no model can fix the answer. Try
`EMBEDDING_PROVIDER=openai` (then `--reset`), which retrieves noticeably better.

---

## Known limitations

Stated plainly, because knowing where a system is weak is part of using it.

- **Scanned pages are not readable.** They are detected and flagged, not
  transcribed. OCR would fix this and is not implemented.
- **Sparsely numbered documents get coarse sections.** With few headings,
  sections are large and get labelled `(part 3 of 16)`. Accurate, but less
  useful than on a densely numbered policy document.
- **Counting and comparison questions are weak.** "How many X in total?" needs
  every chunk; retrieval fetches a handful. This is inherent to RAG.
- **The local embedding model truncates long chunks** at roughly 1000
  characters, which silently affects large tables. `EMBEDDING_PROVIDER=openai`
  has a 32,000-character window and does not have this problem — ingest logs a
  warning for every chunk at risk.
