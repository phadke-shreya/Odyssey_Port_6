# Code walkthrough, line by line

Every file, every meaningful line. Read this next to the actual file with line
numbers on. For the concepts behind it, read [EXPLAINER.md](EXPLAINER.md) first.

**Reading order matters** — each file depends on the one before it:

```
config.py  ->  chunker.py  ->  pdf_parser.py  ->  vector_store.py
           ->  rag_chain.py  ->  main.py  ->  streamlit_app.py
```

**Two conventions used everywhere** (from CLAUDE.md), because a reviewer will ask:

1. **No f-strings** — `"{}".format(x)` throughout, for consistency.
2. **Never `print()`** — always `logger`, so output has a level and can be filtered.

---

# `app/config.py` — every setting in one place

**Why this file exists:** if a chunk size or a model name is written in five
places and you change four of them, the app breaks in a way that is very hard to
find. One home means it cannot drift.

| Lines | What it does |
|---|---|
| `1-5` | Module docstring. States the rule: nothing else in the project reads `os.environ` or hardcodes a number. |
| `7-10` | `os` for environment variables, `Path` for filesystem paths, `load_dotenv` to read `.env`. |
| `12` | `load_dotenv()` — reads `.env` into the environment. **Runs at import**, so every later `os.getenv` sees it. |
| `15` | `PROJECT_ROOT` — `__file__` is `app/config.py`; `.resolve()` makes it absolute; `.parent.parent` climbs from `config.py` → `app/` → project root. Absolute paths mean the app works no matter which directory you run it from. |
| `16-17` | `documents/` (input PDFs) and `chroma_db/` (the database) built from the root with `/`, which `pathlib` overloads to mean "join path". |
| `18` | `CHROMA_COLLECTION` — the table name inside the database. |
| `21` | `OPENAI_API_KEY` — `os.getenv(name, "")` returns `""` rather than `None` if unset, so later code can do a simple truth test. |
| `25` | `OPENAI_BASE_URL` — `.strip()` removes stray spaces; `or None` converts empty string to `None`, because the SDK treats `None` as "use the default host" but would treat `""` as a real (broken) URL. **This one line is what lets a company gateway or Ollama work with no code change.** |
| `34-35` | The two model names, both `os.getenv`-overridable so a gateway or local server can expose different names. |
| `40` | `EMBEDDING_PROVIDER` — `"local"` or `"openai"`. `.strip().lower()` so `" OpenAI "` still works. |
| `45` | `LOCAL_EMBEDDING_MODEL` — the fallback that needs no key. Not overridable: it is whatever ChromaDB ships. |
| `48` | `CHAT_TEMPERATURE = 0` — temperature controls randomness; 0 means "always pick the most likely next word", so the same question gives the same answer. **This is what makes the app consistent (graded: M6B1).** |
| `53-55` | The chunk sizes. Parents 3000 chars (what the AI reads), children 400 with 80 overlap (what gets searched). |
| `59` | `MIN_HEADINGS_FOR_SECTIONS = 3` — below this, a document is treated as unstructured and we fall back to size-based parents. |
| `62` | `MAX_HEADING_CHARS = 80` — a heading is short. Longer means it is prose that merely looks like one. |
| `65` | `MIN_CHARS_FOR_TEXT_PAGE = 100` — a full page with less text than this is almost certainly a scan. |
| `71-75` | `CHUNK_WARN_CHARS` — how long a chunk may be before the embedding model would silently cut it off. **It is a conditional expression**, not a constant, because the limit is a property of the model in use: 1000 for MiniLM, 30000 for the hosted ones. |
| `78` | `TOP_K_CHILDREN = 8` — how many small pieces to fetch. |
| `79` | `TOP_K_PARENTS = 4` — how many unique sections survive de-duplication and reach the AI. Fetching 8 to keep 4 leaves room for several children sharing one parent. |
| `81-98` | `MAX_DISTANCE` — the anti-hallucination threshold, also model-dependent. **The comment records the measured numbers for both models and tells the next person how to re-measure.** This is the single most easily-broken setting in the project. |
| `101` | `MAX_QUERY_CHARS = 2000` — rejects an essay pasted into the question box. |
| `102` | `MAX_UPLOAD_BYTES` — `25 * 1024 * 1024` written as arithmetic rather than `26214400`, so a reader sees "25 MB" immediately. |
| `105` | `API_URL` — where the UI looks for the API. `.rstrip("/")` prevents a double slash when paths are appended. |
| `108-119` | **`embedding_fingerprint()`** — returns e.g. `"openai:nomic-embed-text"`. Stamped into the database so a mismatch can be detected. The docstring explains *why*: querying with the wrong model does not raise an error, it silently returns confident nonsense. |
| `122-134` | **`missing_credentials()`** — returns a problem description or `""`. Returning a *string* rather than raising means the caller decides what to do. Note line `128`: a key is only required when `EMBEDDING_PROVIDER == "openai"`, because the local path needs none. |

---

# `app/chunker.py` — the centrepiece

**Why this file exists:** turning a document into pieces is where the quality of a
RAG system is decided. Everything downstream can only be as good as this.

### Setting up (lines 1–59)

| Lines | What it does |
|---|---|
| `1-12` | Docstring stating the whole strategy in one paragraph, including the deliberate table exception. |
| `14-21` | Imports. `re` for regex, `dataclass` for tidy data holders, `StrEnum` for the content types, `RecursiveCharacterTextSplitter` from LangChain, and our own `config`. |
| `23` | `logger = logging.getLogger(__name__)` — `__name__` is `"app.chunker"`, so log lines say which module they came from. |
| `26-31` | **`ContentType`** — `StrEnum` means the members *are* strings, so `ContentType.TABLE == "table"` is `True`. That lets the value go straight into a database field with no conversion. |
| `34-40` | **`Block`** — one piece of a parsed page: its text, its page number, and its kind. This is the hand-off shape between `pdf_parser` and `chunker`. |
| `43-58` | **`Chunk`** — a *child*, the thing that gets embedded. Note `parent_text` (line 54): each child **carries a full copy of its parent**. That is deliberate — it means retrieval needs no second lookup, and the parent survives a restart because ChromaDB writes metadata to disk. Slight duplication, one fewer moving part. |

### Heading detection (lines 61–232)

This is the fiddliest part of the project, and the comment at `61-65` explains the
governing principle: **a wrong section label in a citation is worse than no label**,
so every rule errs toward saying "not a heading".

| Lines | What it does |
|---|---|
| `68` | `_MULTI_LEVEL_HEADING` — matches `5.2 Remote Work`. Reading the regex: `^` start, `(\d+(?:\.\d+)+)` a number with **at least one** dot-group (so `5.2`, `5.2.1` — but not bare `5`), `\.?` an optional trailing dot, `\s+` whitespace, `([A-Z].*)` text starting with a capital, `$` end. |
| `72` | `_SINGLE_LEVEL_HEADING` — matches `1. Introduction`. The trailing dot after `(\d+)` is **required**. Without it, page footers like `8 Publication 15 (2026)` matched — a real bug that produced 120 fake headings. |
| `75-78` | `_KEYWORD_HEADING` — `SECTION 4`, `Article 3`. `[0-9IVXLC]+` allows Roman numerals. `re.IGNORECASE` so case does not matter. |
| `81` | `_SENTENCE_END` — a tuple of punctuation. A real heading almost never ends with a full stop or comma. |
| `84-108` | `_SMALL_WORDS` — words ignored when judging Title Case ("of", "the", "and"). `frozenset` because it never changes and set membership is fast. |
| `110` | `_WORDS` — finds words. `[A-Za-z]` first character, then letters, apostrophes (including the curly `’`) and hyphens. |
| `115` | `_DOT_LEADER` — matches `. . .`, the dotted line in a table of contents. TOC entries duplicate real headings, so they are rejected. |
| `121-149` | `_GENERIC_LABELS` — ALL-CAPS words that label a *part* of a section rather than naming a topic. Standards documents repeat `DISCUSSION` under every requirement; a citation reading `page 44 \| DISCUSSION` tells the reader nothing. |
| `152-164` | **`is_title_case()`** — the rule that separates a heading from a list item. Line `159` finds words; `160` keeps only "significant" ones (3+ letters, not a small word); `161-162` returns False if none; `163` counts capitalised ones; `164` returns True if **60% or more** are capitalised. So `"Remote Work Policy"` → 3/3 = 100% ✅, while `"If your spouse itemizes deductions"` → 1/4 = 25% ❌. |
| `167-169` | `_numbered_match()` — tries multi-level first, then single. `or` returns the first truthy result, or `None`. |
| `172-220` | **`looks_like_heading()`** — a gauntlet of rejections, cheapest first: `181` empty or too long, `183` ends like a sentence, `187` ends with a hyphen (a word split across lines, so a fragment), `190` contains a comma, `193` contains a slash (print artifacts like `AH XSL/XML`), `196` is a TOC line. Then `199-201` if numbered, it is a heading **only if** the text after the number is Title Case. `203-208` a keyword heading must either have nothing after the number (`SECTION 4`) or Title-Case text — this rejects `"Section 3509 rates aren't available..."`. `210-220` finally, ALL-CAPS lines: must be all uppercase letters, must not be only generic labels, and must have 2+ words or one word of 6+ letters (which rejects the acronym `UTC`). |
| `223-232` | **`heading_depth()`** — counts dot-separated parts: `5` → 1, `5.2` → 2, `5.2.1` → 3. Used to nest the breadcrumb. |

### Building parents and children (lines 235–386)

| Lines | What it does |
|---|---|
| `235-241` | **`build_breadcrumb()`** — joins the trail with `>`. Takes `(depth, text)` pairs and keeps only the text. |
| `244-266` | **`_explode_long_lines()`** — grouping works on *lines*, so one enormous paragraph with no newlines could never be split. This pre-splits any line longer than a whole parent. Each piece **keeps its page number** (line `265`). A test caught this: an 8800-character paragraph sailed past the size cap. |
| `269-311` | **`_emit_parent()`** — turns accumulated `(line, page)` pairs into parents. `280` explode long lines first. `287-296` walk the lines accumulating a batch; when adding the next line would exceed `PARENT_MAX_CHARS`, close the batch and start a new one **with the new line's page** (line `291`). `298-299` flush the last batch. `301` drop empties. `305-310` if a section became several parents, label them `(part 2 of 5)` so a citation says which piece. **The page-per-line tracking here fixed a real bug**: previously every part claimed the heading's page, so text on page 30 was cited as page 1. |
| `314-371` | **`_prose_parents()`** — the main grouping loop. `322-325` flatten all blocks into `(line, page)` pairs, preserving order. `327-333` count headings; **if fewer than 3, give up on sections entirely** and return size-based parents with no label. `339` the trail as `(depth, text)` pairs. `343-366` walk every line: if it is a heading, close the current parent, then fix the trail — `353-354` a numbered heading must not sit under an unnumbered one (this fixed `REFERENCES > 03.05.03 Multi-Factor Authentication`, which wrongly implied the requirement lived in References); `355-356` an unnumbered heading resets the trail; `360-361` **pop every entry at this depth or deeper**, because those are siblings or children, not ancestors (this fixed `3.1 Access Control > 3.5 Identification`, where two siblings were wrongly nested). Otherwise `366` just add the line to the current parent. |
| `374-386` | **`_children_for()`** — slices a parent into children. The `separators` list is the priority order: paragraph break, newline, sentence end, space, and `""` (mid-word) only as a last resort. `chunk_overlap` makes each child repeat the tail of the previous one, so a fact landing on a cut line survives whole in at least one child. |

### The entry point (lines 389–463)

| Lines | What it does |
|---|---|
| `407-408` | Refuse empty input loudly. An empty document is a caller bug, not something to paper over. |
| `415-416` | Split blocks: prose is grouped **across** blocks so a section can span a page break; tables and image placeholders are handled one at a time. `is` compares identity, which is correct for enum members. |
| `418-433` | For each prose parent: `420-421` build a unique `parent_id` like `"p15.pdf::p7"`; `422` slice into children; `423-433` create one `Chunk` per child, every one carrying the **same** `parent_text` and `parent_id`. That shared id is what de-duplication uses later. |
| `435-460` | For each table / image block: `438-447` warn if it exceeds the embedding model's window — this is how you learn that a big table is being silently truncated. `450-460` create **one** chunk where `text` and `parent_text` are **the same value**. That is the table rule in code: its own parent, its own only child, never sliced. |
| `462-463` | Log the count and return. |

---

# `app/pdf_parser.py` — reading the PDF

**Why this file exists:** a PDF is not a text file. It is drawing instructions.
Getting clean, correctly-classified text out is most of the work.

| Lines | What it does |
|---|---|
| `1-17` | Docstring listing the four jobs in order. |
| `19-27` | Imports. `Counter` for counting repeated lines; `pdfplumber` for reading; `Block`/`ContentType` from the chunker, so both files agree on the hand-off shape. |
| `32` | `_EDGE_LINES = 3` — only the top and bottom 3 lines of a page are candidates for being a running header/footer. This stops a genuinely repeated body sentence being deleted. |
| `35` | `_REPEAT_FRACTION = 0.5` — a line must appear on at least half the pages to count as furniture. |
| `38-39` | Regexes to collapse 3+ blank lines and runs of spaces. |
| `42-46` | **`_cell()`** — normalises one table cell. `None` becomes `""`, and `" ".join(value.split())` collapses all whitespace including newlines into single spaces, so a multi-line cell does not break the Markdown row. |
| `49-67` | **`is_real_table()`** — rejects layout boxes. PDFs use borders for callouts constantly, and those come back looking like a one-column table. `57-58` clean and drop empty rows; `59-60` need 2+ rows; `61-63` need 2+ columns; `66-67` need at least two rows that actually fill 2+ columns. **This filter cut 11 detected "tables" down to 6 real ones in one document.** |
| `70-90` | **`table_to_markdown()`** — renders rows as a Markdown table. `82` find the widest row; `83` pad every row to that width so the columns line up (a ragged row would misalign everything); `85-89` first row becomes the header, then the `---` separator, then the body. Markdown is used because it keeps each value **visibly attached to its column header**. |
| `93-122` | **`_caption_for()`** — builds the caption line. `104` crop the page to everything *above* the table (`table.bbox[1]` is its top edge; `max(..., 1)` guards against a table at the very top). `110-113` walk the lines above **upwards** and take the first that starts with a capital or digit — a line starting lowercase is the tail of a wrapped sentence and makes a confusing caption. `114-117` a missing caption must never fail a parse, so any error is swallowed and logged at debug level. **The caption is what makes a table findable**: nothing in `\| Junior \| 15 \| 10 \|` says "vacation". |
| `125-153` | **`_text_outside_tables()`** — the de-duplication trick. Without it every table is stored twice: once as clean Markdown and once as the mangled version `extract_text()` returns. `134` collect the table rectangles. `136-144` `keep()` is a predicate run on every character: return `False` if it sits inside any table rectangle (with a 1-point tolerance for rounding). `147` `page.filter(keep)` gives a page view with those characters removed. `148-153` if filtering fails, fall back to unfiltered text and warn — duplicated text is better than no text. |
| `156-176` | **`_repeated_lines()`** — finds running headers and footers. `162-163` skip documents under 3 pages, where "repeated" means nothing. `167-170` for each page take its first and last 3 lines and count them — `set(edges)` so a line appearing twice on one page still counts once. `172` the threshold is half the pages, minimum 2. `173` keep lines at or above it. |
| `179-190` | **`_clean()`** — drops furniture and normalises whitespace. `184-185` skip repeated lines; `186-187` skip lines that are only a page number; `188` collapse multiple spaces; `190` collapse 3+ blank lines to 2. |
| `193-306` | **`parse_pdf()`**, the entry point. `206-209` validate: missing file raises `FileNotFoundError`, wrong extension raises `ValueError`. `217-224` opening is wrapped so a corrupt file gives *"It may be corrupt or password protected"* instead of a library traceback. `227` iterate pages. `228` find tables **first**, because their text must be excluded from the prose. `230-248` for each table: extract, reject layout boxes, render Markdown, build the caption, and store it with the caption on the **first line**. `236-242` a broken table is skipped with a warning rather than killing the whole document. `250-257` get the text outside the tables, wrapped so a damaged page yields `""` instead of failing. `260-261` an image-only page is one with almost no text **and** no table. `263` find the repeated furniture **after** reading every page — you cannot know what repeats until you have seen them all. `265-270` clean each page into a `PROSE` block, skipping pages that cleaned down to nothing. `272-275` add the table blocks. `277-294` add a placeholder block for each unreadable page, with a `WARNING` log **and** text the user can actually read. `296-297` if nothing came out at all, fail loudly. |

---

# `app/vector_store.py` — the database

**Why this file exists:** it is the only file that imports `chromadb`. Swapping
Chroma for something else touches this file and nothing else.

| Lines | What it does |
|---|---|
| `29-31` | `_SPACE = "cosine"`. Chroma's default is squared L2 distance. **Cosine is set explicitly** because it is bounded and comparable across documents, which is what makes one `MAX_DISTANCE` threshold meaningful. |
| `35` | `_BATCH = 500` — chunks are sent in batches so memory stays flat and progress is visible instead of the app appearing to hang. |
| `37` | The metadata key holding the fingerprint. |
| `40-41` | **`EmbeddingModelMismatch`** — our own exception type, so callers can catch exactly this. |
| `44-60` | **`Retrieved`** — one parent section ready to cite. `55-60` `citation()` builds the display string: always file and page, plus the section **only if there is one** (line `58`) — an empty label would render as an ugly trailing `\|`. |
| `63-77` | **`_embedding_function()`** — picks the embedder from config. `71-75` the OpenAI path passes `api_base=config.OPENAI_BASE_URL`; **that single argument is why a company gateway or Ollama works without a code change.** `77` otherwise ChromaDB's built-in local model. |
| `80-83` | **`_client()`** — `mkdir(parents=True, exist_ok=True)` creates the folder if needed and does not complain if it exists. `PersistentClient` is the on-disk client; the in-memory one would lose everything on restart. |
| `86-126` | **`get_collection()`** — opens or creates the collection. `98-102` `get_or_create` avoids a separate "does it exist" call; the metadata carries both the distance space and our fingerprint. `103-115` Chroma runs its **own** conflict check that fires before ours with an unhelpful message, so it is caught and re-raised as *"Run: python ingest.py --reset"*. `115` `raise` bare re-raises anything unrelated. `117-125` our own check: if the stored fingerprint differs from the current one, refuse. `check_fingerprint=False` exists so `/health` can report on a mismatched database instead of crashing. |
| `129-133` | **`reset()`** — deletes the folder. Required after changing embedding model. |
| `136-178` | **`ingest()`** — `146-147` refuse empty input. `155-167` build three parallel lists, which is the shape Chroma wants: `156` a unique id per child (`parent_id` plus the loop index); `157` **`documents` is the child text — this is what gets embedded**; `158-167` the metadata, including `parent_text`, which is what makes expansion possible later. `169-176` add in batches of 500, logging progress. |
| `181-242` | **`search()`** — the heart of retrieval. `194-202` validate the question: empty and over-long both raise `ValueError` with a readable sentence. `205-207` an empty database returns `[]` rather than erroring. `209-212` query for `TOP_K_CHILDREN` children; `min(..., collection.count())` avoids asking for more than exist. `214-215` unwrap Chroma's nested lists (it supports batched queries, so results come back one list per query). `218-233` **the de-duplication**: for each hit, `220-221` drop anything beyond `MAX_DISTANCE` — *this line is the anti-hallucination guard*; `222-225` keep only the **nearest** child per `parent_id`, so one section never appears twice; `226-233` build the `Retrieved` object from metadata. `235` sort nearest-first. `242` return at most `TOP_K_PARENTS`. |
| `245-266` | **`stats()`** — for `/health` and the sidebar. `247-251` wrapped in a broad `except` because **a health check must never crash**; it returns zeros instead. `256-260` sample up to 5000 chunks to collect the distinct document names. |

---

# `app/rag_chain.py` — the prompt and the model

**Why this file exists:** it owns everything about talking to the model. It does
not know ChromaDB exists — it receives already-retrieved sections.

| Lines | What it does |
|---|---|
| `26-28` | **`DONT_KNOW`** — the exact refusal wording, as a constant so the API, the UI, and the tests all agree. |
| `30-33` | `NO_DOCUMENTS` — a *different* message, because "nothing indexed yet" and "nothing matched" are different problems needing different fixes. |
| `38-75` | **`PROMPT_TEMPLATE`** — a string **literal** with `{named}` placeholders. Nothing is ever `.format()`ed into it: LangChain fills the slots. Mixing the two brace systems causes `KeyError` at runtime. Notable parts: `43-48` formatting rules; `50-56` a **worked GOOD/BAD example pair**, because small models copy the shape of what they are shown far more reliably than they follow described rules; `59-61` forbids the document's own `[50]`-style markers; `62-66` the refusal must be the **entire** reply, never appended to an answer. |
| `78-89` | **`Answer`** — text and sources in **one object**. That is structural: there is no code path that can return text without citations. `83` `field(default_factory=list)` because a mutable default like `[]` would be shared between instances — a classic Python bug. `87-89` `is_dont_know` as a property so callers ask a question rather than string-matching themselves. |
| `92-93` | `GenerationUnavailable` — a distinct exception for "no model configured", which is not an error the user caused. |
| `96-108` | **`format_context()`** — numbers the sections for the model. Labelled `"Source 1"`, **not `"[1]"`**, on purpose: NIST documents are full of their own `[50]` markers, and a model told to cite with `[n]` copies those straight out of the text. This was a real bug. |
| `111-135` | **`_build_llm()`** — `118-122` no key means `GenerationUnavailable`, with a message noting retrieval still works. `124-126` the import is **inside the function** so the module can be imported (and tested) without the dependency loaded. `128-135` build the model: `temperature=0` for consistency, `base_url` for the gateway, `timeout=60` so a hung request cannot wedge the app, `max_retries=1` because a user is waiting. |
| `138-163` | **`_friendly_error()`** — translates a provider exception into something actionable. It matches on the exception **class name** and message text rather than importing provider-specific exception types, so it works across OpenAI, gateways and Ollama. Each branch names the setting to check. |
| `166-223` | **`answer_question()`** — the orchestrator. `180-182` **if nothing was retrieved, return `DONT_KNOW` immediately — the model is never called.** That is the anti-hallucination guarantee. `184-193` if no model is available, return the **sources anyway** with an explanation, because retrieval is the useful half. `197-198` `prompt \| llm` is LangChain's pipe: feed the filled prompt into the model. `200-215` the call is wrapped; `209` `logger.exception` writes the **full traceback to the log** while `214` gives the user **one friendly sentence**. `217-222` the raw text passes through four cleanup functions, then `223` returns text and sources together. |
| `226-241` | **`_strip_contradictory_hedge()`** — removes a trailing "I don't know" that follows a real answer. `235-236` `position <= 0` means either not found (`-1`) or at the very start (`0`, a genuine refusal) — both are left alone. `238-239` if less than 40 characters preceded it, treat the whole thing as a refusal rather than an answer. |
| `247-260` | **`_strip_document_markers()`** — deletes `[50]`-style markers the model copied from the source. Safe by construction: valid citations are `(Source N)`, so any `[digits]` is by definition an artifact. `260` also tidies the double spaces left behind. |
| `265-280` | **`_strip_echoed_source_header()`** — drops `Source 2: file.pdf \| page 55 \| ...` lines the model copied. `re.MULTILINE` makes `^` match at each line start. `276-277` **if stripping left nothing, return the original** — never show an empty answer just because it was badly formatted. |
| `287-304` | **`_normalise_citations()`** — rewrites `[Source 2]`, `Source 2`, `(source 2)` all as `(Source 2)`. Two regex alternatives on purpose (`288`): the bracketed form may have padding inside the brackets, but the bare form must **not** swallow the spaces around it — a bug the tests caught, which produced `see(Source 3)for detail`. `300-302` `group(1) or group(2)` picks whichever alternative matched. |

---

# `app/main.py` — the HTTP layer

**Why this file exists:** it does HTTP only. Validate, call the modules that hold
the logic, turn any failure into a clean message.

| Lines | What it does |
|---|---|
| `21` | `logging.basicConfig` — set up once, here, because this is the entry point. |
| `24-28` | The FastAPI app. Title and description appear in the free interactive docs at `/docs`. |
| `31-32` | **`AskRequest`** — a Pydantic model. `Field(...)` with `...` means **required**; FastAPI rejects a malformed body with a 422 before your code runs. |
| `35-49` | `SourceOut` and `AskResponse` — the response shapes. Declaring them means the JSON is validated on the way out and documented automatically. |
| `52-63` | **`GET /health`** — what is indexed, which embedding model built it, whether a key is set. `61` reports `api_key_configured` rather than "generation works", because a key existing is **not** the same as a key working. |
| `66-104` | **`POST /ask`**. `70` search. `71-73` a `ValueError` is the *user's* input problem, so its message is returned as the answer with no traceback. `74-81` anything else is logged with a full traceback and turned into a short notice. `83-85` distinguish "no documents at all" from "no match". `87` generate. `88-104` convert `Retrieved` objects into `SourceOut`; `97` rounds the distance for display. |
| `107-153` | **POST /upload**. `108` `Annotated[UploadFile, File()]` is the modern FastAPI idiom (the older `= File(...)` default trips linters). `114` **`Path(...).name` strips any directory part — the uploaded filename is never trusted as a path**, which is what stops `../../etc/passwd`. `115-116` extension check. `118-121` write to a temporary directory that cleans itself up. `123-131` size check **after** writing, because a client-supplied length cannot be trusted. `133-136` parse, chunk, ingest. `137-138` a `ValueError` carries a readable message, so pass it through. `139-145` anything else is logged and generalised. `147-153` report how many chunks were added and how many pages were unreadable — **surfacing the gap rather than hiding it**. |

---

# `streamlit_app.py` — the screen

**Why this file exists:** display only. It calls the API over HTTP, so the same
endpoints could serve a Slack bot.

| Lines | What it does |
|---|---|
| `13` | `TIMEOUT = 120` — generous, because a local model can be slow. |
| `15` | `set_page_config` — must be the first Streamlit call. |
| `18-25` | **`api_get()`** — returns `{}` on any failure, so callers test truthiness rather than handling exceptions. `raise_for_status()` turns a 4xx/5xx into an exception. |
| `28-35` | **`api_post()`** — returns `{"_transport_error": ...}` on failure. The underscore marks it as an internal signal rather than a field the API sent. |
| `40` | One `/health` call feeds the whole sidebar. |
| `45-50` | If the API is unreachable, show **the exact command to fix it**. This message previously named the wrong command — a real bug, because following it started the API on the wrong port. |
| `52-61` | Chunk count, document list, and which embedding model built the index. |
| `63-67` | Warn when no key is configured — and state that search still works, so the user knows what they *can* do. |
| `71-95` | Upload. `72` the button only acts when a file is chosen. `73` a spinner, because indexing takes seconds. `78-81` transport failure and API-reported failure are different messages. `88-94` if pages were unreadable, say so. `95` `st.rerun()` refreshes the sidebar so the new document appears immediately. |
| `106-110` | The question box and Ask button. |
| `112-114` | `question.strip()` so spaces alone do not count as a question. |
| `124-128` | A refusal renders as neutral info (`st.info`), not as an answer — visually different on purpose. |
| `132-138` | If the model could not be called, warn **and** say the sections were still found, because that is the useful half. |
| `141-155` | **Citations, always shown.** `144` numbered to match the `(Source N)` labels in the answer. `145-148` tables and unreadable pages are tagged in the label. `149-155` each source is expandable, showing the distance and the exact text — **this is what makes the answer verifiable**, which is the whole point. |
| `156-157` | If it refused and there are no sources, explain why in a caption. |
| `159-160` | Pressing Ask with an empty box gets a nudge, not a crash. |

---

# `ingest.py` and the run scripts

| File | What it does |
|---|---|
| **`ingest.py`** | Loads every PDF in `documents/`. `argparse` gives `--reset`, which wipes the database first — **required after changing embedding model**. Each document is wrapped in its own try/except so **one bad PDF does not stop the rest**, and failures are collected and reported at the end with a non-zero exit code, so CI would notice. |
| **`run_api.sh`** | Sources `.env`, reads `API_PORT` (default 8006), and `exec`s uvicorn. `exec` replaces the shell process so Ctrl+C reaches uvicorn directly. |
| **`run_ui.sh`** | Same, but exports `API_URL` so the UI and API can never disagree about the port. |

---

# The five questions a reviewer is most likely to ask

**"Why parent/child instead of one chunk size?"**
> Searching wants small chunks for a sharp match; answering wants big ones for
> context. Two sizes gets both: children are embedded, parents are what the model
> reads. `vector_store.search()` lines 218–233 is where a matched child is
> expanded to its parent and duplicates are dropped.

**"How do you stop it making things up?"**
> Two layers. `vector_store.search()` line 220 drops anything beyond
> `MAX_DISTANCE`, and `rag_chain.answer_question()` line 180 returns "I don't
> know" **without calling the model at all** when nothing survives. The prompt
> also forbids outside knowledge, but that is the second line of defence, not the
> first.

**"Where are the embeddings stored?"**
> `chroma_db/` on disk, via `chromadb.PersistentClient` (`vector_store.py:83`).
> Kill the app, restart, query — nothing is re-read. The collection also records
> *which model* built it and refuses to answer on a mismatch, because querying
> with a different model returns confident nonsense rather than an error.

**"What was hardest?"**
> Heading detection. The first version found 120 "headings" in two documents and
> essentially all were wrong — page footers, numbered list items, wrapped
> fragments. The fix was a Title-Case test (`chunker.py:152`) that distinguishes
> `5.2 Remote Work Policy` from `11. If your spouse itemizes deductions`. Getting
> from 120 false positives to 11 real headings took several rounds, each pinned by
> a regression test.

**"What would you do differently?"**
> Build the evaluation set first. I tuned chunk size and the distance threshold by
> eyeballing output, which is slow and subjective. And I would have learned sooner
> that `MAX_DISTANCE` is a property of the embedding model — switching models broke
> eight tests because all the distances shifted, and only measuring caught it.
