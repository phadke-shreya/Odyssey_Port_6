# CLAUDE.md — Rules for this project

Instructions for any AI assistant (and any human) writing code in this repo.
**Read this before writing code. These rules override default habits.**

---

## 0. Project context

**SmartDoc** — a RAG document assistant. Users ask plain-English questions and get cited answers from a library of PDFs.

Stack: FastAPI + LangChain + ChromaDB + Streamlit + OpenAI (`text-embedding-3-small` to embed, `gpt-4o-mini` to answer).
See [plan.md](plan.md) for the architecture, the chunking strategy, and the grading rubric.

⏰ **Hard constraint: this is a ONE-DAY build.** Every rule below is written with that in mind. When a rule and the deadline conflict, the deadline usually wins — but never at the cost of the five non-negotiables in §11.

---

# PART A — Behavioral rules

These bias toward **caution over speed**. For genuinely trivial tasks, use judgment.

## 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- **State assumptions explicitly.** If uncertain, ask.
- **If multiple interpretations exist, present them** — don't silently pick one.
- **If a simpler approach exists, say so.** Push back when warranted.
- **If something is unclear, stop.** Name exactly what's confusing. Ask.

A clarifying question before writing costs 30 seconds. A wrong assumption discovered after 200 lines costs an hour — and today, an hour is 12% of the entire build.

## 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you wrote 200 lines and it could be 50, **rewrite it**.

Ask: *"Would a senior engineer call this overcomplicated?"* If yes, simplify.

### Concrete bans for this project

No plugin systems. No abstract base classes with one implementation. No caching layers, retry decorators, or async unless a real problem demands it. No config for things that will never change. No `**kwargs` pass-through "for later."

## 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor what isn't broken.
- **Match existing style**, even if you'd do it differently.
- If you spot unrelated dead code, **mention it — don't delete it.**

When your changes create orphans:

- Remove imports/variables/functions that **your** changes made unused.
- Don't remove **pre-existing** dead code unless asked.

**The test:** every changed line must trace directly to the request. If you can't explain why a line changed, revert it.

## 4. Goal-driven execution

**Define success criteria. Loop until verified.**

Turn vague tasks into verifiable ones:

| Vague | Verifiable |
|---|---|
| "Add validation" | "Write tests for invalid inputs, then make them pass" |
| "Fix the bug" | "Write a test that reproduces it, then make it pass" |
| "Refactor X" | "Tests pass before and after" |
| "Make chunking better" | "A table survives whole in one chunk — assert it" |

For multi-step work, state the plan first:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

Strong criteria let you loop independently. Weak criteria ("make it work") force constant clarification.

**Verify by running things.** Don't report success on unexecuted code. If you couldn't run it, say so plainly.

---

# PART B — Code quality rules

## 5. Boring code over clever code ⭐

**The guiding principle of this repo:** when two constructs do the same job, use the one that is **harder to get wrong**, even if the clever one is shorter or prettier. Readability is not worth a class of bug.

### Banned constructs

| Don't | Do | Why |
|---|---|---|
| f-strings — `f"Page {n} of {total}"` | `"Page {} of {}".format(n, total)` | **Project convention** (see note below) |
| Walrus — `if (n := len(x)) > 5:` | `n = len(x)` then `if n > 5:` | Two plain lines beat one clever one |
| Nested comprehensions | A `for` loop | Unreadable at 2 levels, unfixable at 3 |
| Comprehension with `if` *and* a call *and* an unpack | A `for` loop | Do one thing per line |
| `lambda` assigned to a name | `def` | `def` gives a real name in tracebacks |
| Chained ternaries | `if` / `elif` / `else` | |
| Mutable default args — `def f(x=[])` | `def f(x=None)` then check | Classic Python bug |
| Bare `except:` | `except SpecificError:` | Bare except swallows Ctrl-C |
| `from module import *` | Explicit imports | |
| Single-letter names (except `i`, `_`) | Real names | |

> **Note on the f-string rule.** This is a deliberate project convention, chosen for consistency, not a claim that f-strings are broken. Follow it uniformly — a codebase that mixes both is worse than either choice made consistently. The rules directly below are the ones that catch real bugs, and they hold regardless.

### 🔴 Three string rules that are genuinely load-bearing here

**1. Never interpolate into a LangChain prompt template.**
LangChain templates use `{}` for their own variables. Interpolating into that string makes the two brace systems collide, and you get `KeyError` at runtime — or worse, user text silently swallowed as a template variable.

```python
# WRONG — the braces fight each other
template = "Answer using {}.".format(context) + " Question: {question}"

# RIGHT — the template stays a literal; LangChain fills every slot
template = "Answer using this context:\n{context}\n\nQuestion: {question}"
prompt = ChatPromptTemplate.from_template(template)
```

**Rule: prompt templates are string literals with `{named}` placeholders. Nothing is interpolated into them.** Keep them in one module so they're auditable.

**2. Never interpolate into a log call.** Pass args and let the logger do it.

```python
logger.warning("Page %s of %s has no text", page_num, filename)   # right
logger.warning("Page " + str(page_num) + " has no text")          # wrong
```

Lazy formatting: if the level is off, the string is never built. It also keeps logs greppable by shape.

**3. Never build a query string by concatenation.** Not an issue with Chroma's Python API, but the habit matters: pass parameters, never assemble.

## 6. DRY — and its limits

### The rule

**Third occurrence = extract it.** Put it in `app/utils.py` (or the module that owns the concept), give it a real name and a docstring.

- **1st time:** write it inline.
- **2nd time:** copy it, but leave a `# TODO: dedupe with X` if it's clearly going to recur.
- **3rd time:** extract. No debate.

If the *second* occurrence is more than ~10 lines, or is a rule that must stay identical everywhere (a size limit, a cleanup step, a citation format), extract on the second.

### ⚠️ The limit — this rule and §2 pull against each other

Rule 2 says *no abstractions for single-use code*. Rule 6 says *extract repeats*. They collide when two blocks merely **look** alike.

**The test is not "does this look similar?" — it's "will these always change together?"**

If block A and block B would change for **different reasons**, they are **not** duplication. They're two things that currently resemble each other. Merging them creates a function with a `mode` flag that both callers fight over — worse than the copy.

```
Same shape, same reason to change  → extract.        (cleaning text in 3 places)
Same shape, different reasons      → leave separate.  (validating a query vs. validating a filename)
```

When unsure on a one-day build: **extract if it's mechanical, leave it if it's judgment.**

### Things that must exist exactly once in this repo

| Thing | Lives in |
|---|---|
| Embedding model name | `config.py` — **critical**, see §11 |
| Chunk sizes / overlap / `top_k` / distance threshold | `config.py` |
| The prompt template | `rag_chain.py` |
| Citation formatting | one function, used by API and UI |
| Text cleanup (headers, whitespace) | `utils.py` |
| ChromaDB client creation | `vector_store.py` |

## 7. Consistency ⭐

**Pick one way. Use it everywhere.** A reader should never have to ask "why is this done differently here?" — because the answer is always "no reason," which erodes trust in the whole file.

| Decision | The one way |
|---|---|
| Iteration | **`for` loops.** `while` only for genuinely unbounded loops (there are none here) |
| Loop over index | `for item in items:` / `enumerate()` — never `for i in range(len(items))` |
| Strings | `.format()`, per §5 |
| Paths | `pathlib.Path` — never `os.path` or string joins |
| Quotes | `"double"` |
| Naming | `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants |
| Type hints | On **every** function signature |
| Docstrings | Triple-double-quote on every public function |
| Returns | One shape per function. Never `dict` on success and `None` on failure — raise instead |
| Errors | Raise a specific exception; catch at the boundary (§9) |
| Imports | stdlib → third-party → local, blank line between, alphabetical within |
| Logging | `logging` module. **Never `print()`** outside `ingest.py` CLI output |
| Data passing | One dataclass (`Chunk`) end to end — not dicts in some places and objects in others |

**When touching existing code, the surrounding style wins over this table** (§3). Consistency with what's there beats consistency with the rules.

## 8. Comments and docstrings

**Comment the WHY, never the WHAT.** The code says what. If the what isn't clear, fix the naming.

```python
# BAD — restates the code
# loop through the pages
for page in pages:

# GOOD — explains a decision the code can't
# pdfplumber returns table text inside page.extract_text() too, so we strip it
# out after extracting tables — otherwise every table is indexed twice.
```

### Comment these

- **Non-obvious decisions** and the tradeoff behind them (chunk sizes, thresholds).
- **Magic numbers.** Every constant gets a why: `CHILD_CHUNK_SIZE = 400  # small for precise embedding; parents carry the context`
- **Workarounds**, with the reason.
- **Known limitations** — `# NOTE: fails on scanned pages; see plan.md §4E`
- **Anything you'd have to re-derive** in a week.

### Don't comment

Obvious lines. Commented-out code (**delete it — that's what git is for**). Changelogs in comments. Decorative banners.

### Docstring shape

```python
def chunk_document(pages: list[Page], filename: str) -> list[Chunk]:
    """Split parsed pages into parent/child chunks ready for embedding.

    Parents come from detected section headings; falls back to size-based
    parents when a document has fewer than MIN_HEADINGS. Tables are never
    split -- each becomes a single chunk that is its own parent and child.

    Args:
        pages: Parsed pages from pdf_parser.
        filename: Source name, stored on every chunk for citations.

    Returns:
        Child chunks, each carrying its parent's full text in metadata.

    Raises:
        ValueError: If pages is empty.
    """
```

Note: **`--` not `—` inside docstrings**, and no other non-ASCII. Encoding surprises aren't worth the typography.

## 9. Error handling

**Fail loudly in code you control. Fail gracefully at the edges the user touches.**

Three layers:

1. **Deep code (parsers, chunkers):** validate inputs, `raise` specific exceptions. Don't catch what you can't fix.
2. **Boundary (FastAPI routes, Streamlit callbacks):** catch, log with traceback, return a clean message.
3. **External calls (OpenAI API, file I/O, Chroma):** always wrapped, always with a message that says what the user can *do*.

```python
# The user must never see a stack trace. Graded: M6B3.
try:
    answer = rag_chain.ask(question)
except AuthenticationError:
    st.error("API key is invalid. Check OPENAI_API_KEY in your .env file.")
except APIConnectionError:
    st.error("Could not reach the AI service. Check your internet connection.")
except Exception:
    logger.exception("Unexpected failure answering question")
    st.error("Something went wrong. Check the logs for details.")
```

**Rules**

- Error messages state **what happened** and **what to do**. Never "Error: None".
- Never `except: pass`.
- Log the full traceback (`logger.exception`), show the user the short version.
- Validate at the entry point: empty query, over-long query, wrong file type, no documents indexed.
- **No error handling for impossible cases** (§2). If a function can't get `None`, don't check for it.

## 10. Security — graded (M6B5, M6E3)

- 🔴 **No secrets in code. Ever.** `os.getenv("OPENAI_API_KEY")`, loaded from `.env` via `python-dotenv`. `OPENAI_BASE_URL` too — blank means talk to OpenAI directly, set means route via the company gateway. Both listed blank in `.env.example`.
- 🔴 **`.env` in `.gitignore` before the first commit.** A key in git history is a key that must be rotated.
- Commit `.env.example` with **empty** values.
- **Never log or print the key**, not even truncated. Not in errors, not in debug output.
- **Validate every user input:** query length, uploaded file extension *and* size, filename sanitized before it touches the filesystem (`Path(name).name` — never trust an uploaded path).
- Never `eval`, `exec`, `pickle.load`, or `subprocess` with user input.
- On startup, fail with a clear message if the key is missing — don't discover it mid-query.

---

# PART C — Project invariants

## 11. The five things that must never break

These are graded deliverables. **No refactor, simplification, or time pressure may violate them.** If a change would, stop and say so.

1. **One embedding model name, in `config.py`, used by both ingest and query.**
   Ingest and query must embed with the *same* model or retrieval silently returns garbage — vectors from different models are incomparable. Never hardcode a model name at a call site.
   Model: OpenAI `text-embedding-3-small`. This is a **different model from the answering one** (`gpt-4o-mini`) — never mix them up. If the embedding model ever changes, **delete `chroma_db/` and re-ingest**; old vectors live in a different vector space and are unusable. If the fallback local model (`all-MiniLM-L6-v2`) is used instead, its input window is only ~256 tokens — log a warning on any chunk over ~1000 chars, because the remainder is silently dropped.

2. **ChromaDB persists to `./chroma_db/`.** Never an in-memory client, never a Python list. Kill the app, restart, query without re-ingesting — that must work. Parent text lives in child metadata so it persists too.

3. **Every answer carries a citation** — document + page + section. An answer path that can return text without sources is a bug, not a shortcut. Structure it so it's impossible: the answer and its sources come back together.

4. **Out-of-scope questions get "I don't know."** Enforced in two places: the distance threshold before the call, and the prompt itself. Never loosen both.

5. **A table is never split.** One table = one chunk = its own parent and child. There is a test. Keep it passing.

Also: **`temperature=0`** on every generation call, so the same question gives the same answer (M6B1).

## 12. SOLID, concretely

Abstract SOLID is useless. Here it is in terms of this repo.

**S — Single responsibility.** One module, one job:

| Module | Does | Must NOT |
|---|---|---|
| `pdf_parser.py` | read + clean + classify | chunk, embed |
| `chunker.py` | text → parent/child chunks | touch PDFs or Chroma |
| `vector_store.py` | embed, store, retrieve, expand to parents | build prompts or call the LLM |
| `rag_chain.py` | prompt → LLM → answer + citations | know Chroma exists |
| `main.py` | HTTP in/out, validation | contain business logic |
| `streamlit_app.py` | display | contain retrieval logic |

A function named `parse_and_chunk_and_embed` violates this. Split it.

**O — Open/closed.** Adding a new content type (say code blocks) should mean **adding** a handler, not editing five functions. Dispatch on `content_type`; don't scatter `if is_table` checks through the pipeline.

**L — Substitutability.** Light here, since there's little inheritance. But if content handlers share an interface, every one returns the same shape — `list[Chunk]`, never sometimes a bare string.

**I — Interface segregation.** Don't force callers to pass what they don't use. If `format_citation()` needs three fields, take three fields, not the whole app config.

**D — Dependency inversion.** `rag_chain` receives a retriever; it does not `import chromadb`. Swapping Chroma for FAISS should touch **one file**. Config is passed in or imported from `config.py` — never read from `os.environ` deep inside logic.

## 13. Python conventions (M6E4 grades this)

- **PEP 8.** 4 spaces, ~88 char lines, two blank lines between top-level defs.
- Run `black .` and `ruff check .` before committing. Non-negotiable — it's free marks.
- Type hints on every signature. Modern style: `list[str]`, `dict[str, int]`, `str | None`.
- `pathlib` over `os.path`. `dataclass` for structured data. `enum` for fixed sets like `content_type`.
- Constants `UPPER_SNAKE_CASE` at module top or in `config.py`.
- Prefix truly internal helpers with `_`.
- No mutable global state. `config.py` holds constants only.

## 14. Commits

- Small and frequent. Present tense, specific: `add section-aware parent chunking`, not `updates`.
- One logical change per commit. Don't mix a feature with a reformat.
- Never commit `.env`, `chroma_db/`, `venv/`, `__pycache__/`, or PDFs you don't have rights to.
- **Never commit code you haven't run.**

---

## 15. Before saying "done"

- [ ] It **runs**, and I ran it
- [ ] Every changed line traces to the request (§3)
- [ ] No stray debug `print()`s
- [ ] No secrets, no keys, nothing sensitive logged
- [ ] `black` + `ruff` clean
- [ ] All five invariants in §11 hold
- [ ] Type hints and docstrings on new functions
- [ ] No f-strings, no walrus, no nested comprehensions (§5)
- [ ] Consistent with §7 — same loop style, same string style, same error style
- [ ] Assumptions and known limitations stated out loud, not buried

**If something is broken, incomplete, or untested, say so plainly.** A known gap is fine. A hidden one is not — and it's exactly what a mentor's third question will find.
