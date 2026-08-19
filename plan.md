# SmartDoc — My Plan (explained super simply)

## 1. What am I building?

Imagine your school has a HUGE pile of boring books: the rules book, the "how to use the printer" book, the "what to do on your first day" book. If you want to know *"how many days off do I get?"*, you would have to read all the books to find the one line that answers it. That is annoying and slow.

**I am building a robot librarian.**

You type a question in normal English:
> "How many vacation days do I get?"

And the robot answers:
> "You get 20 vacation days per year."
> *— from `HR_Policy.pdf`, page 4, section "Leave"*

Two important parts:
1. It **answers in plain words**, like a person would.
2. It **shows where it found the answer** (which file, which page). This is called a **citation**. My project must show a citation on *every single answer*. No exceptions.

And if you ask it something silly that is not in any of the books, like *"who won the football match yesterday?"*, it must be brave and say **"I don't know, that's not in these documents."** It must NOT make something up. Making stuff up is called **hallucination**, and it is the #1 thing that gets my project marked down.

---

## 2. The magic trick behind it: RAG

The fancy name for my robot librarian is **RAG** = **R**etrieval **A**ugmented **G**eneration.

Say it in kid words:

> **Retrieval** = go find the right pages.
> **Augmented** = staple those pages onto my question.
> **Generation** = the AI reads only those pages and writes the answer.

### Why not just ask ChatGPT directly?

Because ChatGPT has never read *my company's* HR policy. It was never allowed to see it. So if I just ask it, it will guess — confidently and wrongly.

### Why not just paste the whole PDF into the AI?

Because the AI has a small brain-window (a "context window"). You cannot shove 500 pages in. And even if you could, it costs a lot of money and the AI gets confused and misses things.

So: **first find the 4 most useful paragraphs, then only show those 4 to the AI.** That is the whole idea. That's RAG. That's it.

### The one-line version I will say to my mentor

> "RAG is: I give the AI an open-book exam instead of a memory test. I find the right pages first, hand only those pages to the AI, and make it answer using only what's on them — and cite them."

---

## 3. How does the computer "find the right pages"?

This is the part that sounds like magic, so let me explain it slowly.

### Old way: keyword search (like Ctrl+F)

Ctrl+F looks for the *exact letters* you typed. So if the document says **"annual leave entitlement"** but you type **"vacation days"**, Ctrl+F finds **nothing**. Zero results. Even though the document literally answers your question. Different words, same meaning — Ctrl+F is blind to that.

### New way: embeddings (meaning-search)

An **embedding** turns a piece of text into a long list of numbers, like `[0.12, -0.88, 0.34, ...]` (about 384 or 1536 numbers long).

Think of it as **an address on a giant map of meaning.**

- "vacation days" and "annual leave" land at almost the **same spot** on the map, because they mean the same thing.
- "vacation days" and "fire extinguisher location" land **very far apart**.

So to answer a question, I turn the *question* into numbers too, drop it on the map, and grab whatever paragraphs are **standing nearest to it**. Nearest = most likely to be about the same topic. Meaning-based, not letter-based.

That's why this beats Ctrl+F: **it understands synonyms.**

### Where do these numbers live? The vector database

I need a place to keep thousands of these number-lists and search them *fast*. That place is a **vector database**. I'm using **ChromaDB**.

A vector DB is like a library where books are shelved **by what they're about**, not alphabetically. So "give me the 4 things nearest to this question" is one quick lookup instead of comparing against all 5,000 chunks one by one.

**Very important for my marks (M6S2):** ChromaDB must **save to disk** in a folder (I'll use `./chroma_db/`). If I keep the embeddings in a Python list or in memory, they vanish the moment I close the app and I lose those points. I must be able to point at the folder and say *"there. that's where they're persisted."* And I must be able to restart the app and have it still work **without re-reading the PDFs**.

---

## 4. Chunking — the most important part (my mentor's note)

### Why chunk at all?

I can't store a whole 50-page PDF as one blob of numbers. If I did, the "meaning address" would be a mush of every topic in the book — vacation, fire safety, laptops, dress code, all averaged together into one meaningless smudge. Searching it would be useless.

So I **cut the document into small pieces** ("chunks"), and give each piece its own address on the meaning-map. Now each address is about *one* thing.

### The Goldilocks problem (chunk size)

| Chunk size | What goes wrong |
|---|---|
| **Too small** (~100 characters) | Answer gets cut in half. Chunk says "employees are entitled to" and stops. The AI sees half a sentence and can't answer. |
| **Too big** (~5000 characters) | One chunk covers 6 different topics. Its meaning-address becomes mush, so search stops finding it. Also I waste money sending junk to the AI. |
| **Just right** (~800–1000 characters) | About 1–2 paragraphs. One idea per chunk. Enough context to actually answer. |

But notice something annoying about that table. **Searching wants small chunks and answering wants big chunks.**

- **Small is better for searching.** A tiny chunk is about exactly one thing, so its meaning-address is sharp and it matches the question precisely.
- **Big is better for answering.** The AI needs surrounding context — the sentence before, the exception clause after — to give a complete answer.

If I pick one number, I lose one side. So don't pick one number. **Use two.**

---

### 🌟 Parent chunks and child chunks (small-to-big retrieval)

This is the trick I heard about, and it's the right call. **Search with small chunks, answer with big ones.**

Picture a book:

- The **parent** is a whole section — like all of "5.2 Remote Work Policy". Big. ~2000–3000 characters. Full context.
- The **children** are small slices of that parent. ~400 characters each. Sharp and specific.

Then:

1. I embed and store **only the children** in ChromaDB. Small = precise matching.
2. A question comes in. Chroma finds the best-matching **child**.
3. But I **don't** send that child to the AI. I look up **which parent it came from** and send the *whole parent section* instead.

So I get precise *finding* and complete *answering* at the same time. The name for this is **small-to-big retrieval**.

**A real example.** Question: *"can I work from home on Fridays?"*

- The winning child is one sentence: `"Fridays are designated as optional remote days."` Great match — but on its own it doesn't mention that you need manager approval.
- So I send up the whole parent, section 5.2, which also contains: *"...subject to written manager approval and a minimum of 3 office days per week."*
- The answer is now **correct and complete** instead of technically-true-but-misleading.

A single medium-sized chunk would probably have missed one half or the other. That example is exactly what I should tell my mentor.

---

### 🌟 How I decide where a parent starts and stops: section detection

Here's the second good idea — and it's why this project stops looking generic.

Most real company documents are **already** cut into sections *by a human*, and they numbered them for me:

```
5. Working Arrangements
5.1 Office Hours
5.2 Remote Work Policy
5.3 Overtime
```

That numbering is a **free, human-made table of contents**. The person who wrote the document already decided where one idea ends and the next begins — far better than any character count I could guess. It would be silly to ignore that and blindly count to 1000.

**So my parents are the real sections, not arbitrary character blocks.**

How I find them: a regex over the extracted text looking for lines that look like headings —

- `1.` / `1.1` / `1.1.1` (numbered)
- `SECTION 4:` or `ARTICLE 3`
- `ALL CAPS SHORT LINES` (a common heading style)
- A short line (under ~80 chars) with no full stop at the end, followed by a blank line

Every heading I find starts a new parent, and it hands me a **breadcrumb** for free: `5 > 5.2 Remote Work Policy`.

**Two big wins from this:**

1. **Better retrieval.** Each parent is one genuine topic, so its content is coherent.
2. **Citations that actually score marks.** M6S4 wants "exact document *and section*". I can print:
   > 📄 `HR_Policy.pdf` · page 12 · **§5.2 Remote Work Policy**

   That's a real citation a human can go verify. "page 12" alone is much weaker.

**Fallback (important — documents are messy):** plenty of PDFs have no numbering at all. If I find fewer than ~3 headings in a document, I don't fight it — I fall back to plain size-based parents (~2000 chars, cut on paragraph breaks). **Section detection is a bonus when the document cooperates, never a requirement.** Never let a nice feature be the thing that crashes on a document I've never seen (M6S6 is literally a fresh document, so this fallback matters).

**Guard the other direction too:** if a section is enormous (say 10,000 characters), it's too big to send to the AI. So I cap parents at ~3000 characters and split oversized sections into `5.2 (part 1 of 3)`, still cutting on paragraph breaks.

---

### 📛 What is all this actually called? (naming it properly)

Useful to know, because "I built hierarchical chunking with small-to-big retrieval" lands very differently from "I split it up into bits."

| Name | What it means | Am I doing it? |
|---|---|---|
| **Fixed-size chunking** | Cut every N characters, ignore everything else | No — this is the lazy default I'm beating |
| **Recursive chunking** | Cut at the nicest available seam: paragraph → sentence → word | ✅ yes, for children |
| **Structure-aware chunking** *(a.k.a. document-based / layout-aware)* | Cut on the document's **own** structure — headings, `1.1`, `## markdown`, HTML tags | ✅ **yes — this is my section detection** |
| **Hierarchical chunking** | Keep chunks at **several levels** (document → section → paragraph) instead of one flat list | ✅ **yes — this is the umbrella term for what I built** |
| **Small-to-big retrieval** *(a.k.a. parent-child, parent document retrieval)* | Embed the small chunks, but **return the big ones** to the model | ✅ **yes — my retrieval strategy** |
| **Semantic chunking** | Cut where the *meaning* shifts, by comparing sentence embeddings | ❌ no — slower, needs embedding calls just to chunk. Nice-to-have |
| **RAPTOR** | Cluster chunks, have an LLM summarise each cluster, recurse into a tree of summaries | ❌ no — this is the heavy-duty version of hierarchical. Way beyond one day |

### ⭐ The important distinction: two separate decisions

This is the bit that shows real understanding, because these two are **orthogonal** — they solve different problems and you choose them independently:

1. **Hierarchical / structure-aware chunking** answers **"where do I cut?"**
   → *At real section headings.* A **chunking** decision, made once at ingest time.

2. **Small-to-big / parent-child** answers **"what do I embed vs. what do I send to the model?"**
   → *Embed children, send parents.* A **retrieval** decision, made at query time.

You could do either one alone:

- Parent/child with **dumb** size-based parents → still works, but the parents are arbitrary blocks that may start mid-topic.
- Hierarchical chunking with **flat** retrieval → nice boundaries, but you send the model whatever small piece matched, so it lacks surrounding context.

**I'm doing both, and they compound**: the hierarchy decides *where the parent boundaries are*, and small-to-big decides *which level gets used for what*. Structure gives me meaningful units and free citations; small-to-big gives me precise search with complete context.

### If my mentor asks "what's this technique called?"

> "**Hierarchical chunking** — I keep two levels rather than one flat list. Specifically it's **structure-aware** at the top level, because I derive parent boundaries from the document's own numbered headings instead of a character count, and then **small-to-big retrieval** at query time: children get embedded, parents get sent to the model. The heavier version of this idea is RAPTOR, which builds a tree of LLM-generated summaries — I didn't need that here, and it's expensive to build."

⚠️ One honest caveat if pushed: mine is a **two-level** hierarchy (section → slice). A stricter reading of "hierarchical" means a **deeper tree** — document → chapter → section → subsection → paragraph, with retrieval able to climb to whatever level fits. Mine is the shallow, practical version. Saying that unprompted is better than being caught by it.

---

### My final numbers

| | Size | Overlap | Stored in Chroma? | Sent to the AI? |
|---|---|---|---|---|
| **Parent** (a section) | ~2000–3000 chars | none needed | text kept as metadata | ✅ **yes** — this is what the AI reads |
| **Child** (a slice) | ~400 chars | 80 chars | ✅ **yes** — this is what gets embedded | ❌ no |

Parents don't need overlap, because a section is already a complete thought with its own natural boundary. Children need overlap for the reason below.

### What is overlap and why do I need it?

**Overlap** means each child repeats the last little bit of the child before it.

Without overlap, imagine the cut lands right in the middle of the sentence I need:

```
Child 1: "...Employees may carry forward unused leave"
Child 2: "up to a maximum of 5 days into the next year."
```

Neither child would match well on "how many days can I carry forward?" — the idea got sliced in two, so *both* halves have a fuzzy meaning-address. With 80 characters of overlap, child 2 *starts* by repeating the tail of child 1, so the full sentence survives inside one child and matches properly. It's a safety net for sentences that fall on a cut line.

Cost of overlap: slightly more storage, slight duplication. Worth it.

### ⚠️ Two traps in parent/child that I must not fall into

**Trap 1 — LangChain's `ParentDocumentRetriever` does NOT persist parents by default.**
It ships with an `InMemoryStore` for the parent text. That means **the parents vanish the moment I close the app**, and on restart my retriever finds children and can't resolve them to anything. That would quietly break my app *and* contradict M6S2 ("persisted, not in-memory"). 

**My fix, and it's the simple one:** store the **parent's full text directly in each child's ChromaDB metadata** (`parent_text`, `parent_id`, `section`, `page`, `source`). Chroma persists metadata to disk along with the vectors, so one folder holds everything and a restart is safe. It duplicates the parent text a few times — a few extra KB, completely irrelevant at this scale — and it removes an entire moving part. Worth it on a one-day build.

**Trap 2 — the same parent can win twice.**
If I fetch the top 8 children, three of them might come from the same section. If I don't check, I paste that section into the prompt **three times** — wasting context and making the AI think it's extra important. **So: fetch ~8 children → map them to parents → de-duplicate by `parent_id` → keep the top ~4 unique parents.**

### What I must be able to say out loud (M6S1)

> "I use **parent/child chunking**. Children are ~400 characters with 80 overlap, and they're the only thing I embed — small chunks give a precise meaning-match. But I never send a child to the model; I send its **parent**, which is the full document section, ~2000–3000 characters. Small for finding, big for answering.
>
> And parents aren't arbitrary character blocks — I **detect real section headings** (`5.2 Remote Work Policy`) by regex and use those as parent boundaries, because the document's author already decided where one topic ends. That also gives me section-level citations. If a document has no numbering, I fall back to ~2000-character paragraph-aligned parents.
>
> Then per content type: tables stay whole with their headers, lists stay whole, and image-only pages get flagged instead of silently dropped."

That answer covers chunk size, overlap, the tradeoff, *and* intentionality in about 40 seconds.

### 🔴 The big one: DIFFERENT CHUNKING FOR DIFFERENT KINDS OF CONTENT

My mentor specifically pointed this out, and it's what separates a default copy-pasted project from an intentional one. **A PDF is not just text.** It has paragraphs, tables, headings, lists, images. Chopping all of them the same way is lazy and it breaks things. Here's my rule for each type:

#### A. Normal paragraphs → split by size, respecting structure

Split into ~400-char children (inside their ~2500-char parent), **cutting at natural seams**, in this order of preference:
1. paragraph break (`\n\n`) — best
2. end of a sentence (`.`)
3. a space
4. only mid-word as a last resort

This is what LangChain's `RecursiveCharacterTextSplitter` does: it tries the nicest cut first and only gets uglier if it has to. Never chop mid-word if a paragraph break is nearby.

Same splitter, used **twice** with different settings — once coarse for parents (only when there are no headings to use), once fine for children. Same tool, two jobs.

#### B. Tables → 🚫 NEVER cut a table in half

This is the classic disaster. A table like:

| Role | Vacation Days | Sick Days |
|---|---|---|
| Junior | 15 | 10 |
| Senior | 20 | 12 |
| Manager | 25 | 15 |

If a size-based splitter cuts it after "Senior", then chunk 2 begins with a naked row `Manager | 25 | 15`. **The column headers are gone.** The AI sees three numbers with no labels, and now "25" could be days, dollars, or someone's age. That's how you get a confidently wrong answer.

My rules for tables:
- **Detect tables first**, before any splitting. I'll use `pdfplumber`'s `extract_tables()`, which finds them properly instead of guessing from spaces. I also **cut the table's text out of the page's plain text** afterwards, so the same table doesn't get stored twice (once as a proper table, once as leftover spaghetti).
- **🔒 One table = one chunk. Always. Never split.** A table is a single unit of meaning; the rows only mean anything *next to* their headers.
- **A table is its own parent, and it is its own only child.** No slicing into 400-character children — that would cut the rows off from the headers, which is exactly the disaster I'm avoiding. So the table gets embedded whole *and* sent to the AI whole. This is the one place where my parent and my child are the same thing, and it's a deliberate exception I can explain.
- If a table is genuinely enormous (bigger than the model can take), only then split it **by rows** — and **repeat the header row at the top of every piece**. Never lose the headers.
- **Convert tables to Markdown** (`| Role | Vacation |`) before embedding. Markdown keeps the row/column relationship readable; raw PDF table text collapses into meaningless spaces where you can no longer tell which number belongs to which column. It's also the format the AI reads best.
- **Add a caption line** on top: `"Table from HR_Policy.pdf, page 4, §3.1 Leave: entitlement by role"`. Now the chunk's meaning-address includes *what the table is about*, so a question about leave can actually find it. A bare grid of numbers embeds to nothing useful — nothing in `| Junior | 15 | 10 |` says "vacation".

#### C. Headings → glue them onto their content, and keep a breadcrumb

A heading like **"5.2 Remote Work Policy"** is only ~20 characters. On its own it's a useless chunk. But it's the single best clue about what the text under it is about.

So: every chunk carries its heading trail in the text and in its metadata, like
`Section: 5 > 5.2 Remote Work Policy`.
Two wins: better meaning-address for searching, and my citation can say **which section** it came from, not just a page number. That's directly worth marks (M6S4).

#### D. Lists → keep the list together

A numbered list of 8 onboarding steps is one idea. Cutting it after step 4 means someone asking "what are the onboarding steps?" gets half an answer and never knows it. So: if a bullet/numbered block fits, keep it whole, and keep the sentence that introduces it ("Follow these steps:") attached — otherwise the list is a bunch of orphan fragments.

#### E. Images / scanned pages → detect, and be honest

Embeddings only work on *text*. An image is not text.

- If a page has almost **no extractable text** (say under 100 characters) but is a full page, it's probably a **scan or a screenshot**. Real text-based PDFs give you plenty of characters.

**My decision, in priority order (I have one day, so this is a time-boxed call):**

**Plan A — the minimum, and I do this no matter what.** Detect the page, and produce a chunk that says so honestly:

> `⚠️ Page 7 of product_manual.pdf appears to be an image or scanned page (no extractable text). Its content could not be indexed. Please view the original document for this page.`

Two things this buys me. The ingest log tells me straight away which pages I'm blind to. And the UI can *show* the user "this document has 3 image-only pages" rather than pretending the document was fully absorbed. **A known gap is fine. A silent gap is not.**

**Plan B — only if the core pipeline is done and working.** Add **OCR** with `pytesseract` to read the words out of the picture, and tag those chunks `source_type: "ocr"` so I know they're less trustworthy (OCR misreads `5` as `S`, `0` as `O`). If OCR is on, citations for those chunks get an "OCR — may contain errors" note.

⚠️ **OCR is genuinely a trap on a one-day build**: `pytesseract` needs the *Tesseract binary* installed separately (`brew install tesseract`), plus `pdf2image`, which needs `poppler`. That's system-level installs that can eat an hour and fail. **So OCR is strictly last, after everything else works and is committed.** If I run out of time, Plan A is a perfectly good answer — arguably a *better* demo answer, because it shows I know where my system's limits are.

**Silently dropping content is the worst outcome.** The system looks like it works and quietly has a hole in it. Knowing about my own blind spot is exactly what M6A4 ("where does your system fail?") is asking about, and I get to answer it honestly.

#### F. Headers / footers / page numbers → strip them out

Every page repeats "Confidential — Acme Corp — Page 3 of 40". If I keep that, it's noise glued to every chunk, and it slightly poisons every meaning-address. My cleanup step: find lines that repeat on most pages and delete them.

### Chunking summary table

| Content type | Parent (what the AI reads) | Child (what gets embedded) | Why |
|---|---|---|---|
| **Paragraphs** | The detected section (`§5.2`), capped ~3000 chars | ~400 chars / 80 overlap, cut at paragraph → sentence → space | Precise search, complete answers |
| **Tables** | The whole table, as Markdown + caption | **same as parent — not sliced** | Rows are meaningless without their headers |
| **Headings** | Start a new parent; text glued to its content | Breadcrumb prefixed onto every child | Better search + section-level citations |
| **Lists** | Kept whole with its intro line | The whole list as one child if it fits | A split list is a silently half-wrong answer |
| **Images / scans** | A `⚠️ not indexed` placeholder chunk (or OCR text if time allows) | same | Embeddings can't see pictures — flag, never hide |
| **Headers / footers** | Deleted before anything else | — | Repeated noise pollutes every chunk |

---

## 5. What happens when I hit Enter? (walkthrough — M6A3 asks this exactly)

Two separate journeys. Don't mix them up.

### Journey 1 — Loading a document (happens once per PDF, slow, ~10–60 seconds)

```
PDF file
  ↓ 1. READ        pdfplumber pulls out text, tables, page numbers
  ↓ 2. CLEAN       strip repeated headers/footers, fix broken spacing
  ↓ 3. CLASSIFY    paragraph? table? heading? image-only page?
  ↓ 4. SPLIT INTO  detect section headings (1.1, 1.2, SECTION 4...) → each
       PARENTS     section is one parent. No headings found? → ~2000-char
                   paragraph-aligned parents instead. Tables = own parent.
  ↓ 5. SPLIT INTO  slice each parent into ~400-char children, 80 overlap.
       CHILDREN    (Tables skip this — the table IS its own child.)
  ↓ 6. TAG         every child carries: source file, page, section breadcrumb,
                   content_type, parent_id, and the FULL PARENT TEXT
  ↓ 7. EMBED       each CHILD → list of numbers.  Parents are never embedded.
  ↓ 8. STORE       child vectors + metadata → ChromaDB on disk (./chroma_db/)
Done. The PDF is now searchable forever. Never needs re-reading.
```

### Journey 2 — Asking a question (every time, fast, ~2–5 seconds)

```
"How many vacation days do I get?"
  ↓ 1. VALIDATE    empty? too long? gibberish? → handle nicely, don't crash
  ↓ 2. EMBED       question → numbers, using the SAME model as step 7 above
  ↓ 3. SEARCH      ChromaDB returns the top ~8 nearest CHILDREN + metadata
  ↓ 4. CHECK       are they actually close enough? all far away → "I don't know"
  ↓ 5. EXPAND      ⭐ children → their parents. De-dupe by parent_id.
       (small→big)    Keep the top ~4 unique parent SECTIONS.
  ↓ 6. PROMPT      "Using ONLY the context below, answer. If the answer isn't
                    there, say you don't know. Context: <the 4 parent sections>"
  ↓ 7. GENERATE    the LLM reads that and writes the answer
  ↓ 8. DISPLAY     answer + expandable citations:
                    📄 HR_Policy.pdf · page 4 · §3.1 Leave Entitlement
                       [▸ show the exact source text]
```

Step 5 is the whole parent/child idea in one line: **I searched with the small thing, then answered with the big thing.**

⚠️ **Step 2 must use the exact same embedding model as step 6 of Journey 1.** Different models draw different maps. Asking for directions using a map of Paris while your documents are pinned on a map of Tokyo — nothing will ever match. This is a real bug people hit; I'll set the model in one place in config so it can't drift.

---

## 6. My tech stack (and why each piece)

| Piece | What I'm using | What it does, in kid words |
|---|---|---|
| PDF reading | **pdfplumber** | Opens PDFs. Picked over PyPDF2 because it actually extracts *tables* properly, which I need for section 4B. |
| Chunking | **LangChain** splitters | Gives me the "cut at the nicest seam" splitter for free, plus a standard `Document` object with metadata baked in. |
| Embeddings | **OpenAI `text-embedding-3-small`** (or a local model — see §6b) | Turns text into meaning-numbers. Separate model from the answering one. |
| Vector DB | **ChromaDB** (persisted to `./chroma_db/`) | The meaning-library. Saves to disk so it survives restarts. |
| The answering AI | **OpenAI `gpt-4o-mini`** (via LangChain) | Reads the found chunks and writes the human answer. |
| API layer | **FastAPI** | The kitchen. Does the real work: `/upload`, `/ask`, `/health`. |
| UI | **Streamlit** | The dining room. Upload box, chat box, citations. Chat UI fits doc Q&A (M6C3). |
| Secrets | **.env** + `python-dotenv` | API key lives here. `.env` goes in `.gitignore`. I commit a `.env.example` with blank values instead. |

## 6b. Which embedding model? (and one trap it creates)

### ⚠️ First: one provider is doing two different jobs

My app needs **two different kinds of AI model**, and they are not interchangeable:

| Job | What it does | Model |
|---|---|---|
| **Embedding** (at ingest) | Text → meaning-numbers, so I can search by meaning | `text-embedding-3-small` |
| **Generation** (at question time) | Reads the found chunks → writes the human answer | `gpt-4o-mini` |

Both come from OpenAI, so **one API key covers both**. Worth knowing that this isn't always true — Anthropic's Claude, for example, has **no embeddings endpoint at all**, so a Claude-based build needs a second provider just for embeddings. Mine doesn't. One key, one provider, one failure mode to handle.

### 🔴 Open question: my key is a gateway key, not a plain OpenAI key

My key was **rejected by `api.openai.com` with `401 invalid_api_key`**, and it doesn't start with `sk-`. That means it's almost certainly a **company gateway key** — an internal proxy that speaks the OpenAI API but lives at a different address.

**Three things I need to find out** from whoever issued the key:

1. The **base URL** (e.g. `https://<gateway>/v1`)
2. The exact **model names** it exposes — gateways often rename them (`gpt-4o-mini` might be `gpt4o-mini` or an Azure deployment name)
3. Whether it's **OpenAI-compatible** — if yes, `langchain-openai` works unchanged with just `base_url` set

That's why `config.py` reads **both** `OPENAI_API_KEY` and `OPENAI_BASE_URL` from `.env`. When `OPENAI_BASE_URL` is blank the code talks to OpenAI directly; when it's set, everything routes through the gateway. **One variable, no code change.** Writing it this way now costs nothing and means the answer, whenever it arrives, is a one-line `.env` edit rather than a refactor.

### 🅱️ Plan B if the gateway never materialises

**The important realisation: most of this project needs no API key at all.**

| Piece | Needs a key? |
|---|---|
| PDF parsing, cleaning, table extraction | ❌ no |
| Section detection, parent/child chunking | ❌ no |
| `test_chunker.py` | ❌ no |
| Embedding + ChromaDB storage | ⚠️ only if hosted — a **local** model needs none |
| Retrieval + citations | ❌ no (once embedded) |
| **Writing the answer in plain English** | ✅ **yes — this is the only part** |

So my build order already protects me: Block 3 (the chunker, where most of the marks are) is entirely offline. If the key stays broken I swap the embedding model to **ChromaDB's local built-in** (`all-MiniLM-L6-v2`, no key, no network) and I still have a working, demoable retrieval system with real citations. Only the final answer-writing step would be missing.

⚠️ **The local model's catch:** it reads only the **first ~256 tokens (~1000 chars)** of a chunk and silently discards the rest — which collides with my "a whole table is one chunk" rule, because a big table would lose its bottom half with no error. If I fall back to local, the table caption **must** be the chunk's first line, and I must log a warning on any chunk over ~1000 characters. `text-embedding-3-small` takes 8191 tokens and has no such problem — that window is the main reason I prefer it.

### Why `text-embedding-3-small` when it works

| | Local `all-MiniLM-L6-v2` | **`text-embedding-3-small`** |
|---|---|---|
| **Input limit** | ~256 tokens (~1000 chars) | **8191 tokens** ⬅️ the reason |
| Retrieval quality | Decent | Noticeably better |
| Dimensions | 384 | 1536 |
| Cost | Free | ~$0.02 / 1M tokens |
| Needs network | No | Yes |

**Cost is a rounding error:** 5 PDFs × ~40 pages × ~500 tokens ≈ 100,000 tokens ≈ **$0.002**. Fifty re-ingests while debugging is ten cents. Cost is not a reason to avoid it.

Better retrieval also lands directly on a graded line: **M6S3** is the mentor asking three questions from a document they've read and checking whether my citations match.

### Why `gpt-4o-mini` for the answering step

The generation job here is deliberately easy — the model isn't recalling facts, it's rewriting 4 paragraphs I already handed it. `gpt-4o-mini` is cheap, fast, and entirely capable of that. `gpt-4o` is the upgrade if answers come out weak, but I should suspect **retrieval** before blaming the model: if the right chunk never got fetched, no model can save the answer.

**`temperature=0`** on every call, so the same question gives the same answer (M6B1).

### Names to know but not use

**Voyage AI** is the embedding provider usually paired with Claude. **`text-embedding-3-large`** (3072 dims) beats `-small` but costs ~6× more — unnecessary at 5 documents.

### 🔒 The rule that must never break

Whatever I pick, **the ingest side and the query side must use the same model** — set once in `config.py`, never hardcoded at a call site. Different models draw different maps, and nothing will ever match (§5, §11.1). If I ever switch models, **I must delete `chroma_db/` and re-ingest** — the old vectors are from the old map and are worthless.

---

### Why FastAPI *and* Streamlit, instead of just Streamlit?

Streamlit alone could technically do it. But splitting them means the brain (retrieval + answering) is separate from the face (buttons). So a mobile app or Slack bot could use the same `/ask` endpoint later, and I can test the logic without clicking through a UI. It also matches the required stack.

### ⚠️ Python version note

I have **Python 3.14**, which is very new. ChromaDB and some LangChain packages often lag behind on the newest Python and may fail to install. **First thing I'll do:** try installing into a venv. If it breaks, I'll create the venv with **Python 3.11 or 3.12** instead — the well-supported versions. Better to find this out in 5 minutes than to lose a day to install errors.

---

## 7. Planned folder layout

```
Odyssey Port 6/
├── plan.md                    ← this file
├── README.md                  ← how to run it (a mentor must follow it cold — M6E2)
├── requirements.txt
├── .env.example               ← OPENAI_API_KEY= / OPENAI_BASE_URL= (blank, safe to commit)
├── .env                       ← real key. GITIGNORED. never committed.
├── .gitignore
│
├── documents/                 ← my 5+ test PDFs
├── chroma_db/                 ← the persisted vector DB (gitignored, it's data)
│
├── app/
│   ├── config.py              ← ALL settings in ONE place: parent/child sizes,
│   │                            overlap, model names, top_k, distance threshold
│   ├── pdf_parser.py          ← Journey 1 steps 1–3: read, clean, classify,
│   │                            detect tables, detect image-only pages
│   ├── chunker.py             ← ⭐ THE CENTREPIECE. Journey 1 steps 4–6:
│   │                            section detection → parents → children,
│   │                            plus the per-content-type rules
│   ├── vector_store.py        ← Journey 1 steps 7–8 + Journey 2 steps 3+5:
│   │                            ChromaDB, and the child→parent de-dupe expand
│   ├── rag_chain.py           ← Journey 2 steps 6–8: prompt + LLM + citations
│   └── main.py                ← FastAPI endpoints: /upload /ask /health /stats
│
├── streamlit_app.py           ← the UI
├── ingest.py                  ← CLI: load every PDF in documents/ into ChromaDB
└── tests/
    └── test_chunker.py        ← proves (a) a table never gets split, and
                                 (b) a child always resolves to its parent
```

One idea per file. A mentor should understand each in under 2 minutes (M6E1).

Everything tunable lives in `config.py` — **especially the embedding model name**, so the ingest side and the query side physically cannot drift apart (the map-of-Paris bug from section 5).

---

## 8. Build order — I have ONE DAY ⏰

### The one rule for today

**Get a working end-to-end answer FIRST, then make it good.** An ugly pipeline that answers one question with a citation is worth infinitely more than a beautiful chunker with no UI attached. So I build the thinnest possible full path early, then improve each piece in place.

**Commit after every block.** Roughly 10 commits with real messages ("add section-aware parent chunking", "handle empty query") beats one giant `final` commit. On the two-week question (M6D3) I just tell the truth — I had a day, and here's how I sequenced it. Honesty scores better under "Learning Demonstrated" than a faked history, and faked timestamps are obvious anyway.

---

### 🌅 Block 1 · Setup (~45 min)

- venv + install. **Resolve the Python 3.14 question in the first 10 minutes** — if `chromadb` or `langchain` won't install, immediately rebuild the venv on **Python 3.11/3.12** and move on. Don't debug it, just switch.
- `.gitignore` (with `.env`, `chroma_db/`, `venv/`) and `.env.example` **before the first commit**, so a key can never enter git history.
- Drop 5+ PDFs into `documents/`. **Deliberately include one with a real table and one that's a scan/image-heavy** — otherwise my best work has nothing to prove itself against.
- `git init`, first commit.

### ☀️ Block 2 · Thin end-to-end spike (~2 hr) — the most important block of the day

Cheat deliberately here. **Dumb chunking on purpose**: plain `RecursiveCharacterTextSplitter`, no parents, no sections, no tables.

`pdf_parser.py` → naive chunks → ChromaDB (persisted) → retrieve top 4 → prompt the LLM → print answer + filename in the terminal.

By lunch, **the whole path works**. Every remaining block is now an upgrade to a working system rather than a bet that it will come together. If I run out of time later, I still have something to demo. *Commit.*

### 🍜 Block 3 · The real chunker (~2.5 hr) — where the marks are

Now upgrade the middle. In `chunker.py`:

1. **Section detection** — regex for `1.` / `1.1` / `SECTION 4` / ALL-CAPS lines → parent boundaries + breadcrumbs. **With the size-based fallback** when a doc has fewer than ~3 headings.
2. **Parents → children** — ~2000–3000 char parents, ~400/80 children, parent text into child metadata.
3. **Tables** — `extract_tables()`, whole table = one chunk = its own parent and child, Markdown + caption, and remove the table text from the plain-text stream so it isn't stored twice.
4. **Image-only pages** — detect, emit the `⚠️ not indexed` placeholder, log a warning.
5. **`test_chunker.py`** — two tests: a table never splits, and every child resolves to a parent. These are quick and they're what I point at when asked "is your chunking intentional?"
6. **Child→parent de-dupe on retrieval** in `vector_store.py`.

Re-ingest, ask the same questions as Block 2, and **see the answers get better**. That comparison is a great demo moment. *Commit (2–3 commits).*

### 🌇 Block 4 · FastAPI + Streamlit (~2 hr)

- `main.py`: `/upload`, `/ask`, `/health`. Keep it thin — the logic already exists.
- `streamlit_app.py`: upload box, chat, and **expandable citation cards showing file · page · §section · exact source text**. Citations are a graded deliverable, so this is not the place to cut corners.
- Show the ingest warnings in the UI ("3 image-only pages were skipped"). *Commit.*

### 🌃 Block 5 · Break it on purpose (~1 hr) — cheap, easy marks

Straight down the rubric. Test each, fix each:

- Empty query → friendly "please type a question", not a crash (M6B2)
- 10,000-word query → truncate with a notice
- A question in Hindi, and one that's pure emoji
- **Wrong API key** → friendly red box, never a stack trace (M6B3)
- Deleted `chroma_db/` → "no documents indexed yet, upload one"
- Uploading a `.txt` or a corrupt PDF → rejected politely
- `temperature=0` set, then **ask the same question twice** and diff it (M6B1)

*Commit.*

### 🌙 Block 6 · Anti-hallucination + fresh doc (~45 min)

- Ask ~8 out-of-scope questions ("who won the World Cup?", "what's the capital of Peru?"). Tune the prompt and the distance threshold until it reliably says **"I don't know, that's not in these documents."** (M6S5 — this is 5% of my grade in one line of prompt.)
- Then **ingest a PDF I have never used** and run 3 questions on it. This is literally the M6S6 test, so rehearsing it means no surprises. *Commit.*

### 🛏️ Block 7 · README + rehearse (~45 min)

- README: install → `.env` → `ingest.py` → run. Then **follow it cold in a clean folder** — this is the single most commonly broken deliverable (M6E2).
- Read section 10 out loud twice. Most of my score is *explaining*, not code.

### If I'm running out of time, cut in this order

1. ❌ OCR (Plan A message is a fine answer)
2. ❌ `/upload` in the UI — pre-ingest from `documents/` and demo the chat only
3. ❌ FastAPI — put the logic straight in Streamlit *(costs a little on stack-compliance, but a working app beats a broken split)*
4. ❌ Extra polish, confidence scores, chat memory

### 🚫 Never cut these — they're graded directly

- Persisted ChromaDB (M6S2)
- Citations on **every** answer with document + section (M6S4)
- "I don't know" on out-of-scope (M6S5)
- Intentional, content-type-aware chunking I can explain (M6S1)
- `.env`, no hardcoded keys (M6B5, M6E3)

---

## 9. Things that will go wrong (and my plan for each)

Being able to name my own weak spots is worth real marks (M6A4, M6D4). Honest > polished.

| What breaks | Why | What I do |
|---|---|---|
| **Scanned PDFs give empty text** | It's a picture, not text | Detect, warn loudly, optionally OCR, tag as low-confidence |
| **Tables lose their headers** | Naive size-splitting | Never split tables; repeat headers; Markdown format |
| **Section detection finds nothing** | Loads of PDFs have no numbering at all | Automatic fallback to ~2000-char paragraph-aligned parents. **Must test this on a messy doc**, or a fresh document could break the demo |
| **Section detection finds too much** | A page of ALL-CAPS text, or `3.5 kg` looking like a heading | Require headings to be short lines; sanity-check the parent count. If a doc yields 400 "sections", something's wrong — fall back |
| **A parent is enormous** | One section runs 10,000 chars | Cap at ~3000 and split into `§5.2 (part 1 of 3)` on paragraph breaks |
| **The same section is pasted 3× into the prompt** | 3 children from one parent all rank highly | De-dupe by `parent_id` before building the prompt |
| **"Compare A and B" questions** | The answer needs 2 far-apart chunks and I only fetch the nearest few | Know the limit. Say so honestly. Fetching more chunks helps a little, not fully. |
| **"How many X in total?"** | Counting needs *all* chunks, retrieval gives me 4 | Genuinely a weak spot for RAG. Name it, don't hide it. |
| **Two documents disagree** | Old policy + new policy both loaded | Cite both, flag the conflict, let the human decide. Don't silently pick one. |
| **Confidently wrong answer** | The AI fills gaps when context is thin | Strict prompt: answer only from context. Show the source text so the user can verify me. |
| **Question uses words the doc never uses** | Embeddings are good at synonyms, not perfect | Note it; a hybrid keyword+vector search would be the fix (see section 11) |
| **API key invalid / no internet** | It happens | try/except around every API call → friendly Streamlit error, never a red stack trace (M6B3) |
| **Same question, different answer** | LLMs are random by default | **Set `temperature=0`.** Makes it as repeatable as possible. M6B1 literally runs the same input twice 5 min apart and compares. |

---

## 10. Questions my mentor will ask, and my honest answers

**"Explain RAG like I'm a PM."**
> Open-book exam instead of a memory test. I find the right pages first, hand only those to the AI, and make it answer from them — with a citation so you can check my work.

**"Why ChromaDB and not keyword search?"**
> Keyword search matches letters. If the doc says "annual leave entitlement" and you type "vacation days", Ctrl+F returns nothing even though the answer is right there. Embeddings match *meaning*, so synonyms still hit. Chroma stores those meaning-vectors and searches them fast, and it persists to disk so I don't re-process PDFs on every restart.

**"Walk me through submit → answer."**
> Validate the input, embed the question with the same model I embedded the docs with, ask Chroma for the nearest 4 chunks, check they're actually close enough (if not: "I don't know"), paste them into a strict prompt, the LLM writes the answer, UI shows it with page-and-section citations you can expand.

**"What chunk size and why?"** ← the big one, answer it like this
> "Two sizes, because searching and answering want opposite things. **Children are ~400 characters with 80 overlap and they're the only thing I embed** — small means a sharp, precise meaning-match. But I never send a child to the model; I send its **parent**, the full section, ~2000–3000 characters. Small for finding, big for answering — small-to-big retrieval.
>
> And my parents are **real document sections**, not arbitrary character counts. I regex out headings like `5.2 Remote Work Policy` and cut there, because the author already decided where one topic ends — and it gives me section-level citations for free. If a document has no numbering, I fall back to ~2000-character paragraph-aligned parents.
>
> Overlap on the children catches facts that straddle a cut line. And that's just prose — tables stay whole with their headers because rows are meaningless without them, lists stay intact, and image-only pages get a visible 'not indexed' flag instead of being silently dropped."

**"Which embedding model, and why that one?"**
> "Two models, both from OpenAI: `text-embedding-3-small` to embed, `gpt-4o-mini` to answer. They're different jobs — embedding turns text into searchable numbers, generation writes the prose — and it's worth knowing one provider doesn't always cover both: Claude has no embeddings endpoint at all, so a Claude build needs a second provider just for vectors. Mine needs one key.
>
> I chose `-small` for the **input window**, not the benchmark score. The local alternative reads only ~256 tokens and silently drops the rest, which would break my rule that a table is never split — a big table would lose its bottom half with no error. `-small` takes 8191 tokens, so my chunking strategy and my embedding model stop contradicting each other. Better retrieval quality is a bonus, and it happens to be what M6S3 grades.
>
> Cost isn't a factor: my whole corpus is ~100,000 tokens, so about two-tenths of a cent to ingest.
>
> `gpt-4o-mini` for answering because that step is easy — the model isn't recalling anything, it's rewriting four paragraphs I already handed it. If answers come out weak, I'd suspect retrieval before the model."

**"Show me a case where parent/child actually mattered."**
> "Ask it 'can I work from home on Fridays?'. The best-matching child is one sentence — *'Fridays are designated as optional remote days'* — which alone would give you a technically-true but misleading 'yes'. Because I expand to the parent section, the model also sees *'subject to written manager approval and a minimum of 3 office days per week'*, so the answer is complete. One medium-sized chunk would likely have caught one half or the other."

**"Why did you duplicate the parent text into the metadata?"**
> "LangChain's `ParentDocumentRetriever` keeps parents in an `InMemoryStore` by default, so they'd disappear on restart — my children would find nothing to expand into, and it'd contradict the whole point of persisting. Putting the parent text in the child's Chroma metadata means one folder holds everything and a restart is safe. It costs a few KB of duplication and removes a moving part."

**"Your commit history is one day, not two weeks."**
> "Correct — I had one day for this. I sequenced it as a thin end-to-end spike first so I always had something working, then upgraded the chunker, then hardened the edge cases, committing at each step. If I'd had the full two weeks the first thing I'd have added is an evaluation set."

**"Where will it be wrong?"**
> Scanned PDFs — no text to embed. Counting and comparison questions — retrieval gives me a few chunks, not the whole corpus. And if two documents contradict each other, I cite both rather than pretending there's one answer.

**"Who uses this and what does it save them?"**
> New joiners and anyone in HR/support. Today they either read 40 pages or interrupt a colleague. This turns a 20-minute hunt into a 10-second question — and the citation means they can trust it, which a plain chatbot can't offer.

**"What was hardest?"**
> Tables. My first splitter cut one in half and the second chunk was rows with no column headers — so the AI saw "25" with no idea it meant vacation days. Fixed it by detecting tables before splitting, keeping them whole, and re-printing the header row on every piece.

**"What would you do differently?"**
> Build the evaluation set first. I tuned chunk size by eyeballing outputs, which is slow and subjective. With 20 question/expected-answer pairs written up front I could have measured each change instead of guessing.

**"What are you least confident about?"**
> Picking the "I don't know" threshold. It's a number I tuned by feel — too strict and it refuses questions it could answer, too loose and it answers from irrelevant chunks. I'd want real usage data to set it properly.

---

## 11. If I have extra time (nice-to-haves, only after the core works)

- **Hybrid search** — combine keyword + vector. Catches exact things like product codes ("Model X-42") that embeddings are surprisingly bad at.
- **Re-ranking** — fetch 20 chunks, then have a small model pick the best 4. Notably better retrieval.
- **OCR** for scanned pages (`pytesseract`).
- **An eval set** — 20 question/expected-answer pairs, so I can measure whether a change helped instead of vibe-checking.
- **Show a confidence score** next to each answer.
- **Chat memory** so follow-ups like "and for managers?" work.

---

## 12. My definition of done

Not done until **all** of these are true:

- [ ] 5+ PDFs parsed, including one with a real table and one image-heavy one
- [ ] Chunking is **visibly different** per content type, and I can point at the code that does it
- [ ] **Parent/child working**: children embedded, parents sent to the model, de-duped by parent
- [ ] **Section detection working** on at least one document, **with the fallback proven** on a document that has no numbering
- [ ] A table is provably **never split** (there's a test)
- [ ] Image-only pages produce a **visible warning**, not silence
- [ ] Embeddings **persist** in `./chroma_db/` — I can kill the app, restart, and query without re-ingesting (and parents survive too)
- [ ] Every answer shows document + page + section
- [ ] Out-of-scope questions get **"I don't know"**, tested on 10 of them
- [ ] Empty / very long / non-English input all handled without a crash
- [ ] Invalid API key shows a friendly message, not a stack trace
- [ ] No secrets anywhere in the repo; `.env` is gitignored, `.env.example` is committed
- [ ] `temperature=0` so the same question gives the same answer
- [ ] README works when followed cold in a clean folder
- [ ] Works on a brand-new PDF I've never tested
- [ ] ~10 real commits across the day, not one `final` dump
- [ ] I can answer every question in section 10 without notes

---

## 13. What actually got built (written after the fact)

The plan above was written before any code existed. This section records where
reality differed, because that is the interesting part.

### Changed from the plan

| Plan said | What happened | Why |
|---|---|---|
| Embeddings via OpenAI `text-embedding-3-small`, answers via `gpt-4o-mini` | **Exactly that in the end -- but only after a detour through `nomic-embed-text` and `llama3.2` on Ollama** | The key appeared not to work, so both models moved local rather than lose build time to debugging credentials. Once it was working, both moved back. The detour paid for itself: running two different embedding models over the same corpus is what exposed that `MAX_DISTANCE` is a property of the model, not of the app -- see the row below and bug 6. The Ollama path is still supported and still scores 100% on the eval; README documents it as the free, no-key option. |
| Vector search only | **Vector + exact-identifier search** | Measured: identifier questions like "what does Table 2-2 show?" scored 62% Hit@1 with vectors alone, 100% with a literal-match pass. |
| One `MAX_DISTANCE` constant | **Model-dependent, chosen from a sweep** | The number is a property of the embedding model. Switching models broke 8 tests because every distance shifted. |
| "I don't know" from distance alone | **Distance + a named-entity check** | Distance cannot tell "topically close" from "answers the question". A GDPR question scored 0.36 against a document that never mentions GDPR. |
| No evaluation planned | **`eval/run_eval.py`, 31 ground-truth questions** | Without it, every tuning decision was guesswork. This should have been built first. |
| One API key for everything | **Separate credentials per job** | Sharing one variable meant configuring a local chat model overwrote the embedding key and silently disabled it. Embeddings and chat are different jobs and may use different providers. |
| One relevance threshold | **Keyed by embedding model name** | The number is a property of the model, not the app: `nomic-embed-text` needs 0.375 and `text-embedding-3-small` needs 0.52. Running one at the other's value refused 10 of 23 answerable questions. |
| IRS tax publications as test documents | **HR policy, lab safety SOP, product manual, compliance standard** | The first set worked technically but told the wrong story: the brief is about a company document library. |

### The bugs worth remembering

1. **Heading detection found 120 headings in two documents, and essentially all
   were wrong** — page footers, numbered list items, wrapped fragments. Fixed with
   a Title-Case test. This took the most iterations of anything in the project.
2. **Citations pointed at page 1 for text on page 30**, because a parent inherited
   its heading's page rather than tracking pages per line.
3. **Sibling sections nested under one another** (`3.1 Access Control > 3.5
   Identification`) because the breadcrumb truncated by list position instead of
   heading depth.
4. **The model cited `[50]`** — a reference marker copied out of the NIST text,
   because our citations used the same `[n]` format the documents use.
5. **A line-based parent splitter could not split a single 8800-character
   paragraph.** Caught by an existing test the moment the change was made.
6. **Switching embedding model silently broke retrieval.** The threshold was
   still the previous model's, so the app answered "I don't know" to 10 of 23
   questions the documents clearly answer. Nothing crashed and no test failed --
   only the eval caught it. This is the single best argument for having built it.

7. **The literal-match pass matched a phone number.** Chroma's `$contains` is a
   plain substring filter, so asking "tell me about section 3.5" returned
   `3.5mm from module edge` (a screw-hole dimension), a current rating of `3.5`,
   and `206.543.5677` -- three different documents, none of which has a section
   3.5. Found only because someone asked what happens when the question is broad.
   Fixed by re-checking each hit for the boundaries a real identifier has. Still
   imperfect: a bare `3.5` alone in a table cell is a whole token, so telling it
   from a section number needs the surrounding structure, not the string.

8. **An identifier question dragged in the neighbouring identifier.** "What does
   control 03.05.03 require?" also retrieved `03.04.03 Configuration Change
   Control` at distance 0.4331 -- inside the threshold, because NIST controls are
   near-identical in shape and one digit barely moves an embedding. Given both, a
   small model can blend them into an answer that is wrong and correctly cited,
   which is the worst failure shape here. Now an identifier question returns only
   its exact matches; measured first, so the rule provably cannot reach the 23
   ordinary or 14 refusal questions, none of which produce an exact match at all.

### What I would still do differently

- **Build the eval set first.** Everything before it was tuned by eye, and two
  decisions (chunk size, threshold) had to be revisited once numbers existed.
- **Choose the documents for the demo story first.** Rebuilding the index around
  the right corpus late meant re-running everything.

### OCR (built after the fact)

The plan listed OCR as "only if I have time", with a warning that the system
installs could eat an hour. In the end it took about two hours, and the install
was a non-issue -- page rendering was already available through `pypdfium2`, a
pdfplumber dependency, so no `poppler` was needed.

Two bugs in my own OCR code, both found by measuring rather than assuming:

1. **Joining every word with a space destroyed the page structure.** The chunker
   cuts on paragraph and line breaks first, so a page flattened to one line could
   not be split at all. Tesseract reports block, paragraph and line numbers per
   word; I had discarded them.
2. **One chunk holding six unrelated facts blurs its own embedding.** Measured on
   a scanned notice: the query "when must employees collect their laptops?" scored
   0.582 against the whole page (refused) but 0.229 against just the sentence that
   answers it. A dilution cost of 0.353 -- more than the entire threshold margin.
   Fixed by splitting OCR text on the page's own paragraph breaks, the same
   principle as using headings for prose.

The second is the more interesting finding: it is the parent/child design being
justified by numbers on a case the plan never anticipated.
