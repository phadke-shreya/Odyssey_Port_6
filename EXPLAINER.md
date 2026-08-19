# SmartDoc explained simply

For understanding **what** this project does, **why** each tool was chosen, and
**what happens** when you press Ask. For the code itself, see
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

---

## 1. What it is, in one breath

A company has hundreds of pages of PDFs — HR policies, manuals, standards. Finding
one fact means reading forty pages or interrupting a colleague.

**This app is a robot librarian.** You ask a normal question. It finds the right
paragraphs, writes a short answer, and **shows you exactly which document, page
and section it used** so you can check it yourself.

And if the answer isn't in the documents, it says **"I don't know"** instead of
inventing something.

---

## 2. The trick: RAG (Retrieval Augmented Generation)

> **Retrieval** = go find the right pages
> **Augmented** = staple those pages onto the question
> **Generation** = the AI reads only those pages and writes the answer

**Why not just ask ChatGPT?** It has never seen your company's HR policy. It will
guess, confidently and wrongly.

**Why not paste the whole PDF in?** The AI has a limited reading window, it costs
money, and it gets lost. So: find the 4 best sections first, show only those.

**The one-line version:** *RAG gives the AI an open-book exam instead of a memory
test.*

---

## 3. How a computer "finds the right pages"

### The old way: Ctrl+F

Ctrl+F matches **letters**. If the document says *"annual leave entitlement"* and
you type *"vacation days"*, you get zero results — even though the answer is right
there.

### The new way: embeddings

An **embedding** turns text into a long list of numbers, like
`[0.12, -0.88, 0.34, ...]`. Think of it as **an address on a map of meaning**:

- *"vacation days"* and *"annual leave"* land in almost the **same place**
- *"vacation days"* and *"fire extinguisher"* land **far apart**

To answer a question, we turn the *question* into numbers too, drop it on the map,
and grab whatever is standing nearest. **That is why it beats Ctrl+F: it
understands synonyms.**

### Where the numbers live

A **vector database** — ours is **ChromaDB**. It's a library where things are
shelved by *what they are about* rather than alphabetically, so "give me the 8
nearest things" is one fast lookup instead of comparing 2600 items one by one.

---

## 4. The clever bit: parent and child chunks

Searching and answering want **opposite** things:

- **Small pieces are better for finding.** One idea per piece = a sharp address.
- **Big pieces are better for answering.** The AI needs the surrounding context.

So we don't choose. We use **two sizes**:

| | Size | Embedded? | Sent to the AI? |
|---|---|---|---|
| **Child** | ~400 characters | ✅ yes | ❌ no |
| **Parent** (a whole section) | ~3000 characters | ❌ no | ✅ yes |

We search the children, then **hand the AI the parent** the winning child came
from. This is called **small-to-big retrieval**.

**Why it matters — a real example.** Ask *"can I work from home on Fridays?"*

- The best-matching child is one sentence: *"Fridays are designated as optional
  remote days."* On its own that means **yes**.
- Expand to the parent section and it also contains: *"...subject to written
  manager approval and a minimum of 3 office days per week."*
- The answer is now **complete** instead of technically-true-but-misleading.

### Where a parent starts and stops

Not at an arbitrary character count. **At the document's own headings** —
`3.6 Incident Response`, `5.2 Remote Work Policy`. The person who wrote the
document already decided where one topic ends, and that beats any guess. It also
gives citations that name a **section**, not just a page.

If a document has no numbering, we fall back to splitting by size. A nice feature
must never be the thing that breaks on an unfamiliar document.

### Different content, different rules

| Content | Rule | Why |
|---|---|---|
| **Paragraphs** | ~400-char children inside section parents | precise search, complete answers |
| **Tables** | 🔒 **never split** — one table is one chunk | a row without its column headers is meaningless numbers |
| **Headings** | become the citation breadcrumb | tells the reader where they are |
| **Scanned pages** | flagged as unreadable, not dropped | a known gap is fine; a silent one is not |
| **Headers/footers** | deleted | repeated noise would pollute every chunk |

---

## 4b. Two searches, not one

Embeddings understand **meaning**, and that is exactly why they are bad at
**codes**. `Table 2-2` and `03.05.03` mean almost nothing as text, so their
"address on the map of meaning" is close to noise. Ask *"what does Table 2-2
show?"* and pure vector search returns nothing useful.

So the app runs **two** searches and merges them:

| Pass | Finds | Good at |
|---|---|---|
| **Vector** | things that *mean* the same | synonyms, paraphrases, concepts |
| **Exact identifier** | things that *literally contain* `Table 2-2` | codes, policy numbers, part numbers, pin names |

Measured on this corpus, identifier questions went from **62% correct to 100%**.

**The safety catch.** A literal search must not become a back door. If any number
counted as an identifier, then *"Who won the World Cup in 2022?"* would look up
`2022`, find it somewhere, and answer an out-of-scope question. So an identifier
must be **structured** — it must contain `.`, `-`, `#`, or mix letters with digits.
`03.05.03` qualifies; a bare `2022` does not.

## 4c. Two guards on "I don't know"

**Guard 1 — distance.** If nothing is close enough, refuse. Measured, not guessed
— and the number **belongs to the embedding model, not the app**:

| Model | In-scope tops out | Nearest excluded | Threshold |
|---|---|---|---|
| `nomic-embed-text` | 0.245 | 0.435 | 0.375 |
| `text-embedding-3-small` | 0.479 | 0.562 | 0.52 |

Those are 40% apart on the same corpus. Using one model's threshold with the
other made the app answer "I don't know" to **10 of 23 questions the documents
genuinely answer** — a silent failure caught only because the eval existed.

**Guard 2 — named entities.** Distance alone cannot tell *"topically close"* from
*"actually answers"*. Asking about **GDPR** scored 0.36 against a document about
US compliance that never mentions GDPR — close enough to pass, but not an answer.
So: if a question names an acronym (GDPR, HIPAA) that appears **nowhere** in what
was retrieved, refuse.

Deliberately narrow — only acronyms. Requiring ordinary words to match would
break the entire point of embeddings, which is finding *"annual leave"* when
someone asks about *"vacation days"*.

---

## 5. Why each piece of the tech stack

| Tool | Job | Why this one |
|---|---|---|
| **pdfplumber** | Read PDFs | It extracts **tables as tables**. PyPDF2 gives you text soup where you cannot tell which number belongs to which column. |
| **LangChain** | Split text, talk to the model | Its splitter cuts at the *nicest available seam* (paragraph → sentence → space → mid-word only as a last resort). It also gives one interface for any OpenAI-compatible model, which is why swapping OpenAI for a local model was a config change, not a rewrite. |
| **ChromaDB** | Store and search the numbers | Persists to a folder on disk, so restarting doesn't lose the work. Understands "nearest neighbour" natively. Runs in-process — no separate database server to install. |
| **OpenAI-compatible API** | Embeddings + answers | A standard shape that OpenAI, company gateways, and local servers (Ollama) all speak — so one code path serves all three. Embedding and chat have **separate** credentials, so you can pay for one and run the other free. |
| **FastAPI** | The brain, as HTTP | Separating logic from screen means a Slack bot or mobile app could use the same `/ask` endpoint. Also gives free interactive docs at `/docs`. |
| **Streamlit** | The screen | Chat-style interface in very little code, which is the right shape for document Q&A. |
| **pytest** | Prove it works | 126 tests. Most exist because a real bug happened and must not come back. |
| **eval harness** | Prove *retrieval* works | 31 ground-truth questions plus out-of-scope and near-miss sets. Turns "retrieval seems good" into a number that can be compared before and after a change. |
| **black + ruff** | Format and lint | Consistent style with no arguing about it. |

### Why FastAPI *and* Streamlit, rather than only Streamlit?

Streamlit alone could do it. Splitting them means the **brain** (finding and
answering) is independent of the **face** (buttons). The logic is testable without
clicking anything, and another product could reuse it. It's also what the brief
asked for.

---

## 6. What happens when you press Ask

### Journey 1 — loading a document (once per PDF, ~5 seconds)

```
PDF file
 ↓ 1. READ      pdfplumber pulls out text, tables, page numbers
 ↓ 2. CLEAN     strip repeated headers/footers
 ↓ 3. CLASSIFY  paragraph? table? image-only page?
 ↓ 4. PARENTS   detect headings -> one parent per section
 ↓ 5. CHILDREN  slice each parent into ~400-char children
 ↓ 6. TAG       each child carries: file, page, section, parent's full text
 ↓ 7. EMBED     each CHILD becomes numbers (parents never are)
 ↓ 8. STORE     numbers + text + tags -> ChromaDB on disk
Done. Searchable forever, no re-reading needed.
```

### Journey 2 — asking a question (~2–10 seconds)

```
"What are the rules for remote access?"
 ↓ 1. VALIDATE  empty? too long? -> friendly message, no crash
 ↓ 2. EMBED     question -> numbers, using the SAME model as step 7
 ↓ 3. SEARCH    ChromaDB returns the 8 nearest CHILDREN
 ↓ 4. CHECK     all too far away? -> "I don't know", and STOP HERE
 ↓ 5. EXPAND    children -> their parents, de-duplicated, keep best 4
 ↓ 6. PROMPT    "Answer using ONLY these sources..." + the 4 parents
 ↓ 7. GENERATE  the model writes the answer
 ↓ 8. CLEAN     strip anything the model formatted wrongly
 ↓ 9. DISPLAY   answer + expandable citations
```

**Step 4 is the anti-hallucination guarantee.** If nothing is close enough, the
model is **never called**, so it has no opportunity to invent anything.

⚠️ **Step 2 must use the same model as step 7 of Journey 1.** Different models
draw different maps — asking for directions with a map of Paris while your
documents are pinned to a map of Tokyo. That is why the database records which
model built it and refuses to answer on a mismatch.

---

## 7. Where this system is weakest

Knowing this is part of understanding it. These are the gaps that remain **after**
hybrid retrieval, the second guard, and the eval harness.

| Weakness | Why | What we do |
|---|---|---|
| **Scanned pages** | Embeddings cannot read pictures | Detected and flagged; OCR not implemented |
| **Counting questions** | "How many X in total?" needs every chunk; we fetch 8 | Inherent to RAG. Say so honestly. |
| **Single turn only** | No conversation memory | A follow-up like "and for managers?" is treated as a fresh question |
| **The entity guard only knows acronyms** | "the European privacy regulation" names GDPR without saying GDPR | That near-miss falls back to the distance threshold alone |
| **The threshold is corpus-specific** | It was measured on these five documents | Re-measure on a very different corpus: `python eval/run_eval.py --sweep` |
| **Small model, sloppy inline citations** | `llama3.2` is 3B parameters | The Sources list is built by **code**, so it is always right; only the model's inline pointer can be wrong |
| **Sparsely numbered documents** | Few headings means huge sections labelled `(part 3 of 16)` | Accurate but coarse; better on densely numbered documents |

### What used to be on this list, and is not any more

- ~~No way to measure retrieval quality~~ → `eval/run_eval.py`, 31 ground-truth
  questions, run in CI on every push
- ~~Exact identifiers retrieve badly~~ → the literal-match pass took identifier
  questions from 62% to 100%
- ~~Topical near-misses slip through~~ → the named-entity guard catches them
- ~~The threshold was tuned by eye~~ → chosen from a measured sweep, and the
  numbers are recorded in `config.py`

## 8. Who would use this, and what it saves

**New joiners, HR, support staff, anyone answering the same policy question
repeatedly.**

Today they either read 40 pages or interrupt someone who knows. This turns a
20-minute hunt into a 10-second question — and because every answer carries a
citation, they can **verify** it, which a plain chatbot can never offer. That
verifiability is the difference between a demo and something a company would
actually trust.
