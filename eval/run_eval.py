"""Measure retrieval quality against a fixed question set.

Run it after any change that could affect retrieval -- chunk sizes, embedding
model, the distance threshold, the search strategy:

    python eval/run_eval.py                 # measure
    python eval/run_eval.py --sweep         # also sweep the distance threshold

Why this exists: without it, "retrieval is good" is an impression. Every number
below is reproducible, so a change can be shown to have helped or hurt rather
than argued about.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, vector_store  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

QUESTIONS = Path(__file__).resolve().parent / "questions.json"


def load() -> dict:
    """Read the question set from disk."""
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def validate_ground_truth(cases: list[dict]) -> list[str]:
    """Check the eval set itself before trusting any metric it produces.

    A `must_contain` phrase that appears nowhere in the expected document is a
    bug in the question set, not a failure of the system. Catching that here
    stops a bad expectation from silently scoring the app down.
    """
    from app.chunker import ContentType
    from app.pdf_parser import parse_pdf

    text_by_doc: dict[str, str] = {}
    problems: list[str] = []

    for case in cases:
        doc = case["expected_doc"]
        if doc not in text_by_doc:
            path = config.DOCUMENTS_DIR / doc
            if not path.exists():
                problems.append("{}: document is missing".format(doc))
                text_by_doc[doc] = ""
                continue
            blocks = parse_pdf(path)
            text_by_doc[doc] = "\n".join(
                b.text for b in blocks if b.content_type is not ContentType.IMAGE_ONLY
            ).lower()

        phrase = case["must_contain"].lower()
        if phrase and phrase not in text_by_doc[doc]:
            problems.append(
                "{}: '{}' not found in {}".format(case["q"][:40], phrase, doc)
            )
    return problems


def measure_in_scope(cases: list[dict]) -> dict:
    """Retrieval metrics for questions the documents genuinely answer."""
    hit_at_1 = 0
    recall_at_k = 0
    grounded = 0
    empty = 0
    reciprocal_ranks: list[float] = []
    failures: list[tuple[str, str]] = []

    for case in cases:
        hits = vector_store.search(case["q"])
        if not hits:
            empty += 1
            failures.append((case["q"], "returned nothing"))
            reciprocal_ranks.append(0.0)
            continue

        docs = [h.source for h in hits]
        expected = case["expected_doc"]

        if docs[0] == expected:
            hit_at_1 += 1
        if expected in docs:
            recall_at_k += 1
            reciprocal_ranks.append(1.0 / (docs.index(expected) + 1))
        else:
            reciprocal_ranks.append(0.0)
            failures.append((case["q"], "wrong doc: got {}".format(docs[0])))

        phrase = case["must_contain"].lower()
        if any(phrase in h.text.lower() for h in hits):
            grounded += 1

    total = len(cases)
    return {
        "total": total,
        "hit_at_1": hit_at_1 / total,
        "recall_at_k": recall_at_k / total,
        "mrr": sum(reciprocal_ranks) / total,
        "grounded": grounded / total,
        "returned_nothing": empty,
        "failures": failures,
    }


def measure_refusals(questions: list[str], label: str) -> dict:
    """How often questions the documents cannot answer are correctly refused."""
    refused = 0
    leaked: list[tuple[str, float, str]] = []
    for question in questions:
        hits = vector_store.search(question)
        if not hits:
            refused += 1
        else:
            leaked.append((question, hits[0].distance, hits[0].source))
    return {
        "label": label,
        "total": len(questions),
        "refused": refused / len(questions) if questions else 0.0,
        "leaked": leaked,
    }


def sweep_threshold(cases: list[dict], out_of_scope: list[str]) -> None:
    """Show the recall/refusal tradeoff across candidate thresholds.

    The right threshold is the one that keeps in-scope recall high while refusing
    everything out of scope. Printing the whole curve makes that a measurement
    rather than a guess -- and shows how much margin there is.
    """
    original = config.MAX_DISTANCE
    print()
    print("Threshold sweep (higher recall is better, higher refusal is better)")
    print(
        "  {:>9} {:>14} {:>16}".format(
            "threshold", "in-scope kept", "out-scope refused"
        )
    )
    print("  " + "-" * 43)
    try:
        for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            config.MAX_DISTANCE = threshold
            kept = sum(1 for c in cases if vector_store.search(c["q"]))
            refused = sum(1 for q in out_of_scope if not vector_store.search(q))
            marker = "  <- current" if abs(threshold - original) < 1e-9 else ""
            print(
                "  {:>9.2f} {:>13.0f}% {:>15.0f}%{}".format(
                    threshold,
                    100 * kept / len(cases),
                    100 * refused / len(out_of_scope),
                    marker,
                )
            )
    finally:
        config.MAX_DISTANCE = original


def main() -> int:
    """Parse arguments, run the job, and return a process exit code."""
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--sweep", action="store_true", help="sweep the threshold")
    args = parser.parse_args()

    data = load()
    cases = data["in_scope"]
    hard = data.get("hard", [])

    print("=" * 66)
    print("RETRIEVAL EVALUATION")
    print("=" * 66)
    print("  embedding model : {}".format(config.embedding_fingerprint()))
    print("  max distance    : {}".format(config.MAX_DISTANCE))
    print(
        "  children / parents fetched: {} / {}".format(
            config.TOP_K_CHILDREN, config.TOP_K_PARENTS
        )
    )
    summary = vector_store.stats()
    print(
        "  indexed         : {} chunks from {} documents".format(
            summary["chunks"], len(summary["sources"])
        )
    )

    problems = validate_ground_truth(cases + hard)
    if problems:
        print()
        print("GROUND TRUTH PROBLEMS (fix the question set, not the app):")
        for problem in problems:
            print("  - {}".format(problem))
        return 1
    print(
        "  ground truth    : all {} expectations verified".format(
            len(cases) + len(hard)
        )
    )

    scores = measure_in_scope(cases)
    print()
    print("In-scope questions ({}):".format(scores["total"]))
    print(
        "  Hit@1        {:>5.0f}%   top source is the right document".format(
            100 * scores["hit_at_1"]
        )
    )
    print(
        "  Recall@{}     {:>5.0f}%   right document appears in the results".format(
            config.TOP_K_PARENTS, 100 * scores["recall_at_k"]
        )
    )
    print(
        "  MRR          {:>5.2f}    1.00 means always ranked first".format(
            scores["mrr"]
        )
    )
    print(
        "  Grounded     {:>5.0f}%   expected phrase present in retrieved text".format(
            100 * scores["grounded"]
        )
    )
    if scores["returned_nothing"]:
        print(
            "  ** {} answerable question(s) returned nothing **".format(
                scores["returned_nothing"]
            )
        )

    if hard:
        hard_scores = measure_in_scope(hard)
        print()
        print("Hard questions -- exact identifiers ({}):".format(hard_scores["total"]))
        print("  Hit@1        {:>5.0f}%".format(100 * hard_scores["hit_at_1"]))
        print(
            "  Recall@{}     {:>5.0f}%".format(
                config.TOP_K_PARENTS, 100 * hard_scores["recall_at_k"]
            )
        )
        print("  MRR          {:>5.2f}".format(hard_scores["mrr"]))
        print(
            "  Grounded     {:>5.0f}%   <- the identifier itself was retrieved".format(
                100 * hard_scores["grounded"]
            )
        )
        for question, reason in hard_scores["failures"]:
            print("    miss: {:<46} {}".format(question[:46], reason))

    for group, label in [("out_of_scope", "Out-of-scope"), ("near_miss", "Near-miss")]:
        result = measure_refusals(data[group], label)
        print()
        print("{} questions ({}):".format(label, result["total"]))
        print("  Refused      {:>5.0f}%".format(100 * result["refused"]))
        for question, distance, source in result["leaked"][:5]:
            print(
                "    let through: {:<44} {:.3f}  {}".format(
                    question[:44], distance, source[:28]
                )
            )

    if scores["failures"]:
        print()
        print("In-scope failures:")
        for question, reason in scores["failures"]:
            print("  - {:<48} {}".format(question[:48], reason))

    if args.sweep:
        sweep_threshold(cases, data["out_of_scope"])

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
