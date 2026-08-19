"""Check that CODE_WALKTHROUGH.md still points at the code it describes.

The walkthrough is read line-by-line with the source open beside it, so a stale
line number is not cosmetic -- it sends the reader to the wrong function and
quietly destroys their trust in the rest of the document.

This exists because the references drifted twice without anyone noticing: once
when data types moved out into models.py, and again when two functions were
inserted above the ones already documented. Both times every number below the
edit was wrong and nothing failed. Range-checking the numbers was not enough --
they were all still inside the file, just pointing at the wrong thing.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# "| `309-324` | **`_validated()`** | ..." -- only rows where the function is the
# row's subject, in bold. A row that merely mentions a name in passing is prose,
# not a reference, and carries no promise about line numbers.
_ROW = re.compile(r"^\| `(\d+)(?:[-–](\d+))?` \| \*\*`([\w_]+)(?:\(\))?`\*\*")
_FILE_HEADING = re.compile(r"^# `(?:app/)?([\w.]+\.py)`")


def _positions(path: Path) -> dict[str, tuple[int, int]]:
    """Where every top-level definition and constant in a file starts and ends."""
    found: dict[str, tuple[int, int]] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            found[node.name] = (node.lineno, node.end_lineno or node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = (node.lineno, node.end_lineno or node.lineno)
    return found


def test_every_line_reference_points_at_what_it_claims() -> None:
    """Each documented range must match the real span of that definition."""
    sources: dict[str, dict[str, tuple[int, int]]] = {}
    for path in list((ROOT / "app").glob("*.py")) + [
        ROOT / "streamlit_app.py",
        ROOT / "ingest.py",
    ]:
        sources[path.name] = _positions(path)

    lines = (ROOT / "CODE_WALKTHROUGH.md").read_text(encoding="utf-8").split("\n")
    current = ""
    checked = 0
    wrong: list[str] = []

    for number, line in enumerate(lines, start=1):
        heading = _FILE_HEADING.match(line)
        if heading:
            current = heading.group(1)
            continue
        row = _ROW.match(line)
        if not row or not current:
            continue
        real = sources.get(current, {}).get(row.group(3))
        if real is None:
            continue
        claimed_start = int(row.group(1))
        claimed_end = int(row.group(2)) if row.group(2) else claimed_start
        checked += 1
        if (claimed_start, claimed_end) != real:
            wrong.append(
                "CODE_WALKTHROUGH.md:{} says {} is at {}-{}, but it is at "
                "{}-{}".format(
                    number, row.group(3), claimed_start, claimed_end, real[0], real[1]
                )
            )

    assert checked > 30, "the checker matched too few rows; has the format changed?"
    assert not wrong, "stale line references:\n  " + "\n  ".join(wrong)
