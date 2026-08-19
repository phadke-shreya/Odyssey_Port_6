"""Check the codebase against the conventions in CLAUDE.md.

Some of those rules are matters of judgement and cannot be automated. The ones
below can, so they are enforced here rather than left to whoever reviews a diff:

    python tools/check_conventions.py

Exits non-zero on any finding, so CI fails rather than the rule quietly rotting.
Rules that need a human eye -- "does this comment explain why", "is this the
simplest thing that works" -- are deliberately out of scope.
"""

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Objects whose .info()/.warning()/.error() calls are logging. Streamlit's
# st.error() has the same method names but renders to the screen, where a
# formatted string is exactly right.
LOGGER_NAMES = frozenset({"logger", "log", "logging"})

LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})

# print() is for a command-line tool talking to its user; everything else logs.
CLI_FILES = frozenset({"ingest.py", "run_eval.py", "check_conventions.py"})

# Names short enough to be meaningless, minus the two the rule allows.
ALLOWED_SHORT_NAMES = frozenset({"i", "_"})


@dataclass
class Finding:
    """One convention violation, ready to print."""

    rule: str
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return "  {}:{}  {}".format(
            self.path.relative_to(ROOT), self.line, self.message
        )


def python_files() -> list[Path]:
    """Every source file the conventions apply to."""
    found: list[Path] = []
    for folder in ("app", "tests", "eval", "tools"):
        found.extend(sorted((ROOT / folder).glob("*.py")))
    found.extend([ROOT / "ingest.py", ROOT / "streamlit_app.py"])
    return [path for path in found if path.exists() and path.stat().st_size > 0]


def _is_logger_call(node: ast.Call) -> bool:
    """Whether this call is logging, as opposed to rendering to a screen."""
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in LOG_METHODS:
        return False
    target = node.func.value
    return isinstance(target, ast.Name) and target.id in LOGGER_NAMES


def _is_formatted(node: ast.expr) -> bool:
    """Whether an expression builds a string rather than being a literal."""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    )


def check_banned_constructs(tree: ast.AST, path: Path) -> list[Finding]:
    """Section 5: constructs this project does not use."""
    found: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            found.append(Finding("5", path, node.lineno, "f-string; use .format()"))
        elif isinstance(node, ast.NamedExpr):
            found.append(Finding("5", path, node.lineno, "walrus operator"))
        elif isinstance(node, ast.IfExp) and isinstance(node.orelse, ast.IfExp):
            found.append(Finding("5", path, node.lineno, "chained ternary"))
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            found.append(Finding("5", path, node.lineno, "bare except"))
        elif isinstance(node, ast.ImportFrom) and any(
            alias.name == "*" for alias in node.names
        ):
            found.append(Finding("5", path, node.lineno, "wildcard import"))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            found.append(
                Finding("5", path, node.lineno, "lambda assigned to a name; use def")
            )
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            if isinstance(node.elt, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                found.append(
                    Finding(
                        "5", path, node.lineno, "comprehension inside a comprehension"
                    )
                )
        elif isinstance(node, ast.FunctionDef):
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    found.append(
                        Finding("5", path, node.lineno, "mutable default argument")
                    )
    return found


def check_logging(tree: ast.AST, path: Path) -> list[Finding]:
    """Section 5: let the logger do the formatting, so it stays lazy."""
    found: list[Finding] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _is_logger_call(node)
            and node.args
            and _is_formatted(node.args[0])
        ):
            found.append(
                Finding(
                    "5",
                    path,
                    node.lineno,
                    "formatted string passed to the logger; pass args instead",
                )
            )
    return found


def check_consistency(tree: ast.AST, path: Path) -> list[Finding]:
    """Section 7: one way of doing each thing."""
    found: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            found.append(
                Finding("7", path, node.lineno, "while loop; this project uses for")
            )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            found.append(Finding("7", path, node.lineno, "os.path; use pathlib"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "range" and node.args:
                first = node.args[0]
                if (
                    isinstance(first, ast.Call)
                    and isinstance(first.func, ast.Name)
                    and first.func.id == "len"
                ):
                    found.append(
                        Finding(
                            "7", path, node.lineno, "range(len(...)); use enumerate"
                        )
                    )
            elif node.func.id == "print" and path.name not in CLI_FILES:
                found.append(
                    Finding("7", path, node.lineno, "print() outside a CLI; use logger")
                )

    return found


def check_variable_names(tree: ast.AST, path: Path) -> list[Finding]:
    """Section 5: no single-letter names, in loops and comprehensions too.

    Added after a review found 21 of these that the first version of this checker
    missed, because it only inspected function parameters.
    """
    found: list[Finding] = []
    targets: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            for generator in node.generators:
                if isinstance(generator.target, ast.Name):
                    targets.append((node.lineno, generator.target.id))
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            targets.append((node.lineno, node.target.id))
    for line, name in targets:
        if len(name) == 1 and name not in ALLOWED_SHORT_NAMES:
            found.append(
                Finding(
                    "5", path, line, "single-letter loop variable '{}'".format(name)
                )
            )
    return found


def check_signatures(tree: ast.AST, path: Path) -> list[Finding]:
    """Sections 7 and 13: type hints everywhere, docstrings on public functions."""
    found: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.returns is None:
            found.append(
                Finding(
                    "13", path, node.lineno, "{}() has no return type".format(node.name)
                )
            )
        for argument in node.args.args + node.args.kwonlyargs:
            if argument.arg in ("self", "cls"):
                continue
            if argument.annotation is None:
                found.append(
                    Finding(
                        "13",
                        path,
                        node.lineno,
                        "{}(): '{}' has no type".format(node.name, argument.arg),
                    )
                )
            if len(argument.arg) == 1 and argument.arg not in ALLOWED_SHORT_NAMES:
                found.append(
                    Finding(
                        "5",
                        path,
                        node.lineno,
                        "{}(): single-letter parameter '{}'".format(
                            node.name, argument.arg
                        ),
                    )
                )
        public = not node.name.startswith("_") and not node.name.startswith("test_")
        if public and ast.get_docstring(node) is None:
            found.append(
                Finding(
                    "7",
                    path,
                    node.lineno,
                    "public {}() has no docstring".format(node.name),
                )
            )
    return found


# A comment that is really disabled code, rather than an explanation.
_COMMENTED_CODE = re.compile(
    r"^#\s*(import |from \S+ import |def |class |return |if .+:$|for .+:$|while .+:$)"
)


def check_comments(source: str, path: Path) -> list[Finding]:
    """Section 8: no commented-out code. That is what version control is for."""
    found: list[Finding] = []
    for number, line in enumerate(source.split("\n"), start=1):
        stripped = line.strip()
        if _COMMENTED_CODE.match(stripped):
            found.append(
                Finding(
                    "8", path, number, "commented-out code: {}".format(stripped[:50])
                )
            )
    return found


def main() -> int:
    """Check every file and report, returning a process exit code."""
    findings: list[Finding] = []
    for path in python_files():
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            findings.append(Finding("-", path, error.lineno or 0, "syntax error"))
            continue
        findings.extend(check_banned_constructs(tree, path))
        findings.extend(check_logging(tree, path))
        findings.extend(check_consistency(tree, path))
        findings.extend(check_variable_names(tree, path))
        findings.extend(check_signatures(tree, path))
        findings.extend(check_comments(source, path))

    if not findings:
        print(
            "CLAUDE.md conventions: no violations in {} files".format(
                len(python_files())
            )
        )
        return 0

    findings.sort(key=lambda item: (item.rule, str(item.path), item.line))
    print("{} convention violation(s):".format(len(findings)))
    current = ""
    for finding in findings:
        if finding.rule != current:
            print("\n--- CLAUDE.md section {} ---".format(finding.rule))
            current = finding.rule
        print(finding)
    return 1


if __name__ == "__main__":
    sys.exit(main())
