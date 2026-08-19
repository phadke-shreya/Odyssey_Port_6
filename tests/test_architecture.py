"""Tests for the module boundaries in CLAUDE.md section 12.

These exist because the boundary was silently broken once. Retrieved lived in
vector_store, so importing rag_chain pulled in chromadb -- even though rag_chain
never touches the database and the rule says it must not know Chroma exists.
Nothing failed; the coupling was simply invisible until someone checked.

A comment cannot enforce that. A test can.
"""

import ast
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"


def _app_imports(module: str) -> set[str]:
    """Which other app modules a given app module imports."""
    tree = ast.parse((APP / "{}.py".format(module)).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("app."):
                found.add(node.module.split(".", 1)[1])
            elif node.module == "app":
                for alias in node.names:
                    found.add(alias.name)
    return found


def _imports_cleanly_without(module: str, forbidden: str) -> bool:
    """Whether importing a module avoids loading a heavy dependency.

    Run in a fresh interpreter, because anything already imported by the test
    session would make this pass regardless.
    """
    code = (
        "import importlib, sys; importlib.import_module('app.{}'); "
        "sys.exit(1 if '{}' in sys.modules else 0)".format(module, forbidden)
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=str(APP.parent),
        check=False,
    )
    return result.returncode == 0


# --- the data layer depends on nothing -----------------------------------


def test_models_and_config_depend_on_no_other_module() -> None:
    """They are the bottom of the stack, so the arrows all point one way."""
    assert _app_imports("models") == set()
    assert _app_imports("config") == set()


# --- the stated boundaries ------------------------------------------------


def test_rag_chain_does_not_know_the_database_exists() -> None:
    """CLAUDE.md section 12: rag_chain must not know Chroma exists.

    Regression: it imported Retrieved from vector_store, so `import app.rag_chain`
    loaded chromadb.
    """
    assert "vector_store" not in _app_imports("rag_chain")
    assert _imports_cleanly_without("rag_chain", "chromadb")


def test_chunker_does_not_read_pdfs() -> None:
    """Chunking works on text; reading files is the parser's job."""
    assert "pdf_parser" not in _app_imports("chunker")
    assert _imports_cleanly_without("chunker", "pdfplumber")


def test_heading_detection_stands_alone() -> None:
    """headings.py answers "is this line a heading" and nothing else.

    It was split out of the chunker precisely so it could be tested on bare
    strings. If it ever imports the chunker or the models, that property is gone
    and the split bought nothing.
    """
    assert _app_imports("headings") <= {"config"}


def test_the_parser_does_not_depend_on_the_chunker() -> None:
    """Regression: the producer imported its own consumer for a data type."""
    assert "chunker" not in _app_imports("pdf_parser")


def test_vector_store_does_not_build_prompts_or_call_a_model() -> None:
    """Storage and retrieval only. The prompt belongs to rag_chain."""
    assert "rag_chain" not in _app_imports("vector_store")
    assert _imports_cleanly_without("vector_store", "langchain_openai")


def _top_level_imports(path: Path) -> set[str]:
    """The third-party packages a file imports, by AST rather than by text.

    A text search would match the word in a docstring: models.py explains why the
    types moved out of vector_store and names chromadb while importing nothing.
    That is the same mistake the conventions checker made in its first draft.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_only_vector_store_imports_chroma() -> None:
    """Swapping the database should mean editing exactly one file."""
    users = [
        path.name
        for path in sorted(APP.glob("*.py"))
        if "chromadb" in _top_level_imports(path)
    ]

    assert users == ["vector_store.py"], users


def test_only_the_parser_and_ocr_read_pdfs() -> None:
    """Reading files is their job; nothing downstream should need the library."""
    users = [
        path.name
        for path in sorted(APP.glob("*.py"))
        if "pdfplumber" in _top_level_imports(path)
    ]

    assert users == ["ocr.py", "pdf_parser.py"], users


def test_there_are_no_import_cycles() -> None:
    """A cycle means two modules cannot be understood or reused separately."""
    modules = [path.stem for path in APP.glob("*.py") if path.stem != "__init__"]
    graph = {name: _app_imports(name) & set(modules) for name in modules}
    for name, direct in graph.items():
        for other in direct:
            assert name not in graph[other], "{} and {} import each other".format(
                name, other
            )
