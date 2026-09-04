"""No function may reference a name the module never binds.

Written after a real outage: commit 0e9a090 added `SectionPipeline(...)` to
`ui/stream.py` and never added its import, so **watch-along through the server
raised `NameError` for a day** while every test and every benchmark passed.
The benchmark passed because `bench.py` constructs `SectionPipeline` itself,
with its own import — the measured path and the served path had diverged, and
nothing compared them.

The class of bug is "a name used inside a function body that no test ever
enters". Type checkers and linters catch it, but this repo has neither
installed, and adding a dependency for it is a bigger decision than adding a
test. `symtable` is in the standard library and knows, per function, which
names are global rather than local; anything global that the module does not
bind and that is not a builtin cannot resolve at runtime.

Deliberately static: it parses, and never imports. Importing `patmir.*` pulls
torch, funasr and NeMo, which would make this the slowest test in the suite
and would skip exactly the modules whose heavy dependencies are missing.
"""
from __future__ import annotations

import ast
import builtins
import symtable
import unittest
from pathlib import Path

_BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
    "__builtins__", "__debug__",
}


def _module_bindings(tree: ast.Module) -> set[str]:
    """Every name the module could bind at import time.

    Walked over the whole tree rather than just the top level: a name bound
    inside `if TYPE_CHECKING:` or a `try/except ImportError` fallback is still
    a module-level binding as far as a function body is concerned.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _globals_referenced(table: symtable.SymbolTable, out: set[str]) -> None:
    for sym in table.get_symbols():
        # `is_global` means the name is not local to this function; excluding
        # assigned ones avoids flagging a `global x` write.
        if sym.is_global() and not sym.is_assigned():
            out.add(sym.get_name())
    for child in table.get_children():
        _globals_referenced(child, out)


def unresolved_names(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, str(path))
    top = symtable.symtable(src, str(path), "exec")
    bound = _module_bindings(tree) | _BUILTINS
    used: set[str] = set()
    for child in top.get_children():          # skip module scope itself
        _globals_referenced(child, used)
    return sorted(n for n in used if n not in bound)


class NoUndefinedNamesTest(unittest.TestCase):
    def test_every_module_resolves_its_names(self):
        root = Path(__file__).resolve().parent.parent
        offenders = {}
        for path in sorted(root.rglob("*.py")):
            missing = unresolved_names(path)
            if missing:
                offenders[str(path.relative_to(root.parent))] = missing
        self.assertEqual(
            offenders, {},
            "names referenced but never bound — these raise NameError when the "
            "line is reached:\n" + "\n".join(
                f"  {f}: {', '.join(n)}" for f, n in offenders.items()),
        )

    def test_the_check_catches_the_bug_it_was_written_for(self):
        """A guard that cannot detect its own motivating failure is decoration.

        This is `ui/stream.py`'s shape: a name used deep inside a method, with
        the import absent.
        """
        import tempfile
        src = (
            "from os import path\n"
            "\n"
            "def run():\n"
            "    return SectionPipeline(path)\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            tmp = Path(fh.name)
        try:
            self.assertEqual(unresolved_names(tmp), ["SectionPipeline"])
        finally:
            tmp.unlink()

    def test_it_does_not_flag_a_name_that_is_imported(self):
        import tempfile
        src = (
            "from queue import Queue\n"
            "\n"
            "def run():\n"
            "    return Queue()\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            tmp = Path(fh.name)
        try:
            self.assertEqual(unresolved_names(tmp), [])
        finally:
            tmp.unlink()

    def test_it_does_not_flag_builtins_or_late_module_globals(self):
        import tempfile
        src = (
            "LIMIT = 3\n"
            "\n"
            "def run(xs):\n"
            "    return len(xs) + LIMIT + int('1')\n"
            "\n"
            "def later():\n"
            "    return HELPER()\n"
            "\n"
            "def HELPER():\n"
            "    return 1\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            tmp = Path(fh.name)
        try:
            self.assertEqual(unresolved_names(tmp), [])
        finally:
            tmp.unlink()


if __name__ == "__main__":
    unittest.main()
