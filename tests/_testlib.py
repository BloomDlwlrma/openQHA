"""Shared helpers for the repository-hygiene tests. Not a test itself.

Two of the checks here scan source text for a forbidden pattern, and both meet the
same trap: a docstring that EXPLAINS why a pattern was removed contains that pattern,
so documenting a fix would make the check fail. `code_only` is the answer both of them
need, so it lives here once rather than twice.

Nothing in this file is imported by the package. It exists only for `tests/`.
"""
import ast
import io
import tokenize
from pathlib import Path


def repo_root(start):
    """Directory holding the openQHA package, walking up from `start`."""
    for p in Path(start).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + str(start))


def code_only(text):
    """`text` with comments and prose strings blanked, line numbers preserved.

    Blanked: comments, and strings that stand alone as a statement (module, class and
    function docstrings, and free-floating prose). NOT blanked: strings used as
    arguments, because those are code.

    That distinction matters: blanking every string literal would hide two
    genuine occurrences of `sys.path.insert(0, ".")`, because the `"."` the pattern
    looks for is itself a string literal. Widening a filter until a check goes quiet
    is how a check stops being one.
    """
    out = text.splitlines()

    def blank(r0, c0, r1, c1):
        for r in range(r0, r1 + 1):
            i = r - 1
            if not (0 <= i < len(out)):
                continue
            line = out[i]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(line)
            out[i] = line[:a] + " " * max(0, b - a) + line[b:]

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                blank(tok.start[0], tok.start[1], tok.end[0], tok.end[1])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text                       # unparsable: fall back to the raw text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "\n".join(out)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            v = node.value
            blank(v.lineno, v.col_offset, v.end_lineno, v.end_col_offset)
    return "\n".join(out)


def uncommented(text, markers=("#", ";", "//")):
    """`text` with whole-line comments removed. For files that are not Python."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        keep.append("" if any(s.startswith(m) for m in markers) else line)
    return "\n".join(keep)
