from __future__ import annotations
from pathlib import Path
from typing import Iterable, Set
import re
import keyword

try:
    import libcst as cst
except Exception:  # pragma: no cover
    cst = None  # type: ignore

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,63}")

def tokens_from_text(text: str) -> Set[str]:
    """Regex fallback: extract plausible code-ish tokens (identifiers) from text."""
    toks = set(m.group(0) for m in _IDENT.finditer(text))
    toks = {t for t in toks if t not in keyword.kwlist and len(t) >= 2}
    return toks

def _libcst_tokens(code: str) -> Set[str]:
    if cst is None:
        return set()
    try:
        mod = cst.parse_module(code)
    except Exception:
        return set()

    names: Set[str] = set()

    def _add_param_name(obj):
        # Accept either a Param with .name, or ParamStar with .name or .param.name (older/newer versions)
        if obj is None:
            return
        n = getattr(obj, "name", None)
        if n is not None and hasattr(n, "value"):
            names.add(n.value)
            return
        p = getattr(obj, "param", None)
        if p is not None and getattr(p, "name", None) is not None:
            names.add(p.name.value)

    class V(cst.CSTVisitor):
        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            # function name
            names.add(node.name.value)
            # regular params
            ps = node.params
            for p in (ps.params + ps.posonly_params + ps.kwonly_params):
                if getattr(p, "name", None):
                    names.add(p.name.value)
            # *args / **kwargs across LibCST versions
            _add_param_name(getattr(ps, "star_arg", None))
            _add_param_name(getattr(ps, "star_kwarg", None))

        def visit_Assign(self, node: cst.Assign) -> None:
            # collect simple assignment targets when possible
            for t in node.targets:
                target = getattr(t, "target", None)
                name = getattr(target, "value", None)
                if isinstance(name, str):
                    names.add(name)

        def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
            for n in node.names:
                asname = getattr(n, "asname", None)
                if asname and getattr(asname, "name", None):
                    names.add(asname.name.value)
                else:
                    base = getattr(n, "name", None)
                    if base and getattr(base, "value", None):
                        names.add(base.value)

        def visit_Import(self, node: cst.Import) -> None:
            for n in node.names:
                asname = getattr(n, "asname", None)
                if asname and getattr(asname, "name", None):
                    names.add(asname.name.value)
                else:
                    base = getattr(n, "name", None)
                    if base and getattr(base, "value", None):
                        names.add(base.value)

    try:
        mod.visit(V())
    except Exception:
        # If a LibCST layout difference trips us, fall back to regex-only for this file
        return set()

    # filter out keywords and very short tokens
    return {n for n in names if n not in keyword.kwlist and len(n) >= 2}

def extract_from_paths(paths: Iterable[Path]) -> Set[str]:
    out: Set[str] = set()
    for p in paths:
        try:
            txt = Path(p).read_text()
        except Exception:
            continue
        toks = _libcst_tokens(txt)
        if not toks:
            toks = tokens_from_text(txt)
        out |= toks
    return out
