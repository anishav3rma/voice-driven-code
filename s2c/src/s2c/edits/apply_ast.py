from __future__ import annotations
from typing import Optional, Union, List, Tuple
import libcst as cst
from libcst import metadata as md
from libcst.metadata import MetadataWrapper, PositionProvider



# ---------- Rename variable (scope-limited to a function if provided) ----------
class _RenameVarTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (md.ScopeProvider,)

    def __init__(self, func_name: Optional[str], old: str, new: str) -> None:
        self.func_name = func_name
        self.old = old
        self.new = new
        self._in_target_func = func_name is None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        if self.func_name is None:
            return True
        self._in_target_func = (node.name.value == self.func_name)
        return True

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> Union[cst.Name, cst.CSTNode]:
        if not self._in_target_func:
            return updated_node
        if original_node.value == self.old:
            return updated_node.with_changes(value=self.new)
        return updated_node


# ---------- Add parameter to a function (with optional default) ----------
def _coerce_literal(val):
    """
    Make the default robust if it comes in as a string: turn "2"->2, "2.5"->2.5,
    "true"/"false"->bool, else keep as-is.
    """
    if isinstance(val, str):
        s = val.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        # int-like
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except Exception:
                pass
        # float-like
        try:
            if any(ch in s for ch in (".", "e", "E")):
                return float(s)
        except Exception:
            pass
    return val

def _mk_param_node(name: str, default_value) -> cst.Param:
    """
    Build a Param node with exact formatting (min_count=2) by parsing a tiny function.
    """
    coerced = _coerce_literal(default_value)
    if coerced is None:
        dummy = f"def _f(_x, {name}):\n    pass\n"
    else:
        dummy = f"def _f(_x, {name}={repr(coerced)}):\n    pass\n"
    mod = cst.parse_module(dummy)
    func = next(n for n in mod.body if isinstance(n, cst.FunctionDef))
    return func.params.params[1]

class _AddParamTransformer(cst.CSTTransformer):
    def __init__(self, func_name: str, name: str, default_value) -> None:
        self.func_name = func_name
        self.name = name
        self.default_value = default_value

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        if updated_node.name.value != self.func_name:
            return updated_node

        # If parameter already exists → no-op
        for p in updated_node.params.params:
            if getattr(p.name, "value", "") == self.name:
                return updated_node

        new_param = _mk_param_node(self.name, self.default_value)
        new_params = list(updated_node.params.params) + [new_param]
        return updated_node.with_changes(
            params=updated_node.params.with_changes(params=tuple(new_params))
        )

def apply_add_param(code: str, function: str, name: str, default_value) -> str:
    mod = cst.parse_module(code)
    tx = _AddParamTransformer(function, name, default_value)
    return mod.visit(tx).code


# ---------- Wrap a contiguous slice of lines in try/except (inside a function) ----------
# ---------- Wrap a contiguous slice of lines in try/except (optionally inside a function) ----------
class _WrapLinesInFunc(cst.CSTTransformer):
    """
    If function is provided, `start_line`/`end_line` are interpreted *relative to the function body*
    (1-based, first real body statement line == 1). If function is None, they are interpreted as
    absolute module line numbers (1-based).
    """
    METADATA_DEPENDENCIES = (md.PositionProvider,)

    def __init__(self, start_line: int, end_line: int, function: Optional[str] = None) -> None:
        self.start = min(start_line, end_line)
        self.end = max(start_line, end_line)
        self.func = function
        self._done = False

    # --- helpers ---
    def _pos(self, node: cst.CSTNode) -> Tuple[int, int]:
        p = self.get_metadata(md.PositionProvider, node)
        return p.start.line, p.end.line

    @staticmethod
    def _simple_print_e_stmt() -> cst.SimpleStatementLine:
        return cst.SimpleStatementLine(
            body=[cst.Expr(value=cst.Call(func=cst.Name("print"), args=[cst.Arg(value=cst.Name("e"))]))]
        )

    # --- module absolute wrapping (when no function is given) ---
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.CSTNode:
        if self._done or self.func is not None:
            return updated_node

        body = list(updated_node.body)
        if not body:
            return updated_node

        # Find first/last indices intersecting the [start..end] absolute line span.
        first_idx = last_idx = None
        for i, stmt in enumerate(original_node.body):
            lo, hi = self._pos(stmt)
            if hi >= self.start and first_idx is None:
                first_idx = i
            if lo <= self.end:
                last_idx = i

        if first_idx is None or last_idx is None or last_idx < first_idx:
            return updated_node  # no overlap → no-op

        mid = body[first_idx : last_idx + 1]
        try_node = cst.Try(
            body=cst.IndentedBlock(body=mid),
            handlers=[
                cst.ExceptHandler(
                    type=cst.Name("Exception"),
                    name=cst.AsName(name=cst.Name("e")),
                    body=cst.IndentedBlock(body=[self._simple_print_e_stmt()]),
                )
            ],
        )
        new_body = body[:first_idx] + [try_node] + body[last_idx + 1:]
        self._done = True
        return updated_node.with_changes(body=new_body)

    # --- function-relative wrapping (preferred) ---
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        if self._done:
            return updated_node
        if self.func is not None and original_node.name.value != self.func:
            return updated_node

        # Derive the first "real" body line to compute relative indices.
        if not original_node.body.body:
            return updated_node

        # Find the first non-empty statement line in the function body
        body_start_abs = None
        for stmt in original_node.body.body:
            lo, _ = self._pos(stmt)
            body_start_abs = lo
            break
        if body_start_abs is None:
            return updated_node

        # Map relative [start..end] to absolute lines within the function
        s_abs = body_start_abs + (self.start - 1)
        e_abs = body_start_abs + (self.end - 1)

        # Identify indices of body statements that intersect the [s_abs..e_abs] window
        first_idx = last_idx = None
        for i, stmt in enumerate(original_node.body.body):
            lo, hi = self._pos(stmt)
            if hi >= s_abs and first_idx is None:
                first_idx = i
            if lo <= e_abs:
                last_idx = i

        if first_idx is None or last_idx is None or last_idx < first_idx:
            return updated_node  # nothing to wrap inside this function

        # Use UPDATED body's statements to preserve other rewrites
        pre  = list(updated_node.body.body[:first_idx])
        mid  = list(updated_node.body.body[first_idx : last_idx + 1])
        post = list(updated_node.body.body[last_idx + 1 :])

        try_node = cst.Try(
            body=cst.IndentedBlock(body=mid),
            handlers=[
                cst.ExceptHandler(
                    type=cst.Name("Exception"),
                    name=cst.AsName(name=cst.Name("e")),
                    body=cst.IndentedBlock(body=[self._simple_print_e_stmt()]),
                )
            ],
        )
        new_body = pre + [try_node] + post
        self._done = True
        return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

def apply_rename_var(code: str, function: Optional[str], old: str, new: str) -> str:
    mod = cst.parse_module(code)
    wrapper = md.MetadataWrapper(mod)
    return wrapper.visit(_RenameVarTransformer(function, old, new)).code

def apply_add_param(code: str, function: str, name: str, default_value: Optional[object]) -> str:
    mod = cst.parse_module(code)
    default_code = None if default_value is None else repr(default_value)
    tx = _AddParamTransformer(function, name, default_code)
    return mod.visit(tx).code

def apply_wrap_lines_old(code: str, start_line: int, end_line: int, function: Optional[str] = None) -> str:
    mod = cst.parse_module(code)
    wrapper = md.MetadataWrapper(mod)
    return wrapper.visit(_WrapLinesInFunc(start_line, end_line, function=function)).code

class NoWrapTargetFound(Exception):
    """No statements overlap the requested line span in the given scope."""
    pass

def apply_wrap_lines(
    src: str,
    *,
    start_line: int,
    end_line: int,
    function: Optional[str] = None,
) -> str:
    """
    Wrap the statements intersecting [start_line, end_line] (GLOBAL file lines)
    in a try/except Exception as e: pass.
    If `function` is given, only consider statements inside that function's body.
    """
    if end_line < start_line:
        start_line, end_line = end_line, start_line

    mod = cst.parse_module(src)
    wrapper = MetadataWrapper(mod)
    positions = wrapper.resolve(PositionProvider)

    # ---- locate target block (module body or a specific function body) ----
    target_func: Optional[cst.FunctionDef] = None
    if function:
        class _FindFunc(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)
            def __init__(self) -> None:
                self.hit: Optional[cst.FunctionDef] = None
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                if node.name.value == function and self.hit is None:
                    self.hit = node

        vf = _FindFunc()
        wrapper.visit(vf)
        target_func = vf.hit
        if target_func is None:
            # function not found → no change
            return src

    # Extract the list of statements (body) to operate on
    if target_func is None:
        # Module-level
        body_list = list(mod.body)
        parent_kind = "module"
    else:
        if not isinstance(target_func.body, cst.IndentedBlock):
            return src
        body_list = list(target_func.body.body)
        parent_kind = "function"

    # ---- choose which statements to wrap based on GLOBAL line span ----
    def stmt_span(stmt: cst.CSTNode) -> Tuple[int, int]:
        pos = positions[stmt]
        return pos.start.line, pos.end.line
    
    # If a function is specified, convert function-relative S/E to global
    # by anchoring to the first body statement's start line.
    if target_func is not None and body_list:
        base_first_stmt_start = positions[body_list[0]].start.line
        # Heuristic: if S/E clearly fall outside the function's global span,
        # assume they are function-relative and lift them.
        func_s, func_e = positions[target_func].start.line, positions[target_func].end.line
        gl_start, gl_end = start_line, end_line
        if not (func_s <= gl_start <= func_e and func_s <= gl_end <= func_e):
            gl_start = start_line + base_first_stmt_start - 1
            gl_end   = end_line   + base_first_stmt_start - 1
    else:
        gl_start, gl_end = start_line, end_line

    # Intersect any statement that overlaps [start_line, end_line]
    chosen_idx: List[int] = []
    for i, stmt in enumerate(body_list):
        s, e = stmt_span(stmt)
        # If scoping to a function, filter still happens inside its body_list only
        if not (e < gl_start or s > gl_end):
            chosen_idx.append(i)

    if not chosen_idx:
        scope = f"function {function}" if function else "module"
        # Optional: include function/global spans for easier debugging
        func_span = None
        if target_func is not None:
            p = positions[target_func]
            func_span = (p.start.line, p.end.line)
        raise NoWrapTargetFound(
            f"No statements overlap [{gl_start}, {gl_end}] in {scope}"
            + (f" (function span {func_span[0]}–{func_span[1]})" if func_span else "")
        )

    i0, i1 = chosen_idx[0], chosen_idx[-1]
    selected_stmts = body_list[i0:i1 + 1]

    # ---- build the Try/Except node wrapping those statements ----
    try_node = cst.Try(
        body=cst.IndentedBlock(selected_stmts),
        handlers=[
            cst.ExceptHandler(
                type=cst.Name("Exception"),
                name=cst.AsName(name=cst.Name("e")),
                body=cst.IndentedBlock([cst.SimpleStatementLine([cst.Pass()])]),
            )
        ],
        orelse=None,
        finalbody=None,
    )

    # ---- splice into the correct parent body ----
    new_body = body_list[:i0] + [try_node] + body_list[i1 + 1:]

    if parent_kind == "module":
        # Optional guard: avoid wrapping across function defs at module scope
        if any(isinstance(s, cst.FunctionDef) for s in selected_stmts):
            raise NoWrapTargetFound("Selection crosses function definition(s) at module scope")
        new_mod = mod.with_changes(body=new_body)
    else:
        # Replace by position, not identity, and use the wrapper for metadata
        tgt_pos = positions[target_func]
        class _ReplaceFunc(cst.CSTTransformer):
            METADATA_DEPENDENCIES = (PositionProvider,)
            def leave_FunctionDef(self, orig_node: cst.FunctionDef, updated_node: cst.FunctionDef):
                pos = self.get_metadata(PositionProvider, orig_node)
                if (orig_node.name.value == function
                    and pos.start == tgt_pos.start
                    and pos.end == tgt_pos.end):
                    return updated_node.with_changes(body=cst.IndentedBlock(new_body))
                return updated_node

        new_mod = wrapper.visit(_ReplaceFunc())   # <-- use wrapper, not mod.visit

        # new_func = target_func.with_changes(body=cst.IndentedBlock(new_body))
        # class _ReplaceFunc(cst.CSTTransformer):
        #     def leave_FunctionDef(self, orig_node: cst.FunctionDef, updated_node: cst.FunctionDef):
        #         if orig_node is target_func:
        #             return new_func
        #         return updated_node
        # new_mod = mod.visit(_ReplaceFunc())

    return new_mod.code
