
import re
from typing import Optional, Dict, Any

_WRAP_TRY_EXC = re.compile(
    r"""
    \bwrap\s+(?:lines?\s+)?      # "wrap" or "wrap lines"
    (?P<start>\d+)               # start line (1-based, relative to function body)
    \s*(?:to|through|[-–—])\s*   # to/through/dash
    (?P<end>\d+)                 # end line
    \s+in\s+try(?:\s*[-_/]?\s*)?except         # "in try except"
    (?:\s+(?:in|inside)?\s*function\s+(?P<func>[A-Za-z_][A-Za-z0-9_]*))?  # optional "in function foo"
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# def _parse_wrap_try_except(s: str):
#     m = _WRAP_TRY_EXC.search(s)
#     if not m:
#         return None
#     start = int(m.group("start"))
#     end   = int(m.group("end"))
#     if end < start:
#         start, end = end, start
#     func = m.group("func")
#     return {"op": "wrap_try_except", "start": start, "end": end, "function": func}

_WS = r"[ \t]+"
_RENAME = re.compile(rf"^rename{_WS}([A-Za-z_][A-Za-z0-9_]*){_WS}to{_WS}([A-Za-z_][A-Za-z0-9_]*)$", re.I)
_RENAME_IN_FUNC = re.compile(rf"^rename{_WS}([A-Za-z_][A-Za-z0-9_]*){_WS}to{_WS}([A-Za-z_][A-Za-z0-9_]*){_WS}in{_WS}function{_WS}([A-Za-z_][A-Za-z0-9_]*)$", re.I)
_ADD_PARAM = re.compile(rf"^add{_WS}(?:param|parameter){_WS}([A-Za-z_][A-Za-z0-9_]*)(?:{_WS}default{_WS}([0-9]+))?{_WS}to{_WS}([A-Za-z_][A-Za-z0-9_]*)$", re.I)
# _WRAP = re.compile(rf"^wrap{_WS}lines{_WS}([0-9]+){_WS}to{_WS}([0-9]+){_WS}in{_WS}try except$", re.I)

def parse_edit(text: str) -> Optional[Dict[str, Any]]:
    s = text.strip()
    m = _RENAME_IN_FUNC.match(s)
    if m:
        old, new, func = m.group(1), m.group(2), m.group(3)
        return {"op": "rename", "target_kind": "var", "target": old, "new_name": new, "function": func}
    m = _RENAME.match(s)
    if m:
        old, new = m.group(1), m.group(2)
        return {"op": "rename", "target_kind": "var", "target": old, "new_name": new, "function": None}
    m = _ADD_PARAM.match(s)
    if m:
        name, default, func = m.group(1), m.group(2), m.group(3)
        default_val = int(default) if default is not None else None
        return {"op": "add_param", "function": func, "name": name, "default": default_val}
    # m = _WRAP.match(s)
    # if m:
    #     start, end = int(m.group(1)), int(m.group(2))
    #     return {"op": "wrap", "start_line": start, "end_line": end, "wrapper": "try_except"}
    
    m = _WRAP_TRY_EXC.search(s)
    if m:
        start = int(m.group("start"))
        end   = int(m.group("end"))
        if end < start:
            start, end = end, start
        func = m.group("func")
        return {"op": "wrap", "start_line": start, "end_line": end, "function": func}

    # out = _parse_wrap_try_except(s)
    # if out:
    #     return out

    return None
