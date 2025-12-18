
from s2c.edits.apply_ast import apply_rename_var, apply_add_param, apply_wrap_lines

CODE = """def foo(x):
    total = x + 1
    counts = [1,2,3]
    for count in counts:
        total += count
    return total
"""

def test_rename_var_in_function():
    out = apply_rename_var(CODE, function="foo", old="counts", new="total_counts")
    assert "total_counts = [1,2,3]" in out
    assert "for count in total_counts" in out

def test_add_param_with_default():
    out = apply_add_param(CODE, function="foo", name="min_count", default_value=2)
    assert "def foo(x, min_count=2):" in out

def test_wrap_lines_try_except():
    out = apply_wrap_lines(CODE, start_line=3, end_line=5)
    assert "try:" in out and "except Exception as e:" in out
