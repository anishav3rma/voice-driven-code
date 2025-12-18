
from s2c.edits.grammar import parse_edit

def test_parse_rename():
    e = parse_edit("rename counts to total_counts")
    assert e and e["op"] == "rename" and e["target"] == "counts"

def test_parse_rename_in_function():
    e = parse_edit("rename counts to total_counts in function foo")
    assert e and e["function"] == "foo"

def test_parse_add_param():
    e = parse_edit("add parameter min_count default 2 to foo")
    assert e and e["op"] == "add_param" and e["default"] == 2

def test_parse_wrap():
    e = parse_edit("wrap lines 10 to 20 in try except")
    assert e and e["op"] == "wrap" and e["start_line"] == 10 and e["end_line"] == 20
