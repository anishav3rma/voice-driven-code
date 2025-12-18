
from importlib import reload
import s2c.user_solution as sol

def test_reverse_basic():
    reload(sol)
    assert sol.reverse_string("abc") == "cba"
    assert sol.reverse_string("") == ""

def test_reverse_unicode():
    reload(sol)
    assert sol.reverse_string("😀ab") == "ba😀"
