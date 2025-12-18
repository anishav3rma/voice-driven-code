
from importlib import reload
import s2c.user_solution as sol

def test_wc_basic():
    reload(sol)
    out = sol.word_count("A a A a b")
    assert out == {"a": 4, "b": 1}

def test_wc_punct():
    reload(sol)
    out = sol.word_count("Hello, HELLO!! hello? It's me.")
    assert out["hello"] == 3
    assert out["it's"] == 1
