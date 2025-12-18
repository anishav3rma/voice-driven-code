
import importlib
import s2c.user_solution as sol

# sol = importlib.import_module('s2c.sample_word_count')

def test_basic():
    text = "A a A a b"
    out = sol.word_count(text)
    assert out == {"a": 4, "b": 1}
    print("test_basic")

def test_punct_and_case():
    text = "Hello, HELLO!! hello? It's me."
    out = sol.word_count(text)
    assert out["hello"] == 3
    assert out["it's"] == 1
    print("test_punc")

def test_empty():
    out = sol.word_count("")
    assert out == {}
    print("test_empty")
