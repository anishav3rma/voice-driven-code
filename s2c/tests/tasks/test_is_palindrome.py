
from importlib import reload
import s2c.user_solution as sol

def test_pal_basic():
    reload(sol)
    assert sol.is_palindrome("A man, a plan, a canal: Panama")
    assert not sol.is_palindrome("race a car")

def test_pal_punct_and_case():
    reload(sol)
    assert sol.is_palindrome("No 'x' in Nixon")
