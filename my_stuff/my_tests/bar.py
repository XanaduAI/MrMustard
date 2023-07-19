import pytest
from hypothesis import given
import hypothesis.strategies as st
import operator as op

def everything_except(excluded_types):
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )

@given(x=everything_except((int, float)))
def test_not_an_int(x):
    #x = 3
    assert isinstance(x, int) == False


def f():
    raise ValueError()


def test_f_raises_ValueError():
    with pytest.raises(ValueError):
        _ = f()

@given(other = st.integers(min_value=9, max_value=11))
@pytest.mark.parametrize("operator", [op.gt, op.lt])
def test_iterate_over_operators(operator, other):
    assert operator( 20 , other ) == True



