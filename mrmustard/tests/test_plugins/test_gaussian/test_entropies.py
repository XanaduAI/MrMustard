from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays

import numpy as np
from mrmustard import *
gp = GaussianPlugin()

@given(x=st.floats(), y=st.floats())
def test_von_neumann_entropy_is_zero_for_pure_states(x, y):
    pure_state = Coherent(x, y)
    assert np.isclose(gp.von_neumann_entropy(pure_state.cov), 0.0)

def test_known_values_of_vne():
    pass


