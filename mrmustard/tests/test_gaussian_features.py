from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from mrmustard import *
from mrmustard.plugins import gaussian as gp
import numpy as np

import tensorflow as tf




def test_von_neumann_entropy_is_zero_for_pure_states(x, y):
    pure_state = Coherent(tf.constant(x), tf.constant(y))
    print("purestate_cov", pure_state.cov, "entropy=", gp.von_neumann_entropy(pure_state.cov))
    assert np.isclose(gp.von_neumann_entropy(pure_state.cov), 0)


"""
@given(x=st.floats(), y=st.floats())
def test_sympletic_diag(x, y):
    pure_state = Coherent(tf.constant(x), tf.constant(y))
    vals = gp.sympletic_eigenvals(pure_state.cov).numpy()
    print('vals' , vals, np.greater_equal(vals, 1), np.isclose(vals, 1))
    assert np.greater_equal(vals, 1) or np.isclose(vals, 1)
"""
