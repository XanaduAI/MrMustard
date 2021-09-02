from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from mrmustard import *
#  Here we write a bunch of strategies so that the test files can generate inputs, states, transformations, etc.

def covariance_matrix(n):
    array = arrays(dtype=np.float64, shape=(2*n, 2*n), elements=st.floats(min_value=-1e6, max_value=1e6))
    return array + array.T

