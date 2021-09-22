from hypothesis import strategies as st, given
from hypothesis.extra.numpy import arrays
from mrmustard import DisplacedSqueezed
from mrmustard.experimental import XPTensor
import numpy as np
from mrmustard.tests.random import random_pure_state

even = st.integers(min_value=2, max_value=10).filter(lambda x: x % 2 == 0)

@st.composite
def matrix(draw):  # rectangular
    return draw(arrays(np.float64, shape=(draw(even), draw(even)), elements=st.floats(allow_nan=False, allow_infinity=False)))

@st.composite
def square_matrix(draw, min_size=2):
    e = draw(st.integers(min_value=min_size, max_value=10).filter(lambda x: x % 2 == 0))
    return draw(arrays(np.float64, shape=(e, e), elements=st.floats(allow_nan=False, allow_infinity=False)))

@st.composite
def rectangular_matrix(draw): # strictly rectangular
    a,b = draw(st.tuples(even, even).filter(lambda x: x[0] != x[1]))
    return draw(arrays(np.float64, shape=(a, b), elements=st.floats(allow_nan=False, allow_infinity=False)))

@st.composite
def vector(draw):
    return draw(arrays(np.float64, shape=(draw(even),), elements=st.floats(allow_nan=False, allow_infinity=False)))

def test_like_1_like_0():
    assert XPTensor(like_0 = False, like_1 = True).like_1
    assert XPTensor(like_0 = True, like_1 = False).like_0
    assert XPTensor(like_0 = True).like_0
    assert XPTensor(like_0 = False).like_1
    assert XPTensor(like_1 = True).like_1
    assert XPTensor(like_1 = False).like_0

@given(matrix=matrix())
def test_from_xxpp_xpxp_matrix(matrix):
    N = matrix.shape[0]//2
    M = matrix.shape[1]//2
    modes = (list(range(N)), list(range(M))) if N == M else (list(range(N)), list(range(N, M+N)))
    x = XPTensor(matrix, modes=modes, like_1=N==M)
    xp1 = XPTensor.from_xxpp(matrix, modes=modes, like_1=N==M)
    xpxp_matrix = np.reshape(np.transpose(np.reshape(matrix, (2,N,2,M)), (1,0,3,2)), (2*N,2*M))
    xp2 = XPTensor.from_xpxp(xpxp_matrix, modes=modes, like_1=N==M)
    assert np.allclose(xp1.tensor, xp2.tensor)

@given(vector())
def test_from_xpxp_xxpp_vector(vector):
    N = vector.shape[0]//2
    xp1 = XPTensor.from_xxpp(vector, like_0=True)
    xpxp_vector = np.reshape(np.transpose(np.reshape(vector, (2,N)), (1,0)), (-1,))
    xp2 = XPTensor.from_xpxp(xpxp_vector, like_0=True)
    assert np.allclose(xp1.tensor, xp2.tensor)

@given(matrix())
def test_from_xpxp_to_xpxp(matrix):
    N = matrix.shape[0]//2
    M = matrix.shape[1]//2
    modes = (list(range(N)), list(range(M))) if N == M else (list(range(N)), list(range(N, M+N)))
    xpxp_matrix = np.reshape(np.transpose(np.reshape(matrix, (2,N,2,M)), (1,0,3,2)), (2*N,2*M))
    xp1 = XPTensor.from_xpxp(xpxp_matrix, modes=modes, like_1=N==M)
    assert np.allclose(xp1.to_xpxp(), xpxp_matrix)

@given(matrix())
def test_from_xxpp_to_xxpp(matrix):
    N = matrix.shape[0]//2
    M = matrix.shape[1]//2
    modes = (list(range(N)), list(range(M))) if N == M else (list(range(N)), list(range(N, M+N)))
    xp1 = XPTensor.from_xxpp(matrix, modes=modes, like_1=N==M)
    assert np.allclose(xp1.to_xxpp(), matrix)

@given(matrix())
def test_xxpp_to_xpxp_to_xxpp_to_xpxp(matrix):
    N = matrix.shape[0]//2
    M = matrix.shape[1]//2
    modes = (list(range(N)), list(range(M))) if N == M else (list(range(N)), list(range(N, M+N)))
    xpxp_matrix = np.reshape(np.transpose(np.reshape(matrix, (2,N,2,M)), (1,0,3,2)), (2*N,2*M))
    xp1 = XPTensor.from_xpxp(xpxp_matrix, modes=modes, like_1=N==M)
    xp2 = XPTensor.from_xxpp(xp1.to_xxpp(), modes=modes, like_1=N==M)
    xp3 = XPTensor.from_xpxp(xp2.to_xpxp(), modes=modes, like_1=N==M)
    assert np.allclose(matrix, xp3.to_xxpp())
    xp4 = XPTensor.from_xxpp(xp3.to_xxpp(), modes=modes, like_1=N==M)
    assert np.allclose(xpxp_matrix, xp4.to_xpxp())

@given(matrix())
def test_matmul_same_modes(matrix):
    N = matrix.shape[0]//2
    M = matrix.shape[1]//2
    modes = (list(range(N)), list(range(M))) if N == M else (list(range(N)), list(range(N, M+N)))
    xp1 = XPTensor.from_xxpp(matrix, modes=modes, like_1=N==M)
    assert np.allclose((xp1 @ xp1.T).to_xxpp(), matrix @ matrix.T)

@given(square_matrix())
def test_matmul_different_modes(xpxp_matrix):
    N = xpxp_matrix.shape[0]//2
    xp1 = XPTensor.from_xpxp(xpxp_matrix, modes=list(range(N)), like_1=True)
    xp2 = XPTensor.from_xpxp(xpxp_matrix, modes=list(range(1,N+1)), like_1=True)
    matrix1 = np.block([[xpxp_matrix, np.zeros((2*N,2))], [np.zeros((2,2*N)), np.eye(2)]])  # add one extra empty mode at the end
    matrix2 = np.block([[np.eye(2), np.zeros((2,2*N))], [np.zeros((2*N,2)), xpxp_matrix]])  # add one extra empty mode at the beginning
    prod = xp1 @ xp2
    assert np.allclose(prod.to_xpxp(), matrix1 @ matrix2)

@given(rectangular_matrix())
def test_matmul_same_modes_coherence(xpxp_matrix):
    N = xpxp_matrix.shape[0]//2
    M = xpxp_matrix.shape[1]//2
    coh = XPTensor.from_xpxp(xpxp_matrix, modes=[list(range(N)), list(range(N,M+N))], like_0=True)
    assert np.allclose((coh @ coh.T).to_xpxp(), xpxp_matrix @ xpxp_matrix.T)

@given(rectangular_matrix().filter(lambda x: x.shape[0] > 2 and x.shape[1] > 2))
def test_matmul_different_modes_coherence(xpxp_matrix):
    N = xpxp_matrix.shape[0]//2
    M = xpxp_matrix.shape[1]//2
    coh1 = XPTensor.from_xpxp(xpxp_matrix, modes=[list(range(N)), list(range(N,M+N))], like_0=True)
    coh2 = XPTensor.from_xpxp(xpxp_matrix, modes=[list(range(N)), list(range(N+1,M+N+1))], like_0=True)
    matrix1 = np.block([[xpxp_matrix, np.zeros((2*N,2))]])  # add a column on the right
    matrix2 = np.block([[np.zeros((2*N,2)), xpxp_matrix]])  # add a column on the left
    assert np.allclose((coh1 @ coh2.T).to_xpxp(), matrix1 @ matrix2.T)

@given(rectangular_matrix().filter(lambda x: x.shape[0] > 2 and x.shape[1] > 2))
def test_matmul_all_different_modes_coherence(xpxp_matrix):
    N = xpxp_matrix.shape[0]//2
    M = xpxp_matrix.shape[1]//2
    coh1 = XPTensor.from_xpxp(xpxp_matrix, modes=[list(range(N)), list(range(N,N+M))], like_0=True)
    coh2 = XPTensor.from_xpxp(xpxp_matrix, modes=[list(range(N)), list(range(N+M+1,N+M+M))], like_0=True)
    assert (coh1 @ coh2.T).to_xpxp() is None
