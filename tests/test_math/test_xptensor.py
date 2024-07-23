# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hypothesis import strategies as st, given
from hypothesis.extra.numpy import arrays
from mrmustard.math.tensor_wrappers.xptensor import XPVector, XPMatrix
import numpy as np

even = st.integers(min_value=2, max_value=10).filter(lambda x: x % 2 == 0)
floats = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def matrix(draw):  # square or rectangular
    return draw(arrays(np.float64, shape=(draw(even), draw(even)), elements=floats))


@st.composite
def square_matrix(draw, min_size=2, max_size=10):  # strictly square
    e = draw(
        st.integers(min_value=min_size, max_value=max_size).filter(lambda x: x % 2 == 0)
    )
    return draw(arrays(np.float64, shape=(e, e), elements=floats))


@st.composite
def rectangular_matrix(draw):  # strictly rectangular
    a, b = draw(st.tuples(even, even).filter(lambda x: x[0] != x[1]))
    return draw(arrays(np.float64, shape=(a, b), elements=floats))


@st.composite
def vector(draw, size=None):
    if size is not None:
        return draw(arrays(np.float64, shape=(size,), elements=floats))
    return draw(arrays(np.float64, shape=(draw(even),), elements=floats))


def test_like_1_like_0():
    assert XPMatrix(like_0=False, like_1=True).like_1
    assert XPMatrix(like_0=True, like_1=False).like_0
    assert XPMatrix(like_0=True).like_0
    assert XPMatrix(like_0=False).like_1
    assert XPMatrix(like_1=True).like_1
    assert XPMatrix(like_1=False).like_0
    assert XPVector().like_0


@given(matrix())
def test_from_xxpp_equals_from_xpxp_matrix(matrix):
    N = matrix.shape[0] // 2
    M = matrix.shape[1] // 2
    modes = (
        (list(range(N)), list(range(M)))
        if N == M
        else (list(range(N)), list(range(N, M + N)))
    )
    xp1 = XPMatrix.from_xxpp(matrix, modes=modes, like_1=N == M)
    xpxp_matrix = np.reshape(
        np.transpose(np.reshape(matrix, (2, N, 2, M)), (1, 0, 3, 2)), (2 * N, 2 * M)
    )
    xp2 = XPMatrix.from_xpxp(xpxp_matrix, modes=modes, like_1=N == M)
    assert np.allclose(xp1.tensor, xp2.tensor)


@given(vector())
def test_from_xpxp_equals_from_xxpp_vector(vector):
    N = vector.shape[0] // 2
    xp1 = XPVector.from_xxpp(vector)
    xpxp_vector = np.reshape(np.transpose(np.reshape(vector, (2, N)), (1, 0)), (-1,))
    xp2 = XPVector.from_xpxp(xpxp_vector)
    assert np.allclose(xp1.tensor, xp2.tensor)


@given(matrix())
def test_from_xpxp_to_xpxp_is_the_same(matrix):
    N = matrix.shape[0] // 2
    M = matrix.shape[1] // 2
    modes = (
        (list(range(N)), list(range(M)))
        if N == M
        else (list(range(N)), list(range(N, M + N)))
    )
    xpxp_matrix = np.reshape(
        np.transpose(np.reshape(matrix, (2, N, 2, M)), (1, 0, 3, 2)), (2 * N, 2 * M)
    )
    xp1 = XPMatrix.from_xpxp(xpxp_matrix, modes=modes, like_1=N == M)
    assert np.allclose(xp1.to_xpxp(), xpxp_matrix)


@given(matrix())
def test_from_xxpp_to_xxpp_is_the_same(matrix):
    N = matrix.shape[0] // 2
    M = matrix.shape[1] // 2
    modes = (
        (list(range(N)), list(range(M)))
        if N == M
        else (list(range(N)), list(range(N, M + N)))
    )
    xp1 = XPMatrix.from_xxpp(matrix, modes=modes, like_1=N == M)
    assert np.allclose(xp1.to_xxpp(), matrix)


@given(matrix())
def test_xxpp_to_xpxp_to_xxpp_to_xpxp(matrix):
    N = matrix.shape[0] // 2
    M = matrix.shape[1] // 2
    modes = (
        (list(range(N)), list(range(M)))
        if N == M
        else (list(range(N)), list(range(N, M + N)))
    )
    xpxp_matrix = np.reshape(
        np.transpose(np.reshape(matrix, (2, N, 2, M)), (1, 0, 3, 2)), (2 * N, 2 * M)
    )
    xp1 = XPMatrix.from_xpxp(xpxp_matrix, modes=modes, like_1=N == M)
    xp2 = XPMatrix.from_xxpp(xp1.to_xxpp(), modes=modes, like_1=N == M)
    xp3 = XPMatrix.from_xpxp(xp2.to_xpxp(), modes=modes, like_1=N == M)
    assert np.allclose(matrix, xp3.to_xxpp())
    xp4 = XPMatrix.from_xxpp(xp3.to_xxpp(), modes=modes, like_1=N == M)
    assert np.allclose(xpxp_matrix, xp4.to_xpxp())


# TESTING MATMUL
@given(matrix())
def test_matmul_all_same_modes(matrix):
    N = matrix.shape[0] // 2
    M = matrix.shape[1] // 2
    modes = (
        (list(range(N)), list(range(M)))
        if N == M
        else (list(range(N)), list(range(N, M + N)))
    )
    xp1 = XPMatrix.from_xxpp(matrix, modes=modes, like_1=N == M)
    assert np.allclose((xp1 @ xp1.T).to_xxpp(), matrix @ matrix.T)


@given(square_matrix())
def test_matmul_few_different_modes(xpxp_matrix):
    N = xpxp_matrix.shape[0] // 2
    xp1 = XPMatrix.from_xpxp(
        xpxp_matrix, modes=(list(range(N)), list(range(N))), like_1=True
    )
    xp2 = XPMatrix.from_xpxp(
        xpxp_matrix, modes=(list(range(1, N + 1)), list(range(1, N + 1))), like_1=True
    )
    matrix1 = np.block(
        [[xpxp_matrix, np.zeros((2 * N, 2))], [np.zeros((2, 2 * N)), np.eye(2)]]
    )  # add one extra empty mode at the end
    matrix2 = np.block(
        [[np.eye(2), np.zeros((2, 2 * N))], [np.zeros((2 * N, 2)), xpxp_matrix]]
    )  # add one extra empty mode at the beginning
    prod = xp1 @ xp2
    assert np.allclose(prod.to_xpxp(), matrix1 @ matrix2)


@given(a=square_matrix(), b=square_matrix())
def test_matmul_all_different_modes(a, b):
    Na = a.shape[0] // 2
    Nb = b.shape[0] // 2
    xp1 = XPMatrix.from_xpxp(a, modes=(list(range(Na)), list(range(Na))), like_1=True)
    xp2 = XPMatrix.from_xpxp(
        b, modes=(list(range(Na, Na + Nb)), list(range(Na, Na + Nb))), like_1=True
    )
    matrix1 = np.block(
        [[a, np.zeros((2 * Na, 2 * Nb))], [np.zeros((2 * Nb, 2 * Na)), np.eye(2 * Nb)]]
    )  # add N empty modes at the end
    matrix2 = np.block(
        [[np.eye(2 * Na), np.zeros((2 * Na, 2 * Nb))], [np.zeros((2 * Nb, 2 * Na)), b]]
    )  # add N empty modes at the beginning
    prod = xp1 @ xp2
    assert np.allclose(prod.to_xpxp(), matrix1 @ matrix2, rtol=1e-5)


@given(rectangular_matrix())
def test_matmul_all_same_modes_coherence(coherence):
    N = coherence.shape[0] // 2
    M = coherence.shape[1] // 2
    coh = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N, M + N))), like_0=True
    )
    assert np.allclose((coh @ coh.T).to_xpxp(), coherence @ coherence.T, rtol=1e-5)


@given(rectangular_matrix().filter(lambda x: x.shape[0] > 2 and x.shape[1] > 2))
def test_matmul_few_different_modes_coherence(coherence):
    N = coherence.shape[0] // 2
    M = coherence.shape[1] // 2
    coh1 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N, M + N))), like_0=True
    )
    coh2 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N + 1, M + N + 1))), like_0=True
    )
    matrix1 = np.block([[coherence, np.zeros((2 * N, 2))]])  # add a column on the right
    matrix2 = np.block([[np.zeros((2 * N, 2)), coherence]])  # add a column on the left
    assert np.allclose((coh1 @ coh2.T).to_xpxp(), (matrix1 @ matrix2.T), rtol=1e-5)


@given(rectangular_matrix().filter(lambda x: x.shape[0] > 2 and x.shape[1] > 2))
def test_matmul_all_different_modes_coherence(coherence):
    N = coherence.shape[0] // 2
    M = coherence.shape[1] // 2
    coh1 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N, N + M))), like_0=True
    )
    coh2 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N + M, N + M + M))), like_0=True
    )
    assert (coh1 @ coh2.T).to_xpxp() is None


# TESTING MATVEC
@st.composite
def mat_vec(draw, compatible: bool):
    mat = draw(square_matrix())
    vec = draw(vector(mat.shape[1])) if compatible else draw(vector())
    return mat, vec


@given(mat_vec(compatible=True))
def test_matvec_all_same_modes(mat_vec):
    mat, vec = mat_vec
    expected = mat @ vec
    N = mat.shape[0] // 2
    mat = XPMatrix.from_xpxp(mat, modes=(list(range(N)), list(range(N))), like_1=True)
    vec = XPVector.from_xpxp(vec, modes=list(range(N)))
    assert np.allclose((mat @ vec).to_xpxp(), expected, rtol=1e-5)


@given(
    mat_vec(compatible=True).filter(lambda x: x[0].shape[0] > 2 and x[0].shape[1] > 2)
)
def test_matvec_few_different_modes(mat_vec):
    mat, vec = mat_vec
    N = mat.shape[0] // 2
    expected = mat[:, 2:] @ vec[2:]  # 2 is skip one mode
    mat = XPMatrix.from_xpxp(
        mat, modes=(list(range(N)), [1000] + list(range(N, 2 * N - 1))), like_0=True
    )
    vec = XPVector.from_xpxp(vec, modes=[500] + list(range(N, 2 * N - 1)))
    assert np.allclose((mat @ vec).to_xpxp(), expected, rtol=1e-5)


# TESTING VECVEC


@given(vector())
def test_vecvec_all_same_modes(vec):
    N = vec.shape[0] // 2
    vec1 = XPVector.from_xpxp(vec, modes=list(range(N)))
    vec2 = XPVector.from_xpxp(vec, modes=list(range(N)))
    assert np.allclose(vec1 @ vec2, vec @ vec, rtol=1e-5)


# TESTING ADDITION


@given(square_matrix())
def test_addition_all_same_modes_cov(xpxp_matrix):
    N = xpxp_matrix.shape[0] // 2
    xp1 = XPMatrix.from_xpxp(
        xpxp_matrix, modes=(list(range(N)), list(range(N))), like_1=True
    )
    xp2 = XPMatrix.from_xpxp(
        xpxp_matrix, modes=(list(range(N)), list(range(N))), like_1=True
    )
    assert np.allclose((xp1 + xp2).to_xpxp(), xpxp_matrix + xpxp_matrix, rtol=1e-5)


@given(matrix())
def test_addition_few_different_modes_coherences(coherence):
    N = coherence.shape[0] // 2
    M = coherence.shape[1] // 2
    coh1 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N, M + N))), like_0=True
    )
    coh2 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N + 1, M + N + 1))), like_0=True
    )
    matrix1 = np.block([[coherence, np.zeros((2 * N, 2))]])  # add a column on the right
    matrix2 = np.block([[np.zeros((2 * N, 2)), coherence]])  # add a column on the left
    assert np.allclose((coh1 + coh2).to_xpxp(), matrix1 + matrix2, rtol=1e-5)


@given(matrix())
def test_addition_all_different_modes_coherences(coherence):
    N = coherence.shape[0] // 2
    M = coherence.shape[1] // 2
    coh1 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N, N + M))), like_0=True
    )
    coh2 = XPMatrix.from_xpxp(
        coherence, modes=(list(range(N)), list(range(N + M, N + M + M))), like_0=True
    )
    matrix1 = np.block([[coherence, np.zeros((2 * N, 2 * M))]])
    matrix2 = np.block([[np.zeros((2 * N, 2 * M)), coherence]])
    assert np.allclose((coh1 + coh2).to_xpxp(), matrix1 + matrix2, rtol=1e-5)
