# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for real and comple gaussian integral functions and related helper functions."""

import pytest

from mrmustard import math, settings
from mrmustard.physics import triples
from mrmustard.physics.utils import join_Abc


@pytest.mark.parametrize("return_log_c", [True, False])
def test_join_Abc_nonbatched(return_log_c):
    """Tests the ``join_Abc`` method for non-batched inputs."""
    A1 = math.astensor([[1, 2], [3, 4]])
    b1 = math.astensor([5, 6])
    c1 = math.astensor(7)

    A2 = math.astensor([[8, 9], [10, 11]])
    b2 = math.astensor([12, 13])
    c2 = math.astensor(10)

    A, b, c = join_Abc(A1, b1, c1, A2, b2, c2, return_log_c=return_log_c)

    assert math.allclose(
        A,
        math.astensor([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 8, 9], [0, 0, 10, 11]]),
    )
    assert math.allclose(b, math.astensor([5, 6, 12, 13]))
    if return_log_c:
        assert math.allclose(c, math.log(math.cast(70, "complex128")))
    else:
        assert math.allclose(c, 70)


@pytest.mark.parametrize("return_log_c", [True, False])
def test_join_Abc_batched_zip(return_log_c):
    """Tests the ``join_Abc`` method for batched inputs with automatic broadcasting."""
    A1 = math.astensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b1 = math.astensor([[5, 6], [7, 8]])
    c1 = math.astensor([7, 8])

    A2 = math.astensor([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])
    b2 = math.astensor([[12, 13], [14, 15]])
    c2 = math.astensor([10, 100])

    A, b, c = join_Abc(A1, b1, c1, A2, b2, c2, return_log_c=return_log_c)

    assert math.allclose(
        A,
        math.astensor(
            [
                [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 8, 9], [0, 0, 10, 11]],
                [[5, 6, 0, 0], [7, 8, 0, 0], [0, 0, 12, 13], [0, 0, 14, 15]],
            ],
        ),
    )
    assert math.allclose(b, math.astensor([[5, 6, 12, 13], [7, 8, 14, 15]]))
    if return_log_c:
        assert math.allclose(c, math.log(math.cast(math.astensor([70, 800]), "complex128")))
    else:
        assert math.allclose(c, math.astensor([70, 800]))


@pytest.mark.parametrize("return_log_c", [True, False])
def test_join_Abc_batched_kron(return_log_c):
    """Tests the ``join_Abc`` method for batched inputs with broadcasting (kron-like behavior)."""
    A1 = math.astensor([[[1, 2], [3, 4]]])
    b1 = math.astensor([[5, 6]])
    c1 = math.astensor([7])

    A2 = math.astensor([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])
    b2 = math.astensor([[12, 13], [14, 15]])
    c2 = math.astensor([10, 100])

    A, b, c = join_Abc(A1, b1, c1, A2, b2, c2, return_log_c=return_log_c)

    assert math.allclose(
        A,
        math.astensor(
            [
                [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 8, 9], [0, 0, 10, 11]],
                [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 12, 13], [0, 0, 14, 15]],
            ],
        ),
    )
    assert math.allclose(b, math.astensor([[5, 6, 12, 13], [5, 6, 14, 15]]))
    if return_log_c:
        assert math.allclose(c, math.log(math.cast(math.astensor([70, 700]), "complex128")))
    else:
        assert math.allclose(c, math.astensor([70, 700]))


def test_complex_gaussian_integral_2_not_batched():
    """Tests the ``complex_gaussian_integral_2`` method for non-batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(0.1 + 0.3j)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(alpha=0.1 + 0.3j)

    res = math.complex_gaussian_integral_2(A1, b1, A2, b2, [0, 1], [0, 1])
    assert math.allclose(res[0], A3)
    assert math.allclose(res[1], b3)
    assert math.allclose(math.exp(res[2]) * c2 * c1, c3)


def test_complex_gaussian_integral_2_batched():
    """tests that the ``complex_gaussian_integral_2`` method works for batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2, b2, c2 = triples.squeezing_gate_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])
    A3, b3, c3 = triples.squeezed_vacuum_state_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])

    A1 = math.astensor([A1, A1, A1])
    b1 = math.astensor([b1, b1, b1])
    c1 = math.astensor([c1, c1, c1])

    res = math.complex_gaussian_integral_2(A1, b1, A2, b2, [0], [1])
    assert math.allclose(res[0], A3)
    assert math.allclose(res[1], b3)
    assert math.allclose(math.exp(res[2]) * c2 * c1, c3)


def test_complex_gaussian_integral_1_not_batched():
    """Tests the ``complex_gaussian_integral_1`` method for non-batched inputs."""
    A, b, c = triples.thermal_state_Abc(nbar=0.5)
    Ar, br, cr = triples.vacuum_state_Abc(0)

    res = math.complex_gaussian_integral_1(A, b, [0, 1])
    assert math.allclose(res[0], Ar)
    assert math.allclose(res[1], br)
    assert math.allclose(math.exp(res[2]) * c, cr)

    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(0.1 + 0.3j)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(0.1 + 0.3j)

    A, b, c = join_Abc(A1, b1, c1, A2, b2, c2)

    res = math.complex_gaussian_integral_1(A, b, [0, 1, 2, 3])
    assert math.allclose(res[0], A3)
    assert math.allclose(res[1], b3)
    assert math.allclose(math.exp(res[2]) * c, c3)


def test_complex_gaussian_integral_1_batched():
    """tests that the ``complex_gaussian_integral_2`` method works for batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2, b2, c2 = triples.squeezing_gate_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])
    A3, b3, c3 = triples.squeezed_vacuum_state_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])

    A1 = math.astensor([A1, A1, A1])
    b1 = math.astensor([b1, b1, b1])
    c1 = math.astensor([c1, c1, c1])

    A, b, c = join_Abc(A1, b1, c1, A2, b2, c2)
    res1 = math.complex_gaussian_integral_1(A, b, [0, 2])
    assert math.allclose(res1[0], A3)
    assert math.allclose(res1[1], b3)
    assert math.allclose(math.exp(res1[2]) * c, c3)


def test_complex_gaussian_integral_1_multidim_batched():
    """tests that the ``complex_gaussian_integral_2`` method works for multi-dimensional batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2, b2, c2 = triples.squeezing_gate_Abc(
        r=[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        phi=[[0.3, 0.4, 0.5], [0.3, 0.4, 0.5]],
    )
    A3, b3, c3 = triples.squeezed_vacuum_state_Abc(
        r=[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        phi=[[0.3, 0.4, 0.5], [0.3, 0.4, 0.5]],
    )

    A1 = math.astensor([[A1, A1, A1], [A1, A1, A1]])
    b1 = math.astensor([[b1, b1, b1], [b1, b1, b1]])
    c1 = math.astensor([[c1, c1, c1], [c1, c1, c1]])

    A, b, c = join_Abc(A1, b1, c1, A2, b2, c2)
    res1 = math.complex_gaussian_integral_1(A, b, [0, 2])
    assert math.allclose(res1[0], A3)
    assert math.allclose(res1[1], b3)
    assert math.allclose(math.exp(res1[2]) * c, c3)


def test_gaussian_integral_poly_batched():
    """Tests that the Gaussian integral works for batched inputs with polynomial c."""
    # batch 4 and 2 polynomial wires
    A = settings.get_rng().random((4, 4, 4))
    b = settings.get_rng().random((4, 4))
    res = math.complex_gaussian_integral_1(A, b, [0, 1])
    assert res[0].shape == (4, 2, 2)
    assert res[1].shape == (4, 2)
    assert res[2].shape == (4,)
