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

import numpy as np
from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.gaussian_integrals import (
    real_gaussian_integral,
    complex_gaussian_integral_2,
    complex_gaussian_integral_1,
    join_Abc,
    join_Abc_real,
    reorder_abc,
)


def test_real_gaussian_integral():
    """Tests the ``real_gaussian_integral`` method with a hard-coded A matric from a Gaussian(3) state."""
    A = math.astensor(
        np.array(
            [
                [
                    0.35307718 - 0.09738001j,
                    -0.01297994 + 0.26050244j,
                    0.05349344 - 0.13728068j,
                ],
                [
                    -0.01297994 + 0.26050244j,
                    0.05696707 - 0.2351408j,
                    0.18954838 - 0.42959383j,
                ],
                [
                    0.05349344 - 0.13728068j,
                    0.18954838 - 0.42959383j,
                    -0.16931712 - 0.09205837j,
                ],
            ]
        )
    )
    b = math.astensor(np.arange(3) + 0j)
    c = 1.0 + 0j
    res = real_gaussian_integral((A, b, c), idx=[0, 1])
    assert np.allclose(res[0], A[2, 2] - A[2:, :2] @ math.inv(A[:2, :2]) @ A[:2, 2:])
    assert np.allclose(res[1], b[2] - math.sum(A[2:, :2] * math.matvec(math.inv(A[:2, :2]), b[:2])))
    assert np.allclose(
        res[2],
        c
        * math.sqrt((-2 * np.pi) ** 2, math.complex128)
        / math.sqrt(math.det(A[:2, :2]))
        * math.exp(-0.5 * math.sum(b[:2] * math.matvec(math.inv(A[:2, :2]), b[:2]))),
    )
    res2 = real_gaussian_integral((A, b, c), idx=[])
    assert np.allclose(res2[0], A)
    assert np.allclose(res2[1], b)
    assert np.allclose(res2[2], c)

    A2 = math.astensor(
        np.array(
            [
                [0.35307718 - 0.09738001j, -0.01297994 + 0.26050244j],
                [-0.01297994 + 0.26050244j, 0.05696707 - 0.2351408j],
            ]
        )
    )
    b2 = math.astensor(np.arange(2) + 0j)
    c2 = 1.0 + 0j
    res3 = real_gaussian_integral((A2, b2, c2), idx=[0, 1])
    assert np.allclose(res3[0], math.astensor([]))
    assert np.allclose(res3[1], math.astensor([]))
    assert np.allclose(
        res3[2],
        c2
        * math.sqrt((-2 * np.pi) ** 2, math.complex128)
        / math.sqrt(math.det(A2[:2, :2]))
        * math.exp(-0.5 * math.sum(b2[:2] * math.matvec(math.inv(A2[:2, :2]), b2[:2]))),
    )


def test_join_Abc_real():
    """Tests the ``join_Abc_real`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)
    idx1 = [0, 1]
    idx2 = [0, 1]

    joined_Abc = join_Abc_real((A1, b1, c1), (A2, b2, c2), idx1, idx2)
    assert math.allclose(joined_Abc[0], A2)
    assert math.allclose(joined_Abc[1], b2)
    assert math.allclose(joined_Abc[2], math.outer(c1, c2))

    joined_Abc1 = join_Abc_real((A2, b2, c2), (A1, b1, c1), idx1, idx2)
    assert math.allclose(joined_Abc1[0], A2)
    assert math.allclose(joined_Abc1[1], b2)
    assert math.allclose(joined_Abc1[2], math.outer(c1, c2))

    joined_Abc2 = join_Abc_real((A2, b2, c2), (A1, b1, c1), [0], [0])
    new_joinedA = np.zeros((5, 5)) + 0j * np.zeros((5, 5))
    new_joinedA[0, 0] = A2[0, 0] + A1[0, 0]
    new_joinedA[0, 1:4] = A2[0, 1:]
    new_joinedA[0, 4] = A1[0, 1]
    new_joinedA[1:4, 0] = A2[1:, 0]
    new_joinedA[1:4, 1:4] = A2[1:, 1:]
    new_joinedA[4:0] = A1[1, 0]
    new_joinedA[4:4] = A1[1, 1]

    new_joinedb = np.zeros(5) + 0j * np.zeros(5)
    new_joinedb[0] = b1[0] + b2[0]
    new_joinedb[1:4] = b2[1:]
    new_joinedb[4] = b1[1]
    assert math.allclose(joined_Abc2[0], new_joinedA)
    assert math.allclose(joined_Abc2[1], new_joinedb)
    assert math.allclose(joined_Abc2[2], math.outer(c1, c2))


def test_join_Abc_nonbatched():
    """Tests the ``join_Abc`` method for non-batched inputs."""
    A1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([5, 6])
    c1 = np.array(7)

    A2 = np.array([[8, 9], [10, 11]])
    b2 = np.array([12, 13])
    c2 = np.array(10)

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2))

    assert np.allclose(A, np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 8, 9], [0, 0, 10, 11]]))
    assert np.allclose(b, np.array([5, 6, 12, 13]))
    assert np.allclose(c, 70)


def test_join_Abc_batched_zip():
    """Tests the ``join_Abc`` method for batched inputs in zip mode (and with polynomial c)."""
    A1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b1 = np.array([[5, 6], [7, 8]])
    c1 = np.array([7, 8])

    A2 = np.array([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])
    b2 = np.array([[12, 13], [14, 15]])
    c2 = np.array([10, 100])

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), mode="zip")

    assert np.allclose(
        A,
        np.array(
            [
                [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 8, 9], [0, 0, 10, 11]],
                [[5, 6, 0, 0], [7, 8, 0, 0], [0, 0, 12, 13], [0, 0, 14, 15]],
            ]
        ),
    )
    assert np.allclose(b, np.array([[5, 6, 12, 13], [7, 8, 14, 15]]))
    assert np.allclose(c, np.array([70, 800]))


def test_join_Abc_batched_kron():
    """Tests the ``join_Abc`` method for batched inputs in kron mode (and with polynomial c)."""
    A1 = np.array([[[1, 2], [3, 4]]])
    b1 = np.array([[5, 6]])
    c1 = np.array([7])

    A2 = np.array([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])
    b2 = np.array([[12, 13], [14, 15]])
    c2 = np.array([10, 100])

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), mode="kron")

    assert np.allclose(
        A,
        np.array(
            [
                [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 8, 9], [0, 0, 10, 11]],
                [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 12, 13], [0, 0, 14, 15]],
            ]
        ),
    )
    assert np.allclose(b, np.array([[5, 6, 12, 13], [5, 6, 14, 15]]))
    assert np.allclose(c, np.array([70, 700]))


def test_reorder_abc():
    """Test that the reorder_abc function works correctly"""
    A = np.array([[1, 2], [2, 3]])
    b = np.array([4, 5])
    c = np.array(6)
    same = reorder_abc((A, b, c), (0, 1))
    assert all(np.allclose(x, y) for x, y in zip(same, (A, b, c)))
    flipped = reorder_abc((A, b, c), (1, 0))
    assert all(np.allclose(x, y) for x, y in zip(flipped, (A[[1, 0], :][:, [1, 0]], b[[1, 0]], c)))

    A = np.array([[[1, 2, 3], [2, 4, 5], [3, 5, 6]]])
    b = np.array([[4, 5, 6]])
    c = np.array([[1, 2, 3]])
    same = reorder_abc((A, b, c), (0, 1))
    assert all(np.allclose(x, y) for x, y in zip(same, (A, b, c)))
    flipped = reorder_abc((A, b, c), (1, 0))
    assert all(
        np.allclose(x, y)
        for x, y in zip(flipped, (A[:, [1, 0, 2], :][:, :, [1, 0, 2]], b[:, [1, 0, 2]], c))
    )


def test_complex_gaussian_integral_2_not_batched():
    """Tests the ``complex_gaussian_integral_2`` method for non-batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(x=[0.1, 0.2], y=0.3)

    res = complex_gaussian_integral_2((A1, b1, c1), (A2, b2, c2), [0, 1], [2, 3])
    assert np.allclose(res[0], A3)
    assert np.allclose(res[1], b3)
    assert np.allclose(res[2], c3)


def test_complex_gaussian_integral_2_batched():
    """tests that the ``complex_gaussian_integral_2`` method works for batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2a, b2a, c2a = triples.squeezing_gate_Abc(r=0.1, delta=0.3)
    A2b, b2b, c2b = triples.squeezing_gate_Abc(r=0.2, delta=0.4)
    A2c, b2c, c2c = triples.squeezing_gate_Abc(r=0.3, delta=0.5)
    A3a, b3a, c3a = triples.squeezed_vacuum_state_Abc(r=0.1, phi=0.3)
    A3b, b3b, c3b = triples.squeezed_vacuum_state_Abc(r=0.2, phi=0.4)
    A3c, b3c, c3c = triples.squeezed_vacuum_state_Abc(r=0.3, phi=0.5)
    A1 = math.astensor([A1, A1, A1])
    A2 = math.astensor([A2a, A2b, A2c])
    A3 = math.astensor([A3a, A3b, A3c])
    b1 = math.astensor([b1, b1, b1])
    b2 = math.astensor([b2a, b2b, b2c])
    b3 = math.astensor([b3a, b3b, b3c])
    c1 = math.astensor([c1, c1, c1])
    c2 = math.astensor([c2a, c2b, c2c])
    c3 = math.astensor([c3a, c3b, c3c])

    res = complex_gaussian_integral_2((A1, b1, c1), (A2, b2, c2), [0], [1], mode="zip")
    assert np.allclose(res[0], A3)
    assert np.allclose(res[1], b3)
    assert np.allclose(res[2], c3)


def test_complex_gaussian_integral_1_not_batched():
    """Tests the ``complex_gaussian_integral_1`` method for non-batched inputs."""
    A, b, c = triples.thermal_state_Abc(nbar=[0.5, 0.9, 1.0])
    Ar, br, cr = triples.vacuum_state_Abc(0)

    res = complex_gaussian_integral_1((A, b, c), [0, 2, 4], [1, 3, 5])
    assert np.allclose(res[0], Ar)
    assert np.allclose(res[1], br)
    assert np.allclose(res[2], cr)

    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(x=[0.1, 0.2], y=0.3)

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), mode="zip")

    res = complex_gaussian_integral_1((A, b, c), [0, 1], [4, 5])
    assert np.allclose(res[0], A3)
    assert np.allclose(res[1], b3)
    assert np.allclose(res[2], c3)


def test_complex_gaussian_integral_1_batched():
    """tests that the ``complex_gaussian_integral_2`` method works for batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2a, b2a, c2a = triples.squeezing_gate_Abc(r=0.1, delta=0.3)
    A2b, b2b, c2b = triples.squeezing_gate_Abc(r=0.2, delta=0.4)
    A2c, b2c, c2c = triples.squeezing_gate_Abc(r=0.3, delta=0.5)
    A3a, b3a, c3a = triples.squeezed_vacuum_state_Abc(r=0.1, phi=0.3)
    A3b, b3b, c3b = triples.squeezed_vacuum_state_Abc(r=0.2, phi=0.4)
    A3c, b3c, c3c = triples.squeezed_vacuum_state_Abc(r=0.3, phi=0.5)
    A1 = math.astensor([A1, A1, A1])
    A2 = math.astensor([A2a, A2b, A2c])
    A3 = math.astensor([A3a, A3b, A3c])
    b1 = math.astensor([b1, b1, b1])
    b2 = math.astensor([b2a, b2b, b2c])
    b3 = math.astensor([b3a, b3b, b3c])
    c1 = math.astensor([c1, c1, c1])
    c2 = math.astensor([c2a, c2b, c2c])
    c3 = math.astensor([c3a, c3b, c3c])

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), mode="zip")
    res1 = complex_gaussian_integral_1((A, b, c), [0], [2])
    assert np.allclose(res1[0], A3)
    assert np.allclose(res1[1], b3)
    assert np.allclose(res1[2], c3)


def test_gaussian_integral_poly_batched():
    """Tests that the Gaussian integral works for batched inputs with polynomial c."""
    # batch 4 and 2 polynomial wires
    A = np.random.random((4, 4, 4))
    b = np.random.random((4, 4))
    c = np.random.random((4, 2, 2))
    res = complex_gaussian_integral_1((A, b, c), [0], [1])  # pylint: disable=pointless-statement
    assert res[0].shape == (4, 2, 2)
    assert res[1].shape == (4, 2)
    assert res[2].shape == (4, 2, 2)
