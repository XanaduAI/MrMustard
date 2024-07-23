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
import pytest
from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.gaussian_integrals import (
    real_gaussian_integral,
    complex_gaussian_integral,
    join_Abc,
    join_Abc_real,
    contract_two_Abc,
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
    assert np.allclose(
        res[1], b[2] - math.sum(A[2:, :2] * math.matvec(math.inv(A[:2, :2]), b[:2]))
    )
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


def test_join_Abc():
    """Tests the ``join_Abc`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)

    joined_Abc = join_Abc((A1, b1, c1), (A2, b2, c2))
    assert np.allclose(joined_Abc[0], math.block_diag(A1, A2))
    assert np.allclose(joined_Abc[1], math.concat([b1, b2], axis=-1))
    assert np.allclose(joined_Abc[2], math.outer(c1, c2))


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


def test_complex_gaussian_integral():
    """Tests the ``complex_gaussian_integral`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(x=[0.1, 0.2], y=0.3)

    joined_Abc = join_Abc((A1, b1, c1), (A2, b2, c2))

    res1 = complex_gaussian_integral(joined_Abc, [], [])
    assert np.allclose(res1[0], joined_Abc[0])
    assert np.allclose(res1[1], joined_Abc[1])
    assert np.allclose(res1[2], joined_Abc[2])

    res2 = complex_gaussian_integral(joined_Abc, [0, 1], [4, 5])
    assert np.allclose(res2[0], A3)
    assert np.allclose(res2[1], b3)
    assert np.allclose(res2[2], c3)

    res3 = complex_gaussian_integral(
        join_Abc((A1, b1, c1), (A1, b1, c1)), [0, 1], [2, 3]
    )
    assert np.allclose(res3[0], 0)
    assert np.allclose(res3[1], 0)
    assert np.allclose(res3[2], 1)


def test_complex_gaussian_integral_error():
    """Tests the error of the ``complex_gaussian_integral`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)

    with pytest.raises(ValueError):
        complex_gaussian_integral(
            join_Abc((A1, b1, c1), (A2, b2, c2)),
            [0, 1],
            [
                4,
            ],
        )


def test_contract_two_Abc():
    """Tests the error of the ``contract_two_Abc`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)

    res1 = contract_two_Abc((A1, b1, c1), (A2, b2, c2), [], [])
    assert np.allclose(res1[0], math.block_diag(A1, A2))
    assert np.allclose(
        res1[1], [0, 0, 0.1 + 0.3j, 0.2 + 0.3j, -0.1 + 0.3j, -0.2 + 0.3j]
    )
    assert np.allclose(res1[2], c1 * c2)

    res2 = contract_two_Abc((A1, b1, c1), (A2, b2, c2), [0, 1], [2, 3])
    assert np.allclose(res2[0], math.zeros((2, 2)))
    assert np.allclose(res2[1], [0.1 + 0.3j, 0.2 + 0.3j])
    assert np.allclose(res2[2], c1 * c2)

    res3 = contract_two_Abc((A1, b1, c1), (A2, b2, c2), [0, 1], [0, 1])
    assert np.allclose(res3[0], math.zeros((2, 2)))
    assert np.allclose(res3[1], [-0.1 + 0.3j, -0.2 + 0.3j])
    assert np.allclose(res3[2], c1 * c2)

    res4 = contract_two_Abc((A1, b1, c1), (A2, b2, c2), [], [])
    assert np.allclose(res4[0], res1[0])
    assert np.allclose(res4[1], res1[1])
    assert np.allclose(res4[2], res1[2])


def test_reorder_abc():
    """Test that the reorder_abc function works correctly"""
    A = np.array([[1, 2], [2, 3]])
    b = np.array([4, 5])
    c = np.array(6)
    same = reorder_abc((A, b, c), (0, 1))
    assert all(np.allclose(x, y) for x, y in zip(same, (A, b, c)))
    flipped = reorder_abc((A, b, c), (1, 0))
    assert all(
        np.allclose(x, y)
        for x, y in zip(flipped, (A[[1, 0], :][:, [1, 0]], b[[1, 0]], c))
    )
    c = np.array([[6, 7], [8, 9]])
    flipped = reorder_abc((A, b, c), (1, 0))  #  test transposition of c
    assert all(
        np.allclose(x, y)
        for x, y in zip(flipped, (A[[1, 0], :][:, [1, 0]], b[[1, 0]], c.T))
    )
