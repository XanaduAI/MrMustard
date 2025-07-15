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

from mrmustard import math, settings
from mrmustard.physics import triples
from mrmustard.physics.gaussian_integrals import (
    complex_gaussian_integral_1,
    complex_gaussian_integral_2,
    join_Abc,
    join_Abc_real,
    real_gaussian_integral,
)


def test_real_gaussian_integral():
    """Tests the ``real_gaussian_integral`` method with a hard-coded A matric from a Gaussian(3) state."""
    A = math.astensor(
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
        ],
    )
    b = math.cast(math.arange(3), dtype=math.complex128)  # tensorflow does not support complex
    c = 1.0 + 0j
    res = real_gaussian_integral((A, b, c), idx=[0, 1])
    assert math.allclose(res[0], A[2, 2] - A[2:, :2] @ math.inv(A[:2, :2]) @ A[:2, 2:])
    assert math.allclose(
        res[1],
        b[2] - math.sum(A[2:, :2] * math.matvec(math.inv(A[:2, :2]), b[:2])),
    )
    assert math.allclose(
        res[2],
        c
        * math.sqrt((-2 * np.pi) ** 2, math.complex128)
        / math.sqrt(math.det(A[:2, :2]))
        * math.exp(-0.5 * math.sum(b[:2] * math.matvec(math.inv(A[:2, :2]), b[:2]))),
    )
    res2 = real_gaussian_integral((A, b, c), idx=[])
    assert math.allclose(res2[0], A)
    assert math.allclose(res2[1], b)
    assert math.allclose(res2[2], c)

    A2 = math.astensor(
        [
            [0.35307718 - 0.09738001j, -0.01297994 + 0.26050244j],
            [-0.01297994 + 0.26050244j, 0.05696707 - 0.2351408j],
        ],
    )
    b2 = math.cast(math.arange(2), dtype=math.complex128)  # tensorflow does not support complex
    c2 = 1.0 + 0j
    res3 = real_gaussian_integral((A2, b2, c2), idx=[0, 1])
    assert math.allclose(res3[0], math.astensor([]))
    assert math.allclose(res3[1], math.astensor([]))
    assert math.allclose(
        res3[2],
        c2
        * math.sqrt((-2 * np.pi) ** 2, math.complex128)
        / math.sqrt(math.det(A2[:2, :2]))
        * math.exp(-0.5 * math.sum(b2[:2] * math.matvec(math.inv(A2[:2, :2]), b2[:2]))),
    )


def test_join_Abc_real():
    """Tests the ``join_Abc_real`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2, b2, c2 = triples.displacement_gate_Abc(x=0.1, y=0.3)
    idx1 = [0]
    idx2 = [0]

    joined_Abc = join_Abc_real((A1, b1, c1), (A2, b2, c2), idx1, idx2)
    assert math.allclose(joined_Abc[0], A2)
    assert math.allclose(joined_Abc[1], b2)
    assert math.allclose(joined_Abc[2], math.outer(c1, c2))

    joined_Abc1 = join_Abc_real((A2, b2, c2), (A1, b1, c1), idx1, idx2)
    assert math.allclose(joined_Abc1[0], A2)
    assert math.allclose(joined_Abc1[1], b2)
    assert math.allclose(joined_Abc1[2], math.outer(c1, c2))


def test_join_Abc_nonbatched():
    """Tests the ``join_Abc`` method for non-batched inputs."""
    A1 = math.astensor([[1, 2], [3, 4]])
    b1 = math.astensor([5, 6])
    c1 = math.astensor(7)

    A2 = math.astensor([[8, 9], [10, 11]])
    b2 = math.astensor([12, 13])
    c2 = math.astensor(10)

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), batch_string=None)

    assert math.allclose(
        A,
        math.astensor([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 8, 9], [0, 0, 10, 11]]),
    )
    assert math.allclose(b, math.astensor([5, 6, 12, 13]))
    assert math.allclose(c, 70)


def test_join_Abc_batched_zip():
    """Tests the ``join_Abc`` method for batched inputs in zip mode (and with polynomial c)."""
    A1 = math.astensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b1 = math.astensor([[5, 6], [7, 8]])
    c1 = math.astensor([7, 8])

    A2 = math.astensor([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])
    b2 = math.astensor([[12, 13], [14, 15]])
    c2 = math.astensor([10, 100])

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), batch_string="i,i->i")

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
    assert math.allclose(c, math.astensor([70, 800]))


def test_join_Abc_batched_kron():
    """Tests the ``join_Abc`` method for batched inputs in kron mode (and with polynomial c)."""
    A1 = math.astensor([[[1, 2], [3, 4]]])
    b1 = math.astensor([[5, 6]])
    c1 = math.astensor([7])

    A2 = math.astensor([[[8, 9], [10, 11]], [[12, 13], [14, 15]]])
    b2 = math.astensor([[12, 13], [14, 15]])
    c2 = math.astensor([10, 100])

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), batch_string="i,j->ij")

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
    assert math.allclose(c, math.astensor([70, 700]))


def test_complex_gaussian_integral_2_not_batched():
    """Tests the ``complex_gaussian_integral_2`` method for non-batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=0.1, y=0.3)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(x=0.1, y=0.3)

    res = complex_gaussian_integral_2((A1, b1, c1), (A2, b2, c2), [0, 1], [0, 1])
    assert math.allclose(res[0], A3)
    assert math.allclose(res[1], b3)
    assert math.allclose(res[2], c3)


def test_complex_gaussian_integral_2_batched():
    """tests that the ``complex_gaussian_integral_2`` method works for batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2, b2, c2 = triples.squeezing_gate_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])
    A3, b3, c3 = triples.squeezed_vacuum_state_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])

    A1 = math.astensor([A1, A1, A1])
    b1 = math.astensor([b1, b1, b1])
    c1 = math.astensor([c1, c1, c1])

    res = complex_gaussian_integral_2((A1, b1, c1), (A2, b2, c2), [0], [1], batch_string="i,i->i")
    assert math.allclose(res[0], A3)
    assert math.allclose(res[1], b3)
    assert math.allclose(res[2], c3)


def test_complex_gaussian_integral_1_not_batched():
    """Tests the ``complex_gaussian_integral_1`` method for non-batched inputs."""
    A, b, c = triples.thermal_state_Abc(nbar=0.5)
    Ar, br, cr = triples.vacuum_state_Abc(0)

    res = complex_gaussian_integral_1((A, b, c), [0], [1])
    assert math.allclose(res[0], Ar)
    assert math.allclose(res[1], br)
    assert math.allclose(res[2], cr)

    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=0.1, y=0.3)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(x=0.1, y=0.3)

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2))

    res = complex_gaussian_integral_1((A, b, c), [0, 1], [2, 3])
    assert math.allclose(res[0], A3)
    assert math.allclose(res[1], b3)
    assert math.allclose(res[2], c3)


def test_complex_gaussian_integral_1_batched():
    """tests that the ``complex_gaussian_integral_2`` method works for batched inputs."""
    A1, b1, c1 = triples.vacuum_state_Abc(1)
    A2, b2, c2 = triples.squeezing_gate_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])
    A3, b3, c3 = triples.squeezed_vacuum_state_Abc(r=[0.1, 0.2, 0.3], phi=[0.3, 0.4, 0.5])

    A1 = math.astensor([A1, A1, A1])
    b1 = math.astensor([b1, b1, b1])
    c1 = math.astensor([c1, c1, c1])

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), batch_string="i,i->i")
    res1 = complex_gaussian_integral_1((A, b, c), [0], [2])
    assert math.allclose(res1[0], A3)
    assert math.allclose(res1[1], b3)
    assert math.allclose(res1[2], c3)


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

    A, b, c = join_Abc((A1, b1, c1), (A2, b2, c2), batch_string="ij,ij->ij")
    res1 = complex_gaussian_integral_1((A, b, c), [0], [2])
    assert math.allclose(res1[0], A3)
    assert math.allclose(res1[1], b3)
    assert math.allclose(res1[2], c3)


def test_gaussian_integral_poly_batched():
    """Tests that the Gaussian integral works for batched inputs with polynomial c."""
    # batch 4 and 2 polynomial wires
    A = settings.rng.random((4, 4, 4))
    b = settings.rng.random((4, 4))
    c = settings.rng.random((4, 2, 2))
    res = complex_gaussian_integral_1((A, b, c), [0], [1])
    assert res[0].shape == (4, 2, 2)
    assert res[1].shape == (4, 2)
    assert res[2].shape == (4, 2, 2)
