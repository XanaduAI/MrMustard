# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the lattice module"""

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab import DM, Dgate, Ket, Sgate, Unitary
from mrmustard.math.lattice.strategies.beamsplitter import (
    apply_BS_schwinger,
    beamsplitter,
    sector_idx,
    sector_u,
)
from mrmustard.math.lattice.strategies.binomial import binomial, binomial_dict
from mrmustard.math.lattice.strategies.displacement import displacement, grad_displacement
from mrmustard.math.lattice.strategies.vanilla import vanilla_numba


def test_binomial_vs_binomialDict():
    """Test that binomial and binomial_dict give the same result."""

    A, b, c = Ket.random((0, 1)).bargmann_triple()

    max_prob = 0.9
    local_cutoffs = (10, 10)
    global_cutoff = 15

    G, _ = binomial(local_cutoffs, A, b, c, max_prob, global_cutoff)
    D = binomial_dict(local_cutoffs, A, b, complex(c), max_prob, global_cutoff)

    for idx in D:
        assert np.isclose(D[idx], G[idx])


def test_bs_schwinger():
    "test that the schwinger method to apply a BS works correctly"
    G = Ket.random((0, 1)).fock_array([20, 20])
    G = math.asnumpy(G)
    BS = beamsplitter((20, 20, 20, 20), 1.0, 1.0)
    manual = np.einsum("ab, cdab", G, BS)
    G = apply_BS_schwinger(1.0, 1.0, 0, 1, G)
    assert np.allclose(manual, G)

    Gg = Unitary.random((0, 1)).fock_array([20, 20, 20, 20])
    Gg = math.asnumpy(Gg)
    BS = beamsplitter((20, 20, 20, 20), 2.0, -1.0)
    manual = np.einsum("cdab, abef", BS, Gg)
    Gg = apply_BS_schwinger(2.0, -1.0, 0, 1, Gg)
    assert np.allclose(manual, Gg)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_diagonalbatchNumba_vs_diagonalNumba(batch_size):
    """Test the batch version works versus the normal diagonal version."""
    state = DM.random((0, 1, 2)) >> Dgate(0, 0.0) >> Dgate(1, 0.1) >> Dgate(2, 0.2)
    A, b, c = state.bargmann_triple()

    cutoffs = (18, 19, 20, batch_size)

    # Diagonal MM
    G_ref = math.hermite_renormalized_diagonal(A, b, c, cutoffs=cutoffs[:-1])

    # replicate the B
    b_batched = math.astensor(np.stack((b,) * batch_size, axis=1))

    G_batched = math.hermite_renormalized_diagonal_batch(A, b_batched, c, cutoffs=cutoffs[:-1])

    for nb in range(batch_size):
        assert np.allclose(G_ref, G_batched[:, :, :, nb])


def test_displacement_grad():
    """Tests the value of the analytic gradient for the Dgate against finite differences"""
    cutoff = 4
    r = 2.0
    theta = np.pi / 8
    T = displacement((cutoff, cutoff), r * np.exp(1j * theta))
    Dr, Dtheta = grad_displacement(T, r, theta)

    dr = 0.001
    dtheta = 0.001
    Drp = displacement((cutoff, cutoff), (r + dr) * np.exp(1j * theta))
    Drm = displacement((cutoff, cutoff), (r - dr) * np.exp(1j * theta))
    Dthetap = displacement((cutoff, cutoff), r * np.exp(1j * (theta + dtheta)))
    Dthetam = displacement((cutoff, cutoff), r * np.exp(1j * (theta - dtheta)))
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)


def test_sector_idx():
    "tests that the indices of a sector are calculated correctly"
    assert sector_idx(1, shape=(4, 4)) == [1, 4]
    assert sector_idx(2, shape=(4, 4)) == [2, 5, 8]
    assert sector_idx(1, shape=(3, 4)) == [1, 4]
    assert sector_idx(1, shape=(4, 3)) == [1, 3]
    assert sector_idx(5, shape=(4, 4)) == [11, 14]


def test_sector_u():
    "tests that the unitary of a few sectors is indeed unitary"
    for i in range(1, 10):
        u = sector_u(i, theta=1.129, phi=0.318)
        assert u @ u.conj().T == pytest.approx(np.eye(i + 1))


def test_vanillaNumba_vs_binomial():
    """Test that the vanilla method and the binomial method give the same result."""
    with settings(SEED=42):
        A, b, c = Ket.random((0, 1)).bargmann_triple()
        A, b, c = math.asnumpy(A), math.asnumpy(b), math.asnumpy(c)

        ket_vanilla = vanilla_numba(shape=(10, 10), A=A, b=b, c=c)[:5, :5]
        ket_binomial = binomial(
            local_cutoffs=(5, 5),
            A=A,
            b=b,
            c=c,
            max_l2=0.9999,
            global_cutoff=12,
        )[0][:5, :5]

        assert np.allclose(ket_vanilla, ket_binomial)


def test_vanilla_stable():
    "tests the vanilla stable against other known stable methods"
    with settings(STABLE_FOCK_CONVERSION=True):
        assert np.allclose(
            Dgate(0, x=4.0, y=4.0).fock_array([1000, 1000]),
            displacement((1000, 1000), 4.0 + 4.0j),
        )
        sgate = Sgate(0, r=4.0, phi=2.0).fock_array([1000, 1000])
        assert np.max(np.abs(sgate)) < 1
