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

import importlib
import numpy as np
import pytest

from mrmustard.lab import Gaussian, Dgate
from mrmustard import lab_dev as mmld
from mrmustard import settings, math
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.math.lattice.strategies.binomial import binomial, binomial_dict
from mrmustard.math.lattice.strategies.beamsplitter import (
    apply_BS_schwinger,
    beamsplitter,
    sector_idx,
    sector_u,
)
from mrmustard.math.lattice.strategies.displacement import displacement
from mrmustard.math.lattice.strategies.vanilla import vanilla_average

original_precision = settings.PRECISION_BITS_HERMITE_POLY

do_julia = bool(importlib.util.find_spec("juliacall"))
precisions = (
    [128, 256, 384, 512]
    if do_julia
    else [
        128,
    ]
)


@pytest.mark.parametrize("precision", precisions)
def test_vanillaNumba_vs_binomial(precision):
    """Test that the vanilla method and the binomial method give the same result.
    Test is repeated for all possible values of PRECISION_BITS_HERMITE_POLY."""

    settings.PRECISION_BITS_HERMITE_POLY = precision
    G = Gaussian(2)

    ket_vanilla = G.ket(cutoffs=[10, 10])[:5, :5]
    ket_binomial = G.ket(max_photons=10)[:5, :5]

    assert np.allclose(ket_vanilla, ket_binomial)

    settings.PRECISION_BITS_HERMITE_POLY = original_precision


def test_binomial_vs_binomialDict():
    """Test that binomial and binomial_dict give the same result."""

    A, b, c = Gaussian(2).bargmann(numpy=True)
    max_prob = 0.9
    local_cutoffs = (10, 10)
    global_cutoff = 15

    G, norm = binomial(local_cutoffs, A, b, c.item(), max_prob, global_cutoff)
    D = binomial_dict(local_cutoffs, A, b, c.item(), max_prob, global_cutoff)

    for idx in D.keys():
        assert np.isclose(D[idx], G[idx])


@pytest.mark.parametrize("batch_size", [1, 3])
def test_vanillabatchNumba_vs_vanillaNumba(batch_size):
    """Test the batch version works versus the normal vanilla version."""
    state = Gaussian(3) >> Dgate([0.0, 0.1, 0.2])
    A, B, C = wigner_to_bargmann_rho(
        state.cov, state.means
    )  # Create random state (M mode Gaussian state with displacement)

    cutoffs = (20, 20, 20, 20)

    # Vanilla MM
    G_ref = math.hermite_renormalized(A, B, C, shape=cutoffs)

    # replicate the B
    B_batched = np.stack((B,) * batch_size, axis=0)

    G_batched = math.hermite_renormalized_batch(A, B_batched, C, shape=cutoffs)

    for nb in range(batch_size):
        assert np.allclose(G_ref, G_batched[nb, :, :, :, :])


@pytest.mark.parametrize("batch_size", [1, 3])
def test_diagonalbatchNumba_vs_diagonalNumba(batch_size):
    """Test the batch version works versus the normal diagonal version."""
    state = Gaussian(3) >> Dgate([0.0, 0.1, 0.2])
    A, B, C = wigner_to_bargmann_rho(
        state.cov, state.means
    )  # Create random state (M mode Gaussian state with displacement)

    cutoffs = (18, 19, 20, batch_size)

    # Diagonal MM
    G_ref = math.hermite_renormalized_diagonal(A, B, C, cutoffs=cutoffs[:-1])

    # replicate the B
    B_batched = np.stack((B,) * batch_size, axis=1)

    G_batched = math.hermite_renormalized_diagonal_batch(A, B_batched, C, cutoffs=cutoffs[:-1])

    for nb in range(batch_size):
        assert np.allclose(G_ref, G_batched[:, :, :, nb])


def test_bs_schwinger():
    "test that the schwinger method to apply a BS works correctly"
    G = mmld.Ket.random([0, 1]).fock([20, 20])
    G = math.asnumpy(G)
    BS = beamsplitter((20, 20, 20, 20), 1.0, 1.0)
    manual = np.einsum("ab, cdab", G, BS)
    G = apply_BS_schwinger(1.0, 1.0, 0, 1, G)
    assert np.allclose(manual, G)

    Gg = mmld.Unitary.random([0, 1]).fock([20, 20, 20, 20])
    Gg = math.asnumpy(Gg)
    BS = beamsplitter((20, 20, 20, 20), 2.0, -1.0)
    manual = np.einsum("cdab, abef", BS, Gg)
    Gg = apply_BS_schwinger(2.0, -1.0, 0, 1, Gg)
    assert np.allclose(manual, Gg)


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


def test_vanilla_average():
    "tests the vanilla average against other known stable methods"
    settings.USE_VANILLA_AVERAGE = True
    assert np.allclose(
        mmld.Dgate([0], x=4.0, y=4.0).fock([1000, 1000]),
        displacement((1000, 1000), 4.0 + 4.0j),
    )
    sgate = mmld.Sgate([0], r=4.0, phi=2.0).fock([1000, 1000])
    assert np.max(np.abs(sgate)) < 1

    settings.USE_VANILLA_AVERAGE = False


def test_vanilla_average_batched():
    "tests the vanilla average against other known stable methods. batched version."
    settings.USE_VANILLA_AVERAGE = True
    A, b, c = mmld.Ket.random([0, 1]).bargmann_triple(batched=True)
    batched = vanilla_average((4, 4), A[0], b, c[0])
    non_batched = vanilla_average((4, 4), A[0], b[0], c[0])

    assert np.allclose(batched[0], non_batched)

    settings.USE_VANILLA_AVERAGE = False
