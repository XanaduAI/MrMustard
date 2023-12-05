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
import pytest
import numpy as np

from mrmustard.lab import Gaussian, Dgate
from mrmustard import settings, math
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.math.lattice.strategies.binomial import binomial, binomial_dict

original_precision = settings.PRECISION_BITS_HERMITE_POLY

do_julia = True if importlib.util.find_spec("julia") else False
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


@pytest.mark.parametrize("n_batches", [1, 3])
def test_vanillabatchNumba_vs_vanillaNumba(n_batches):
    """Test the batch version works versus the normal vanilla version."""
    state = Gaussian(3) >> Dgate([0.0, 0.1, 0.2])
    A, B, C = wigner_to_bargmann_rho(
        state.cov, state.means
    )  # Create random state (M mode Gaussian state with displacement)

    cutoffs = (20, 20, 20, 20, n_batches)

    # Vanilla MM
    G_ref = math.hermite_renormalized(A, B, C, shape=cutoffs[:-1])

    # replicate the B
    B_batched = np.stack((B,) * n_batches, axis=1)

    G_batched = math.hermite_renormalized_batch(A, B_batched, C, shape=cutoffs)

    for nb in range(n_batches):
        assert np.allclose(G_ref, G_batched[:, :, :, :, nb])


@pytest.mark.parametrize("n_batches", [1, 3])
def test_diagonalbatchNumba_vs_diagonalNumba(n_batches):
    """Test the batch version works versus the normal diagonal version."""
    state = Gaussian(3) >> Dgate([0.0, 0.1, 0.2])
    A, B, C = wigner_to_bargmann_rho(
        state.cov, state.means
    )  # Create random state (M mode Gaussian state with displacement)

    cutoffs = (18, 19, 20, n_batches)

    # Diagonal MM
    G_ref = math.hermite_renormalized_diagonal(A, B, C, cutoffs=cutoffs[:-1])

    # replicate the B
    B_batched = np.stack((B,) * n_batches, axis=1)

    G_batched = math.hermite_renormalized_diagonal_batch(A, B_batched, C, cutoffs=cutoffs[:-1])

    for nb in range(n_batches):
        assert np.allclose(G_ref, G_batched[:, :, :, nb])
