# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the lattice/strategies/vanilla module"""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.math.lattice import strategies

from ..conftest import skip_jax, skip_tf


def random_triple(n, batch=(), seed=None):
    r"""
    Generate random triple of A, b, c for testing the vanilla strategy.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    A = rng.random(batch + (n, n)) + 1j * rng.random(batch + (n, n))
    A = A + np.swapaxes(A, -1, -2)
    A /= np.abs(np.linalg.eigvals(A)).max() + 0.2
    b = rng.random(batch + (n,)) + 1j * rng.random(batch + (n,))
    c = rng.random(batch + ()) + 1j * rng.random(batch + ())
    return A, b, c


class TestVanilla:
    r"""
    Test the vanilla strategy for calculating the Fock representation of a Gaussian tensor.
    """

    @pytest.mark.parametrize("n", [2, 3])
    def test_vanilla_vjp(self, n):
        r"""
        Unit test for vanilla_vjp_numba function by comparing with finite difference approximations.
        """
        skip_tf()
        skip_jax()
        epsilon = 1e-9
        A, b, c = random_triple(n, (), seed=673)
        shape = (4,) * n

        G = strategies.vanilla_numba(shape, A, b, c)

        # upstream gradient
        dLdG = np.random.randn(*G.shape)

        # Compute finite difference for c
        dGdc_fd = (strategies.vanilla_numba(shape, A, b, c + epsilon) - G) / epsilon
        dLdc_fd = np.sum(dLdG * dGdc_fd)

        # Compute finite differences for b
        dGdb_fd = np.zeros(G.shape + b.shape, dtype=np.complex128)
        for i in range(b.shape[0]):
            eps = np.zeros_like(b)
            eps[i] = epsilon
            dGdb_fd[..., i] = (strategies.vanilla_numba(shape, A, b + eps, c) - G) / epsilon
        dLdb_fd = np.zeros_like(b)
        for i in range(b.shape[0]):
            dLdb_fd[i] = np.sum(dLdG * dGdb_fd[..., i])

        # Compute finite differences for A
        dGdA_fd = np.zeros(G.shape + A.shape, dtype=np.complex128)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                eps = np.zeros_like(A)
                eps[i, j] = epsilon
                dGdA_fd[..., i, j] = (strategies.vanilla_numba(shape, A + eps, b, c) - G) / epsilon
        dLdA_fd = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                dLdA_fd[i, j] = np.sum(dLdG * dGdA_fd[..., i, j])

        dLdA, dLdb, dLdc = strategies.vanilla_vjp_numba(G, c, dLdG)
        assert np.allclose(dLdc, dLdc_fd)
        assert np.allclose(dLdb, dLdb_fd)
        assert np.allclose(dLdA, (dLdA_fd + dLdA_fd.T) / 2)

    def test_full_batch_vjp(self):
        r"""
        Unit test for vanilla_batch_vjp_numba function by comparing its results with finite difference approximations.
        """
        skip_tf()
        skip_jax()
        # Generate the output tensor G
        epsilon = 1e-9
        A, b, c = random_triple(3, (2,), seed=673)
        shape = (1, 2, 3)
        G = strategies.vanilla_batch_numba(shape, A, b, c)

        # Generate random upstream gradient with same shape as G
        dLdG = np.random.randn(*G.shape) + 1j * np.random.randn(*G.shape)  # upstream gradient

        # Compute finite difference for c
        dGdc_fd = np.zeros(G.shape + c.shape, dtype=np.complex128)
        for i in range(c.shape[0]):
            eps = np.zeros_like(c)
            eps[i] = epsilon
            dGdc_fd[..., i] = (strategies.vanilla_batch_numba(shape, A, b, c + eps) - G) / epsilon

        # Contract with upstream gradient
        dLdc_fd = np.zeros_like(c)
        for i in range(c.shape[0]):
            dLdc_fd[i] = np.sum(dLdG * dGdc_fd[..., i])

        # Compute finite differences for b
        dGdb_fd = np.zeros(G.shape + b.shape, dtype=np.complex128)  # shape: G.shape + b.shape
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                eps = np.zeros_like(b)
                eps[i, j] = epsilon
                dGdb_fd[..., i, j] = (
                    strategies.vanilla_batch_numba(shape, A, b + eps, c) - G
                ) / epsilon

        # Contract with upstream gradient
        dLdb_fd = np.zeros_like(b)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                dLdb_fd[i, j] = np.sum(dLdG * dGdb_fd[..., i, j])

        # Compute finite differences for A
        dGdA_fd = np.zeros(G.shape + A.shape, dtype=np.complex128)  # shape: G.shape + A.shape
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    eps = np.zeros_like(A)
                    eps[i, j, k] = epsilon
                    dGdA_fd[..., i, j, k] = (
                        strategies.vanilla_batch_numba(shape, A + eps, b, c) - G
                    ) / epsilon

        # Contract with upstream gradient
        dLdA_fd = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    dLdA_fd[i, j, k] = np.sum(dLdG * dGdA_fd[..., i, j, k])

        # Use the VJP function to compute gradients
        assert not np.isnan(G).any()
        assert not np.isnan(dLdG).any()
        assert not np.isnan(c).any()
        dLdA, dLdb, dLdc = strategies.vanilla_batch_vjp_numba(G, c, dLdG)
        assert not np.isnan(dLdA).any()
        assert not np.isnan(dLdb).any()
        assert not np.isnan(dLdc).any()

        # Verify results
        assert np.allclose(dLdc, dLdc_fd)
        assert np.allclose(dLdb, dLdb_fd)
        assert np.allclose(dLdA, (dLdA_fd + np.swapaxes(dLdA_fd, -1, -2)) / 2)

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_unbatched(self, stable):
        r"""
        Test the hermite_renormalized function for unbatched inputs.
        """
        A, b, c = random_triple(2, (), seed=673)
        shape = (3, 3)
        G = math.hermite_renormalized(A, b, c, shape, stable=stable)
        assert G.shape == shape

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_b_batched(self, stable):
        r"""
        Test the hermite_renormalized function for batched b inputs.
        """
        A, b, c = random_triple(2, (2, 1), seed=673)
        shape = (4, 5)
        G = math.hermite_renormalized(A[0, 0], b, c[0, 0], shape, stable=stable)
        assert G.shape == (2, 1) + shape
        assert math.allclose(G[0, 0], math.hermite_renormalized(A[0, 0], b[0, 0], c[0, 0], shape))
        assert math.allclose(G[1, 0], math.hermite_renormalized(A[0, 0], b[1, 0], c[0, 0], shape))

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_batched(self, stable):
        r"""
        Test the hermite_renormalized function for batched inputs.
        """
        A, b, c = random_triple(2, (2, 1), seed=673)
        shape = (4, 5)
        G = math.hermite_renormalized(A, b, c, shape, stable=stable)
        assert G.shape == (2, 1) + shape
        assert math.allclose(G[0, 0], math.hermite_renormalized(A[0, 0], b[0, 0], c[0, 0], shape))
        assert math.allclose(G[1, 0], math.hermite_renormalized(A[1, 0], b[1, 0], c[1, 0], shape))
