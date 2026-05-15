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
from mrmustard.mathlib.cython_lattice import (
    vanilla,
    vanilla_batched,
    vanilla_vjp,
    vanilla_vjp_batched,
)
from mrmustard.physics.utils import random_Abc


class TestVanilla:
    r"""
    Test the vanilla strategy for calculating the Fock representation of a Gaussian tensor.
    """

    @pytest.mark.requires_backend("numpy")
    @pytest.mark.parametrize("n", [2, 3])
    def test_vanilla_vjp(self, n):
        r"""
        Unit test for vanilla_vjp function by comparing with finite difference approximations.
        """
        epsilon = 1e-9
        seed = 673
        rng = np.random.default_rng(seed)
        A, b, c = random_Abc(core_vars=n, seed=rng)
        shape = (4,) * n
        stable = False

        G = vanilla(shape, A, b, c, stable=stable)

        # upstream gradient
        dLdG = rng.standard_normal(G.shape) + 1j * rng.standard_normal(G.shape)

        # Compute finite difference for c
        dGdc_fd = (vanilla(shape, A, b, c + epsilon, stable=stable) - G) / epsilon
        dLdc_fd = math.sum(dLdG * dGdc_fd)

        # Compute finite differences for b
        dGdb_fd = math.zeros(G.shape + b.shape, dtype=math.complex128)
        for i in range(b.shape[0]):
            eps = math.zeros_like(b)
            eps[i] = epsilon
            dGdb_fd[..., i] = (vanilla(shape, A, b + eps, c, stable=stable) - G) / epsilon
        dLdb_fd = math.zeros_like(b)
        for i in range(b.shape[0]):
            dLdb_fd[i] = math.sum(dLdG * dGdb_fd[..., i])

        # Compute finite differences for A
        dGdA_fd = math.zeros(G.shape + A.shape, dtype=math.complex128)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                eps = math.zeros_like(A)
                eps[i, j] = epsilon
                dGdA_fd[..., i, j] = (vanilla(shape, A + eps, b, c, stable=stable) - G) / epsilon
        dLdA_fd = math.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                dLdA_fd[i, j] = math.sum(dLdG * dGdA_fd[..., i, j])

        dLdA, dLdb, dLdc = vanilla_vjp(G, c, dLdG)
        assert math.allclose(dLdc, dLdc_fd)
        assert math.allclose(dLdb, dLdb_fd)
        assert math.allclose(dLdA, (dLdA_fd + dLdA_fd.T) / 2)

    @pytest.mark.requires_backend("numpy")
    def test_full_batch_vjp(self):  # noqa: C901
        r"""
        Unit test for vanilla_vjp_batched function by comparing its results with finite difference approximations.
        """
        # Generate the output tensor G
        epsilon = 1e-9
        seed = 673
        rng = np.random.default_rng(seed)
        A, b, c = random_Abc(core_vars=3, batch=(2,), seed=rng)
        shape = (1, 2, 3)
        stable = False
        G = vanilla_batched(shape, A, b, c, stable=stable)

        # Generate random upstream gradient with same shape as G
        dLdG = rng.standard_normal(G.shape) + 1j * rng.standard_normal(
            G.shape,
        )  # upstream gradient

        # Compute finite difference for c
        dGdc_fd = math.zeros(G.shape + c.shape, dtype=math.complex128)
        for i in range(c.shape[0]):
            eps = math.zeros_like(c)
            eps[i] = epsilon
            dGdc_fd[..., i] = (vanilla_batched(shape, A, b, c + eps, stable=stable) - G) / epsilon

        # Contract with upstream gradient
        dLdc_fd = math.zeros_like(c)
        for i in range(c.shape[0]):
            dLdc_fd[i] = math.sum(dLdG * dGdc_fd[..., i])

        # Compute finite differences for b
        dGdb_fd = math.zeros(G.shape + b.shape, dtype=math.complex128)  # shape: G.shape + b.shape
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                eps = math.zeros_like(b)
                eps[i, j] = epsilon
                dGdb_fd[..., i, j] = (
                    vanilla_batched(shape, A, b + eps, c, stable=stable) - G
                ) / epsilon

        # Contract with upstream gradient
        dLdb_fd = math.zeros_like(b)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                dLdb_fd[i, j] = math.sum(dLdG * dGdb_fd[..., i, j])

        # Compute finite differences for A
        dGdA_fd = math.zeros(G.shape + A.shape, dtype=math.complex128)  # shape: G.shape + A.shape
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    eps = math.zeros_like(A)
                    eps[i, j, k] = epsilon
                    dGdA_fd[..., i, j, k] = (
                        vanilla_batched(shape, A + eps, b, c, stable=stable) - G
                    ) / epsilon

        # Contract with upstream gradient
        dLdA_fd = math.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    dLdA_fd[i, j, k] = math.sum(dLdG * dGdA_fd[..., i, j, k])

        # Use the VJP function to compute gradients
        assert not math.any(math.isnan(G))
        assert not math.any(math.isnan(dLdG))
        assert not math.any(math.isnan(c))
        dLdA, dLdb, dLdc = vanilla_vjp_batched(G, c, dLdG)
        assert not math.any(math.isnan(dLdA))
        assert not math.any(math.isnan(dLdb))
        assert not math.any(math.isnan(dLdc))

        # Verify results
        assert math.allclose(dLdc, dLdc_fd, atol=2e-7)
        assert math.allclose(dLdb, dLdb_fd, atol=2e-7)
        assert math.allclose(dLdA, (dLdA_fd + math.swapaxes(dLdA_fd, -1, -2)) / 2, atol=2e-7)

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_unbatched(self, stable):
        r"""
        Test the hermite_renormalized function for unbatched inputs.
        """
        A, b, c = random_Abc(core_vars=2, seed=673)
        shape = (3, 3)
        G = math.hermite_renormalized(A, b, c, shape, stable=stable)
        assert G.shape == shape

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_unbatched_out(self, stable):
        r"""
        Test the hermite_renormalized function for unbatched inputs with out.
        """
        A, b, c = random_Abc(core_vars=2, seed=673)
        shape = (3, 3)
        out_arr = np.zeros(shape, dtype=math.complex128)
        G = math.hermite_renormalized(A, b, c, shape, stable=stable, out=out_arr)
        assert out_arr is G
        assert math.allclose(G, math.hermite_renormalized(A, b, c, shape, stable=stable))

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_b_batched_no_out(self, stable):
        r"""
        Test the hermite_renormalized function for batched b inputs without out parameter.
        """
        A, b, c = random_Abc(core_vars=2, batch=(2, 1), seed=673)
        shape = (4, 5)
        G = math.hermite_renormalized(A[0, 0], b, c[0, 0], shape, stable=stable)
        assert G.shape == (2, 1, *shape)
        assert math.allclose(G[0, 0], math.hermite_renormalized(A[0, 0], b[0, 0], c[0, 0], shape))
        assert math.allclose(G[1, 0], math.hermite_renormalized(A[0, 0], b[1, 0], c[0, 0], shape))

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_b_batched_with_out(self, stable):
        r"""
        Test the hermite_renormalized function for batched b inputs with out parameter.
        """
        A, b, c = random_Abc(core_vars=2, batch=(2, 1), seed=673)
        shape = (4, 5)
        out_arr = np.zeros((2, 1, *shape), dtype=math.complex128)
        G = math.hermite_renormalized(A[0, 0], b, c[0, 0], shape, stable=stable, out=out_arr)
        assert G.shape == (2, 1, *shape)
        assert math.allclose(G[0, 0], math.hermite_renormalized(A[0, 0], b[0, 0], c[0, 0], shape))
        assert math.allclose(G[1, 0], math.hermite_renormalized(A[0, 0], b[1, 0], c[0, 0], shape))

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_batched_no_out(self, stable):
        r"""
        Test the hermite_renormalized function for batched inputs without out parameter.
        """
        A, b, c = random_Abc(core_vars=2, batch=(2, 1), seed=673)
        shape = (4, 5)
        G = math.hermite_renormalized(A, b, c, shape, stable=stable)
        assert G.shape == (2, 1, *shape)
        assert math.allclose(G[0, 0], math.hermite_renormalized(A[0, 0], b[0, 0], c[0, 0], shape))
        assert math.allclose(G[1, 0], math.hermite_renormalized(A[1, 0], b[1, 0], c[1, 0], shape))

    @pytest.mark.parametrize("stable", [True, False])
    def test_hermite_renormalized_batched_with_out(self, stable):
        r"""
        Test the hermite_renormalized function for batched inputs with out parameter.
        """
        A, b, c = random_Abc(core_vars=2, batch=(2, 1), seed=673)
        shape = (4, 5)
        out_arr = np.zeros((2, 1, *shape), dtype=math.complex128)
        G = math.hermite_renormalized(A, b, c, shape, stable=stable, out=out_arr)
        assert G.shape == (2, 1, *shape)
        assert math.allclose(G[0, 0], math.hermite_renormalized(A[0, 0], b[0, 0], c[0, 0], shape))
        assert math.allclose(G[1, 0], math.hermite_renormalized(A[1, 0], b[1, 0], c[1, 0], shape))
