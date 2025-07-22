# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains test for the calculation of the discretized Wigner function."""

import numpy as np
import pytest
from scipy.special import assoc_laguerre

from mrmustard import settings
from mrmustard.lab import Coherent, Ket, Number, SqueezedVacuum
from mrmustard.physics.wigner import wigner_discretized

# ~~~~~~~
# Helpers
# ~~~~~~~


def W_cat(q_vec, p_vec, q0):
    r"""Calculates the discretized Wigner function for a cat state with
    coherent states centered in `(q0, 0)`. See Eq. 3.3 in arXiv:0406015.
    """

    def generator(q, p, q0):
        norm = (1 + np.exp(-(q0**2))) ** -0.5
        W_plus = np.exp(-((q + q0) ** 2) - p**2)
        W_minus = np.exp(-((q - q0) ** 2) - p**2)
        W_int = np.cos(2 * p * q0) * np.exp(-(q**2) - p**2)
        return (W_plus / 2 + W_minus / 2 + W_int) * norm**2 / np.pi / settings.HBAR

    q = q_vec / (settings.HBAR) ** 0.5
    p = p_vec / (settings.HBAR) ** 0.5

    return np.array([[generator(i, j, q0 * 2**0.5) for j in p] for i in q])


def W_coherent(q_vec, p_vec, alpha, s):
    r"""Calculates the discretized Wigner function for a coherent state centered
    around `alpha` and with squeezing `s`. See Eq. 4.12 in arXiv:0406015.
    """

    def generator(q, p, alpha, s):
        q0 = np.real(alpha) * 2**0.5
        p0 = np.imag(alpha) * 2**0.5
        ret = -np.exp(2 * s) * (q - q0) ** 2 - np.exp(-2 * s) * (p - p0) ** 2
        return np.exp(ret) / np.pi / settings.HBAR

    q = q_vec / (settings.HBAR) ** 0.5
    p = p_vec / (settings.HBAR) ** 0.5

    return np.array([[generator(i, j, alpha, s) for j in p] for i in q])


def W_fock(q_vec, p_vec, n):
    r"""Calculates the discretized Wigner function for a fock state.
    See Eq. 4.10 in arXiv:0406015.
    """

    def generator(q, p, n):
        alpha2 = q**2 + p**2
        ret = (-1) ** n * np.exp(-alpha2) * assoc_laguerre(2 * alpha2, n)
        return ret / np.pi / settings.HBAR

    q = q_vec / (settings.HBAR) ** 0.5
    p = p_vec / (settings.HBAR) ** 0.5

    return np.array([[generator(i, j, n) for j in p] for i in q])


# ~~~~~
# Tests
# ~~~~~


class TestWignerDiscretized:
    r"""Tests discretized Wigner functions (DWF) for various states"""

    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    @pytest.mark.parametrize("hbar", [1, 2])
    def test_cat_state(self, method, hbar):
        r"""Tests DWF for cat states"""
        with settings(DISCRETIZATION_METHOD=method, HBAR=hbar):
            q_vec = np.linspace(-4, 4, 100)
            p_vec = np.linspace(-1.5, 1.5, 100)

            q0 = 2.0
            cat_amps = (Coherent(0, q0) + Coherent(0, -q0)).fock_array([20])
            cat_amps = cat_amps / np.linalg.norm(cat_amps)
            state = Ket.from_fock([0], cat_amps).dm()
            W_mm, q_mat, p_mat = wigner_discretized(state.fock_array(), q_vec, p_vec)
            W_th = W_cat(q_vec, p_vec, q0)

            assert np.allclose(W_mm, W_th, atol=1e-4)
            assert np.allclose(q_mat.T, q_vec)
            assert np.allclose(p_mat, p_vec)

    @pytest.mark.parametrize("alpha", [0 + 0j, 3 + 3j])
    @pytest.mark.parametrize("hbar", [2, 3])
    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_coherent_state(self, alpha, hbar, method):
        r"""Tests DWF for coherent states"""

        with settings(
            DISCRETIZATION_METHOD=method,
            HBAR=hbar,
            AUTOSHAPE_MIN=100,
            AUTOSHAPE_MAX=150,
        ):
            # centering the intervals around alpha--away from the center,
            # the values are small and unstable.
            left = (np.real(alpha) * 2**0.5 - 1) * (settings.HBAR) ** 0.5
            right = (np.real(alpha) * 2**0.5 + 1) * (settings.HBAR) ** 0.5
            q_vec = np.linspace(left, right, 50)
            p_vec = np.linspace(left, right, 50)

            state = Coherent(0, np.real(alpha), np.imag(alpha))
            W_mm, q_mat, p_mat = wigner_discretized(
                state.dm().fock_array(100, standard_order=True),
                q_vec,
                p_vec,
            )
            W_th = W_coherent(q_vec, p_vec, alpha, 0)

            assert np.allclose(W_mm, W_th, atol=1e-4)
            assert np.allclose(q_mat.T, q_vec)
            assert np.allclose(p_mat, p_vec)

    @pytest.mark.parametrize("n", [2, 6])
    @pytest.mark.parametrize("hbar", [2, 3])
    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_fock_state(self, n, hbar, method):
        r"""Tests DWF for fock states"""
        with settings(DISCRETIZATION_METHOD=method, HBAR=hbar):
            q_vec = np.linspace(-1, 1, 20)
            p_vec = np.linspace(-1, 1, 20)

            state = Number(0, n)
            W_mm, q_mat, p_mat = wigner_discretized(state.dm().fock_array(), q_vec, p_vec)
            W_th = W_fock(q_vec, p_vec, n)

            assert np.allclose(W_mm, W_th, atol=1e-4)
            assert np.allclose(q_mat.T, q_vec)
            assert np.allclose(p_mat, p_vec)

    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_squeezed_vacuum_both_method_succeed(self, method):
        r"""Tests DWF for a squeezed vacuum state with squeezing s=1.
        Both discretization methods are expected to pass successfully.
        """
        with settings(DISCRETIZATION_METHOD=method, AUTOSHAPE_MIN=100, AUTOSHAPE_MAX=150):
            q_vec = np.linspace(-0.5, 0.5, 50)
            p_vec = np.linspace(-5, 5, 50)

            s = 1
            state = SqueezedVacuum(0, s)
            W_mm, q_mat, p_mat = wigner_discretized(
                state.dm().fock_array(100, standard_order=True),
                q_vec,
                p_vec,
            )
            W_th = W_coherent(q_vec, p_vec, 0j, s)

            assert np.allclose(W_mm, W_th, atol=1e-4)
            assert np.allclose(q_mat.T, q_vec)
            assert np.allclose(p_mat, p_vec)

    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_squeezed_vacuum_iterative_fails(self, method):
        r"""Tests DWF for a squeezed vacuum state with squeezing s=2.
        The iterative method cannot produce a DWF that matched with the analytical one.
        """
        with settings(DISCRETIZATION_METHOD=method, AUTOSHAPE_MIN=100, AUTOSHAPE_MAX=150):
            q_vec = np.linspace(-0.2, 0.2, 50)
            p_vec = np.linspace(-5, 5, 50)

            s = 2
            state = SqueezedVacuum(0, s)
            W_mm, _, _ = wigner_discretized(
                state.dm().fock_array(100, standard_order=True),
                q_vec,
                p_vec,
            )
            W_th = W_coherent(q_vec, p_vec, 0j, s)

            success = np.allclose(W_mm, W_th, atol=1e-4)
            assert success is False if method == "iterative" else True
