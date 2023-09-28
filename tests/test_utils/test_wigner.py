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
from mrmustard.lab import (
    Coherent,
    Fock,
    SqueezedVacuum,
    State,
)
from mrmustard.utils.wigner import wigner_discretized

# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF
autocutoff_min0 = settings.AUTOCUTOFF_MIN_CUTOFF
method0 = settings.DISCRETIZATION_METHOD
hbar0 = settings.HBAR

# ~~~~~~~
# Helpers
# ~~~~~~~


def reset_settings():
    r"""Resets `Settings`"""
    settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0
    settings.AUTOCUTOFF_MIN_CUTOFF = autocutoff_min0
    settings.DISCRETIZATION_METHOD = method0
    settings.HBAR = hbar0


def distance(W_mm, W_th):
    r"""Calculates the distance between the discretized Wigner functions W_mm (generated
    by `mrmustard`) and W_th (computed analytically) as the maximum of `|W_mm-W_th|/|W_th|`,
    where .
    """
    num = np.abs(W_mm - W_th)
    den = np.abs(W_th)
    return (num / den).max()


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
    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    @pytest.mark.parametrize("hbar", [1, 2])
    def test_cat_state(self, method, hbar):
        settings.DISCRETIZATION_METHOD = method
        settings.HBAR = hbar

        q_vec = np.linspace(-4, 4, 100)
        p_vec = np.linspace(-1.5, 1.5, 100)

        q0 = 2.0
        cat_amps = Coherent(q0).ket([20]) + Coherent(-q0).ket([20])
        cat_amps = cat_amps / np.linalg.norm(cat_amps)
        state = State(ket=cat_amps)
        W_mm, _, _ = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_cat(q_vec, p_vec, q0)

        assert np.allclose(distance(W_mm, W_th), 0, atol=10**-1)

        reset_settings()

    @pytest.mark.parametrize("alpha", [0 + 0j, 3 + 3j])
    @pytest.mark.parametrize("hbar", [2, 3])
    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_coherent_state(self, alpha, hbar, method):
        settings.AUTOCUTOFF_MIN_CUTOFF = 100
        settings.AUTOCUTOFF_MAX_CUTOFF = 150
        settings.DISCRETIZATION_METHOD = method
        settings.HBAR = hbar

        # centering the intervals around alpha--away from the center,
        # the values are small and unstable.
        left = (np.real(alpha) * 2**0.5 - 1) * (settings.HBAR) ** 0.5
        right = (np.real(alpha) * 2**0.5 + 1) * (settings.HBAR) ** 0.5
        q_vec = np.linspace(left, right, 50)
        p_vec = np.linspace(left, right, 50)

        state = Coherent(np.real(alpha), np.imag(alpha))
        W_mm, _, _ = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_coherent(q_vec, p_vec, alpha, 0)

        assert np.allclose(distance(W_mm, W_th), 0)

        reset_settings()

    @pytest.mark.parametrize("n", [2, 6])
    @pytest.mark.parametrize("hbar", [2, 3])
    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_fock_state(self, n, hbar, method):
        settings.DISCRETIZATION_METHOD = method
        settings.HBAR = hbar

        q_vec = np.linspace(-1, 1, 20)
        p_vec = np.linspace(-1, 1, 20)

        state = Fock(n)
        W_mm, q, p = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_fock(q_vec, p_vec, n)

        assert np.allclose(distance(W_mm, W_th), 0)

        reset_settings()

    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_squeezed_vacuum_both_method_succeed(self, method):
        settings.AUTOCUTOFF_MIN_CUTOFF = 100
        settings.AUTOCUTOFF_MAX_CUTOFF = 150
        settings.DISCRETIZATION_METHOD = method

        q_vec = np.linspace(-0.5, 0.5, 50)
        p_vec = np.linspace(-5, 5, 50)

        s = 1
        state = SqueezedVacuum(s)
        W_mm, _, _ = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_coherent(q_vec, p_vec, 0j, s)

        assert np.allclose(distance(W_mm, W_th), 0, atol=10**-1)

        reset_settings()

    @pytest.mark.parametrize("method", ["iterative", "clenshaw"])
    def test_squeezed_vacuum_iterative_fails(self, method):
        settings.AUTOCUTOFF_MIN_CUTOFF = 100
        settings.AUTOCUTOFF_MAX_CUTOFF = 150
        settings.DISCRETIZATION_METHOD = method

        q_vec = np.linspace(-0.2, 0.2, 50)
        p_vec = np.linspace(-5, 5, 50)

        s = 2
        state = SqueezedVacuum(s)
        W_mm, _, _ = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_coherent(q_vec, p_vec, 0j, s)

        success = np.allclose(distance(W_mm, W_th), 0, atol=10**-1)
        assert success is False if method == "iterative" else True

        reset_settings()
