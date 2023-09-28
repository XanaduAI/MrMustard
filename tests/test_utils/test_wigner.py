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
# from scipy.stats import multivariate_normal

from mrmustard import settings
from mrmustard.lab import (
    Coherent,
    # DisplacedSqueezed,
    # Fock,
    # Gaussian,
    SqueezedVacuum,
    State,
    # Thermal,
    # Vacuum,
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
    r""" Resets `Settings`
    """
    settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0
    settings.AUTOCUTOFF_MIN_CUTOFF = autocutoff_min0
    settings.DISCRETIZATION_METHOD = method0
    settings.HBAR = hbar0

def distance(W_mm, W_th):
    r""" Calculates the distance between the discretized Wigner functions W_mm (generated
    by `mrmustard`) and W_th (computed analytically) as `(np.abs(W_mm-W_th)/W_th).max()`.
    """
    return (np.abs(W_mm-W_th)/W_th).max()


def W_cat(q_vec, p_vec, q0):
    r""" Calculates the discretized Wigner function for a cat state with
    coherent states centered in `(q0, 0)`. See Eq. 3.3 in arXiv:0406015.
    """
    def generator(q, p, q0):
        norm = (1 + np.exp(-q0**2))**-0.5
        W_plus = np.exp(-(q+q0)**2-p**2)
        W_minus = np.exp(-(q-q0)**2-p**2)
        W_int = np.cos(2*p*q0)*np.exp(-q**2-p**2)
        return (W_plus/2 + W_minus/2 + W_int)*norm**2/np.pi/settings.HBAR

    q = q_vec/(settings.HBAR)**0.5
    p = p_vec/(settings.HBAR)**0.5

    return np.array([[generator(i, j, q0*2**0.5) for j in p] for i in q])


def W_coherent(q_vec, p_vec, alpha, s):
    r""" Calculates the discretized Wigner function for a coherent state centered
    around `alpha` and with squeezing `s`. See Eq. 4.12 in arXiv:0406015.
    """
    def generator(q, p, alpha, s):
        q0 = np.real(alpha)*2**0.5
        p0 = np.imag(alpha)*2**0.5
        ret = -np.exp(2*s)*(q - q0)**2-np.exp(-2*s)*(p - p0)**2
        return np.exp(ret)/np.pi/settings.HBAR

    q = q_vec/(settings.HBAR)**0.5
    p = p_vec/(settings.HBAR)**0.5

    return np.array([[generator(i, j, alpha, s) for j in p] for i in q])

# ~~~~~
# Tests
# ~~~~~

class TestWignerDiscretized:
    @pytest.mark.parametrize("method", ["iterative", "cleanshaw"])
    @pytest.mark.parametrize("hbar", [2, 3])
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

    @pytest.mark.parametrize("alpha", [0+0j, 3+3j])
    @pytest.mark.parametrize("hbar", [2, 3])
    @pytest.mark.parametrize("method", ["iterative", "cleanshaw"])
    def test_coherent_state(self, alpha, hbar, method):
        settings.AUTOCUTOFF_MIN_CUTOFF = 100
        settings.AUTOCUTOFF_MAX_CUTOFF = 150
        settings.DISCRETIZATION_METHOD = method
        settings.HBAR = hbar

        # centering the intervals around alpha--away from the center,
        # the values are small and unstable.
        left = (np.real(alpha)*2**0.5-1)*(settings.HBAR)**0.5
        right = (np.real(alpha)*2**0.5+1)*(settings.HBAR)**0.5
        q_vec = np.linspace(left, right, 50)
        p_vec = np.linspace(left, right, 50)

        state = Coherent(np.real(alpha), np.imag(alpha))
        W_mm, _, _ = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_coherent(q_vec, p_vec, alpha, 0)

        assert np.allclose(distance(W_mm, W_th), 0)

        reset_settings()

    @pytest.mark.parametrize("method", ["iterative", "cleanshaw"])
    def test_squeezed_vacuum_both_method_succeed(self, method):
        settings.AUTOCUTOFF_MIN_CUTOFF = 100
        settings.AUTOCUTOFF_MAX_CUTOFF = 150
        settings.DISCRETIZATION_METHOD = method

        q_vec = np.linspace(-.5, .5, 50)
        p_vec = np.linspace(-5, 5, 50)

        s = 1
        state = SqueezedVacuum(s)
        W_mm, _, _ = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_coherent(q_vec, p_vec, 0j, s)

        assert np.allclose(distance(W_mm, W_th), 0, atol=10**-1)

        reset_settings()

    @pytest.mark.parametrize("method", ["iterative", "cleanshaw"])
    def test_squeezed_vacuum_iterative_fails(self, method):
        settings.AUTOCUTOFF_MIN_CUTOFF = 100
        settings.AUTOCUTOFF_MAX_CUTOFF = 150
        settings.DISCRETIZATION_METHOD = method

        q_vec = np.linspace(-.2, .2, 50)
        p_vec = np.linspace(-5, 5, 50)

        s = 2
        state = SqueezedVacuum(s)
        W_mm, _, _ = wigner_discretized(state.dm(), q_vec, p_vec)
        W_th = W_coherent(q_vec, p_vec, 0j, s)

        success = np.allclose(distance(W_mm, W_th), 0, atol=10**-1)
        assert success is False if method == "iterative" else True

        reset_settings()
          


# def multivariate_normal_pdf(q_vec, p_vec, means, cov):
#     """generates the PDF of a multivariate normal distribution"""
#     mvn = multivariate_normal(means, cov, allow_singular=True)
#     grid = np.meshgrid(q_vec, p_vec)
#     return mvn.pdf(grid)


# @pytest.mark.parametrize(
#     "state",
#     [
#         Vacuum(1),
#         Coherent(0.3, -0.5),
#         SqueezedVacuum(0.5, 0.45),
#         Thermal(0.25),
#         DisplacedSqueezed(0.3, 0.1, -0.1, 0.1),
#         Gaussian(1),
#     ],
# )
# def test_wigner_gaussian_states(state):
#     """test Wigner function for Gaussian states is a standard normal distribution"""
#     # calculate Wigner from state dm
#     q_vec = np.arange(-5, 5, 100)
#     p_vec = q_vec
#     dm = state.dm(cutoffs=[5]).numpy()
#     W_calc, _, _ = wigner_discretized(dm, q_vec, p_vec)

#     # calculate exact
#     cov = state.cov.numpy()
#     means = state.means.numpy()
#     W_exact = multivariate_normal_pdf(q_vec, p_vec, means, cov)

#     assert np.allclose(W_calc, W_exact, atol=0.001, rtol=0)


# # Exact marginal probability distributions for various states
# hbar = settings.HBAR


# def fock1_marginal(q_vec):
#     """q and p marginal distributions for the Fock state |1>"""
#     x = (
#         0.5
#         * np.sqrt(1 / (np.pi * hbar))
#         * np.exp(-1 * (q_vec**2) / hbar)
#         * (4 / hbar)
#         * (q_vec**2)
#     )
#     p = x
#     return x, p


# def vacuum_marginal(q_vec):
#     """q and p marginal distributions for the vacuum state"""
#     x = np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * (q_vec**2) / hbar)
#     p = x
#     return x, p


# def coherent_marginal(q_vec):
#     r"""q and p marginal distributions for the coherent state with `\alpha=1`"""
#     x = np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * ((q_vec - 0.5 * np.sqrt(2 * hbar)) ** 2) / hbar)
#     p = np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * (q_vec**2) / hbar)
#     return x, p


# @pytest.mark.parametrize(
#     "state, f_marginal",
#     [
#         (Vacuum(1), vacuum_marginal),
#         (Coherent(1.0, 0.0), coherent_marginal),
#         (Fock([1]), fock1_marginal),
#     ],
# )
# def test_marginal_wigner(state, f_marginal):
#     """test marginals of Wigner function agree with the expected ones"""

#     # calculate Wigner from state dm
#     q_vec = np.arange(-5, 5, 100)
#     p_vec = q_vec
#     dm = state.dm(cutoffs=[5]).numpy()
#     W_calc, _, _ = wigner_discretized(dm, q_vec, p_vec)

#     # calculate marginals
#     q_marginal = np.sum(W_calc, axis=1)
#     p_marginal = np.sum(W_calc, axis=0)

#     expected_q_marginal, expected_p_marginal = f_marginal(q_vec)

#     assert np.allclose(q_marginal, expected_q_marginal, atol=0.001, rtol=0)
#     assert np.allclose(p_marginal, expected_p_marginal, atol=0.001, rtol=0)
