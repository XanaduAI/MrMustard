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

from hypothesis import given
import pytest
import numpy as np
from scipy.stats import multivariate_normal
from mrmustard.utils.wigner import wigner_discretized
from mrmustard.lab import (
    Vacuum,
    Coherent,
    SqueezedVacuum,
    Thermal,
    DisplacedSqueezed,
    Gaussian,
    Fock,
)
from mrmustard import settings
from tests import random


def multivariate_normal_pdf(qvec, pvec, means, cov):
    """generates the PDF of a multivariate normal distribution"""
    mvn = multivariate_normal(means, cov, allow_singular=True)
    grid = np.meshgrid(qvec, pvec)
    return mvn.pdf(grid)


@pytest.mark.parametrize(
    "state",
    [
        Vacuum(1),
        Coherent(0.3, -0.5),
        SqueezedVacuum(0.5, 0.45),
        Thermal(0.25),
        DisplacedSqueezed(0.3, 0.1, -0.1, 0.1),
        Gaussian(1),
    ],
)
def test_wigner_gaussian_states(state):
    """test Wigner function for Gaussian states is a standard normal distribution"""

    # calculate Wigner from state dm
    qvec = np.arange(-5, 5, 100)
    pvec = qvec
    dm = state.dm(cutoffs=[5]).numpy()
    W_calc, _, _ = wigner_discretized(dm, qvec, pvec)

    # calculate exact
    cov = state.cov.numpy()
    means = state.means.numpy()
    W_exact = multivariate_normal_pdf(qvec, pvec, means, cov)

    assert np.allclose(W_calc, W_exact, atol=0.001, rtol=0)


# Exact marginal probability distributions for various states
hbar = settings.HBAR


def fock1_marginal(qvec):
    """q and p marginal distributions for the Fock state |1>"""
    x = (
        0.5
        * np.sqrt(1 / (np.pi * hbar))
        * np.exp(-1 * (qvec**2) / hbar)
        * (4 / hbar)
        * (qvec**2)
    )
    p = x
    return x, p


def vacuum_marginal(qvec):
    """q and p marginal distributions for the vacuum state"""
    x = np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * (qvec**2) / hbar)
    p = x
    return x, p


def coherent_marginal(qvec):
    r"""q and p marginal distributions for the coherent state with `\alpha=1`"""
    x = np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * ((qvec - 0.5 * np.sqrt(2 * hbar)) ** 2) / hbar)
    p = np.sqrt(1 / (np.pi * hbar)) * np.exp(-1 * (qvec**2) / hbar)
    return x, p


@pytest.mark.parametrize(
    "state, f_marginal",
    [
        (Vacuum(1), vacuum_marginal),
        (Coherent(1.0, 0.0), coherent_marginal),
        (Fock([1]), fock1_marginal),
    ],
)
def test_marginal_wigner(state, f_marginal):
    """test marginals of Wigner function agree with the expected ones"""

    # calculate Wigner from state dm
    qvec = np.arange(-5, 5, 100)
    pvec = qvec
    dm = state.dm(cutoffs=[5]).numpy()
    W_calc, _, _ = wigner_discretized(dm, qvec, pvec)

    # calculate marginals
    q_marginal = np.sum(W_calc, axis=1)
    p_marginal = np.sum(W_calc, axis=0)

    expected_q_marginal, expected_p_marginal = f_marginal(qvec)

    assert np.allclose(q_marginal, expected_q_marginal, atol=0.001, rtol=0)
    assert np.allclose(p_marginal, expected_p_marginal, atol=0.001, rtol=0)
