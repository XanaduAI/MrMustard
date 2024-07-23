"""
Unit tests for mrmustard.math.compactFock.compactFock~
"""

import importlib

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mrmustard import math, settings
from mrmustard.lab import Ggate, SqueezedVacuum, State, Vacuum
from mrmustard.physics import fidelity, normalize
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.training import Optimizer
from tests.random import n_mode_mixed_state

from ..conftest import skip_np

original_precision = settings.PRECISION_BITS_HERMITE_POLY

do_julia = bool(importlib.util.find_spec("juliacall"))
precisions = [128, 256, 384, 512] if do_julia else [128]


@st.composite
def random_ABC(draw, M):
    r"""
    Generates random Bargmann parameters A,B,C
    for an ``M``-mode mixed state.
    """
    state = draw(n_mode_mixed_state(M))
    A, B, G0 = wigner_to_bargmann_rho(state.cov, state.means)
    return A, B, G0


@given(random_ABC(M=3))
@pytest.mark.parametrize("precision", precisions)
def test_compactFock_diagonal(precision, A_B_G0):
    r"""Test getting Fock amplitudes if all modes are
    detected (math.hermite_renormalized_diagonal)
    """
    settings.PRECISION_BITS_HERMITE_POLY = precision
    cutoffs = (5, 5, 5)

    A, B, G0 = A_B_G0  # Create random state (M mode Gaussian state with displacement)

    # Vanilla MM
    G_ref = math.hermite_renormalized(
        math.conj(-A), math.conj(B), math.conj(G0), shape=list(cutoffs) * 2
    )  # note: shape=[C1,C2,C3,...,C1,C2,C3,...]
    G_ref = math.asnumpy(G_ref)

    # Extract diagonal amplitudes from vanilla MM
    ref_diag = np.zeros(cutoffs, dtype=np.complex128)
    for inds in np.ndindex(*cutoffs):
        inds_expanded = list(inds) + list(inds)  # a,b,c,a,b,c
        ref_diag[inds] = G_ref[tuple(inds_expanded)]

    # New MM
    G_diag = math.hermite_renormalized_diagonal(
        math.conj(-A), math.conj(B), math.conj(G0), cutoffs
    )
    assert np.allclose(ref_diag, G_diag)

    settings.PRECISION_BITS_HERMITE_POLY = original_precision


@given(random_ABC(M=3))
@pytest.mark.parametrize("precision", precisions)
def test_compactFock_1leftover(precision, A_B_G0):
    r"""
    Test getting Fock amplitudes if all but the first mode
    are detected (math.hermite_renormalized_1leftoverMode).
    """
    skip_np()

    settings.PRECISION_BITS_HERMITE_POLY = precision
    cutoffs = (5, 5, 5)

    A, B, G0 = A_B_G0  # Create random state (M mode Gaussian state with displacement)

    # New algorithm
    G_leftover = math.hermite_renormalized_1leftoverMode(
        math.conj(-A), math.conj(B), math.conj(G0), cutoffs
    )

    # Vanilla MM
    G_ref = math.hermite_renormalized(
        math.conj(-A), math.conj(B), math.conj(G0), shape=list(cutoffs) * 2
    )  # note: shape=[C1,C2,C3,...,C1,C2,C3,...]
    G_ref = math.asnumpy(G_ref)

    # Extract amplitudes of leftover mode from vanilla MM
    ref_leftover = np.zeros([cutoffs[0]] * 2 + list(cutoffs)[1:], dtype=np.complex128)
    for inds in np.ndindex(*cutoffs[1:]):
        ref_leftover[
            tuple([slice(cutoffs[0]), slice(cutoffs[0])] + list(inds))
        ] = G_ref[
            tuple([slice(cutoffs[0])] + list(inds) + [slice(cutoffs[0])] + list(inds))
        ]
    assert np.allclose(ref_leftover, G_leftover)

    settings.PRECISION_BITS_HERMITE_POLY = original_precision


@pytest.mark.parametrize("precision", precisions)
def test_compactFock_diagonal_gradients(precision):
    r"""
    Test getting Fock amplitudes and gradients if all modes
    are detected (math.hermite_renormalized_diagonal).
    """
    skip_np()

    settings.PRECISION_BITS_HERMITE_POLY = precision
    G = Ggate(num_modes=2, symplectic_trainable=True)

    def cost_fn():
        n1, n2 = 2, 4  # number of detected photons
        state_opt = Vacuum(2) >> G
        A, B, G0 = wigner_to_bargmann_rho(state_opt.cov, state_opt.means)
        probs = math.hermite_renormalized_diagonal(
            math.conj(-A), math.conj(B), math.conj(G0), cutoffs=[n1 + 1, n2 + 1]
        )
        p = probs[n1, n2]
        return -math.real(p)

    opt = Optimizer(symplectic_lr=0.5)
    opt.minimize(cost_fn, by_optimizing=[G], max_steps=50)
    for i in range(2, min(20, len(opt.opt_history))):
        assert opt.opt_history[i - 1] >= opt.opt_history[i]

    settings.PRECISION_BITS_HERMITE_POLY = original_precision


@pytest.mark.parametrize("precision", precisions)
def test_compactFock_1leftover_gradients(precision):
    r"""
    Test getting Fock amplitudes and if all but the first
    mode are detected (math.hermite_renormalized_1leftoverMode).
    """
    skip_np()

    settings.PRECISION_BITS_HERMITE_POLY = precision
    G = Ggate(num_modes=2, symplectic_trainable=True)

    def cost_fn():
        n2 = 3  # number of detected photons
        state_opt = Vacuum(2) >> G
        A, B, G0 = wigner_to_bargmann_rho(state_opt.cov, state_opt.means)
        marginal = math.hermite_renormalized_1leftoverMode(
            math.conj(-A), math.conj(B), math.conj(G0), cutoffs=[8, n2 + 1]
        )
        conditional_state = normalize(State(dm=marginal[..., n2]))
        return -fidelity(conditional_state, SqueezedVacuum(r=1))

    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(cost_fn, by_optimizing=[G], max_steps=50)
    for i in range(2, min(20, len(opt.opt_history))):
        assert opt.opt_history[i - 1] >= opt.opt_history[i]

    settings.PRECISION_BITS_HERMITE_POLY = original_precision
