"""
Unit tests for mrmustard.math.compactFock.compactFock~
"""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab import DM, Ggate, SqueezedVacuum, Vacuum
from mrmustard.lab.transformations.attenuator import Attenuator
from mrmustard.physics import gaussian
from mrmustard.training import Optimizer


def test_compactFock_diagonal():
    r"""Test getting Fock amplitudes if all modes are
    detected (math.hermite_renormalized_diagonal)
    """
    cutoffs = (5, 5, 5)

    A, B, G0 = DM.random([0, 1, 2]).bargmann_triple()

    # Vanilla MM
    G_ref = math.hermite_renormalized(
        math.conj(-A),
        math.conj(B),
        math.conj(G0),
        shape=cutoffs * 2,
    )  # note: shape=[C1,C2,C3,...,C1,C2,C3,...]
    G_ref = math.asnumpy(G_ref)

    # Extract diagonal amplitudes from vanilla MM
    ref_diag = np.zeros(cutoffs, dtype=np.complex128)
    for inds in np.ndindex(*cutoffs):
        inds_expanded = list(inds) + list(inds)  # a,b,c,a,b,c
        ref_diag[inds] = G_ref[tuple(inds_expanded)]

    G_diag = math.hermite_renormalized_diagonal(math.conj(-A), math.conj(B), math.conj(G0), cutoffs)
    assert math.allclose(ref_diag, G_diag)


def test_compactFock_1leftover():
    r"""
    Test getting Fock amplitudes if all but the first mode
    are detected (math.hermite_renormalized_1leftoverMode).
    """
    A, B, G0 = DM.random([0, 1, 2]).bargmann_triple()
    # New algorithm
    G_leftover = math.hermite_renormalized_1leftoverMode(
        A,
        B,
        G0,
        output_cutoff=3,
        pnr_cutoffs=(1, 2),
    )  # shape=(4,4,2,3)
    # Vanilla MM
    G_ref = math.hermite_renormalized(
        A,
        B,
        G0,
        shape=(4, 2, 3, 4, 2, 3),
    )  # shape=[C1,C2,C3,...,C1,C2,C3,...]
    # Extract amplitudes of leftover mode from vanilla MM
    expected = np.diagonal(np.diagonal(G_ref, axis1=1, axis2=4), axis1=1, axis2=3)
    assert np.allclose(expected, G_leftover)


@pytest.mark.requires_backend("tensorflow")
def test_compactFock_diagonal_gradients():
    r"""
    Test getting Fock amplitudes and gradients if all modes
    are detected (math.hermite_renormalized_diagonal).
    """
    G = Ggate(0, symplectic_trainable=True)
    Att = Attenuator(0, 0.9)

    def cost_fn():
        n1 = 2  # number of detected photons
        state_opt = Vacuum([0]) >> G >> Att
        A, B, G0 = state_opt.bargmann_triple()
        probs = math.hermite_renormalized_diagonal(
            math.conj(-A),
            math.conj(B),
            math.conj(G0),
            cutoffs=[n1 + 1],
        )
        p = probs[n1]
        return -math.real(p)

    opt = Optimizer(symplectic_lr=0.5)
    opt.minimize(cost_fn, by_optimizing=[G], max_steps=5)
    for i in range(2, min(20, len(opt.opt_history))):
        assert opt.opt_history[i - 1] >= opt.opt_history[i]


@pytest.mark.requires_backend()  # TODO: implement gradient of hermite_renormalized_1leftoverMode
def test_compactFock_1leftover_gradients():
    r"""
    Test getting Fock amplitudes and if all but the first
    mode are detected (math.hermite_renormalized_1leftoverMode).
    """
    G = Ggate((0, 1), symplectic_trainable=True)
    Att = Attenuator(0, 0.9)

    def cost_fn():
        n2 = 2  # number of detected photons
        state_opt = Vacuum([0, 1]) >> G >> Att
        A, B, G0 = state_opt.bargmann_triple()
        marginal = math.hermite_renormalized_1leftoverMode(
            math.conj(-A),
            math.conj(B),
            math.conj(G0),
            output_cutoff=2,
            pnr_cutoffs=[n2 + 1],
        )
        conditional_state = DM.from_fock([0], marginal[..., n2]).normalize()
        return -gaussian.fidelity(
            *conditional_state.phase_space(0)[:2],
            *SqueezedVacuum(0, r=1).phase_space(0)[:2],
        )

    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(cost_fn, by_optimizing=[G], max_steps=5)
    for i in range(2, len(opt.opt_history)):
        assert opt.opt_history[i - 1] >= opt.opt_history[i]
