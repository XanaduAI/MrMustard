"""
Unit tests for mrmustard.math.numba.compactFock~
"""
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from mrmustard.lab import Ggate, SqueezedVacuum, State, Vacuum
from mrmustard.math import Math
from mrmustard.physics import fidelity, normalize
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.training import Optimizer
from tests.random import n_mode_mixed_state

math = Math()  # use methods in math if you want them to be differentiable


@st.composite
def random_ABC(draw, M):
    r"""
    generate random Bargmann parameters A,B,C
    for a multimode Gaussian state with displacement
    """
    state = draw(n_mode_mixed_state(M))
    A, B, G0 = wigner_to_bargmann_rho(state.cov, state.means)
    return A, B, G0


@given(random_ABC(M=3))
def test_compactFock_diagonal(A_B_G0):
    """Test getting Fock amplitudes if all modes are detected (math.hermite_renormalized_diagonal)"""
    cutoffs = [7, 4, 8]
    A, B, G0 = A_B_G0  # Create random state (M mode Gaussian state with displacement)

    # Vanilla MM
    G_ref = math.hermite_renormalized(
        math.conj(-A), math.conj(B), math.conj(G0), shape=list(cutoffs) * 2
    ).numpy()  # note: shape=[C1,C2,C3,...,C1,C2,C3,...]

    # Extract diagonal amplitudes from vanilla MM
    ref_diag = np.zeros(cutoffs, dtype=np.complex128)
    for inds in np.ndindex(*cutoffs):
        inds_expanded = list(inds) + list(inds)  # a,b,c,a,b,c
        ref_diag[inds] = G_ref[tuple(inds_expanded)]

    # New MM
    G_diag = math.hermite_renormalized_diagonal(math.conj(-A), math.conj(B), math.conj(G0), cutoffs)
    assert np.allclose(ref_diag, G_diag)


@given(random_ABC(M=3))
def test_compactFock_1leftover(A_B_G0):
    """Test getting Fock amplitudes if all but the first mode are detected (math.hermite_renormalized_1leftoverMode)"""
    cutoffs = [7, 4, 8]
    A, B, G0 = A_B_G0  # Create random state (M mode Gaussian state with displacement)

    # New algorithm
    G_leftover = math.hermite_renormalized_1leftoverMode(
        math.conj(-A), math.conj(B), math.conj(G0), cutoffs
    )

    # Vanilla MM
    G_ref = math.hermite_renormalized(
        math.conj(-A), math.conj(B), math.conj(G0), shape=list(cutoffs) * 2
    ).numpy()  # note: shape=[C1,C2,C3,...,C1,C2,C3,...]

    # Extract amplitudes of leftover mode from vanilla MM
    ref_leftover = np.zeros([cutoffs[0]] * 2 + list(cutoffs)[1:], dtype=np.complex128)
    for inds in np.ndindex(*cutoffs[1:]):
        ref_leftover[tuple([slice(cutoffs[0]), slice(cutoffs[0])] + list(inds))] = G_ref[
            tuple([slice(cutoffs[0])] + list(inds) + [slice(cutoffs[0])] + list(inds))
        ]
    assert np.allclose(ref_leftover, G_leftover)


def test_compactFock_diagonal_gradients():
    """Test getting Fock amplitudes AND GRADIENTS if all modes are detected (math.hermite_renormalized_diagonal)"""
    G = Ggate(num_modes=3, symplectic_trainable=True)

    def cost_fn(G):
        n1, n2, n3 = 2, 2, 4  # number of detected photons
        state_opt = Vacuum(3) >> G
        A, B, G0 = wigner_to_bargmann_rho(state_opt.cov, state_opt.means)
        G = math.hermite_renormalized_diagonal(
            math.conj(-A), math.conj(B), math.conj(G0), cutoffs=[n1 + 1, n2 + 1, n3 + 1]
        )
        p = G[n1, n2, n3]
        p_target = 0.5
        return math.abs(p_target - p)

    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(lambda: cost_fn(G), by_optimizing=[G], max_steps=50)
    for i in range(2, min(20, len(opt.opt_history))):
        assert opt.opt_history[i - 1] >= opt.opt_history[i]


def test_compactFock_1leftover_gradients():
    """Test getting Fock amplitudes AND GRADIENTS if all but the first mode are detected (math.hermite_renormalized_1leftoverMode)"""
    G = Ggate(num_modes=3, symplectic_trainable=True)

    def cost_fn(G):
        n2, n3 = 1, 3  # number of detected photons
        state_opt = Vacuum(3) >> G
        A, B, G0 = wigner_to_bargmann_rho(state_opt.cov, state_opt.means)
        G = math.hermite_renormalized_1leftoverMode(
            math.conj(-A), math.conj(B), math.conj(G0), cutoffs=[8, n2 + 1, n3 + 1]
        )
        conditional_state = normalize(State(dm=G[..., n2, n3]))
        return -fidelity(conditional_state, SqueezedVacuum(r=1))

    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(lambda: cost_fn(G), by_optimizing=[G], max_steps=50)
    for i in range(2, min(20, len(opt.opt_history))):
        assert opt.opt_history[i - 1] >= opt.opt_history[i]
