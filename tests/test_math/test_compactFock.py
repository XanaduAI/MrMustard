"""
Unit tests for mrmustard.math.numba.compactFock~
"""
import numpy as np
from mrmustard.lab import Vacuum, State, SqueezedVacuum, Thermal, Sgate, Dgate, Ggate
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.physics import fidelity, normalize
from mrmustard.training import Optimizer
from mrmustard.math import Math

math = Math()  # use methods in math if you want them to be differentiable


def random_ABC(M):
    """
    generate random Bargmann parameters A,B,C
    for a multimode Gaussian state with displacement
    """
    random_vals = np.random.uniform(low=0, high=1, size=[5, M])
    state = (
        Thermal(nbar=random_vals[0] * 10)
        >> Sgate(r=random_vals[1], phi=random_vals[2] * 2 * np.pi)
        >> Dgate(x=random_vals[3], y=random_vals[4])
        >> Ggate(num_modes=M)
    )
    A, B, G0 = wigner_to_bargmann_rho(state.cov, state.means)
    return A, B, G0


def test_compactFock_diagonal():
    """Test getting Fock amplitudes if all modes are detected (math.hermite_renormalized_diagonal)"""
    M = 3
    cutoffs = [7, 4, 8]
    A, B, G0 = random_ABC(M)  # Create random state (M mode Gaussian state with displacement)

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


def test_compactFock_1leftover():
    """Test getting Fock amplitudes if all but the first mode are detected (math.hermite_renormalized_1leftoverMode)"""
    M = 3
    cutoffs = [7, 4, 8]
    A, B, G0 = random_ABC(M)  # Create random state (M mode Gaussian state with displacement)

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

    def cost_fn():
        n1, n2, n3 = 2, 2, 4  # number of detected photons
        state_opt = Vacuum(3) >> I
        A, B, G0 = wigner_to_bargmann_rho(state_opt.cov, state_opt.means)
        G = math.hermite_renormalized_diagonal(
            math.conj(-A), math.conj(B), math.conj(G0), cutoffs=[3, 3, 5]
        )
        p = G[n1, n2, n3]
        p_target = 0.5
        return math.abs(p_target - p)

    I = Ggate(num_modes=3, symplectic_trainable=True)
    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(cost_fn, by_optimizing=[I], max_steps=50)
    for iter in range(2,50):
        assert opt.opt_history[iter-1] >= opt.opt_history[iter]


def test_compactFock_1leftover_gradients():
    """Test getting Fock amplitudes AND GRADIENTS if all but the first mode are detected (math.hermite_renormalized_1leftoverMode)"""

    def cost_fn():
        n2, n3 = 1, 3  # number of detected photons
        state_opt = Vacuum(3) >> I
        A, B, G0 = wigner_to_bargmann_rho(state_opt.cov, state_opt.means)
        G = math.hermite_renormalized_1leftoverMode(
            math.conj(-A), math.conj(B), math.conj(G0), cutoffs=[8, 2, 4]
        )
        G_firstMode = G[:, :, n2, n3]
        conditional_state = normalize(State(dm=G_firstMode))
        return -fidelity(conditional_state, SqueezedVacuum(r=1))

    I = Ggate(num_modes=3, symplectic_trainable=True)
    opt = Optimizer(symplectic_lr=0.1)
    opt.minimize(cost_fn, by_optimizing=[I], max_steps=50)
    for iter in range(2,50):
        assert opt.opt_history[iter-1] >= opt.opt_history[iter]
