"""
Unit tests for mrmustard.math.numba.compactFock~
"""
import numpy as np
from mrmustard.lab.states import Thermal
from mrmustard.lab.gates import Sgate, Dgate, Ggate
from mrmustard.physics.bargmann import wigner_to_bargmann_rho
from mrmustard.math import Math
math = Math()  # use methods in math if you want them to be differentiable


def random_ABC(M):
    '''
    generate random Bargmann parameters A,B,C
    for a multimode Gaussian state with displacement
    '''
    random_vals = np.random.uniform(low=0,high=1,size=[5,M])
    state = Thermal(nbar=random_vals[0]*10) >> Sgate(r=random_vals[1],phi=random_vals[2]*2*np.pi) >> Dgate(x=random_vals[3],y=random_vals[4]) >> Ggate(num_modes=M)
    A,B,G0 = wigner_to_bargmann_rho(state.cov,state.means)
    return A,B,G0

def test_compactFock_diagonal():
    """Test getting Fock amplitudes if all modes are detected (math.hermite_renormalized_diagonal)"""
    M = 3
    cutoffs = [7,4,8]
    A, B, G0 = random_ABC(M) # Create random state (M mode Gaussian state with displacement)

    # Vanilla MM
    G_ref = math.hermite_renormalized(math.conj(-A), math.conj(B), math.conj(G0),
                                      shape=list(cutoffs) * 2).numpy()  # note: shape=[C1,C2,C3,...,C1,C2,C3,...]

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
    cutoffs = [7,4,8]
    A, B, G0 = random_ABC(M) # Create random state (M mode Gaussian state with displacement)

    # New algorithm
    G_leftover = math.hermite_renormalized_1leftoverMode(math.conj(-A), math.conj(B), math.conj(G0), cutoffs)

    # Vanilla MM
    G_ref = math.hermite_renormalized(math.conj(-A), math.conj(B), math.conj(G0),
                                      shape=list(cutoffs) * 2).numpy()  # note: shape=[C1,C2,C3,...,C1,C2,C3,...]

    # Extract amplitudes of leftover mode from vanilla MM
    ref_leftover = np.zeros([cutoffs[0]] * 2 + list(cutoffs)[1:], dtype=np.complex128)
    for inds in np.ndindex(*cutoffs[1:]):
        ref_leftover[tuple([slice(cutoffs[0]), slice(cutoffs[0])] + list(inds))] = G_ref[
            tuple([slice(cutoffs[0])] + list(inds) + [slice(cutoffs[0])] + list(inds))]
    assert np.allclose(ref_leftover, G_leftover)
