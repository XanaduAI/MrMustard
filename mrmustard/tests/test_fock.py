from hypothesis import strategies as st, given, settings
import numpy as np
from scipy.special import factorial
from thewalrus.quantum import total_photon_number_distribution
from mrmustard import Dgate, Sgate, LossChannel, BSgate, S2gate, Ggate, Circuit, Vacuum


@given(n_mean=st.floats(0.0, 3.0), phi=st.floats(0.0, 2 * np.pi), cutoff=st.integers(1, 5))  # deadline = None because of jit compilation on first run
@settings(deadline=None)
def test_two_mode_squeezing_fock(n_mean, phi, cutoff):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state
    Note that this is consistent with the Strawberryfields convention"""
    r = np.arcsinh(np.sqrt(n_mean))
    S = S2gate(modes=[0, 1], r=r, phi=phi)
    amps = S(Vacuum(num_modes=2)).ket(cutoffs=[cutoff, cutoff])
    diag = (1 / np.cosh(r)) * (np.exp(1j * phi) * np.tanh(r)) ** np.arange(cutoff)
    expected = np.diag(diag)
    assert np.allclose(amps, expected)


@given(n_mean=st.floats(0.0, 3.0), phi=st.floats(0.0, 2 * np.pi), varphi=st.floats(0.0, 2 * np.pi), cutoff=st.integers(2, 5))
@settings(deadline=None)
def test_hong_ou_mandel(n_mean, phi, varphi, cutoff):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state"""
    cutoff = 2
    circ = Circuit()
    r = np.arcsinh(np.sqrt(n_mean))
    circ.append(S2gate(r=r, phi=phi+np.pi)[0, 1])
    circ.append(S2gate(r=r, phi=phi+np.pi)[2, 3])
    circ.append(BSgate(theta=np.pi / 4, phi=varphi)[1, 2])
    amps = circ(Vacuum(num_modes=4)).ket(cutoffs=[cutoff, cutoff, cutoff, cutoff])
    assert np.allclose(amps[1, 1, 1, 1], 0.0)


@given(realpha=st.floats(-5.0, 5.0), imalpha=st.floats(-5.0, 5.0))
def test_coherent_state(realpha, imalpha):
    """Test that coherent states have the correct photon number statistics"""
    cutoff = 10
    circ = Circuit()
    circ.append(Dgate(modes=[0], x=realpha, y=imalpha))
    amps = circ(Vacuum(num_modes=1)).ket(cutoffs=[cutoff])
    alpha = realpha + 1j * imalpha
    expected = np.exp(-0.5 * np.abs(alpha) ** 2) * np.array([alpha ** n / np.sqrt(factorial(n)) for n in range(cutoff)])
    assert np.allclose(amps, expected)


@given(r=st.floats(0.0, 5.0), phi=st.floats(0.0, 2 * np.pi), cutoff=st.integers(2, 12))
def test_squeezed_state(r, phi, cutoff):
    """Test that squeezed states have the correct photon number statistics
    Note that we use the same sign with respect to SMSV in https://en.wikipedia.org/wiki/Squeezed_coherent_state"""
    S = Sgate(modes=[0], r=r, phi=phi)
    amps = S(Vacuum(num_modes=1)).ket(cutoffs=[cutoff])
    assert np.allclose(amps[1::2], 0.0)
    non_zero_amps = amps[0::2]
    len_non_zero = len(non_zero_amps)
    amp_pairs = (
        1
        / np.sqrt(np.cosh(r))
        * np.array(
            [(-np.exp(1j * phi) * np.tanh(r)) ** n * np.sqrt(factorial(2 * n)) / (2 ** n * factorial(n)) for n in range(len_non_zero)]
        )
    )
    assert np.allclose(non_zero_amps, amp_pairs)


@given(n_mean=st.floats(0.0, 5.0), phi=st.floats(0.0, 2 * np.pi))
def test_two_mode_squeezing_fock_mean_and_covar(n_mean, phi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state"""
    r = np.arcsinh(np.sqrt(n_mean))
    S = S2gate(modes=[0, 1], r=-r, phi=phi)
    state = S(Vacuum(num_modes=2))
    meanN = state.number_means
    covN = state.number_cov
    expectedN = np.array([n_mean, n_mean])
    expectedCov = n_mean * (n_mean + 1) * np.ones([2, 2])
    assert np.allclose(meanN, expectedN)
    assert np.allclose(covN, expectedCov)


@given(n_mean=st.floats(0.0, 5.0), phi=st.floats(0.0, 2 * np.pi), eta=st.floats(0.0, 1.0))
def test_lossy_squeezing(n_mean, phi, eta):
    """Tests the total photon number distribution of a lossy squeezed state"""
    r = np.arcsinh(np.sqrt(n_mean))
    cutoff = 40
    circ = Circuit()
    r = np.arcsinh(np.sqrt(n_mean))
    circ.append(Sgate(modes=[0], r=-r, phi=0.0))
    circ.append(LossChannel(modes=[0], transmissivity=eta))
    ps = circ(Vacuum(num_modes=1)).fock_probabilities(cutoffs=[cutoff])
    expected = np.array([total_photon_number_distribution(n, 1, r, eta) for n in range(cutoff)])
    assert np.allclose(ps, expected)


@given(n_mean=st.floats(0.0, 1.5), eta_s=st.floats(0.0, 1.0), eta_i=st.floats(0.0, 1.0))
@settings(deadline=None)
def test_lossy_two_mode_squeezing(n_mean, eta_s, eta_i):
    """Tests the total photon number distribution of a lossy two-mode squeezed state"""
    r = np.arcsinh(np.sqrt(n_mean))
    cutoff = 10
    TMSV = S2gate(modes=[0, 1], r=-r, phi=0.0)(Vacuum(2))
    state = LossChannel(modes=[0,1], transmissivity=[eta_s, eta_i])(TMSV)
    ps = state.fock_probabilities(cutoffs=[cutoff, cutoff])
    n = np.arange(cutoff)
    mean_s = n @ np.sum(ps, axis=1)
    mean_i = n @ np.sum(ps, axis=0)
    assert np.allclose(mean_s, n_mean * eta_s, atol=1e-4)
    assert np.allclose(mean_i, n_mean * eta_i, atol=1e-4)

@st.composite
def n_cutoffs(draw):
    n = draw(st.integers(1, 3))
    return (n, draw(st.lists(st.integers(1, 4), min_size=n, max_size=n)))

@given(ncutoffs=n_cutoffs())
@settings(deadline=None)
def test_density_matrix(ncutoffs):
    """Tests the density matrix of a pure state is equal to |psi><psi|"""
    n, cutoffs = ncutoffs
    G = Ggate(modes=[i for i in range(n)])
    L = LossChannel(modes=[i for i in range(n)], transmissivity=1.0)
    rho_legit = L(G(Vacuum(num_modes=n))).dm(cutoffs=cutoffs)
    rho_built = G(Vacuum(num_modes=n)).dm(cutoffs=cutoffs)
    assert np.allclose(rho_legit, rho_built)
