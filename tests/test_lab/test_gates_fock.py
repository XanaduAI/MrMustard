# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=import-outside-toplevel

import numpy as np
import pytest
from hypothesis import given
from thewalrus.fock_gradients import (
    beamsplitter as tw_beamsplitter,
    displacement as tw_displacement,
    mzgate,
    squeezing,
    two_mode_squeezing,
)

from mrmustard import math, settings
from mrmustard.lab import (
    AdditiveNoise,
    Amplifier,
    Attenuator,
    BSgate,
    Coherent,
    Dgate,
    Interferometer,
    MZgate,
    RealInterferometer,
    Rgate,
    S2gate,
    Sgate,
    Gaussian,
    PhaseNoise,
    Thermal,
)
from mrmustard.lab.states import TMSV, Fock, SqueezedVacuum, State
from mrmustard.math.lattice import strategies
from mrmustard.physics import fock
from tests.random import (
    angle,
    array_of_,
    medium_float,
    n_mode_pure_state,
    r,
    single_mode_cv_channel,
    single_mode_unitary_gate,
    two_mode_unitary_gate,
)
from ..conftest import skip_np


@given(state=n_mode_pure_state(num_modes=1), x=medium_float, y=medium_float)
def test_Dgate_1mode(state, x, y):
    state_out = state >> Dgate(x, y) >> Dgate(-x, -y)
    assert state_out == state


def test_attenuator_on_fock():
    "tests that attenuating a fock state makes it mixed"
    assert not (Fock(10) >> Attenuator(0.5)).is_pure


@given(
    state=n_mode_pure_state(num_modes=2),
    xxyy=array_of_(medium_float, minlen=4, maxlen=4),
)
def test_Dgate_2mode(state, xxyy):
    x1, x2, y1, y2 = xxyy
    state_out = state >> Dgate([x1, x2], [y1, y2]) >> Dgate([-x1, -x2], [-y1, -y2])
    assert state_out == state


def test_additive_noise_equal_to_circuit():
    """Test that the channel is equivalent to an amplifier followed by an attenuator."""
    na, nb = np.random.uniform(size=2)
    amp = 1.0 + np.random.uniform()
    c1 = Attenuator(1 / amp, nb) >> Amplifier(amp, na)
    c2 = AdditiveNoise(2 * (amp - 1) * (1 + na + nb))
    assert all(np.allclose(ele1, ele2) for ele1, ele2 in zip(c1.bargmann(), c2.bargmann()))


@given(gate=single_mode_cv_channel())
def test_single_mode_fock_equals_gaussian_dm(gate):
    """Test same state is obtained via fock representation or phase space
    for single mode circuits."""
    cutoffs = [60]
    gaussian_state = SqueezedVacuum(0.5) >> Attenuator(0.5)
    fock_state = State(dm=gaussian_state.dm(cutoffs))

    via_fock_space_dm = (fock_state >> gate).dm(cutoffs)
    via_phase_space_dm = (gaussian_state >> gate).dm(cutoffs)
    assert np.allclose(via_fock_space_dm, via_phase_space_dm)


@given(gate=single_mode_unitary_gate())
def test_single_mode_fock_equals_gaussian_ket(gate):
    """Test same state is obtained via fock representation or phase space
    for single mode circuits."""
    cutoffs = [70]
    gaussian_state = SqueezedVacuum(-0.1)
    fock_state = State(ket=gaussian_state.ket(cutoffs))

    via_fock_space_ket = (fock_state >> gate).ket([10])
    via_phase_space_ket = (gaussian_state >> gate).ket([10])
    phase = np.exp(1j * np.angle(via_fock_space_ket[0]))
    assert np.allclose(via_fock_space_ket, phase * via_phase_space_ket)


@given(gate=single_mode_unitary_gate())
def test_single_mode_fock_equals_gaussian_ket_dm(gate):
    """Test same state is obtained via fock representation or phase space
    for single mode circuits."""
    cutoffs = (70,)
    gaussian_state = SqueezedVacuum(-0.1)
    fock_state = State(ket=gaussian_state.ket(cutoffs))

    via_fock_space_dm = (fock_state >> gate >> Attenuator(0.1)).dm([10])
    via_phase_space_dm = (gaussian_state >> gate >> Attenuator(0.1)).dm([10])
    assert np.allclose(via_fock_space_dm, via_phase_space_dm, atol=1e-5)


@given(gate=two_mode_unitary_gate())
def test_two_mode_fock_equals_gaussian(gate):
    """Test same state is obtained via fock representation or phase space
    for two modes circuits."""
    cutoffs = (20, 20)
    gaussian_state = TMSV(0.1) >> BSgate(np.pi / 2) >> Attenuator(0.5)
    fock_state = State(dm=gaussian_state.dm(cutoffs))

    via_fock_space_dm = (fock_state >> gate).dm(cutoffs)
    via_phase_space_dm = (gaussian_state >> gate).dm(cutoffs)
    assert np.allclose(via_fock_space_dm, via_phase_space_dm)


@pytest.mark.parametrize(
    "cutoffs,x,y",
    [
        [[5, 5], 0.3, 0.5],
        [[5, 5], 0.0, 0.0],
        [[2, 2, 2, 2], [0.1, 0.1], [0.25, -0.2]],
        [[3, 3, 3, 3], [0.0, 0.0], [0.0, 0.0]],
        [[2, 5, 1, 2, 5, 1], [0.1, 5.0, 1.0], [-0.3, 0.1, 0.0]],
        [[3, 3, 3, 3, 3, 3, 3, 3], [0.1, 0.2, 0.3, 0.4], [-0.5, -4, 3.1, 4.2]],
    ],
)
def test_fock_representation_displacement(cutoffs, x, y):
    """Tests that DGate returns the correct unitary."""

    # apply gate
    dgate = Dgate(x, y)
    Ud = dgate.U(cutoffs)

    # compare with the standard way of calculating
    # transformation unitaries using the Choi isomorphism
    X, _, d = dgate.XYd(allow_none=False)
    expected_Ud = fock.wigner_to_fock_U(X, d, cutoffs)

    assert np.allclose(Ud, expected_Ud, atol=1e-5)


@given(x1=medium_float, x2=medium_float, y1=medium_float, y2=medium_float)
def test_parallel_displacement(x1, x2, y1, y2):
    """Tests that parallel Dgate returns the correct unitary."""
    U12 = Dgate([x1, x2], [y1, y2]).U([2, 7, 2, 7])
    U1 = Dgate(x1, y1).U([2, 2])
    U2 = Dgate(x2, y2).U([7, 7])
    assert np.allclose(U12, np.transpose(np.tensordot(U1, U2, [[], []]), [0, 2, 1, 3]))


def test_squeezer_grad_against_finite_differences():
    """tests fock squeezer gradient against finite differences"""
    skip_np()

    cutoffs = (5, 5)
    r = math.new_variable(0.5, None, "r")
    phi = math.new_variable(0.1, None, "phi")
    delta = 1e-6
    dUdr = (Sgate(r + delta, phi).U(cutoffs) - Sgate(r - delta, phi).U(cutoffs)) / (2 * delta)
    dUdphi = (Sgate(r, phi + delta).U(cutoffs) - Sgate(r, phi - delta).U(cutoffs)) / (2 * delta)
    _, (gradr, gradphi) = math.value_and_gradients(
        lambda: fock.squeezer(r, phi, shape=cutoffs), [r, phi]
    )
    assert np.allclose(gradr, 2 * np.real(np.sum(dUdr)))
    assert np.allclose(gradphi, 2 * np.real(np.sum(dUdphi)))


def test_displacement_grad():
    """tests fock displacement gradient against finite differences"""
    cutoffs = [5, 5]
    x = math.new_variable(0.1, None, "x")
    y = math.new_variable(0.1, None, "y")
    alpha = math.asnumpy(math.make_complex(x, y))
    delta = 1e-6
    dUdx = (fock.displacement(x + delta, y, cutoffs) - fock.displacement(x - delta, y, cutoffs)) / (
        2 * delta
    )
    dUdy = (fock.displacement(x, y + delta, cutoffs) - fock.displacement(x, y - delta, cutoffs)) / (
        2 * delta
    )

    D = fock.displacement(x, y, shape=cutoffs)
    dD_da, dD_dac = strategies.jacobian_displacement(math.asnumpy(D), alpha)
    assert np.allclose(dD_da + dD_dac, dUdx)
    assert np.allclose(1j * (dD_da - dD_dac), dUdy)


def test_fock_representation_displacement_rectangular():
    """Tests that DGate returns the correct unitary."""
    x, y = 0.3, 0.5
    cutoffs = 5, 10
    # apply gate
    dgate = Dgate(x, y)
    Ud = dgate.U(cutoffs)

    # compare with tw implementation
    expected_Ud = tw_displacement(np.sqrt(x * x + y * y), np.arctan2(y, x), 10)[:5, :10]

    assert np.allclose(Ud, expected_Ud, atol=1e-5)


def test_fock_representation_displacement_rectangular2():
    """Tests that DGate returns the correct unitary."""
    x, y = 0.3, 0.5
    cutoffs = 10, 5
    # apply gate
    dgate = Dgate(x, y)
    Ud = dgate.U(cutoffs)

    # compare with tw implementation
    expected_Ud = tw_displacement(np.sqrt(x * x + y * y), np.arctan2(y, x), 10)[:10, :5]

    assert np.allclose(Ud, expected_Ud, atol=1e-5)


@given(r=r, phi=angle)
def test_fock_representation_squeezing(r, phi):
    S = Sgate(r=r, phi=phi)
    expected = squeezing(r=r, theta=phi, cutoff=20)
    assert np.allclose(expected, S.U(cutoffs=[20, 20]), atol=1e-5)


@given(r1=r, phi1=angle, r2=r, phi2=angle)
def test_parallel_squeezing(r1, phi1, r2, phi2):
    """Tests that two parallel squeezers return the correct unitary."""
    U12 = Sgate([r1, r2], [phi1, phi2]).U([5, 7, 5, 7])
    U1 = Sgate(r1, phi1).U([5, 5])
    U2 = Sgate(r2, phi2).U([7, 7])
    assert np.allclose(U12, np.transpose(np.tensordot(U1, U2, [[], []]), [0, 2, 1, 3]))


@given(theta=angle, phi=angle)
def test_fock_representation_beamsplitter(theta, phi):
    BS = BSgate(theta=theta, phi=phi)
    expected = tw_beamsplitter(theta=theta, phi=phi, cutoff=10)
    assert np.allclose(expected, BS.U(cutoffs=[10, 10, 10, 10]), atol=1e-5)


@given(r=r, phi=angle)
def test_fock_representation_two_mode_squeezing(r, phi):
    S2 = S2gate(r=r, phi=phi)
    expected = two_mode_squeezing(r=r, theta=phi, cutoff=10)
    assert np.allclose(expected, S2.U(cutoffs=[10, 10]), atol=1e-5)


@given(phi_a=angle, phi_b=angle)
def test_fock_representation_mzgate(phi_a, phi_b):
    MZ = MZgate(phi_a=phi_a, phi_b=phi_b, internal=False)
    expected = mzgate(theta=phi_b, phi=phi_a, cutoff=10)
    assert np.allclose(expected, MZ.U(cutoffs=[10, 10]), atol=1e-5)


@pytest.mark.parametrize(
    "cutoffs,angles,modes",
    [
        [[5, 4, 3], [np.pi, np.pi / 2, np.pi / 4], None],
        [[3, 4], [np.pi / 3, np.pi / 2], [0, 1]],
        [[3], np.pi / 6, [0]],
    ],
)
def test_fock_representation_rgate(cutoffs, angles, modes):
    """Tests that DGate returns the correct unitary."""

    # apply gate
    rgate = Rgate(angles, modes=modes)
    R = rgate.U(cutoffs)

    # compare with the standard way of calculating
    # transformation unitaries using the Choi isomorphism
    d = np.zeros(len(cutoffs) * 2)
    expected_R = fock.wigner_to_fock_U(rgate.X_matrix, d, tuple(cutoffs + cutoffs))
    assert np.allclose(R, expected_R, atol=1e-5)


def test_raise_interferometer_error():
    """test Interferometer raises an error when both `modes` and `num_modes` don't match"""
    num_modes = 3
    modes = [0, 2]
    with pytest.raises(ValueError):
        Interferometer(num_modes=num_modes, modes=modes)
    with pytest.raises(ValueError):
        RealInterferometer(num_modes=num_modes, modes=modes)
    modes = [2, 5, 6, 7]
    with pytest.raises(ValueError):
        Interferometer(num_modes=num_modes, modes=modes)
    with pytest.raises(ValueError):
        RealInterferometer(num_modes=num_modes, modes=modes)


def test_choi_cutoffs():
    output = State(dm=Coherent([1.0, 1.0]).dm([5, 8])) >> Attenuator(0.5, modes=[1])
    assert output.cutoffs == [5, 8]  # cutoffs are respected by the gate


@pytest.mark.parametrize(
    "gate, kwargs",
    [
        (Sgate, {"r": 1}),
        (Rgate, {"angle": 0.1}),
        (Dgate, {"x": 0.1}),
    ],
)
@pytest.mark.parametrize("cutoff", [2, 5])
@pytest.mark.parametrize("modes", [[0], [1, 2]])
def test_choi_for_unitary(gate, kwargs, cutoff, modes):
    """tests the `choi` method for unitary transformations"""
    gate = gate(**kwargs)

    gate = gate[modes]
    N = gate.num_modes
    cutoffs = [cutoff] * N

    choi = math.asnumpy(gate.choi(cutoffs=cutoffs)).reshape(cutoff ** (2 * N), cutoff ** (2 * N))

    t = math.asnumpy(gate.U(cutoffs=cutoffs))
    row = t.flatten().reshape(1, cutoff ** (2 * N))
    col = t.flatten().reshape(cutoff ** (2 * N), 1)
    expected = np.dot(col, row)

    assert np.allclose(expected, choi)


@given(theta=angle, phi=angle)
def test_schwinger_bs_equals_vanilla_bs_for_small_cutoffs(theta, phi):
    """Tests that the Schwinger boson BS gate is equivalent to the vanilla BS gate for low cutoffs."""
    U_vanilla = BSgate(theta, phi).U([10, 10, 10, 10], method="vanilla")
    U_schwinger = BSgate(theta, phi).U([10, 10, 10, 10], method="schwinger")

    assert np.allclose(U_vanilla, U_schwinger, atol=1e-6)


# pylint: disable=protected-access
@given(phase_stdev=medium_float.filter(lambda x: x > 0))
def test_phasenoise_creates_dm(phase_stdev):
    """test that the phase noise gate is correctly applied"""
    assert (Coherent(1.0) >> PhaseNoise(phase_stdev))._dm is not None
    assert (Fock(10) >> PhaseNoise(phase_stdev))._dm is not None


@given(phase_stdev=medium_float.filter(lambda x: x > 0))
def test_phasenoise_symmetry(phase_stdev):
    "tests that symmetric states are not affected by phase noise"
    assert (Fock(1) >> PhaseNoise(phase_stdev)) == Fock(1)
    settings.AUTOCUTOFF_MIN_CUTOFF = 100
    assert (Thermal(1) >> PhaseNoise(phase_stdev)) == Thermal(1)
    settings.AUTOCUTOFF_MIN_CUTOFF = 1


@given(phase_stdev=medium_float.filter(lambda x: x > 0))
def test_phasenoise_on_multimode(phase_stdev):
    "tests that phase noise can be used on multimode states"
    G2 = Gaussian(2) >> Attenuator(0.1, modes=[0, 1])
    P = PhaseNoise(phase_stdev, modes=[1])
    settings.AUTOCUTOFF_MIN_CUTOFF = 20
    assert (G2 >> P).get_modes(0) == G2.get_modes(0)
    assert (G2 >> P).get_modes(1) == G2.get_modes(1) >> P
    settings.AUTOCUTOFF_MIN_CUTOFF = 1


def test_phasenoise_large_noise():
    "tests that large phase noise kills the off-diagonal elements"
    G1 = Gaussian(1)
    P = PhaseNoise(1000)
    assert (G1 >> P) == State(dm=math.diag(math.diag_part(G1.dm())))


def test_phasenoise_zero_noise():
    "tests that zero phase noise is equal to the identity"
    G1 = Gaussian(1)
    P = PhaseNoise(0.0)
    assert (G1 >> P) == State(dm=G1.dm())
