# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Optimizer class"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from thewalrus.symplectic import two_mode_squeezing

from mrmustard import math, settings
from mrmustard.lab import (
    BSgate,
    Coherent,
    Dgate,
    DisplacedSqueezed,
    GaussianKet,
    Ggate,
    Interferometer,
    Number,
    RealInterferometer,
    S2gate,
    Sgate,
    SqueezedVacuum,
    TwoModeSqueezedVacuum,
    Vacuum,
)
from mrmustard.parameters import ParameterDict, Variable
from mrmustard.training import Optimizer


@pytest.mark.requires_backend("jax")
class TestOptimizer:
    r"""
    Tests for the ``Optimizer`` class.
    """

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_bsgate_grad_from_fock(self, batch_shape):
        """Test that the gradient of a beamsplitter gate is computed from the fock representation."""
        r_var = Variable(value=math.ones(batch_shape), name="r")

        def cost_fn(r_var):
            vac = Vacuum(0)
            num = Number(1, 1)
            bs = BSgate((0, 1), 0.5)
            sq = SqueezedVacuum(0, r=r_var)
            norm = 1 / sq.ansatz.batch_size if sq.ansatz.batch_shape else 1
            return -math.real(
                norm * math.sum(sq >> num >> bs >> (vac >> num).dual) ** 2,
            )

        opt = Optimizer(euclidean_lr=0.05)
        (optimized_r_var,) = opt.minimize(cost_fn, by_optimizing=[r_var], max_steps=100)

        assert math.all(optimized_r_var.value != r_var.value)

    def test_bsgate_optimization(self):
        """Test that BSgate is optimized correctly."""

        params = ParameterDict(Variable(0.05, "theta"), Variable(0.1, "phi"))

        target_gate = BSgate((0, 1), 0.1, 0.2).fock_array(30)

        def cost_fn(*variables):
            params = ParameterDict(*variables)
            bsgate = BSgate((0, 1), params.theta, params.phi)
            return -(math.abs(math.sum(math.conj(bsgate.fock_array(30)) * target_gate)) ** 2)

        opt = Optimizer()
        params = opt.minimize(cost_fn, by_optimizing=params)

        assert math.allclose(params.theta.value, 0.1, atol=0.01)
        assert math.allclose(params.phi.value, 0.2, atol=0.01)

    def test_cat_state_optimization(self):
        # Note: we need to intitialize the cat state with a non-zero value. This is because
        # the gradients are zero when x is zero.
        alpha_var1 = Variable(value=0.1, name="alpha", dtype=math.complex128)
        alpha_var2 = Variable(value=-0.1, name="alpha", dtype=math.complex128)
        expected_cat = Coherent(0, alpha=np.sqrt(np.pi)) + Coherent(0, alpha=-np.sqrt(np.pi))

        def cost_fn(alpha_var1, alpha_var2):
            cat_state = (Coherent(0, alpha=alpha_var1) + Coherent(0, alpha=alpha_var2)).normalize()
            return -math.abs(cat_state.fidelity(expected_cat.normalize()))

        # stable_threshold and max_steps are set to whatever gives us optimized parameters
        # that are within the default ATOL=1e-8 of the expected values
        opt = Optimizer(stable_threshold=1e-12)
        (alpha_var1, alpha_var2) = opt.minimize(
            cost_fn, by_optimizing=[alpha_var1, alpha_var2], max_steps=6000
        )

        assert math.allclose(
            [alpha_var1.value, alpha_var2.value], [np.sqrt(np.pi), -np.sqrt(np.pi)]
        )

    @pytest.mark.parametrize("alpha", [0.2 + 0.4j, -0.1 - 0.2j, 0.1j, 0.4])
    def test_complex_dgate_optimization_bargmann(self, alpha):
        alpha_var = Variable(value=alpha, name="alpha", dtype=math.complex128)

        def cost_fn(alpha_var):
            dgate = Dgate(0, alpha=alpha_var)
            target_state = Coherent(0, alpha=alpha)
            state_out = Vacuum(0) >> dgate
            return 1 - math.real(state_out.expectation(target_state))

        opt = Optimizer(euclidean_lr=0.05)
        (alpha_var,) = opt.minimize(cost_fn, by_optimizing=[alpha_var], max_steps=200)
        assert math.allclose(alpha_var.value, alpha, atol=0.01)

    @pytest.mark.parametrize("alpha", [0.2 + 0.4j, -0.1 - 0.2j, 0.1j, 0.4])
    def test_complex_dgate_optimization_fock(self, alpha):
        alpha_var = Variable(value=alpha, name="alpha", dtype=math.complex128)

        def cost_fn(alpha_var):
            dgate = Dgate(0, alpha=alpha_var)
            target_state = Coherent(0, alpha=alpha)
            state_out = dgate.fock_array((80, 1))[:, 0]
            return (
                1 - math.abs(math.sum(math.conj(state_out) * target_state.fock_array((80,)))) ** 2
            )

        opt = Optimizer(euclidean_lr=0.01)
        (alpha_var,) = opt.minimize(cost_fn, by_optimizing=[alpha_var], max_steps=200)
        assert math.allclose(alpha_var.value, alpha, atol=0.01)

    def test_dgate_optimization(self):
        """Test that Dgate is optimized correctly."""
        alpha_var = Variable(value=0, name="alpha", dtype=math.complex128)

        def cost_fn(alpha_var):
            dgate = Dgate(0, alpha=alpha_var)
            target_state = DisplacedSqueezed(0, r=0.0, alpha=0.1 + 0.2j)
            state_out = Vacuum(0) >> dgate
            return -math.real(state_out.expectation(target_state))

        opt = Optimizer()
        (alpha_var,) = opt.minimize(cost_fn, by_optimizing=[alpha_var])

        assert math.allclose(alpha_var.value, 0.1 + 0.2j, atol=0.01)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_displacement_grad_from_fock(self, batch_shape):
        """Test that the gradient of a displacement gate is computed from the fock representation."""
        alpha_var = Variable(
            value=math.ones(batch_shape) + 0.5, name="alpha", dtype=math.complex128
        )

        def cost_fn(alpha_var):
            vac = Vacuum(0).dual
            num = Number(0, 2)
            disp = Dgate(0, alpha=alpha_var)
            norm = 1 / disp.ansatz.batch_size if disp.ansatz.batch_shape else 1
            return -math.real(norm * math.sum(num >> disp >> vac) ** 2)

        opt = Optimizer(euclidean_lr=0.05)
        (optimized_alpha_var,) = opt.minimize(cost_fn, by_optimizing=[alpha_var], max_steps=100)
        assert math.all(alpha_var.value != optimized_alpha_var.value)
        assert optimized_alpha_var.value.shape == batch_shape

    def test_hong_ou_mandel_optimizer(self):
        """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip
        This generalizes the single photon Hong-Ou-Mandel effect to the many photon setting
        see Eq. 20 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.043065
        which lacks a square root in the right hand side.
        """
        r = np.arcsinh(1.0)
        i = settings.get_rng().integers(1, 5)
        k = settings.get_rng().integers(1, 5)
        cutoff = 1 + i + k

        tmsq_phi = Variable(value=0, name="phi", dtype=math.float64)
        bs_theta = Variable(
            value=np.arccos(np.sqrt(k / (i + k))) + 0.1 * settings.get_rng().normal(),
            name="theta",
            dtype=math.float64,
        )
        bs_phi = Variable(value=settings.get_rng().normal(), name="phi", dtype=math.float64)

        def cost_fn(tmsq_phi, bs_theta, bs_phi):
            state = TwoModeSqueezedVacuum((0, 1), r=r, phi=tmsq_phi)
            bs = BSgate(
                (1, 2),
                theta=bs_theta,
                phi=bs_phi,
            )
            return (
                math.abs(
                    (state >> state.on((2, 3)) >> bs).fock_array((cutoff,) * 4)[i, 1, i + k - 1, k]
                )
                ** 2
            )

        opt = Optimizer(euclidean_lr=0.01)
        (tmsq_phi, bs_theta, bs_phi) = opt.minimize(
            cost_fn,
            by_optimizing=[tmsq_phi, bs_theta, bs_phi],
            max_steps=300,
        )
        assert math.allclose(math.cos(bs_theta.value) ** 2, k / (i + k), atol=1e-2)

    def test_learning_four_mode_Interferometer(self):
        """Finding the optimal Interferometer to make a NOON state with N=2"""
        solution_U = np.array(
            [
                [
                    -0.47541806 + 0.00045878j,
                    -0.41513474 - 0.27218387j,
                    -0.11065812 - 0.39556922j,
                    -0.29912017 + 0.51900235j,
                ],
                [
                    -0.05246398 + 0.5209089j,
                    -0.29650069 - 0.40653082j,
                    0.57434638 - 0.04417284j,
                    0.28230532 - 0.24738672j,
                ],
                [
                    0.28437557 + 0.08773767j,
                    0.18377764 - 0.66496587j,
                    -0.5874942 - 0.19866946j,
                    0.2010813 - 0.10210844j,
                ],
                [
                    -0.63173183 - 0.11057324j,
                    -0.03468292 + 0.15245454j,
                    -0.25390362 - 0.2244298j,
                    0.18706333 - 0.64375049j,
                ],
            ],
        )
        perturbed = (
            Interferometer((0, 1, 2, 3), unitary=solution_U)
            >> BSgate((0, 1), settings.get_rng().normal(scale=0.01))
            >> BSgate((2, 3), settings.get_rng().normal(scale=0.01))
            >> BSgate((1, 2), settings.get_rng().normal(scale=0.01))
            >> BSgate((0, 3), settings.get_rng().normal(scale=0.01))
        )
        X = perturbed.symplectic
        perturbed_U = X[:4, :4] + 1j * X[4:, :4]

        s_gate_r = Variable(
            value=settings.get_rng().normal(loc=np.arcsinh(1.0), scale=0.01),
            name="r",
            dtype=math.float64,
        )
        int_unitary = Variable(value=perturbed_U, update_fn="update_unitary")

        def cost_fn(s_gate_r, int_unitary):
            state_in = Vacuum((0, 1, 2, 3))
            s_gate = Sgate(0, r=s_gate_r)
            interferometer = Interferometer((0, 1, 2, 3), unitary=int_unitary)
            amps = (
                state_in >> s_gate >> s_gate.on(1) >> s_gate.on(2) >> s_gate.on(3) >> interferometer
            ).fock_array((3, 3, 3, 3))
            return -(math.abs((amps[1, 1, 2, 0] + amps[1, 1, 0, 2]) / np.sqrt(2)) ** 2)

        opt = Optimizer(unitary_lr=0.05)
        (s_gate_r, int_unitary) = opt.minimize(
            cost_fn, by_optimizing=[s_gate_r, int_unitary], max_steps=200
        )
        assert math.allclose(-cost_fn(s_gate_r, int_unitary), 0.0625, atol=1e-5)

    def test_learning_four_mode_RealInterferometer(self):
        """Finding the optimal Interferometer to make a NOON state with N=2"""
        solution_O = np.array(
            [
                [0.5, -0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5, -0.5],
            ],
        )
        pertubed = (
            RealInterferometer((0, 1, 2, 3), orthogonal=solution_O)
            >> BSgate((0, 1), settings.get_rng().normal(scale=0.01))
            >> BSgate((2, 3), settings.get_rng().normal(scale=0.01))
            >> BSgate((1, 2), settings.get_rng().normal(scale=0.01))
            >> BSgate((0, 3), settings.get_rng().normal(scale=0.01))
        )
        perturbed_O = pertubed.symplectic[:4, :4]

        s_gate0_r = Variable(
            value=np.arcsinh(1.0) + settings.get_rng().normal(scale=0.01),
            name="r",
            dtype=math.float64,
        )
        s_gate0_phi = Variable(
            value=settings.get_rng().normal(scale=0.01), name="phi", dtype=math.float64
        )
        s_gate1_r = Variable(
            value=np.arcsinh(1.0) + settings.get_rng().normal(scale=0.01),
            name="r",
            dtype=math.float64,
        )
        s_gate1_phi = Variable(
            value=(np.pi / 2) + settings.get_rng().normal(scale=0.01),
            name="phi",
            dtype=math.float64,
        )
        s_gate2_r = Variable(
            value=np.arcsinh(1.0) + settings.get_rng().normal(scale=0.01),
            name="r",
            dtype=math.float64,
        )
        s_gate2_phi = Variable(
            value=-np.pi + settings.get_rng().normal(scale=0.01), name="phi", dtype=math.float64
        )
        s_gate3_r = Variable(
            value=np.arcsinh(1.0) + settings.get_rng().normal(scale=0.01),
            name="r",
            dtype=math.float64,
        )
        s_gate3_phi = Variable(
            value=(-np.pi / 2) + settings.get_rng().normal(scale=0.01),
            name="phi",
            dtype=math.float64,
        )
        perturbed_O = Variable(value=perturbed_O, update_fn="update_orthogonal")
        opt_params = (
            s_gate0_r,
            s_gate0_phi,
            s_gate1_r,
            s_gate1_phi,
            s_gate2_r,
            s_gate2_phi,
            s_gate3_r,
            s_gate3_phi,
            perturbed_O,
        )

        def cost_fn(
            s_gate0_r,
            s_gate0_phi,
            s_gate1_r,
            s_gate1_phi,
            s_gate2_r,
            s_gate2_phi,
            s_gate3_r,
            s_gate3_phi,
            perturbed_O,
        ):
            state_in = Vacuum((0, 1, 2, 3))
            s_gate0 = Sgate(0, r=s_gate0_r, phi=s_gate0_phi)
            s_gate1 = Sgate(1, r=s_gate1_r, phi=s_gate1_phi)
            s_gate2 = Sgate(2, r=s_gate2_r, phi=s_gate2_phi)
            s_gate3 = Sgate(3, r=s_gate3_r, phi=s_gate3_phi)
            r_inter = RealInterferometer((0, 1, 2, 3), orthogonal=perturbed_O)
            amps = (state_in >> s_gate0 >> s_gate1 >> s_gate2 >> s_gate3 >> r_inter).fock_array(
                (2, 2, 3, 3)
            )
            return -(math.abs((amps[1, 1, 0, 2] + amps[1, 1, 2, 0]) / np.sqrt(2)) ** 2)

        opt = Optimizer()
        opt_params = opt.minimize(cost_fn, by_optimizing=opt_params, max_steps=200)
        assert math.allclose(-cost_fn(*opt_params), 0.0625, atol=1e-5)

    def test_learning_two_mode_Ggate(self):
        """Finding the optimal Ggate to make a pair of single photons"""
        symplectic_var = Variable.symplectic(N=2)

        def cost_fn(symplectic_var):
            G = GaussianKet((0, 1), symplectic=symplectic_var)
            amps = G.fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

        (symplectic_var,) = opt.minimize(cost_fn, by_optimizing=[symplectic_var], max_steps=500)
        assert math.allclose(-cost_fn(symplectic_var), 0.25, atol=1e-4)

    def test_learning_two_mode_Interferometer(self):
        """Finding the optimal Interferometer to make a pair of single photons"""
        s_gate_r = Variable(settings.get_rng().uniform(0.1, 1.0), name="r", dtype=math.float64)
        s_gate_phi = Variable(
            settings.get_rng().uniform(0, 2 * np.pi), name="phi", dtype=math.float64
        )
        int_unitary = Variable.unitary(N=2)

        def cost_fn(s_gate_r, s_gate_phi, int_unitary):
            state_in = Vacuum((0, 1))
            s_gate = Sgate(
                0,
                r=s_gate_r,
                phi=s_gate_phi,
            )
            interferometer = Interferometer((0, 1), unitary=int_unitary)
            amps = (state_in >> s_gate >> s_gate.on(1) >> interferometer).fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(unitary_lr=0.1, euclidean_lr=0.001)

        (s_gate_r, s_gate_phi, int_unitary) = opt.minimize(
            cost_fn, by_optimizing=[s_gate_r, s_gate_phi, int_unitary], max_steps=1000
        )
        assert math.allclose(-cost_fn(s_gate_r, s_gate_phi, int_unitary), 0.25, atol=1e-5)

    def test_learning_two_mode_RealInterferometer(self):
        """Finding the optimal Interferometer to make a pair of single photons"""
        s_gate0_r = Variable(settings.get_rng().uniform(0.1, 1.0), name="r", dtype=math.float64)
        s_gate0_phi = Variable(
            settings.get_rng().uniform(0, 2 * np.pi), name="phi", dtype=math.float64
        )
        s_gate1_r = Variable(settings.get_rng().uniform(0.1, 1.0), name="r", dtype=math.float64)
        s_gate1_phi = Variable(
            settings.get_rng().uniform(0, 2 * np.pi), name="phi", dtype=math.float64
        )
        inter_orth = Variable.orthogonal(N=2)
        opt_params = (
            s_gate0_r,
            s_gate0_phi,
            s_gate1_r,
            s_gate1_phi,
            inter_orth,
        )

        def cost_fn(s_gate0_r, s_gate0_phi, s_gate1_r, s_gate1_phi, inter_orth):
            state_in = Vacuum((0, 1))
            s_gate0 = Sgate(0, r=s_gate0_r, phi=s_gate0_phi)
            s_gate1 = Sgate(1, r=s_gate1_r, phi=s_gate1_phi)
            r_inter = RealInterferometer((0, 1), orthogonal=inter_orth)

            amps = (state_in >> s_gate0 >> s_gate1 >> r_inter).fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(orthogonal_lr=0.1, euclidean_lr=0.001)

        opt_params = opt.minimize(cost_fn, by_optimizing=opt_params, max_steps=1000)
        assert math.allclose(-cost_fn(*opt_params), 0.25, atol=1e-5)

    def test_learning_two_mode_squeezing(self):
        """Finding the optimal beamsplitter transmission to make a pair of single photons"""
        s_gate_r = Variable(settings.get_rng().uniform(0.1, 1.0), name="r", dtype=math.float64)
        s_gate_phi = Variable(
            settings.get_rng().uniform(0, 2 * np.pi), name="phi", dtype=math.float64
        )

        bs_theta = Variable(
            settings.get_rng().uniform(0, 2 * np.pi), name="theta", dtype=math.float64
        )
        bs_phi = Variable(settings.get_rng().uniform(0, 2 * np.pi), name="phi", dtype=math.float64)

        def cost_fn(s_gate_r, s_gate_phi, bs_theta, bs_phi):
            state_in = Vacuum((0, 1))
            s_gate = Sgate(
                0,
                r=s_gate_r,
                phi=s_gate_phi,
            )
            bs_gate = BSgate(
                (0, 1),
                theta=bs_theta,
                phi=bs_phi,
            )
            amps = (state_in >> s_gate >> s_gate.on(1) >> bs_gate).fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(euclidean_lr=0.05)

        (s_gate_r, s_gate_phi, bs_theta, bs_phi) = opt.minimize(
            cost_fn, by_optimizing=[s_gate_r, s_gate_phi, bs_theta, bs_phi], max_steps=300
        )
        assert math.allclose(-cost_fn(s_gate_r, s_gate_phi, bs_theta, bs_phi), 0.25, atol=1e-5)

    def test_making_thermal_state_as_one_half_two_mode_squeezed_vacuum(self):
        """Optimizes a Ggate on two modes to match a target two-mode squeezed covariance.

        The target state has mean photon number nbar=1.4 per mode and corresponds
        to a two-mode squeezed vacuum with specific squeezing parameter.
        """
        nbar = 1.4
        S_init = two_mode_squeezing(np.arcsinh(1.0), 0.0)
        target_cov = two_mode_squeezing(2 * np.arcsinh(np.sqrt(nbar)), 0.0)
        # phase_space(s=0) returns cov = S @ S.T * hbar/2, so normalize target
        target_cov_normalized = target_cov * settings.HBAR / 2

        symplectic_var = Variable(value=S_init, update_fn="update_symplectic")

        def cost_fn(symplectic_var):
            G = Ggate((0, 1), symplectic=symplectic_var)
            state = Vacuum((0, 1)) >> G
            cov, _, _ = state.phase_space(s=0)
            diff = cov - target_cov_normalized
            return math.sum(math.real(diff) ** 2)

        opt = Optimizer(symplectic_lr=0.2)
        (symplectic_var,) = opt.minimize(cost_fn, by_optimizing=[symplectic_var], max_steps=200)
        S = math.asnumpy(symplectic_var.value)
        cov = S @ S.T
        assert math.allclose(cov, target_cov, atol=1e-3)

    def test_parameter_passthrough(self):
        r = np.arcsinh(1.0)
        r_var = Variable(r, "r")
        phi_var = Variable(settings.get_rng().normal(), "phi")
        s2_gate0_phi = Variable(0, "phi", dtype=math.float64)
        s2_gate1_phi = Variable(0, "phi", dtype=math.float64)

        def cost_fn(s2_gate0_phi, s2_gate1_phi, r_var, phi_var):
            state_in = Vacuum((0, 1, 2, 3))
            s2_gate0 = S2gate((0, 1), r=r, phi=s2_gate0_phi)
            s2_gate1 = S2gate((2, 3), r=r, phi=s2_gate1_phi)
            s2_gate2 = S2gate((1, 2), r=r_var, phi=phi_var)

            amps = (state_in >> s2_gate0 >> s2_gate1 >> s2_gate2).fock_array((2, 2, 2, 2))
            return math.abs(amps[1, 1, 1, 1]) ** 2

        opt = Optimizer(euclidean_lr=0.001)
        (s2_gate0_phi, s2_gate1_phi, r_var, phi_var) = opt.minimize(
            cost_fn, by_optimizing=[s2_gate0_phi, s2_gate1_phi, r_var, phi_var], max_steps=300
        )
        assert math.allclose(math.sinh(r_var.value) ** 2, 1, atol=1e-2)

    def test_reuse_optimizer(self):
        """Test that the same optimizer instance can be reused."""
        sgate_r = Variable(0.2, "r")
        sgate_phi = Variable(0.1, "phi")

        def cost_fn(sgate_r, sgate_phi):
            sgate = Sgate(0, r=sgate_r, phi=sgate_phi)
            target_state = SqueezedVacuum(0, r=0.1, phi=0.2).fock_array((40,))
            state_out = Vacuum(0) >> sgate
            return -(math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2)

        opt = Optimizer()
        (sgate_r, sgate_phi) = opt.minimize(cost_fn, by_optimizing=[sgate_r, sgate_phi])

        assert math.allclose(sgate_r.value, 0.1, atol=0.01)
        assert math.allclose(sgate_phi.value, 0.2, atol=0.01)

        sgate_r_reused = Variable(0.2, "r")
        sgate_phi_reused = Variable(0.1, "phi")

        (sgate_r_reused, sgate_phi_reused) = opt.minimize(
            cost_fn, by_optimizing=[sgate_r_reused, sgate_phi_reused]
        )

        assert math.allclose(sgate_r_reused.value, sgate_r.value)
        assert math.allclose(sgate_phi_reused.value, sgate_phi.value)

    @given(n=st.integers(0, 3))
    def test_S2gate_coincidence_prob(self, n):
        """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
        S_r = Variable(abs(settings.get_rng().normal(loc=1.0, scale=0.1)), "r", dtype=math.float64)

        def cost_fn(S_r):
            S = TwoModeSqueezedVacuum(
                (0, 1),
                r=S_r,
            )
            return -(math.abs(S.fock_array((n + 1, n + 1))[n, n]) ** 2)

        opt = Optimizer(euclidean_lr=0.01)
        (S_r,) = opt.minimize(cost_fn, by_optimizing=[S_r], max_steps=300)

        expected = 1 / (n + 1) * (n / (n + 1)) ** n
        assert math.allclose(-cost_fn(S_r), expected, atol=1e-5)

    def test_sgate_optimization(self):
        """Test that Sgate is optimized correctly."""
        sgate_r = Variable(0.2, "r", dtype=math.float64)
        sgate_phi = Variable(0.1, "phi", dtype=math.float64)

        def cost_fn(sgate_r, sgate_phi):
            sgate = Sgate(0, r=sgate_r, phi=sgate_phi)
            target_state = SqueezedVacuum(0, r=0.1, phi=0.2).fock_array((40,))
            state_out = Vacuum(0) >> sgate
            return -(math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2)

        opt = Optimizer()
        (sgate_r, sgate_phi) = opt.minimize(cost_fn, by_optimizing=[sgate_r, sgate_phi])

        assert math.allclose(sgate_r.value, 0.1, atol=0.01)
        assert math.allclose(sgate_phi.value, 0.2, atol=0.01)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_squeezing_grad_from_fock(self, batch_shape):
        """Test that the gradient of a squeezing gate is computed from the fock representation."""
        squeezing_r = Variable(math.ones(batch_shape), "r", dtype=math.float64)

        def cost_fn(squeezing_r):
            squeezing = Sgate(0, r=squeezing_r)
            num = Number(0, 2)
            vac = Vacuum(0).dual
            norm = 1 / squeezing.ansatz.batch_size if squeezing.ansatz.batch_shape else 1
            return -math.real(norm * math.sum(num >> squeezing >> vac) ** 2)

        opt = Optimizer(euclidean_lr=0.05)
        (squeezing_r_optimized,) = opt.minimize(cost_fn, by_optimizing=[squeezing_r], max_steps=100)

        assert math.all(squeezing_r_optimized.value != squeezing_r.value)

    def test_squeezing_hong_ou_mandel_optimizer(self):
        """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
        see https://www.pnas.org/content/117/52/33107/tab-article-info
        """
        r = np.arcsinh(1.0)

        S_01_phi = Variable(0.0, "phi", dtype=math.float64)
        S_23_phi = Variable(0.0, "phi", dtype=math.float64)
        S_12_r = Variable(1.0, "r", dtype=math.float64)
        S_12_phi = Variable(settings.get_rng().normal(), "phi", dtype=math.float64)

        def cost_fn(S_01_phi, S_23_phi, S_12_r, S_12_phi):
            state_in = Vacuum((0, 1, 2, 3))
            S_01 = S2gate((0, 1), r=r, phi=S_01_phi)
            S_23 = S2gate((2, 3), r=r, phi=S_23_phi)
            S_12 = S2gate(
                (1, 2),
                r=S_12_r,
                phi=S_12_phi,
            )

            amps = (state_in >> S_01 >> S_23 >> S_12).fock_array((2, 2, 2, 2))
            return math.abs(amps[1, 1, 1, 1]) ** 2

        opt = Optimizer(euclidean_lr=0.001)
        (S_01_phi, S_23_phi, S_12_r, S_12_phi) = opt.minimize(
            cost_fn, by_optimizing=[S_01_phi, S_23_phi, S_12_r, S_12_phi], max_steps=300
        )
        assert math.allclose(math.sinh(S_12_r.value) ** 2, 1, atol=1e-2)
