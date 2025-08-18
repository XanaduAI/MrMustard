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
    Circuit,
    Coherent,
    Dgate,
    DisplacedSqueezed,
    Ggate,
    GKet,
    Interferometer,
    Number,
    RealInterferometer,
    S2gate,
    Sgate,
    SqueezedVacuum,
    TwoModeSqueezedVacuum,
    Vacuum,
)
from mrmustard.math.parameters import Variable
from mrmustard.physics.gaussian import number_means, von_neumann_entropy
from mrmustard.training import Optimizer


@pytest.mark.requires_backend("jax")
class TestOptimizer:
    r"""
    Tests for the ``Optimizer`` class.
    """

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_bsgate_grad_from_fock(self, batch_shape):
        """Test that the gradient of a beamsplitter gate is computed from the fock representation."""
        r_var = Variable(math.ones(batch_shape), "r")
        og_r = math.asnumpy(r_var.value)
        num = Number(1, 1)
        vac = Vacuum(0)
        bs = BSgate((0, 1), 0.5)

        def cost_fn(r):
            sq = SqueezedVacuum(0, r=r)
            norm = 1 / sq.ansatz.batch_size if sq.ansatz.batch_shape else 1
            return -math.real(
                norm * math.sum(sq >> num >> bs >> (vac >> num).dual) ** 2,
            )

        opt = Optimizer(euclidean_lr=0.05)
        (r_var,) = opt.minimize(cost_fn, by_optimizing=[r_var], max_steps=100)

        assert math.all(og_r != r_var.value)

    def test_bsgate_optimization(self):
        """Test that BSgate is optimized correctly."""
        theta_var = Variable(0.05, "theta")
        phi_var = Variable(0.1, "phi")
        target_gate = BSgate((0, 1), 0.1, 0.2).fock_array(40)

        def cost_fn(theta, phi):
            bsgate = BSgate((0, 1), theta=theta, phi=phi)
            return -(math.abs(math.sum(math.conj(bsgate.fock_array(40)) * target_gate)) ** 2)

        opt = Optimizer()
        (theta_var, phi_var) = opt.minimize(cost_fn, by_optimizing=[theta_var, phi_var])

        assert math.allclose(theta_var.value, 0.1, atol=0.01)
        assert math.allclose(phi_var.value, 0.2, atol=0.01)

    def test_cat_state_optimization(self):
        # Note: we need to intitialize the cat state with a non-zero value. This is because
        # the gradients are zero when x is zero.
        alpha1_var = Variable(0.1 + 0.0j, "alpha1")
        alpha2_var = Variable(-0.1 + 0.0j, "alpha2")
        expected_alpha = np.sqrt(np.pi)
        expected_cat = Coherent(0, alpha=expected_alpha) + Coherent(0, alpha=-expected_alpha)

        def cost_fn(alpha1, alpha2):
            cat_state = Coherent(0, alpha=alpha1) + Coherent(0, alpha=alpha2)
            cat_state = cat_state.normalize()
            return -math.abs(cat_state.fidelity(expected_cat.normalize()))

        # stable_threshold and max_steps are set to whatever gives us optimized parameters
        # that are within the default ATOL=1e-8 of the expected values
        opt = Optimizer(stable_threshold=1e-12)
        (alpha1_var, alpha2_var) = opt.minimize(
            cost_fn, by_optimizing=[alpha1_var, alpha2_var], max_steps=6000
        )

        assert math.allclose(alpha1_var.value, expected_alpha, atol=1e-6)
        assert math.allclose(alpha2_var.value, -expected_alpha, atol=1e-6)

    @pytest.mark.parametrize("alpha", [0.2 + 0.4j, -0.1 - 0.2j, 0.1j, 0.4])
    def test_complex_dgate_optimization_bargmann(self, alpha):
        alpha_var = Variable(0.0 + 0.0j, "alpha")

        def cost_fn(alpha_opt):
            dgate = Dgate(0, alpha=alpha_opt)
            target_state = Coherent(0, alpha=alpha)
            state_out = Vacuum(0) >> dgate
            return 1 - math.real(state_out.expectation(target_state))

        opt = Optimizer(euclidean_lr=0.05)
        (alpha_var,) = opt.minimize(cost_fn, by_optimizing=[alpha_var], max_steps=200)
        assert math.allclose(alpha_var.value, alpha, atol=0.01)

    @pytest.mark.parametrize("alpha", [0.2 + 0.4j, -0.1 - 0.2j, 0.1j, 0.4])
    def test_complex_dgate_optimization_fock(self, alpha):
        alpha_var = Variable(alpha, "alpha")

        def cost_fn(alpha_opt):
            dgate = Dgate(0, alpha=alpha_opt)
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
        alpha_var = Variable(0.0 + 0.0j, "alpha")
        target_state = DisplacedSqueezed(0, r=0.0, alpha=0.1 + 0.2j)

        def cost_fn(alpha):
            dgate = Dgate(0, alpha=alpha)
            state_out = Vacuum(0) >> dgate
            return -math.real(state_out.expectation(target_state))

        opt = Optimizer()
        (alpha_var,) = opt.minimize(cost_fn, by_optimizing=[alpha_var])

        assert math.allclose(alpha_var.value, 0.1 + 0.2j, atol=0.01)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_displacement_grad_from_fock(self, batch_shape):
        """Test that the gradient of a displacement gate is computed from the fock representation."""
        alpha_var = Variable(math.ones(batch_shape) + 0.5, "alpha")
        og_alpha = math.asnumpy(alpha_var.value)
        num = Number(0, 2)
        vac = Vacuum(0).dual

        def cost_fn(alpha):
            disp = Dgate(0, alpha=alpha)
            norm = 1 / disp.ansatz.batch_size if disp.ansatz.batch_shape else 1
            return -math.real(norm * math.sum(num >> disp >> vac) ** 2)

        opt = Optimizer(euclidean_lr=0.05)
        (alpha_var,) = opt.minimize(cost_fn, by_optimizing=[alpha_var], max_steps=100)
        assert math.all(og_alpha != alpha_var.value)

    @given(i=st.integers(1, 5), k=st.integers(1, 5))
    def test_hong_ou_mandel_optimizer(self, i, k):
        """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip
        This generalizes the single photon Hong-Ou-Mandel effect to the many photon setting
        see Eq. 20 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.043065
        which lacks a square root in the right hand side.
        """
        r = np.arcsinh(1.0)
        cutoff = 1 + i + k

        phi_var = Variable(0.0, "phi")
        theta_var = Variable(np.arccos(np.sqrt(k / (i + k))) + 0.1 * settings.rng.normal(), "theta")
        bs_phi_var = Variable(settings.rng.normal(), "bs_phi")

        def cost_fn(phi, theta, bs_phi):
            state1 = TwoModeSqueezedVacuum((0, 1), r=r, phi=phi)
            state2 = TwoModeSqueezedVacuum((2, 3), r=r, phi=phi)
            bs = BSgate((1, 2), theta=theta, phi=bs_phi)
            circ = Circuit([state1, state2, bs])
            return math.abs(circ.contract().fock_array((cutoff,) * 4)[i, 1, i + k - 1, k]) ** 2

        opt = Optimizer(euclidean_lr=0.01)
        (phi_var, theta_var, bs_phi_var) = opt.minimize(
            cost_fn,
            by_optimizing=[phi_var, theta_var, bs_phi_var],
            max_steps=300,
        )
        assert math.allclose(math.cos(theta_var.value) ** 2, k / (i + k), atol=1e-2)

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
            >> BSgate((0, 1), settings.rng.normal(scale=0.01))
            >> BSgate((2, 3), settings.rng.normal(scale=0.01))
            >> BSgate((1, 2), settings.rng.normal(scale=0.01))
            >> BSgate((0, 3), settings.rng.normal(scale=0.01))
        )
        X = perturbed.symplectic
        perturbed_U = X[:4, :4] + 1j * X[4:, :4]

        r_var = Variable(settings.rng.normal(loc=np.arcsinh(1.0), scale=0.01), "r")
        unitary_var = Variable.unitary(perturbed_U, "unitary")

        def cost_fn(r, unitary):
            state_in = Vacuum((0, 1, 2, 3))
            s_gate0 = Sgate(0, r=r)
            s_gate1 = Sgate(1, r=r)
            s_gate2 = Sgate(2, r=r)
            s_gate3 = Sgate(3, r=r)
            interferometer = Interferometer((0, 1, 2, 3), unitary=unitary)
            circ = Circuit([state_in, s_gate0, s_gate1, s_gate2, s_gate3, interferometer])
            amps = circ.contract().fock_array((3, 3, 3, 3))
            return -(math.abs((amps[1, 1, 2, 0] + amps[1, 1, 0, 2]) / np.sqrt(2)) ** 2)

        opt = Optimizer(unitary_lr=0.05)
        (r_var, unitary_var) = opt.minimize(
            cost_fn, by_optimizing=[r_var, unitary_var], max_steps=200
        )
        assert math.allclose(-cost_fn(r_var.value, unitary_var.value), 0.0625, atol=1e-5)

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
            >> BSgate((0, 1), settings.rng.normal(scale=0.01))
            >> BSgate((2, 3), settings.rng.normal(scale=0.01))
            >> BSgate((1, 2), settings.rng.normal(scale=0.01))
            >> BSgate((0, 3), settings.rng.normal(scale=0.01))
        )
        perturbed_O = pertubed.symplectic[:4, :4]

        r0_var = Variable(np.arcsinh(1.0) + settings.rng.normal(scale=0.01), "r0")
        phi0_var = Variable(settings.rng.normal(scale=0.01), "phi0")
        r1_var = Variable(np.arcsinh(1.0) + settings.rng.normal(scale=0.01), "r1")
        phi1_var = Variable((np.pi / 2) + settings.rng.normal(scale=0.01), "phi1")
        r2_var = Variable(np.arcsinh(1.0) + settings.rng.normal(scale=0.01), "r2")
        phi2_var = Variable(-np.pi + settings.rng.normal(scale=0.01), "phi2")
        r3_var = Variable(np.arcsinh(1.0) + settings.rng.normal(scale=0.01), "r3")
        phi3_var = Variable((-np.pi / 2) + settings.rng.normal(scale=0.01), "phi3")
        orthogonal_var = Variable.orthogonal(perturbed_O, "orthogonal")

        def cost_fn(r0, phi0, r1, phi1, r2, phi2, r3, phi3, orthogonal):
            state_in = Vacuum((0, 1, 2, 3))
            s_gate0 = Sgate(0, r=r0, phi=phi0)
            s_gate1 = Sgate(1, r=r1, phi=phi1)
            s_gate2 = Sgate(2, r=r2, phi=phi2)
            s_gate3 = Sgate(3, r=r3, phi=phi3)
            r_inter = RealInterferometer((0, 1, 2, 3), orthogonal=orthogonal)
            circ = Circuit([state_in, s_gate0, s_gate1, s_gate2, s_gate3, r_inter])
            amps = circ.contract().fock_array((2, 2, 3, 3))
            return -(math.abs((amps[1, 1, 0, 2] + amps[1, 1, 2, 0]) / np.sqrt(2)) ** 2)

        opt = Optimizer()

        (r0_var, phi0_var, r1_var, phi1_var, r2_var, phi2_var, r3_var, phi3_var, orthogonal_var) = (
            opt.minimize(
                cost_fn,
                by_optimizing=[
                    r0_var,
                    phi0_var,
                    r1_var,
                    phi1_var,
                    r2_var,
                    phi2_var,
                    r3_var,
                    phi3_var,
                    orthogonal_var,
                ],
                max_steps=200,
            )
        )
        assert math.allclose(
            -cost_fn(
                r0_var.value,
                phi0_var.value,
                r1_var.value,
                phi1_var.value,
                r2_var.value,
                phi2_var.value,
                r3_var.value,
                phi3_var.value,
                orthogonal_var.value,
            ),
            0.0625,
            atol=1e-5,
        )

    def test_learning_two_mode_Ggate(self):
        """Finding the optimal Ggate to make a pair of single photons"""
        symplectic_var = Variable.symplectic(math.random_symplectic(2), "symplectic")

        def cost_fn(symplectic):
            G = GKet((0, 1), symplectic=symplectic)
            amps = G.fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(symplectic_lr=0.5, euclidean_lr=0.01)

        (symplectic_var,) = opt.minimize(cost_fn, by_optimizing=[symplectic_var], max_steps=500)
        assert math.allclose(-cost_fn(symplectic_var.value), 0.25, atol=1e-4)

    def test_learning_two_mode_Interferometer(self):
        """Finding the optimal Interferometer to make a pair of single photons"""
        r_var = Variable(settings.rng.normal() ** 2, "r")
        phi_var = Variable(settings.rng.normal(), "phi")
        unitary_var = Variable.unitary(math.random_unitary(2), "unitary")

        def cost_fn(r, phi, unitary):
            state_in = Vacuum((0, 1))
            s_gate0 = Sgate(0, r=r, phi=phi)
            s_gate1 = Sgate(1, r=r, phi=phi)
            interferometer = Interferometer((0, 1), unitary=unitary)
            circ = Circuit([state_in, s_gate0, s_gate1, interferometer])
            amps = circ.contract().fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(unitary_lr=0.5, euclidean_lr=0.01)

        (r_var, phi_var, unitary_var) = opt.minimize(
            cost_fn, by_optimizing=[r_var, phi_var, unitary_var], max_steps=1000
        )
        assert math.allclose(
            -cost_fn(r_var.value, phi_var.value, unitary_var.value), 0.25, atol=1e-5
        )

    def test_learning_two_mode_RealInterferometer(self):
        """Finding the optimal Interferometer to make a pair of single photons"""
        r0_var = Variable(settings.rng.normal() ** 2, "r0")
        phi0_var = Variable(settings.rng.normal(), "phi0")
        r1_var = Variable(settings.rng.normal() ** 2, "r1")
        phi1_var = Variable(settings.rng.normal(), "phi1")
        orthogonal_var = Variable.orthogonal(math.random_orthogonal(2), "orthogonal")

        def cost_fn(r0, phi0, r1, phi1, orthogonal):
            state_in = Vacuum((0, 1))
            s_gate0 = Sgate(0, r=r0, phi=phi0)
            s_gate1 = Sgate(1, r=r1, phi=phi1)
            r_inter = RealInterferometer((0, 1), orthogonal=orthogonal)
            circ = Circuit([state_in, s_gate0, s_gate1, r_inter])
            amps = circ.contract().fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(orthogonal_lr=0.5, euclidean_lr=0.01)

        (r0_var, phi0_var, r1_var, phi1_var, orthogonal_var) = opt.minimize(
            cost_fn,
            by_optimizing=[r0_var, phi0_var, r1_var, phi1_var, orthogonal_var],
            max_steps=1000,
        )
        assert math.allclose(
            -cost_fn(
                r0_var.value, phi0_var.value, r1_var.value, phi1_var.value, orthogonal_var.value
            ),
            0.25,
            atol=1e-5,
        )

    def test_learning_two_mode_squeezing(self):
        """Finding the optimal beamsplitter transmission to make a pair of single photons"""
        r_var = Variable(abs(settings.rng.normal()), "r")
        phi_var = Variable(settings.rng.normal(), "phi")
        theta_var = Variable(settings.rng.normal(), "theta")
        bs_phi_var = Variable(settings.rng.normal(), "bs_phi")

        def cost_fn(r, phi, theta, bs_phi):
            state_in = Vacuum((0, 1))
            s_gate0 = Sgate(0, r=r, phi=phi)
            s_gate1 = Sgate(1, r=r, phi=phi)
            bs_gate = BSgate((0, 1), theta=theta, phi=bs_phi)
            circ = Circuit([state_in, s_gate0, s_gate1, bs_gate])
            amps = circ.contract().fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = Optimizer(euclidean_lr=0.05)

        (r_var, phi_var, theta_var, bs_phi_var) = opt.minimize(
            cost_fn, by_optimizing=[r_var, phi_var, theta_var, bs_phi_var], max_steps=300
        )
        assert math.allclose(
            -cost_fn(r_var.value, phi_var.value, theta_var.value, bs_phi_var.value), 0.25, atol=1e-5
        )

    def test_making_thermal_state_as_one_half_two_mode_squeezed_vacuum(self):
        """Optimizes a Ggate on two modes so as to prepare a state with the same entropy
        and mean photon number as a thermal state"""

        def thermal_entropy(nbar):
            return -(nbar * np.log((nbar) / (1 + nbar)) - np.log(1 + nbar))

        nbar = 1.4
        S_init = two_mode_squeezing(np.arcsinh(1.0), 0.0)
        S = thermal_entropy(nbar)

        symplectic_var = Variable.symplectic(S_init, "symplectic")

        def cost_fn(symplectic):
            G = Ggate((0, 1), symplectic=symplectic)
            state = Vacuum((0, 1)) >> G

            state0 = state[0]
            state1 = state[1]

            cov0, mean0, _ = state0.phase_space(s=0)
            cov1, mean1, _ = state1.phase_space(s=0)

            num_mean0 = number_means(cov0, mean0)[0]
            num_mean1 = number_means(cov1, mean1)[0]

            entropy = von_neumann_entropy(cov0)
            return math.abs((num_mean0 - nbar) ** 2 + (entropy - S) ** 2 + (num_mean1 - nbar) ** 2)

        opt = Optimizer(symplectic_lr=0.1)
        (symplectic_var,) = opt.minimize(cost_fn, by_optimizing=[symplectic_var], max_steps=50)
        S_result = math.asnumpy(symplectic_var.value)
        cov = S_result @ S_result.T
        assert math.allclose(cov, two_mode_squeezing(2 * np.arcsinh(np.sqrt(nbar)), 0.0))

    def test_parameter_passthrough(self):
        """Same as the test above, but with param passthrough"""
        r = np.arcsinh(1.0)
        r_var = Variable(r, "r", (0.0, None))
        phi_var = Variable(settings.rng.normal(), "phi", (None, None))
        phi01_var = Variable(0.0, "phi01")
        phi23_var = Variable(0.0, "phi23")

        def cost_fn(phi01, phi23, r_opt, phi_opt):
            state_in = Vacuum((0, 1, 2, 3))
            s2_gate0 = S2gate((0, 1), r=r, phi=phi01)
            s2_gate1 = S2gate((2, 3), r=r, phi=phi23)
            s2_gate2 = S2gate((1, 2), r=r_opt, phi=phi_opt)
            circ = Circuit([state_in, s2_gate0, s2_gate1, s2_gate2])
            return math.abs(circ.contract().fock_array((2, 2, 2, 2))[1, 1, 1, 1]) ** 2

        opt = Optimizer(euclidean_lr=0.001)
        (phi01_var, phi23_var, r_var, phi_var) = opt.minimize(
            cost_fn, by_optimizing=[phi01_var, phi23_var, r_var, phi_var], max_steps=300
        )
        assert math.allclose(math.sinh(r_var.value) ** 2, 1, atol=1e-2)

    def test_reuse_optimizer(self):
        """Test that the same optimizer instance can be reused."""
        r_var = Variable(0.2, "r")
        phi_var = Variable(0.1, "phi")
        target_state = SqueezedVacuum(0, r=0.1, phi=0.2).fock_array((40,))

        def cost_fn(r, phi):
            sgate = Sgate(0, r=r, phi=phi)
            state_out = Vacuum(0) >> sgate
            return -(math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2)

        opt = Optimizer()
        (r_var, phi_var) = opt.minimize(cost_fn, by_optimizing=[r_var, phi_var])

        assert math.allclose(r_var.value, 0.1, atol=0.01)
        assert math.allclose(phi_var.value, 0.2, atol=0.01)

        r_var_reused = Variable(0.2, "r")
        phi_var_reused = Variable(0.1, "phi")
        (r_var_reused, phi_var_reused) = opt.minimize(
            cost_fn, by_optimizing=[r_var_reused, phi_var_reused]
        )

        assert math.allclose(r_var_reused.value, r_var.value)
        assert math.allclose(phi_var_reused.value, phi_var.value)

    @given(n=st.integers(0, 3))
    def test_S2gate_coincidence_prob(self, n):
        """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
        r_var = Variable(abs(settings.rng.normal(loc=1.0, scale=0.1)), "r")

        def cost_fn(r):
            S = TwoModeSqueezedVacuum((0, 1), r=r)
            return -(math.abs(S.fock_array((n + 1, n + 1))[n, n]) ** 2)

        opt = Optimizer(euclidean_lr=0.01)
        (r_var,) = opt.minimize(cost_fn, by_optimizing=[r_var], max_steps=300)

        expected = 1 / (n + 1) * (n / (n + 1)) ** n
        assert math.allclose(-cost_fn(r_var.value), expected, atol=1e-5)

    def test_sgate_optimization(self):
        """Test that Sgate is optimized correctly."""
        r_var = Variable(0.2, "r")
        phi_var = Variable(0.1, "phi")
        target_state = SqueezedVacuum(0, r=0.1, phi=0.2).fock_array((40,))

        def cost_fn(r, phi):
            sgate = Sgate(0, r=r, phi=phi)
            state_out = Vacuum(0) >> sgate
            return -(math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2)

        opt = Optimizer()
        (r_var, phi_var) = opt.minimize(cost_fn, by_optimizing=[r_var, phi_var])

        assert math.allclose(r_var.value, 0.1, atol=0.01)
        assert math.allclose(phi_var.value, 0.2, atol=0.01)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_squeezing_grad_from_fock(self, batch_shape):
        """Test that the gradient of a squeezing gate is computed from the fock representation."""
        r_var = Variable(math.ones(batch_shape), "r")
        og_r = math.asnumpy(r_var.value)
        num = Number(0, 2)
        vac = Vacuum(0).dual

        def cost_fn(r):
            squeezing = Sgate(0, r=r)
            norm = 1 / squeezing.ansatz.batch_size if squeezing.ansatz.batch_shape else 1
            return -math.real(norm * math.sum(num >> squeezing >> vac) ** 2)

        opt = Optimizer(euclidean_lr=0.05)
        (r_var,) = opt.minimize(cost_fn, by_optimizing=[r_var], max_steps=100)

        assert math.all(r_var.value != og_r)

    def test_squeezing_hong_ou_mandel_optimizer(self):
        """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
        see https://www.pnas.org/content/117/52/33107/tab-article-info
        """
        r = np.arcsinh(1.0)

        phi01_var = Variable(0.0, "phi01")
        phi23_var = Variable(0.0, "phi23")
        r12_var = Variable(1.0, "r12")
        phi12_var = Variable(settings.rng.normal(), "phi12")

        def cost_fn(phi01, phi23, r12, phi12):
            state_in = Vacuum((0, 1, 2, 3))
            S_01 = S2gate((0, 1), r=r, phi=phi01)
            S_23 = S2gate((2, 3), r=r, phi=phi23)
            S_12 = S2gate((1, 2), r=r12, phi=phi12)
            circ = Circuit([state_in, S_01, S_23, S_12])
            return math.abs(circ.contract().fock_array((2, 2, 2, 2))[1, 1, 1, 1]) ** 2

        opt = Optimizer(euclidean_lr=0.001)
        (phi01_var, phi23_var, r12_var, phi12_var) = opt.minimize(
            cost_fn, by_optimizing=[phi01_var, phi23_var, r12_var, phi12_var], max_steps=300
        )
        assert math.allclose(math.sinh(r12_var.value) ** 2, 1, atol=1e-2)
