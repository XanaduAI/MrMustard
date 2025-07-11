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

from mrmustard import math, settings
from mrmustard.lab import (
    BSgate,
    Circuit,
    Dgate,
    DisplacedSqueezed,
    Number,
    S2gate,
    Sgate,
    SqueezedVacuum,
    TwoModeSqueezedVacuum,
    Vacuum,
)
from mrmustard.math.parameters import Variable
from mrmustard.training import OptimizerJax


@pytest.mark.requires_backend("jax")
class TestOptimizerJax:
    r"""
    Tests for the ``OptimizerJax`` class.
    """

    @given(n=st.integers(0, 3))
    def test_S2gate_coincidence_prob(self, n):
        """Testing the optimal probability of obtaining |n,n> from a two mode squeezed vacuum"""
        S = TwoModeSqueezedVacuum(
            (0, 1),
            r=abs(settings.rng.normal(loc=1.0, scale=0.1)),
            r_trainable=True,
        )

        def cost_fn(S):
            return -(math.abs(S.fock_array((n + 1, n + 1))[n, n]) ** 2)

        opt = OptimizerJax(learning_rate=0.01)
        (S,) = opt.minimize(cost_fn, by_optimizing=[S], max_steps=300)

        expected = 1 / (n + 1) * (n / (n + 1)) ** n
        assert math.allclose(-cost_fn(S), expected, atol=1e-5)

    @given(i=st.integers(1, 5), k=st.integers(1, 5))
    def test_hong_ou_mandel_optimizer(self, i, k):
        """Finding the optimal beamsplitter transmission to get Hong-Ou-Mandel dip
        This generalizes the single photon Hong-Ou-Mandel effect to the many photon setting
        see Eq. 20 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.043065
        which lacks a square root in the right hand side.
        """
        r = np.arcsinh(1.0)
        cutoff = 1 + i + k

        state = TwoModeSqueezedVacuum((0, 1), r=r, phi_trainable=True)
        bs = BSgate(
            (1, 2),
            theta=np.arccos(np.sqrt(k / (i + k))) + 0.1 * settings.rng.normal(),
            phi=settings.rng.normal(),
            theta_trainable=True,
            phi_trainable=True,
        )
        circ = Circuit([state, state.on((2, 3)), bs])

        def cost_fn(circ):
            return math.abs(circ.contract().fock_array((cutoff,) * 4)[i, 1, i + k - 1, k]) ** 2

        opt = OptimizerJax(learning_rate=0.01)
        (circ,) = opt.minimize(
            cost_fn,
            by_optimizing=[circ],
            max_steps=300,
        )
        bs = circ.components[2]
        assert math.allclose(math.cos(bs.parameters.theta.value) ** 2, k / (i + k), atol=1e-2)

    def test_learning_two_mode_squeezing(self):
        """Finding the optimal beamsplitter transmission to make a pair of single photons"""
        state_in = Vacuum((0, 1))
        s_gate = Sgate(
            0,
            r=abs(settings.rng.normal()),
            phi=settings.rng.normal(),
            r_trainable=True,
            phi_trainable=True,
        )
        bs_gate = BSgate(
            (0, 1),
            theta=settings.rng.normal(),
            phi=settings.rng.normal(),
            theta_trainable=True,
            phi_trainable=True,
        )
        circ = Circuit([state_in, s_gate, s_gate.on(1), bs_gate])

        def cost_fn(circ):
            amps = circ.contract().fock_array((2, 2))
            return -(math.abs(amps[1, 1]) ** 2) + math.abs(amps[0, 1]) ** 2

        opt = OptimizerJax(learning_rate=0.05)

        (circ,) = opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
        assert math.allclose(-cost_fn(circ), 0.25, atol=1e-5)

    # def test_squeezing_hong_ou_mandel_optimizer(self):
    #     """Finding the optimal squeezing parameter to get Hong-Ou-Mandel dip in time
    #     see https://www.pnas.org/content/117/52/33107/tab-article-info
    #     """
    #     r = np.arcsinh(1.0)

    #     state_in = Vacuum((0, 1, 2, 3))
    #     S_01 = S2gate((0, 1), r=r, phi=0.0, phi_trainable=True)
    #     S_23 = S2gate((2, 3), r=r, phi=0.0, phi_trainable=True)
    #     S_12 = S2gate(
    #         (1, 2),
    #         r=1.0,
    #         phi=settings.rng.normal(),
    #         r_trainable=True,
    #         phi_trainable=True,
    #     )

    #     circ = Circuit([state_in, S_01, S_23, S_12])

    #     def cost_fn(circ):
    #         return math.abs(circ.contract().fock_array((2, 2, 2, 2))[1, 1, 1, 1]) ** 2

    #     opt = OptimizerJax(learning_rate=0.001)
    #     (circ,) = opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
    #     S_12 = circ.components[3]
    #     assert math.allclose(math.sinh(S_12.parameters.r.value) ** 2, 1, atol=1e-2)

    def test_parameter_passthrough(self):
        """Same as the test above, but with param passthrough"""
        r = np.arcsinh(1.0)
        r_var = Variable(r, "r", (0.0, None))
        phi_var = Variable(settings.rng.normal(), "phi", (None, None))

        state_in = Vacuum((0, 1, 2, 3))
        s2_gate0 = S2gate((0, 1), r=r, phi=0.0, phi_trainable=True)
        s2_gate1 = S2gate((2, 3), r=r, phi=0.0, phi_trainable=True)
        s2_gate2 = S2gate((1, 2), r=r_var, phi=phi_var)

        circ = Circuit([state_in, s2_gate0, s2_gate1, s2_gate2])

        def cost_fn(circ):
            return math.abs(circ.contract().fock_array((2, 2, 2, 2))[1, 1, 1, 1]) ** 2

        opt = OptimizerJax(learning_rate=0.001)
        (circ,) = opt.minimize(cost_fn, by_optimizing=[circ], max_steps=300)
        r_var = circ.components[3].parameters.r
        assert math.allclose(math.sinh(r_var.value) ** 2, 1, atol=1e-2)

    def test_dgate_optimization(self):
        """Test that Dgate is optimized correctly."""
        dgate = Dgate(0, x_trainable=True, y_trainable=True)
        target_state = DisplacedSqueezed(0, r=0.0, x=0.1, y=0.2).fock_array((40,))

        def cost_fn(dgate):
            state_out = Vacuum(0) >> dgate
            return -(math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2)

        opt = OptimizerJax()
        (dgate,) = opt.minimize(cost_fn, by_optimizing=[dgate])

        assert math.allclose(dgate.parameters.x.value, 0.1, atol=0.01)
        assert math.allclose(dgate.parameters.y.value, 0.2, atol=0.01)

    def test_sgate_optimization(self):
        """Test that Sgate is optimized correctly."""
        sgate = Sgate(0, r=0.2, phi=0.1, r_trainable=True, phi_trainable=True)
        target_state = SqueezedVacuum(0, r=0.1, phi=0.2).fock_array((40,))

        def cost_fn(sgate):
            state_out = Vacuum(0) >> sgate
            return -(math.abs(math.sum(math.conj(state_out.fock_array((40,))) * target_state)) ** 2)

        opt = OptimizerJax()
        (sgate,) = opt.minimize(cost_fn, by_optimizing=[sgate])

        assert math.allclose(sgate.parameters.r.value, 0.1, atol=0.01)
        assert math.allclose(sgate.parameters.phi.value, 0.2, atol=0.01)

    def test_bsgate_optimization(self):
        """Test that BSgate is optimized correctly."""
        bsgate = BSgate((0, 1), 0.05, 0.1, theta_trainable=True, phi_trainable=True)
        target_gate = BSgate((0, 1), 0.1, 0.2).fock_array(40)

        def cost_fn(bsgate):
            return -(math.abs(math.sum(math.conj(bsgate.fock_array(40)) * target_gate)) ** 2)

        opt = OptimizerJax()
        (bsgate,) = opt.minimize(cost_fn, by_optimizing=[bsgate])

        assert math.allclose(bsgate.parameters.theta.value, 0.1, atol=0.01)
        assert math.allclose(bsgate.parameters.phi.value, 0.2, atol=0.01)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_squeezing_grad_from_fock(self, batch_shape):
        """Test that the gradient of a squeezing gate is computed from the fock representation."""
        squeezing = Sgate(0, r=math.ones(batch_shape), r_trainable=True)
        og_r = math.asnumpy(squeezing.parameters.r.value)
        num = Number(0, 2)
        vac = Vacuum(0).dual

        def cost_fn(squeezing):
            norm = 1 / squeezing.ansatz.batch_size if squeezing.ansatz.batch_shape else 1
            return -math.real(norm * math.sum(num >> squeezing >> vac) ** 2)

        opt = OptimizerJax(learning_rate=0.05)
        (squeezing,) = opt.minimize(cost_fn, by_optimizing=[squeezing], max_steps=100)

        assert math.all(squeezing.parameters.r.value != og_r)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_displacement_grad_from_fock(self, batch_shape):
        """Test that the gradient of a displacement gate is computed from the fock representation."""
        disp = Dgate(0, x=math.ones(batch_shape), y=0.5, x_trainable=True, y_trainable=True)
        og_x = math.asnumpy(disp.parameters.x.value)
        og_y = math.asnumpy(disp.parameters.y.value)
        num = Number(0, 2)
        vac = Vacuum(0).dual

        def cost_fn(disp):
            norm = 1 / disp.ansatz.batch_size if disp.ansatz.batch_shape else 1
            return -math.real(norm * math.sum(num >> disp >> vac) ** 2)

        opt = OptimizerJax(learning_rate=0.05)
        (disp,) = opt.minimize(cost_fn, by_optimizing=[disp], max_steps=100)
        assert math.all(og_x != disp.parameters.x.value)
        assert math.all(og_y != disp.parameters.y.value)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_bsgate_grad_from_fock(self, batch_shape):
        """Test that the gradient of a beamsplitter gate is computed from the fock representation."""
        sq = SqueezedVacuum(0, r=math.ones(batch_shape), r_trainable=True)
        og_r = math.asnumpy(sq.parameters.r.value)
        num = Number(1, 1)
        vac = Vacuum(0)
        bs = BSgate((0, 1), 0.5)

        def cost_fn(sq):
            norm = 1 / sq.ansatz.batch_size if sq.ansatz.batch_shape else 1
            return -math.real(
                norm * math.sum(sq >> num >> bs >> (vac >> num).dual) ** 2,
            )

        opt = OptimizerJax(learning_rate=0.05)
        (sq,) = opt.minimize(cost_fn, by_optimizing=[sq], max_steps=100)

        assert math.all(og_r != sq.parameters.r.value)
