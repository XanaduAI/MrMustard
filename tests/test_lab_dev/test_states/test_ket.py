# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ket."""

# pylint: disable=protected-access, unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

from itertools import product
import numpy as np
import pytest

from ipywidgets import Box, HBox, VBox, HTML
from plotly.graph_objs import FigureWidget

from mrmustard import math, settings
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.math.parameters import Constant, Variable
from mrmustard.lab_dev.circuit_components_utils import TraceOut
from mrmustard.physics.gaussian import vacuum_cov, vacuum_means, squeezed_vacuum_cov
from mrmustard.physics.triples import coherent_state_Abc
from mrmustard.lab_dev.states import Coherent, DM, Ket, Number, Vacuum, DisplacedSqueezed
from mrmustard.lab_dev.transformations import Attenuator, Dgate, Sgate
from mrmustard.lab_dev.wires import Wires
from mrmustard.widgets import state as state_widget


def coherent_state_quad(q, x, y, phi=0):
    """From https://en.wikipedia.org/wiki/Coherent_state#The_wavefunction_of_a_coherent_state"""
    scale = np.sqrt(2 * settings.HBAR)
    alpha = (x + 1j * y) * np.exp(-1j * phi)
    phase = np.sin(2 * phi + 2 * np.arctan2(x, y)) * (x**2 + y**2) / 2
    return (
        math.exp(-1j * phase)  # This global phase allows Coherent >> BtoQ to be equal
        * math.exp(1j * q * alpha.imag * 2 / scale)
        * math.exp(-((q - scale * alpha.real) ** 2) / (scale**2))
        / (np.pi * settings.HBAR) ** 0.25
    )


class TestKet:  # pylint: disable=too-many-public-methods
    r"""
    Tests for the ``Ket`` class.
    """

    modes = [[[0]], [[0]]]
    x = [[1.0], [1.0, -1.0]]
    y = [[1.0], [1.0, -1.0]]
    coeff = [0.5, 0.3]

    @pytest.mark.parametrize("name", [None, "my_ket"])
    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_init(self, name, modes):
        state = Ket(modes, None, name)

        assert state.name in ("Ket0", "Ket01", "Ket2319") if not name else name
        assert list(state.modes) == sorted(modes)
        assert state.wires == Wires(modes_out_ket=set(modes))

    def test_manual_shape(self):
        ket = Coherent([0, 1], x=[1, 2])
        assert ket.manual_shape == [None, None]
        ket.manual_shape[0] = 19
        assert ket.manual_shape == [19, None]

    def test_auto_shape(self):
        ket = Coherent([0, 1], x=[1, 2])
        assert ket.auto_shape() == (8, 15)
        ket.manual_shape[0] = 19
        assert ket.auto_shape() == (19, 15)

        ket = Coherent([0, 1], x=1) >> Number([1], 10).dual
        assert ket.auto_shape() == (settings.AUTOSHAPE_MAX,)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_bargmann(self, modes):
        x = 1
        y = 2
        xs = [x] * len(modes)
        ys = [y] * len(modes)

        state_in = Coherent(modes, x, y)
        triple_in = state_in.bargmann_triple()  # automatically batched

        assert np.allclose(triple_in[0], coherent_state_Abc(xs, ys)[0])
        assert np.allclose(triple_in[1], coherent_state_Abc(xs, ys)[1])
        assert np.allclose(triple_in[2], coherent_state_Abc(xs, ys)[2])

        state_out = Ket.from_bargmann(modes, triple_in, "my_ket")
        assert state_in == state_out

    def test_from_bargmann_error(self):
        state01 = Coherent([0, 1], 1)
        with pytest.raises(ValueError):
            Ket.from_bargmann([0], state01.bargmann_triple(), "my_ket")

    def test_bargmann_triple_error(self):
        with pytest.raises(AttributeError):
            Number([0], n=10).bargmann_triple()

    @pytest.mark.parametrize("modes,x,y,coeff", zip(modes, x, y, coeff))
    def test_normalize(self, modes, x, y, coeff):
        state = Coherent(modes[0], x[0], y[0])
        for i in range(1, len(modes)):
            state += Coherent(modes[i], x[i], y[i])
        state = coeff * state
        # Bargmann
        normalized = state.normalize()
        assert np.isclose(normalized.probability, 1.0)
        # Fock
        state = state.to_fock(5)  # truncated
        normalized = state.normalize()
        assert np.isclose(normalized.probability, 1.0)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_fock(self, modes):
        state_in = Coherent(modes, x=1, y=2)
        state_in_fock = state_in.to_fock(5)
        array_in = state_in.fock(5, batched=True)

        assert math.allclose(array_in, state_in_fock.representation.array)

        state_out = Ket.from_fock(modes, array_in, "my_ket", True)
        assert state_in_fock == state_out

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_phase_space(self, modes):
        cov, means, coeff = Coherent([0], x=1, y=2).phase_space(s=0)
        assert math.allclose(coeff[0], 1.0)
        assert math.allclose(cov[0], np.eye(2) * settings.HBAR / 2)
        assert math.allclose(means[0], np.array([1.0, 2.0]) * np.sqrt(2 * settings.HBAR))
        n_modes = len(modes)

        state1 = Ket.from_phase_space(modes, (vacuum_cov(n_modes), vacuum_means(n_modes), 1.0))
        assert state1 == Vacuum(modes)

        r = [i / 10 for i in range(n_modes)]
        phi = [(i + 1) / 10 for i in range(n_modes)]
        state2 = Ket.from_phase_space(
            modes, (squeezed_vacuum_cov(r, phi), vacuum_means(n_modes), 1.0)
        )
        assert state2 == Vacuum(modes) >> Sgate(modes, r, phi)

    def test_to_from_quadrature(self):
        modes = [0]
        A0 = np.array([[0]])
        b0 = np.array([0.2j])
        c0 = np.exp(-0.5 * 0.04)  # z^*

        state0 = Ket.from_bargmann(modes, (A0, b0, c0))
        Atest, btest, ctest = state0.quadrature_triple()
        state1 = Ket.from_quadrature(modes, (Atest[0], btest[0], ctest[0]))
        Atest2, btest2, ctest2 = state1.bargmann_triple()
        assert math.allclose(Atest2, A0)
        assert math.allclose(btest2, b0)
        assert math.allclose(ctest2, c0)

    def test_L2_norm(self):
        state = Coherent([0], x=1)
        assert state.L2_norm == 1

    def test_probability(self):
        state1 = Coherent([0], x=1) / 3
        assert math.allclose(state1.probability, 1 / 9)
        assert math.allclose(state1.to_fock(20).probability, 1 / 9)

        state2 = Coherent([0], x=1) / 2**0.5 + Coherent([0], x=-1) / 2**0.5
        assert math.allclose(state2.probability, 1.13533528)
        assert math.allclose(state2.to_fock(20).probability, 1.13533528)

        state3 = Number([0], n=1, cutoffs=2) / 2**0.5 + Number([0], n=2) / 2**0.5
        assert math.allclose(state3.probability, 1)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_purity(self, modes):
        state = Ket(modes, None, "my_ket")
        assert state.purity == 1
        assert state.is_pure

    def test_dm(self):
        ket = Coherent([0, 1], x=1, y=[2, 3])
        dm = ket.dm()

        assert dm.name == ket.name
        assert dm.representation == (ket @ ket.adjoint).representation
        assert dm.wires == (ket @ ket.adjoint).wires

    @pytest.mark.parametrize("phi", [0, 0.3, np.pi / 4, np.pi / 2])
    def test_quadrature_single_mode_ket(self, phi):
        x, y = 1, 2
        state = Coherent(modes=[0], x=x, y=y)
        q = np.linspace(-10, 10, 100)
        quad = math.transpose(math.astensor([q]))
        psi_phi = coherent_state_quad(q, x, y, phi)
        assert math.allclose(state.quadrature(quad, phi=phi), psi_phi)
        assert math.allclose(state.quadrature_distribution(q, phi=phi), abs(psi_phi) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(quad, phi=phi), psi_phi)
        assert math.allclose(
            state.to_fock(40).quadrature_distribution(q, phi=phi), abs(psi_phi) ** 2
        )

    def test_quadrature_multimode_ket(self):
        x, y = 1, 2
        state = Coherent(modes=[0, 1], x=x, y=y)
        q = np.linspace(-10, 10, 100)
        quad = math.astensor(list(product(q, repeat=state.n_modes)))
        psi_q = math.kron(coherent_state_quad(q, x, y), coherent_state_quad(q, x, y))
        assert math.allclose(state.quadrature(quad), psi_q)
        assert math.allclose(state.quadrature_distribution(q), abs(psi_q) ** 2)
        assert math.allclose(state.to_fock(100).quadrature(quad), psi_q)
        assert math.allclose(state.to_fock(100).quadrature_distribution(q), abs(psi_q) ** 2)

    def test_quadrature_multivariable_ket(self):
        x, y = 1, 2
        state = Coherent(modes=[0, 1], x=x, y=y)
        q1 = np.linspace(-10, 10, 100)
        q2 = np.linspace(-10, 10, 100)
        quad = np.array([[qa, qb] for qa in q1 for qb in q2])
        psi_q = math.outer(coherent_state_quad(q1, x, y), coherent_state_quad(q2, x, y))
        assert math.allclose(state.quadrature_distribution(quad).reshape(100, 100), abs(psi_q) ** 2)

    def test_quadrature_batch(self):
        x1, y1, x2, y2 = 1, 2, -1, -2
        state = Coherent(modes=[0], x=x1, y=y1) + Coherent(modes=[0], x=x2, y=y2)
        q = np.linspace(-10, 10, 100)
        quad = math.transpose(math.astensor([q]))
        psi_q = coherent_state_quad(q, x1, y1) + coherent_state_quad(q, x2, y2)
        assert math.allclose(state.quadrature(quad), psi_q)
        assert math.allclose(state.quadrature_distribution(q), abs(psi_q) ** 2)
        assert math.allclose(state.to_fock(40).quadrature(quad), psi_q)
        assert math.allclose(state.to_fock(40).quadrature_distribution(q), abs(psi_q) ** 2)

    def test_expectation_bargmann(self):
        ket = Coherent([0, 1], x=1, y=[2, 3])

        assert math.allclose(ket.expectation(ket), 1.0)

        k0 = Coherent([0], x=1, y=2)
        k1 = Coherent([1], x=1, y=3)
        k01 = Coherent([0, 1], x=1, y=[2, 3])

        res_k0 = (ket @ k0.dual) >> TraceOut([1])
        res_k1 = (ket @ k1.dual) >> TraceOut([0])
        res_k01 = ket @ k01.dual

        assert math.allclose(ket.expectation(k0), res_k0)
        assert math.allclose(ket.expectation(k1), res_k1)
        assert math.allclose(ket.expectation(k01), math.sum(res_k01.representation.c))

        dm0 = Coherent([0], x=1, y=2).dm()
        dm1 = Coherent([1], x=1, y=3).dm()
        dm01 = Coherent([0, 1], x=1, y=[2, 3]).dm()

        res_dm0 = (ket @ ket.adjoint @ dm0.dual) >> TraceOut([1])
        res_dm1 = (ket @ ket.adjoint @ dm1.dual) >> TraceOut([0])
        res_dm01 = ket @ ket.adjoint @ dm01.dual

        assert math.allclose(ket.expectation(dm0), res_dm0)
        assert math.allclose(ket.expectation(dm1), res_dm1)
        assert math.allclose(ket.expectation(dm01), math.sum(res_dm01.representation.c))

        u0 = Dgate([1], x=0.1)
        u1 = Dgate([0], x=0.2)
        u01 = Dgate([0, 1], x=[0.3, 0.4])

        res_u0 = ket @ u0 >> ket.dual
        res_u1 = ket @ u1 >> ket.dual
        res_u01 = ket @ u01 >> ket.dual

        assert math.allclose(ket.expectation(u0), res_u0)
        assert math.allclose(ket.expectation(u1), res_u1)
        assert math.allclose(ket.expectation(u01), res_u01)

    def test_expectation_fock(self):
        ket = Coherent([0, 1], x=1, y=[2, 3]).to_fock(10)

        assert math.allclose(ket.expectation(ket), np.abs(ket >> ket.dual) ** 2)

        k0 = Coherent([0], x=1, y=2).to_fock(10)
        k1 = Coherent([1], x=1, y=3).to_fock(10)
        k01 = Coherent([0, 1], x=1, y=[2, 3]).to_fock(10)

        res_k0 = (ket @ k0.dual) >> TraceOut([1])
        res_k1 = (ket @ k1.dual) >> TraceOut([0])
        res_k01 = (ket >> k01.dual) ** 2

        assert math.allclose(ket.expectation(k0), res_k0)
        assert math.allclose(ket.expectation(k1), res_k1)
        assert math.allclose(ket.expectation(k01), res_k01)

        dm0 = Coherent([0], x=1, y=0.2).dm().to_fock(10)
        dm1 = Coherent([1], x=1, y=0.3).dm().to_fock(10)
        dm01 = Coherent([0, 1], x=1, y=[0.2, 0.3]).dm().to_fock(10)

        res_dm0 = (ket @ ket.adjoint @ dm0.dual) >> TraceOut([1])
        res_dm1 = (ket @ ket.adjoint @ dm1.dual) >> TraceOut([0])
        res_dm01 = (ket @ ket.adjoint @ dm01.dual).to_fock(10).representation.array

        assert math.allclose(ket.expectation(dm0), res_dm0)
        assert math.allclose(ket.expectation(dm1), res_dm1)
        assert math.allclose(ket.expectation(dm01), res_dm01[0])

        u0 = Dgate([1], x=0.1)
        u1 = Dgate([0], x=0.2)
        u01 = Dgate([0, 1], x=[0.3, 0.4])

        res_u0 = (ket @ u0 @ ket.dual).to_fock(10).representation.array
        res_u1 = (ket @ u1 @ ket.dual).to_fock(10).representation.array
        res_u01 = (ket @ u01 @ ket.dual).to_fock(10).representation.array

        assert math.allclose(ket.expectation(u0), res_u0[0])
        assert math.allclose(ket.expectation(u1), res_u1[0])
        assert math.allclose(ket.expectation(u01), res_u01[0])

    def test_expectation_error(self):
        ket = Coherent([0, 1], x=1, y=[2, 3])

        op1 = Attenuator([0])
        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            ket.expectation(op1)

        op2 = CircuitComponent(wires=[(), (), (1,), (0,)])
        with pytest.raises(ValueError, match="different modes"):
            ket.expectation(op2)

        op3 = Dgate([2])
        with pytest.raises(ValueError, match="Expected an operator defined on"):
            ket.expectation(op3)

    def test_rshift(self):
        ket = Coherent([0, 1], 1)
        unitary = Dgate([0], 1)
        u_component = CircuitComponent._from_attributes(
            unitary.representation, unitary.wires, unitary.name
        )  # pylint: disable=protected-access
        channel = Attenuator([1], 1)
        ch_component = CircuitComponent._from_attributes(
            channel.representation,
            channel.wires,
            channel.name,
        )  # pylint: disable=protected-access

        # gates
        assert isinstance(ket >> unitary, Ket)
        assert isinstance(ket >> channel, DM)
        assert isinstance(ket >> unitary >> channel, DM)
        assert isinstance(ket >> channel >> unitary, DM)
        assert isinstance(ket >> u_component, CircuitComponent)
        assert isinstance(ket >> ch_component, CircuitComponent)

        # measurements
        assert isinstance(ket >> Coherent([0], 1).dual, Ket)
        assert isinstance(ket >> Coherent([0], 1).dm().dual, DM)

    @pytest.mark.parametrize("modes", [[3, 30, 98]])
    @pytest.mark.parametrize("m", [[3], [30], [98], [3, 98]])
    def test_get_item(self, modes, m):
        ket = Vacuum(modes) >> Dgate(modes, x=[0, 1, 2])
        dm = ket.dm()

        assert ket[m] == dm[m]

    @pytest.mark.parametrize("modes", [[3, 30, 98]])
    @pytest.mark.parametrize("m", [[3], [30], [98], [3, 98]])
    def test_get_item_builtin_kets(self, modes, m):
        idx = [modes.index(s) for s in m]

        x = math.asnumpy([0, 1, 2])
        s = DisplacedSqueezed(modes, x=x, y=3, y_trainable=True, y_bounds=(0, 6))

        assert np.all(s.y.value == 3)
        assert s.y.value.shape == (len(modes),)
        assert s.r.value.shape == (len(modes),)
        assert s.phi.value.shape == (len(modes),)

        si = s[m]
        assert isinstance(si, DisplacedSqueezed)
        assert si == DisplacedSqueezed(m, x=x[idx], y=3, y_trainable=True, y_bounds=(0, 6))

        assert isinstance(si.x, Constant)
        assert math.allclose(si.x.value, x[idx])

        assert isinstance(si.y, Variable)
        assert np.all(si.y.value == 3)
        assert si.y.value.shape == (len(idx),)
        assert si.y.bounds == s.y.bounds

        assert isinstance(si.r, Constant)
        assert np.all(si.r.value == 0)
        assert si.r.value.shape == (len(idx),)

        assert isinstance(si.phi, Constant)
        assert np.all(si.phi.value == 0)
        assert si.phi.value.shape == (len(idx),)

    def test_private_batched_properties(self):
        cat = Coherent([0], x=1.0) + Coherent([0], x=-1.0)  # used as a batch
        assert np.allclose(cat._probabilities, np.ones(2))
        assert np.allclose(cat._L2_norms, np.ones(2))

    def test_unsafe_batch_zipping(self):
        cat = Coherent([0], x=1.0) + Coherent([0], x=-1.0)  # used as a batch
        displacements = Dgate([0], x=1.0) + Dgate([0], x=-1.0)
        settings.UNSAFE_ZIP_BATCH = True
        better_cat = cat >> displacements
        settings.UNSAFE_ZIP_BATCH = False
        assert better_cat == Coherent([0], x=2.0) + Coherent([0], x=-2.0)

    @pytest.mark.parametrize("max_sq", [1, 2, 3])
    def test_random_states(self, max_sq):
        psi = Ket.random([1, 22], max_sq)
        A = psi.representation.A[0]
        assert np.isclose(psi.probability, 1)  # checks if the state is normalized
        assert np.allclose(A - np.transpose(A), np.zeros(2))  # checks if the A matrix is symmetric

    def test_ipython_repr(self):
        """
        Test the widgets.state function.
        Note: could not mock display because of the states.py file name conflict.
        """
        hbox = state_widget(Number([0], n=1), True, True)
        assert isinstance(hbox, HBox)

        [left, viz_2d] = hbox.children
        assert isinstance(left, VBox)
        assert isinstance(viz_2d, FigureWidget)

        [table, dm] = left.children
        assert isinstance(table, HTML)
        assert isinstance(dm, FigureWidget)

    def test_ipython_repr_too_many_dims(self):
        """Test the widgets.state function when the Ket has too many dims."""
        vbox = state_widget(Number([0, 1], n=1), True, True)
        assert isinstance(vbox, Box)

        [table, wires] = vbox.children
        assert isinstance(table, HTML)
        assert isinstance(wires, HTML)

    def test_is_physical(self):
        assert Ket.random([0, 1]).is_physical
