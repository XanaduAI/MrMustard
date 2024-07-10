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

"""Tests for the state subpackage."""

# pylint: disable=protected-access, unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import json
import os
import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.math.parameters import Constant, Variable
from mrmustard.physics.representations import Bargmann
from mrmustard.physics.fock import fock_state
from mrmustard.physics.gaussian import vacuum_cov, vacuum_means, squeezed_vacuum_cov
from mrmustard.physics.triples import coherent_state_Abc, thermal_state_Abc
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.circuit_components_utils import TraceOut
from mrmustard.lab_dev.states import (
    Coherent,
    DisplacedSqueezed,
    DM,
    Ket,
    Number,
    SqueezedVacuum,
    TwoModeSqueezedVacuum,
    Thermal,
    Vacuum,
    Sauron,
)
from mrmustard.lab_dev.transformations import Attenuator, Dgate, Sgate, S2gate
from mrmustard.lab_dev.wires import Wires

# original settings
autocutoff_max0 = settings.AUTOCUTOFF_MAX_CUTOFF


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

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_bargmann(self, modes):
        x = 1
        y = 2
        xs = [x] * len(modes)
        ys = [y] * len(modes)

        state_in = Coherent(modes, x, y)
        triple_in = state_in.bargmann

        assert np.allclose(triple_in[0], coherent_state_Abc(xs, ys)[0])
        assert np.allclose(triple_in[1], coherent_state_Abc(xs, ys)[1])
        assert np.allclose(triple_in[2], coherent_state_Abc(xs, ys)[2])

        state_out = Ket.from_bargmann(modes, triple_in, "my_ket")
        assert state_in == state_out

    def test_from_bargmann_error(self):
        state01 = Coherent([0, 1], 1)
        with pytest.raises(ValueError):
            Ket.from_bargmann([0], state01.bargmann, "my_ket")

    def test_bargmann_triple_error(self):
        with pytest.raises(AttributeError):
            Number([0], n=10).bargmann

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
        array_in = state_in.fock(5)

        assert math.allclose(array_in, state_in_fock.representation.array)

        state_out = Ket.from_fock(modes, array_in, "my_ket", True)
        assert state_in_fock == state_out

    def test_from_fock_error(self):
        state01 = Coherent([0, 1], 1).to_fock(5)
        with pytest.raises(ValueError):
            Ket.from_fock([0], state01.fock(5), "my_ket", True)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_phase_space(self, modes):
        cov, means, coeff = Coherent([0], x=1, y=2).phase_space(s=0)
        assert math.allclose(coeff[0], 1.0)
        assert math.allclose(cov[0], np.eye(2))
        assert math.allclose(means[0], np.array([2.0, 4.0]))

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
        Atest, btest, ctest = state0.quadrature()
        state1 = Ket.from_quadrature(modes, (Atest[0], btest[0], ctest[0]))
        Atest2, btest2, ctest2 = state1.bargmann
        assert math.allclose(Atest2[0], A0)
        assert math.allclose(btest2[0], b0)
        assert math.allclose(ctest2[0], c0)

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
        settings.AUTOCUTOFF_MAX_CUTOFF = 30

        ket = Coherent([0, 1], x=1, y=[0.2, 0.3]).to_fock()

        assert math.allclose(ket.expectation(ket), (ket >> ket.dual) ** 2)

        k0 = Coherent([0], x=1, y=0.2)
        k1 = Coherent([1], x=1, y=0.3)
        k01 = Coherent([0, 1], x=1, y=[0.2, 0.3])

        res_k0 = (ket @ k0.dual) >> TraceOut([1])
        res_k1 = (ket @ k1.dual) >> TraceOut([0])
        res_k01 = (ket >> k01.dual) ** 2

        assert math.allclose(ket.expectation(k0), res_k0)
        assert math.allclose(ket.expectation(k1), res_k1)
        assert math.allclose(ket.expectation(k01), res_k01)

        dm0 = Coherent([0], x=1, y=0.2).dm()
        dm1 = Coherent([1], x=1, y=0.3).dm()
        dm01 = Coherent([0, 1], x=1, y=[0.2, 0.3]).dm()

        res_dm0 = (ket @ ket.adjoint @ dm0.dual) >> TraceOut([1])
        res_dm1 = (ket @ ket.adjoint @ dm1.dual) >> TraceOut([0])
        res_dm01 = (ket @ ket.adjoint @ dm01.dual).representation.array

        assert math.allclose(ket.expectation(dm0), res_dm0)
        assert math.allclose(ket.expectation(dm1), res_dm1)
        assert math.allclose(ket.expectation(dm01), res_dm01[0])

        u0 = Dgate([1], x=0.1)
        u1 = Dgate([0], x=0.2)
        u01 = Dgate([0, 1], x=[0.3, 0.4])

        res_u0 = (ket @ u0 @ ket.dual).representation.array
        res_u1 = (ket @ u1 @ ket.dual).representation.array
        res_u01 = (ket @ u01 @ ket.dual).representation.array

        assert math.allclose(ket.expectation(u0), res_u0[0])
        assert math.allclose(ket.expectation(u1), res_u1[0])
        assert math.allclose(ket.expectation(u01), res_u01[0])

        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0

    def test_expectation_error(self):
        ket = Coherent([0, 1], x=1, y=[2, 3])

        op1 = Attenuator([0])
        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            ket.expectation(op1)

        op2 = CircuitComponent(None, modes_in_ket=[0], modes_out_ket=[1])
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

        si = s[m]
        assert isinstance(si, DisplacedSqueezed)
        assert si == DisplacedSqueezed(m, x=x[idx], y=3, y_trainable=True, y_bounds=(0, 6))

        assert isinstance(si.x, Constant)
        assert math.allclose(si.x.value, x[idx])

        assert isinstance(si.y, Variable)
        assert si.y.value == s.y.value
        assert si.y.bounds == s.y.bounds

        assert isinstance(si.r, Constant)
        assert si.r.value == s.r.value

        assert isinstance(si.phi, Constant)
        assert si.phi.value == s.phi.value

    def test_private_batched_properties(self):
        cat = Coherent([0], x=1.0) + Coherent([0], x=-1.0)  # used as a batch
        assert np.allclose(cat._purities, np.ones(2))
        assert np.allclose(cat._probabilities, np.ones(2))
        assert np.allclose(cat._L2_norms, np.ones(2))

    def test_unsafe_batch_zipping(self):
        cat = Coherent([0], x=1.0) + Coherent([0], x=-1.0)  # used as a batch
        displacements = Dgate([0], x=1.0) + Dgate([0], x=-1.0)
        settings.UNSAFE_ZIP_BATCH = True
        better_cat = cat >> displacements
        settings.UNSAFE_ZIP_BATCH = False
        assert better_cat == Coherent([0], x=2.0) + Coherent([0], x=-2.0)


class TestDM:
    r"""
    Tests for the ``DM`` class.
    """

    modes = [[[0]], [[0]]]
    x = [[1], [1, -1]]
    y = [[1], [1, -1]]
    coeff = [0.5, 0.3]

    @pytest.mark.parametrize("name", [None, "my_dm"])
    @pytest.mark.parametrize("modes", [{0}, {0, 1}, {3, 19, 2}])
    def test_init(self, name, modes):
        state = DM(modes, None, name)

        assert state.name in ("DM0", "DM01", "DM2319") if not name else name
        assert list(state.modes) == sorted(modes)
        assert state.wires == Wires(modes_out_bra=modes, modes_out_ket=modes)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_bargmann(self, modes):
        state_in = Coherent(modes, 1, 2) >> Attenuator([modes[0]], 0.7)
        triple_in = state_in.bargmann

        state_out = DM.from_bargmann(modes, triple_in, "my_dm")
        assert state_in == state_out

    def test_from_bargmann_error(self):
        state01 = Coherent([0, 1], 1).dm()
        with pytest.raises(ValueError):
            DM.from_bargmann(
                [0],
                state01.bargmann,
                "my_dm",
            )

    def test_from_fock_error(self):
        state01 = Coherent([0, 1], 1).dm()
        state01 = state01.to_fock(2)
        with pytest.raises(ValueError):
            DM.from_fock([0], state01.fock(5), "my_dm", True)

    def test_bargmann_triple_error(self):
        fock = Number([0], n=10).dm()
        with pytest.raises(AttributeError):
            fock.bargmann

    @pytest.mark.parametrize("modes,x,y,coeff", zip(modes, x, y, coeff))
    def test_normalize(self, modes, x, y, coeff):
        state = Coherent(modes[0], x[0], y[0]).dm()
        for i in range(1, len(modes)):
            state += Coherent(modes[i], x[i], y[i]).dm()
        state *= coeff
        # Bargmann
        normalized = state.normalize()
        assert np.isclose(normalized.probability, 1.0)
        # Fock
        state = state.to_fock(5)  # truncated
        normalized = state.normalize()
        assert np.isclose(normalized.probability, 1.0)

    @pytest.mark.parametrize("modes", [[0], [0, 1], [3, 19, 2]])
    def test_to_from_fock(self, modes):
        state_in = Coherent(modes, x=1, y=2) >> Attenuator([modes[0]], 0.8)
        state_in_fock = state_in.to_fock(5)
        array_in = state_in.fock(5)

        assert math.allclose(array_in, state_in_fock.representation.array)

        state_out = DM.from_fock(modes, array_in, "my_dm", True)
        assert state_in_fock == state_out

    def test_to_from_phase_space(self):
        state0 = Coherent([0], x=1, y=2) >> Attenuator([0], 1.0)
        cov, means, coeff = state0.phase_space(s=0)  # batch = 1
        assert coeff[0] == 1.0
        assert math.allclose(cov[0], np.eye(2))
        assert math.allclose(means[0], np.array([2.0, 4.0]))

        # test error
        with pytest.raises(ValueError):
            DM.from_phase_space([0, 1], (cov, means, 1.0))

        cov = vacuum_cov(1)
        means = [1.78885438, 3.57770876]
        state1 = DM.from_phase_space([0], (cov, means, 1.0))
        assert state1 == Coherent([0], 1, 2) >> Attenuator([0], 0.8)

    def test_to_from_quadrature(self):
        modes = [0]
        A0 = np.array([[0, 0], [0, 0]])
        b0 = np.array([0.1 - 0.2j, 0.1 + 0.2j])
        c0 = 0.951229424500714  # z, z^*

        state0 = DM.from_bargmann(modes, (A0, b0, c0))
        Atest, btest, ctest = state0.quadrature()
        state1 = DM.from_quadrature(modes, (Atest[0], btest[0], ctest[0]))
        Atest2, btest2, ctest2 = state1.bargmann
        assert math.allclose(Atest2[0], A0)
        assert math.allclose(btest2[0], b0)
        assert math.allclose(ctest2[0], c0)

    def test_L2_norms(self):
        state = Coherent([0], x=1).dm() + Coherent([0], x=-1).dm()  # incoherent
        assert len(state._L2_norms) == 2

    def test_L2_norm(self):
        state = Coherent([0], x=1).dm()
        assert state.L2_norm == 1

    def test_probability(self):
        state1 = Coherent([0], x=1).dm()
        assert state1.probability == 1
        assert state1.to_fock(20).probability == 1

        state2 = Coherent([0], x=1).dm() / 3 + 2 * Coherent([0], x=-1).dm() / 3
        assert state2.probability == 1
        assert math.allclose(state2.to_fock(20).probability, 1)

        state3 = Number([0], n=1, cutoffs=2).dm() / 2 + Number([0], n=2).dm() / 2
        assert math.allclose(state3.probability, 1)

    def test_purity(self):
        state = Coherent([0], 1, 2).dm()
        assert math.allclose(state.purity, 1)
        assert state.is_pure

    def test_expectation_bargmann(self):
        ket = Coherent([0, 1], x=1, y=[2, 3])
        dm = ket.dm()

        k0 = Coherent([0], x=1, y=2)
        k1 = Coherent([1], x=1, y=3)
        k01 = Coherent([0, 1], x=1, y=[2, 3])

        res_k0 = (dm @ k0.dual @ k0.dual.adjoint) >> TraceOut([1])
        res_k1 = (dm @ k1.dual @ k1.dual.adjoint) >> TraceOut([0])
        res_k01 = dm @ k01.dual @ k01.dual.adjoint

        assert math.allclose(dm.expectation(k0), res_k0)
        assert math.allclose(dm.expectation(k1), res_k1)
        assert math.allclose(dm.expectation(k01), res_k01.representation.c[0])

        dm0 = Coherent([0], x=1, y=2).dm()
        dm1 = Coherent([1], x=1, y=3).dm()
        dm01 = Coherent([0, 1], x=1, y=[2, 3]).dm()

        res_dm0 = (dm @ dm0.dual) >> TraceOut([1])
        res_dm1 = (dm @ dm1.dual) >> TraceOut([0])
        res_dm01 = dm @ dm01.dual

        assert math.allclose(dm.expectation(dm0), res_dm0)
        assert math.allclose(dm.expectation(dm1), res_dm1)
        assert math.allclose(dm.expectation(dm01), res_dm01.representation.c[0])

        assert math.allclose(dm.expectation(k0), res_k0)
        assert math.allclose(dm.expectation(k1), res_k1)
        assert math.allclose(dm.expectation(k01), res_k01.representation.c[0])

        dm0 = Coherent([0], x=1, y=2).dm()
        dm1 = Coherent([1], x=1, y=3).dm()
        dm01 = Coherent([0, 1], x=1, y=[2, 3]).dm()

        res_dm0 = (dm @ dm0.dual) >> TraceOut([1])
        res_dm1 = (dm @ dm1.dual) >> TraceOut([0])
        res_dm01 = dm @ dm01.dual

        assert math.allclose(dm.expectation(dm0), res_dm0)
        assert math.allclose(dm.expectation(dm1), res_dm1)
        assert math.allclose(dm.expectation(dm01), res_dm01.representation.c[0])

        u0 = Dgate([0], x=0.1)
        u1 = Dgate([1], x=0.2)
        u01 = Dgate([0, 1], x=[0.3, 0.4])

        res_u0 = (dm @ u0) >> TraceOut([0, 1])
        res_u1 = (dm @ u1) >> TraceOut([0, 1])
        res_u01 = (dm @ u01) >> TraceOut([0, 1])

        assert math.allclose(dm.expectation(u0), res_u0)
        assert math.allclose(dm.expectation(u1), res_u1)
        assert math.allclose(dm.expectation(u01), res_u01)

        settings.AUTOCUTOFF_MAX_CUTOFF = autocutoff_max0

    def test_expectation_error(self):
        dm = Coherent([0, 1], x=1, y=[2, 3]).dm()

        op1 = Attenuator([0])
        with pytest.raises(ValueError, match="Cannot calculate the expectation value"):
            dm.expectation(op1)

        op2 = CircuitComponent(None, modes_in_ket=[0], modes_out_ket=[1])
        with pytest.raises(ValueError, match="different modes"):
            dm.expectation(op2)

        op3 = Dgate([2])
        with pytest.raises(ValueError, match="Expected an operator defined on"):
            dm.expectation(op3)

    def test_rshift(self):
        ket = Coherent([0, 1], 1)
        unitary = Dgate([0], 1)
        u_component = CircuitComponent._from_attributes(
            unitary.representation, unitary.wires, unitary.name
        )  # pylint: disable=protected-access
        channel = Attenuator([1], 1)
        ch_component = CircuitComponent._from_attributes(
            channel.representation, channel.wires, channel.name
        )  # pylint: disable=protected-access

        dm = ket >> channel

        # gates
        assert isinstance(dm, DM)
        assert isinstance(dm >> unitary >> channel, DM)
        assert isinstance(dm >> channel >> unitary, DM)
        assert isinstance(dm >> u_component, CircuitComponent)
        assert isinstance(dm >> ch_component, CircuitComponent)

        # measurements
        assert isinstance(dm >> Coherent([0], 1).dual, DM)
        assert isinstance(dm >> Coherent([0], 1).dm().dual, DM)


class TestCoherent:
    r"""
    Tests for the ``Coherent`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    x = [[1], 1, [1, 2]]
    y = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,x,y", zip(modes, x, y))
    def test_init(self, modes, x, y):
        state = Coherent(modes, x, y)

        assert state.name == "Coherent"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``x``"):
            Coherent(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="Length of ``y``"):
            Coherent(modes=[0, 1], x=1, y=[2, 3, 4])

    def test_trainable_parameters(self):
        state1 = Coherent([0], 1, 1)
        state2 = Coherent([0], 1, 1, x_trainable=True, x_bounds=(-2, 2))
        state3 = Coherent([0], 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.x.value = 3

        state2.x.value = 2
        assert state2.x.value == 2

        state3.y.value = 2
        assert state3.y.value == 2

    def test_representation(self):
        rep1 = Coherent(modes=[0], x=0.1, y=0.2).representation
        assert math.allclose(rep1.A, np.zeros((1, 1, 1)))
        assert math.allclose(rep1.b, [[0.1 + 0.2j]])
        assert math.allclose(rep1.c, [0.97530991])

        rep2 = Coherent(modes=[0, 1], x=0.1, y=[0.2, 0.3]).representation
        assert math.allclose(rep2.A, np.zeros((1, 2, 2)))
        assert math.allclose(rep2.b, [[0.1 + 0.2j, 0.1 + 0.3j]])
        assert math.allclose(rep2.c, [0.9277434863])

        rep3 = Coherent(modes=[1], x=0.1).representation
        assert math.allclose(rep3.A, np.zeros((1, 1, 1)))
        assert math.allclose(rep3.b, [[0.1]])
        assert math.allclose(rep3.c, [0.9950124791926823])

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Coherent(modes=[0], x=[0.1, 0.2]).representation

    def test_linear_combinations(self):
        state1 = Coherent([0], x=1, y=2)
        state2 = Coherent([0], x=2, y=3)
        state3 = Coherent([0], x=3, y=4)

        lc = state1 + state2 - state3
        assert lc.representation.ansatz.batch_size == 3

        assert (lc @ lc.dual).representation.ansatz.batch_size == 9
        settings.UNSAFE_ZIP_BATCH = True
        assert (lc @ lc.dual).representation.ansatz.batch_size == 3  # not 9
        settings.UNSAFE_ZIP_BATCH = False


class TestDisplacedSqueezed:
    r"""
    Tests for the ``DisplacedSqueezed`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    x = [[1], 1, [1, 2]]
    y = [[3], [3, 4], [3, 4]]
    r = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,x,y,r,phi", zip(modes, x, y, r, phi))
    def test_init(self, modes, x, y, r, phi):
        state = DisplacedSqueezed(modes, x, y, r, phi)

        assert state.name == "DisplacedSqueezed"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``x``"):
            DisplacedSqueezed(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="Length of ``y``"):
            DisplacedSqueezed(modes=[0, 1], x=1, y=[2, 3, 4])

    def test_trainable_parameters(self):
        state1 = DisplacedSqueezed([0], 1, 1)
        state2 = DisplacedSqueezed([0], 1, 1, x_trainable=True, x_bounds=(-2, 2))
        state3 = DisplacedSqueezed([0], 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.x.value = 3

        state2.x.value = 2
        assert state2.x.value == 2

        state3.y.value = 2
        assert state3.y.value == 2

    @pytest.mark.parametrize("modes,x,y,r,phi", zip(modes, x, y, r, phi))
    def test_representation(self, modes, x, y, r, phi):
        rep = DisplacedSqueezed(modes, x, y, r, phi).representation
        exp = (Vacuum(modes) >> Sgate(modes, r, phi) >> Dgate(modes, x, y)).representation
        assert rep == exp

    def test_representation_error(self):
        with pytest.raises(ValueError):
            DisplacedSqueezed(modes=[0], x=[0.1, 0.2]).representation


class TestNumber:
    r"""
    Tests for the ``Number`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    n = [[3], 4, [5, 6]]
    cutoffs = [None, [5], [6, 7]]

    @pytest.mark.parametrize("modes,n,cutoffs", zip(modes, n, cutoffs))
    def test_init(self, modes, n, cutoffs):
        state = Number(modes, n, cutoffs)

        assert state.name == "N"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``n``"):
            Number(modes=[0, 1], n=[2, 3, 4])

        with pytest.raises(ValueError, match="Length of ``cutoffs``"):
            Number(modes=[0, 1], n=[2, 3], cutoffs=[4, 5, 6])

    @pytest.mark.parametrize("n", [2, [2, 3], [4, 4]])
    @pytest.mark.parametrize("cutoffs", [None, [4, 5], [5, 5]])
    def test_representation(self, n, cutoffs):
        rep1 = Number([0, 1], n, cutoffs).representation.array
        exp1 = fock_state((n,) * 2 if isinstance(n, int) else n, cutoffs)
        assert math.allclose(rep1, math.asnumpy(exp1).reshape(1, *exp1.shape))

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Coherent(modes=[0], x=[0.1, 0.2]).representation


class TestSqueezedVacuum:
    r"""
    Tests for the ``SqueezedVacuum`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    r = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_init(self, modes, r, phi):
        state = SqueezedVacuum(modes, r, phi)

        assert state.name == "SqueezedVacuum"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``r``"):
            SqueezedVacuum(modes=[0, 1], r=[2, 3, 4])

        with pytest.raises(ValueError, match="Length of ``phi``"):
            SqueezedVacuum(modes=[0, 1], r=1, phi=[2, 3, 4])

    def test_modes_slice_params(self):
        psi = SqueezedVacuum([0, 1], r=[1, 2], phi=[3, 4])
        assert psi[0] == SqueezedVacuum([0], r=1, phi=3)

    def test_trainable_parameters(self):
        state1 = SqueezedVacuum([0], 1, 1)
        state2 = SqueezedVacuum([0], 1, 1, r_trainable=True, r_bounds=(-2, 2))
        state3 = SqueezedVacuum([0], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.r.value = 3

        state2.r.value = 2
        assert state2.r.value == 2

        state3.phi.value = 2
        assert state3.phi.value == 2

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_representation(self, modes, r, phi):
        rep = SqueezedVacuum(modes, r, phi).representation
        exp = (Vacuum(modes) >> Sgate(modes, r, phi)).representation
        assert rep == exp

    def test_representation_error(self):
        with pytest.raises(ValueError):
            SqueezedVacuum(modes=[0], r=[0.1, 0.2]).representation


class TestTwoModeSqueezedVacuum:
    r"""
    Tests for the ``TwoModeSqueezedVacuum`` class.
    """

    modes = [[0, 1], [1, 2], [1, 5]]
    r = [[1], 1, [2]]
    phi = [3, [4], 1]

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_init(self, modes, r, phi):
        state = TwoModeSqueezedVacuum(modes, r, phi)

        assert state.name == "TwoModeSqueezedVacuum"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``r``"):
            TwoModeSqueezedVacuum(modes=[0, 1], r=[2, 3, 4])

        with pytest.raises(ValueError, match="Length of ``phi``"):
            SqueezedVacuum(modes=[0, 1], r=1, phi=[2, 3, 4])

    def test_trainable_parameters(self):
        state1 = TwoModeSqueezedVacuum([0, 1], 1, 1)
        state2 = TwoModeSqueezedVacuum([0, 1], 1, 1, r_trainable=True, r_bounds=(0, 2))
        state3 = TwoModeSqueezedVacuum([0, 1], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.r.value = 3

        state2.r.value = 2
        assert state2.r.value == 2

        state3.phi.value = 2
        assert state3.phi.value == 2

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_representation(self, modes, r, phi):
        rep = TwoModeSqueezedVacuum(modes, r, phi).representation
        exp = (Vacuum(modes) >> S2gate(modes, r, phi)).representation
        assert rep == exp

    def test_representation_error(self):
        with pytest.raises(ValueError):
            TwoModeSqueezedVacuum(modes=[0], r=[0.1, 0.2]).representation


class TestVacuum:
    r"""
    Tests for the ``Vacuum`` class.
    """

    @pytest.mark.parametrize("modes", [(0,), (0, 1), (3, 19, 2)])
    def test_init(self, modes):
        state = Vacuum(modes)

        assert state.name == "Vac"
        assert list(state.modes) == sorted(modes)
        assert state.n_modes == len(modes)

    @pytest.mark.parametrize("n_modes", [1, 3])
    def test_representation(self, n_modes):
        rep = Vacuum(range(n_modes)).representation

        assert math.allclose(rep.A, np.zeros((1, n_modes, n_modes)))
        assert math.allclose(rep.b, np.zeros((1, n_modes)))
        assert math.allclose(rep.c, [1.0])


class TestThermal:
    r"""
    Tests for the ``Thermal`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    nbar = [[3], 4, [5, 6]]

    @pytest.mark.parametrize("modes,nbar", zip(modes, nbar))
    def test_init(self, modes, nbar):
        state = Thermal(modes, nbar)

        assert state.name == "Thermal"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="Length of ``nbar``"):
            Thermal(modes=[0, 1], nbar=[2, 3, 4])

    @pytest.mark.parametrize("nbar", [1, [2, 3], [4, 4]])
    def test_representation(self, nbar):
        rep = Thermal([0, 1], nbar).representation
        exp = Bargmann(*thermal_state_Abc([nbar, nbar] if isinstance(nbar, int) else nbar))
        assert rep == exp

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Thermal(modes=[0], nbar=[0.1, 0.2]).representation


class TestVisualization:
    r"""
    Tests the functions to visualize states.
    """

    # set to ``True`` to regenerate the assets
    regenerate_assets = False

    # path
    path = os.path.dirname(__file__) + "/assets"

    def test_visualize_2d(self):
        st = Coherent([0], y=1) + Coherent([0], y=-1)
        fig = st.visualize_2d(resolution=20, xbounds=(-3, 3), pbounds=(-4, 4), return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path + "/visualize_2d.json", remove_uids=True)

        with open(self.path + "/visualize_2d.json") as file:
            ref_data = json.load(file)

        assert math.allclose(data["data"][0]["x"], ref_data["data"][0]["x"])
        assert math.allclose(data["data"][0]["y"], ref_data["data"][0]["y"])
        assert math.allclose(data["data"][0]["z"], ref_data["data"][0]["z"])
        assert math.allclose(data["data"][1]["x"], ref_data["data"][1]["x"])
        assert math.allclose(data["data"][1]["y"], ref_data["data"][1]["y"])
        assert math.allclose(data["data"][2]["x"], ref_data["data"][2]["x"])
        assert math.allclose(data["data"][2]["y"], ref_data["data"][2]["y"])

    def test_visualize_2d_error(self):
        with pytest.raises(ValueError):
            Coherent([0, 1]).visualize_2d(20)

    def test_visualize_3d(self):
        st = Coherent([0], y=1) + Coherent([0], y=-1)
        fig = st.visualize_3d(resolution=20, xbounds=(-3, 3), pbounds=(-4, 4), return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path + "/visualize_3d.json", remove_uids=True)

        with open(self.path + "/visualize_3d.json") as file:
            ref_data = json.load(file)

        assert math.allclose(data["data"][0]["x"], ref_data["data"][0]["x"])
        assert math.allclose(data["data"][0]["y"], ref_data["data"][0]["y"])
        assert math.allclose(data["data"][0]["z"], ref_data["data"][0]["z"])

    def test_visualize_3d_error(self):
        with pytest.raises(ValueError):
            Coherent([0, 1]).visualize_3d(20)

    def test_visualize_dm(self):
        st = Coherent([0], y=1) + Coherent([0], y=-1)
        fig = st.visualize_dm(20, return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path + "/visualize_dm.json", remove_uids=True)

        with open(self.path + "/visualize_dm.json") as file:
            ref_data = json.load(file)

        assert math.allclose(data["data"][0]["z"], ref_data["data"][0]["z"])

    def test_visualize_dm_error(self):
        with pytest.raises(ValueError):
            Coherent([0, 1]).visualize_dm(20)


class TestSauron:
    r"""
    Tests the Sauron states.
    """

    def test_overlap_with_fock(self):
        n1 = Number([0], n=1)
        s1 = Sauron([0], n=1, r=0.1)
        s1b = Sauron([0], n=1, r=0.5)
        assert abs(s1.expectation(n1)) > abs(
            s1b.expectation(n1)
        )  # s1 is a better approx to n1 than s1b

        n2 = Number([0], n=2)
        s2 = Sauron([0], n=2, r=0.1)
        assert np.isclose(s2.expectation(n2), 1.0)
