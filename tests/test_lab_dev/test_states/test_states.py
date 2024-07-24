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
from pathlib import Path
import numpy as np
import pytest


from mrmustard import math, settings
from mrmustard.physics.representations import Bargmann
from mrmustard.physics.fock import fock_state
from mrmustard.physics.triples import thermal_state_Abc
from mrmustard.lab_dev.states import (
    Coherent,
    DisplacedSqueezed,
    Number,
    SqueezedVacuum,
    TwoModeSqueezedVacuum,
    Thermal,
    Vacuum,
)
from mrmustard.lab_dev.transformations import Dgate, Sgate, S2gate


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
        with pytest.raises(ValueError, match="x"):
            Coherent(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="y"):
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

    def test_vacuum_shape(self):
        assert Coherent([0], x=0.0, y=0.0).auto_shape() == (1,)


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
        with pytest.raises(ValueError, match="x"):
            DisplacedSqueezed(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="y"):
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
    cutoffs = [None, 5, [6, 7]]

    @pytest.mark.parametrize("modes,n,cutoffs", zip(modes, n, cutoffs))
    def test_init(self, modes, n, cutoffs):
        state = Number(modes, n, cutoffs)

        assert state.name == "N"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="n"):
            Number(modes=[0, 1], n=[2, 3, 4])

        with pytest.raises(ValueError, match="cutoffs"):
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
        with pytest.raises(ValueError, match="r"):
            SqueezedVacuum(modes=[0, 1], r=[2, 3, 4])

        with pytest.raises(ValueError, match="phi"):
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
        with pytest.raises(ValueError, match="r"):
            TwoModeSqueezedVacuum(modes=[0, 1], r=[2, 3, 4])

        with pytest.raises(ValueError, match="phi"):
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
        with pytest.raises(ValueError, match="nbar"):
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
    path = Path(__file__).parent.parent / "assets"

    def test_visualize_2d(self):
        st = Coherent([0], y=1) + Coherent([0], y=-1)
        fig = st.visualize_2d(resolution=20, xbounds=(-3, 3), pbounds=(-4, 4), return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path / "visualize_2d.json", remove_uids=True)

        with open(self.path / "visualize_2d.json") as file:
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
            fig.write_json(self.path / "visualize_3d.json", remove_uids=True)

        with open(self.path / "visualize_3d.json") as file:
            ref_data = json.load(file)

        assert math.allclose(data["data"][0]["x"], ref_data["data"][0]["x"])
        assert math.allclose(data["data"][0]["y"], ref_data["data"][0]["y"])
        assert math.allclose(data["data"][0]["z"], ref_data["data"][0]["z"])

    def test_visualize_3d_error(self):
        with pytest.raises(ValueError):
            Coherent([0, 1]).visualize_3d(20)

    def test_visualize_dm(self):
        st = Coherent([0], y=1) + Coherent([0], y=-1)
        st.manual_shape[0] = 20
        fig = st.visualize_dm(20, return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path / "visualize_dm.json", remove_uids=True)

        with open(self.path / "visualize_dm.json") as file:
            ref_data = json.load(file)
        assert math.allclose(data["data"][0]["z"], ref_data["data"][0]["z"])

    def test_visualize_dm_error(self):
        with pytest.raises(ValueError):
            Coherent([0, 1]).visualize_dm(20)
