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

"""This module contains tests for the tensors.py module."""

from mrmustard.math.tensor_networks import *

import numpy as np
import pytest


class TBad(Tensor):
    r"""
    A tensor without value.
    """
    pass


class TId(Tensor):
    r"""
    A tensor whose value is the identity matrix of size ``cutoff``.
    """

    def value(self, cutoff):
        return np.eye(cutoff)


class TComplex(Tensor):
    r"""
    A tensor whose value is a complex matrix of size ``[1, cutoff]``.
    """

    def value(self, cutoff):
        return np.arange(cutoff) + 1j * np.arange(2, cutoff + 2)


class TestWire:
    r"""
    Tests the Wire class.
    """

    @pytest.mark.parametrize("is_input", [True, False])
    @pytest.mark.parametrize("is_ket", [True, False])
    def test_init(self, is_input, is_ket):
        r"""
        Tests the init of wires.
        """
        id = 123
        mode = 5
        wire = Wire(id, mode, is_input, is_ket)

        assert wire.id == id
        assert wire.mode == mode
        assert wire.is_input is is_input
        assert wire.is_ket is is_ket
        assert wire.is_connected is False
        assert isinstance(wire.contraction_id, int)


class TestTensor:
    r"""
    Tests the Tensor class.
    """

    @pytest.mark.parametrize("modes_in_ket", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_ket", [None, [4, 5, 6]])
    @pytest.mark.parametrize("modes_in_bra", [None, [7, 8]])
    @pytest.mark.parametrize("modes_out_bra", [None, [9]])
    def test_init(self, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra):
        r"""
        Tests the init of tensors.
        """
        name = "t"
        t = TId(name, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

        assert t.name == name

        assert len(t.input.ket.items()) == 0 if modes_in_ket is None else len(modes_in_ket)
        assert len(t.output.ket.items()) == 0 if modes_out_ket is None else len(modes_out_ket)
        assert len(t.input.bra.items()) == 0 if modes_in_bra is None else len(modes_in_bra)
        assert len(t.output.bra.items()) == 0 if modes_out_bra is None else len(modes_out_bra)

    def test_adjoint(self):
        t = TComplex("t", [1], [2], [3])
        t_adj = t.adjoint

        cutoff = 12
        assert np.allclose(t.value(cutoff), t_adj.value(cutoff).T)
        assert t.input.ket.keys() == t_adj.input.bra.keys()
        assert t.input.bra.keys() == t_adj.input.ket.keys()
        assert t.output.ket.keys() == t_adj.output.bra.keys()
        assert t.output.bra.keys() == t_adj.output.ket.keys()

    def test_modes_in_out(self):
        t1 = TId("t", [1], [2])
        assert t1.modes_in == [1]
        assert t1.modes_out == [2]

        t1 = TId("t", [1], [2], [1], [2])
        assert t1.modes_in == [1]
        assert t1.modes_out == [2]

        t1 = TId("t", [3], [1], [2], [4])
        assert t1.modes_in == list(set([3, 2]))
        assert t1.modes_out == list(set([1, 4]))

    @pytest.mark.parametrize("modes_in_ket", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_ket", [None, [4, 5, 6]])
    @pytest.mark.parametrize("modes_in_bra", [None, [7, 8]])
    @pytest.mark.parametrize("modes_out_bra", [None, [9]])
    def test_wires(self, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra):
        r"""
        Tests the init of tensors.
        """
        name = "t"
        t = TId(name, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)
        wires = np.array(t.wires)

        list_modes = [] if modes_in_ket is None else modes_in_ket
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == len(list_modes)

        list_modes = [] if modes_out_ket is None else modes_out_ket
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == len(list_modes)

        list_modes = [] if modes_in_bra is None else modes_in_bra
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == len(list_modes)

        list_modes = [] if modes_out_bra is None else modes_out_bra
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == len(list_modes)

    @pytest.mark.parametrize("cutoff", [1, 2, 10])
    def test_value(self, cutoff):
        r"""
        Tests the value property.
        """
        assert np.allclose(TId("t").value(cutoff), np.eye(cutoff))

        with pytest.raises(TypeError, match="abstract method value"):
            TBad("t_bad")
