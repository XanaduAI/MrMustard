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

from mrmustard.math.tensor_networks import Wire, Tensor, connect

import numpy as np
import pytest

# ~~~~~~~
# Helpers
# ~~~~~~~


class TBad(Tensor):
    r"""
    A tensor without value.
    """


class TComplex(Tensor):
    r"""
    A tensor whose value is a complex matrix of given shape.
    """

    def value(self, shape):
        return np.random.rand(*shape) + 1j * np.random.rand(*shape)


# ~~~~~~~
# Tests
# ~~~~~~~


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
        assert wire.dim is None
        assert isinstance(wire.contraction_id, int)

    def test_dim_error(self):
        r"""
        Tests that ``dim`` cannot be set more than once.
        """
        wire = Wire(0, 0, True, True)
        wire.dim = 18

        with pytest.raises(ValueError, match="Cannot change"):
            wire.dim = 20


class TestTensor:
    r"""
    Tests the Tensor class.
    """

    @pytest.mark.parametrize("modes_in_ket", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_ket", [None, [4]])
    @pytest.mark.parametrize("modes_in_bra", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_bra", [None, [4]])
    def test_init(self, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra):
        r"""
        Tests the init of tensors.
        """
        name = "t"
        t = TComplex(name, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

        assert t.name == name

        assert (
            len(t.input.ket.items()) == 0 if modes_in_ket is None else len(modes_in_ket)
        )
        assert (
            len(t.output.ket.items()) == 0
            if modes_out_ket is None
            else len(modes_out_ket)
        )
        assert (
            len(t.input.bra.items()) == 0 if modes_in_bra is None else len(modes_in_bra)
        )
        assert (
            len(t.output.bra.items()) == 0
            if modes_out_bra is None
            else len(modes_out_bra)
        )

    @pytest.mark.parametrize("modes_in_ket", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_ket", [None, [4]])
    @pytest.mark.parametrize("modes_in_bra", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_bra", [None, [4]])
    def test_ids_in_same_tensor(
        self, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra
    ):
        r"""
        Tests that tensors generate wires with different ``id``s.
        """
        t = TComplex("t", modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

        all_ids = [w.id for w in t.input.ket.values()]
        all_ids += [w.id for w in t.output.ket.values()]
        all_ids += [w.id for w in t.input.bra.values()]
        all_ids += [w.id for w in t.output.bra.values()]

        assert len(all_ids) == len(set(all_ids))

    @pytest.mark.parametrize("modes_in_ket", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_ket", [None, [4]])
    @pytest.mark.parametrize("modes_in_bra", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_bra", [None, [4]])
    def test_ids_in_different_tensor(
        self, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra
    ):
        r"""
        Tests that different tensors generate wires with different ``id``s.
        """
        t1 = TComplex("t1", modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)
        t2 = TComplex("t2", modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

        all_ids1 = [w.id for w in t1.input.ket.values()]
        all_ids1 += [w.id for w in t1.output.ket.values()]
        all_ids1 += [w.id for w in t1.input.bra.values()]
        all_ids1 += [w.id for w in t1.output.bra.values()]

        all_ids2 = [w.id for w in t2.input.ket.values()]
        all_ids2 += [w.id for w in t2.output.ket.values()]
        all_ids2 += [w.id for w in t2.input.bra.values()]
        all_ids2 += [w.id for w in t2.output.bra.values()]

        assert len(all_ids1 + all_ids2) == len(set(all_ids1 + all_ids2))

    def test_adjoint(self):
        r"""
        Tests the adjoint method.
        """
        t = TComplex("t", [1, 2], [2, 3])
        t_adj = t.adjoint

        shape = (3, 4, 8, 1)

        assert t_adj.value(shape).shape == shape
        assert t.input.ket.keys() == t_adj.input.bra.keys()
        assert t.input.bra.keys() == t_adj.input.ket.keys()
        assert t.output.ket.keys() == t_adj.output.bra.keys()
        assert t.output.bra.keys() == t_adj.output.ket.keys()

    def test_modes_in_out(self):
        r"""
        Tests the modes_in and modes_out methods.
        """
        t1 = TComplex("t", [1], [2])
        assert t1.modes_in == [1]
        assert t1.modes_out == [2]

        t2 = TComplex("t", [1], [2], [1], [2])
        assert t2.modes_in == [1]
        assert t2.modes_out == [2]

        t3 = TComplex("t", [], [], [3], [4])
        assert t3.modes_in == [3]
        assert t3.modes_out == [4]

    @pytest.mark.parametrize("modes_in_ket", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_ket", [None, [4]])
    @pytest.mark.parametrize("modes_in_bra", [None, [1, 2, 3]])
    @pytest.mark.parametrize("modes_out_bra", [None, [4]])
    def test_wires(self, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra):
        r"""
        Tests the init of tensors.
        """
        name = "t"
        t = TComplex(name, modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)
        wires = np.array(t.wires)

        list_modes = [] if modes_in_ket is None else modes_in_ket
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == 0 or len(modes_in_ket)

        list_modes = [] if modes_out_ket is None else modes_out_ket
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == 0 or len(modes_out_ket)

        list_modes = [] if modes_in_bra is None else modes_in_bra
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == 0 or len(modes_in_bra)

        list_modes = [] if modes_out_bra is None else modes_out_bra
        mask = [w.mode in list_modes for w in wires]
        assert len(wires[mask]) == 0 or len(modes_out_bra)

    def test_value_error(self):
        r"""
        Tests the error for the value property.
        """
        with pytest.raises(TypeError, match="abstract method value"):
            TBad("t_bad")

    def test_change_modes(self):
        r"""
        Tests the function to change modes.
        """
        t = TComplex("t")

        modes_in_ket = [1]
        modes_out_ket = [2, 3]
        t.change_modes(modes_in_ket, modes_out_ket)

        assert list(t.input.ket.keys()) == modes_in_ket
        assert not t.input.bra
        assert list(t.output.ket.keys()) == modes_out_ket
        assert not t.output.bra

    def test_change_modes_errors(self):
        r"""
        Tests the errors of the function to change modes.
        """
        t1 = TComplex("t1", [1])
        t2 = TComplex("t2", None, [1])

        with pytest.raises(ValueError, match="Input modes"):
            t1.change_modes([2], None, [3])

        with pytest.raises(ValueError, match="Output modes"):
            t1.change_modes(None, [2], None, [1])

        connect(t1.input.ket[1], t2.output.ket[1], 1)
        with pytest.raises(ValueError, match="already connected"):
            t1.change_modes([2])
