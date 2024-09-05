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

""" Classes for constructing tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import uuid

from mrmustard.math.backend_manager import BackendManager

math = BackendManager()


__all__ = ["Wire", "Tensor"]


def random_int() -> int:
    r"""
    Returns a random integer obtained from a UUID
    """
    return uuid.uuid1().int


# pylint: disable=too-many-instance-attributes
@dataclass
class Wire:
    r"""Represents a wire in a tensor network.

    Each wire is characterized by a unique identifier ``id``, which must be different from
    the identifiers of all the other wires in the tensor network. Additionally, it owns a
    label ``mode`` that represents the mode of light that this wire is acting on.

    Args:
        id: A numerical identifier for this wire.
        mode: The mode of light that this wire is acting on.
        is_input: Whether this wire is an input to a tensor or an output.
        is_ket: Whether this wire is on the ket or on the bra side.

    """

    id: int
    mode: int
    is_input: bool
    is_ket: bool

    def __post_init__(self):
        self._contraction_id: int = random_int()
        self._dim = None
        self._is_connected = False

    @property
    def contraction_id(self) -> int:
        r"""
        A numerical identifier for the contraction involving this wire.
        """
        return self._contraction_id

    @contraction_id.setter
    def contraction_id(self, value: int):
        self._contraction_id = value

    @property
    def dim(self):
        r"""
        The dimension of this wire.
        """
        return self._dim

    @dim.setter
    def dim(self, value: int):
        if self._dim:
            raise ValueError("Cannot change the dimension of wire with specified dimension.")
        self._dim = value

    @property
    def is_connected(self) -> bool:
        r"""
        Whether or not this wire is connected with another wire.
        """
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value


@dataclass
class WireGroup:
    r"""A group of wires in a tensor network.

    Args:
        ket: A dictionary containing the wires on the ket side.
        bra: A dictionary containing the wires on the bra side.

    """

    ket: dict = field(default_factory=dict)
    bra: dict = field(default_factory=dict)


class Tensor(ABC):
    r"""An abstract class representing a tensor in a tensor network.

    In Mr Mustard, tensors are used to represent a state or a transformation on a given set
    of modes in the Fock representation. For example, a single-mode unitary matrix
    :math:`U=\sum_{i,j=1}^Nu_{i,j}|i\rangle\langle{j}|` acting on mode ``3`` in an
    N-dimensional Fock basis corresponds to the following ``Tensor`` object:

    .. code-block::
        class U(Tensor):
            def value(self, shape):
                # specify the value of the tensor
                pass

        U("my_unitary", [3], [3], [3], [3])

    Args:
        name: The name of this tensor.
        modes_in_ket: The input modes on the ket side.
        modes_out_ket: The output modes on the ket side.
        modes_in_bra: The input modes on the bra side.
        modes_out_bra: The output modes on the bra side.
    """

    _repr_markdown_ = None  # otherwise it takes over the repr due to mro

    def __init__(
        self,
        name: str,
        modes_in_ket: list[int] | None = None,
        modes_out_ket: list[int] | None = None,
        modes_in_bra: list[int] | None = None,
        modes_out_bra: list[int] | None = None,
    ) -> None:
        self._name = name
        self._update_modes(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

    def _update_modes(
        self,
        modes_in_ket: list[int] | None = None,
        modes_out_ket: list[int] | None = None,
        modes_in_bra: list[int] | None = None,
        modes_out_bra: list[int] | None = None,
    ) -> None:
        r"""
        Updates the modes in this tensor by setting:

          * self._modes_in_ket, a list of input modes on the ket side
          * self._modes_out_ket, a list of output modes on the ket side
          * self._modes_in_bra, a list of input modes on the bra side
          * self._modes_out_bra, a list of output modes on the bra side
          * self.self._input, a WireGroup containing all the input modes
          * self.self._output, a WireGroup containing all the output modes

        It computes a new ``id`` for every wire.

        Raises:
            ValueError: if `modes_in_ket` and `modes_in_bra` are not equal, and neither
                of them is `None`.
            ValueError: if `modes_out_ket` and `modes_out_bra` are not equal, and neither
                of them is `None`.
        """
        msg = "modes on ket and bra sides must be equal, unless either of them is `None`."
        if modes_in_ket and modes_in_bra:
            if modes_in_ket != modes_in_bra:
                msg = f"Input {msg}"
                raise ValueError(msg)
        if modes_out_ket and modes_out_bra:
            if modes_out_ket != modes_out_bra:
                msg = f"Output {msg}"
                raise ValueError(msg)

        self._modes_in_ket = modes_in_ket if modes_in_ket else []
        self._modes_out_ket = modes_out_ket if modes_out_ket else []
        self._modes_in_bra = modes_in_bra if modes_in_bra else []
        self._modes_out_bra = modes_out_bra if modes_out_bra else []

        # initialize ket and bra wire dicts
        self._input = WireGroup()
        for mode in self._modes_in_ket:
            self._input.ket |= {mode: Wire(random_int(), mode, True, True)}
        for mode in self._modes_in_bra:
            self._input.bra |= {mode: Wire(random_int(), mode, True, False)}

        self._output = WireGroup()
        for mode in self._modes_out_ket:
            self._output.ket |= {mode: Wire(random_int(), mode, False, True)}
        for mode in self._modes_out_bra:
            self._output.bra |= {mode: Wire(random_int(), mode, False, False)}

    @property
    def adjoint(self) -> AdjointView:
        r"""The adjoint view of this Tensor (with new ``id``s). That is, ket <-> bra."""
        return AdjointView(self)

    @property
    def input(self):
        r"""
        A dictionary mapping the input modes to their respective wires.
        """
        return self._input

    @property
    def modes(self) -> list[int]:
        r"""
        For backward compatibility. Don't overuse.
        It returns a list of modes for this Tensor, unless it's ambiguous.
        """
        if self.modes_in == self.modes_out:  # transformation on same modes
            return list(self.modes_in)
        elif len(self.modes_in) == 0:  # state
            return list(self.modes_out)
        elif len(self.modes_out) == 0:  # measurement
            return list(self.modes_in)
        else:
            raise ValueError("modes are ambiguous for this Tensor.")

    @property
    def modes_in(self) -> list[int]:
        r"""
        The list of input modes that are used by this Tensor.

        If this tensor has no input modes on the bra side, or if the input modes are equal
        on both ket and bra sides, it returns the list of modes. Otherwise, it performs the
        ``set()`` operation before returning the list (and hence, the order may be unexpected).
        """
        if self._modes_in_ket:
            return self._modes_in_ket
        return self._modes_in_bra

    @property
    def modes_out(self) -> list[int]:
        r"""
        The list of output modes that are used by this Tensor.

        If this tensor has no output modes on the bra side, or if the output modes are equal
        on both ket and bra sides, it returns the list of modes. Otherwise, it performs the
        ``set()`` operation before returning the list (and hence, the order may be unexpected).
        """
        if self._modes_out_ket:
            return self._modes_out_ket
        return self._modes_out_bra

    @property
    def name(self) -> int:
        r"""
        The name of this tensor.
        """
        return self._name

    @property
    def output(self):
        r"""
        A dictionary mapping the output modes to their respective wires.
        """
        return self._output

    def unpack_shape(self, shape: tuple[int]):
        r"""
        Unpack the given ``shape`` into the shapes of the input and output wires on ket and bra sides.

        Args:
            shape: A shape.

        Returns:
            shape_in_ket: The shape of the input wires on the ket side.
            shape_out_ket: The shape of the output wires on the ket side.
            shape_in_bra: The shape of the input wires on the bra side.
            shape_out_bra: The shape of the output wires on the bra side.
        """
        idx1 = 0
        idx2 = len(self._modes_in_ket)
        shape_in_ket = shape[idx1:idx2]

        idx1 = idx2
        idx2 += len(self._modes_out_ket)
        shape_out_ket = shape[idx1:idx2]

        idx1 = idx2
        idx2 += len(self._modes_in_bra)
        shape_in_bra = shape[idx1:idx2]

        idx1 = idx2
        idx2 += len(self._modes_out_bra)
        shape_out_bra = shape[idx1:idx2]

        return shape_in_ket, shape_out_ket, shape_in_bra, shape_out_bra

    @property
    def wires(self) -> list[Wire]:
        r"""
        The list of all wires in this tensor, sorted as ``[ket_in, ket_out, bra_in, bra_out]``.
        """
        return (
            list(self.input.ket.values())
            + list(self.output.ket.values())
            + list(self.input.bra.values())
            + list(self.output.bra.values())
        )

    @abstractmethod
    def value(self, shape: tuple[int]):
        r"""The value of this tensor.

        Args:
            shape: the shape of this tensor

        Returns:
            ComplexTensor: the unitary matrix in Fock representation
        """

    def change_modes(
        self,
        modes_in_ket: list[int] | None = None,
        modes_out_ket: list[int] | None = None,
        modes_in_bra: list[int] | None = None,
        modes_out_bra: list[int] | None = None,
    ) -> None:
        r"""
        Changes the modes in this tensor.

        Args:
            name: The name of this tensor.
            modes_in_ket: The input modes on the ket side.
            modes_out_ket: The output modes on the ket side.
            modes_in_bra: The input modes on the bra side.
            modes_out_bra: The output modes on the bra side.

        Raises:
            ValueError: if one or more wires in this tensor are already connected.
        """
        for wire in self.wires:
            if wire.is_connected:
                msg = (
                    "Cannot change nodes in a tensor when some of its wires are already connected."
                )
                raise ValueError(msg)
        self._update_modes(modes_in_ket, modes_out_ket, modes_in_bra, modes_out_bra)

    def shape(self, default_dim: int | None = None, out_in=False):
        r"""
        Returns the shape of the underlying tensor, as inferred from the dimensions of the individual
        wires.

        If ``out_in`` is ``False``, the shape returned is in the order ``(in_ket, in_bra, out_ket, out_bra)``.
        Otherwise, it is in the order ``(out_ket, out_bra, in_ket, in_bra)``.

        Args:
            default_dim: The default dimension of wires with unspecified dimension.
            out_in: Whether to return output shapes followed by input shapes or viceversa.
        """

        def _sort_shapes(*args):
            for arg in args:
                if arg:
                    yield arg

        shape_in_ket = [w.dim if w.dim else default_dim for w in self.input.ket.values()]
        shape_out_ket = [w.dim if w.dim else default_dim for w in self.output.ket.values()]
        shape_in_bra = [w.dim if w.dim else default_dim for w in self.input.bra.values()]
        shape_out_bra = [w.dim if w.dim else default_dim for w in self.output.bra.values()]

        if out_in:
            ret = _sort_shapes(shape_out_ket, shape_out_bra, shape_in_ket, shape_in_bra)
        ret = _sort_shapes(shape_in_ket, shape_in_bra, shape_out_ket, shape_out_bra)

        # pylint: disable=consider-using-generator
        return tuple([item for sublist in ret for item in sublist])


class AdjointView(Tensor):
    r"""
    Adjoint view of a tensor. It swaps the ket and bra wires of a Tensor.
    """

    def __init__(self, tensor):
        self._original = tensor
        super().__init__(
            name=self._original.name,
            modes_in_ket=self._original.input.bra.keys(),
            modes_out_ket=self._original.output.bra.keys(),
            modes_in_bra=self._original.input.ket.keys(),
            modes_out_bra=self._original.output.ket.keys(),
        )

    def value(self, shape: tuple[int]):
        r"""The value of this tensor.

        Args:
            shape: the shape of the adjoint tensor.

        Returns:
            ComplexTensor: the unitary matrix in Fock representation
        """
        # converting the given shape into a shape for the original tensor
        (
            shape_in_ket,
            shape_out_ket,
            shape_in_bra,
            shape_out_bra,
        ) = self._original.unpack_shape(shape)
        shape_ret = shape_in_bra + shape_out_bra + shape_in_ket + shape_out_ket

        ret = math.conj(math.astensor(self._original.value(shape_ret)))
        return ret


class DualView(Tensor):
    r"""
    Dual view of a tensor. It swaps the input and output wires of a Tensor.
    """

    def __init__(self, tensor):
        self._original = tensor
        super().__init__(
            name=self._original.name,
            modes_in_ket=self._original.output.ket.keys(),
            modes_out_ket=self._original.input.ket.keys(),
            modes_in_bra=self._original.output.bra.keys(),
            modes_out_bra=self._original.input.bra.keys(),
        )

    def value(self, shape: tuple[int]):
        r"""The value of this tensor.

        Args:
            shape: the shape of the dual tensor.

        Returns:
            ComplexTensor: the unitary matrix in Fock representation.
        """
        # converting the given shape into a shape for the original tensor
        shape_in_ket, shape_out_ket, shape_in_bra, shape_out_bra = self.unpack_shape(shape)
        shape_ret = shape_out_ket + shape_in_ket + shape_out_bra, shape_in_bra

        return math.conj(self._original.value(shape_ret))
