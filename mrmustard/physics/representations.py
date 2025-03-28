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

"""
This module contains the class for representations.
"""

from __future__ import annotations
from typing import Sequence
from mrmustard import math
from mrmustard.utils.typing import (
    ComplexTensor,
    ComplexMatrix,
    ComplexVector,
    Batch,
)

from .ansatz import Ansatz, PolyExpAnsatz, ArrayAnsatz
from .triples import identity_Abc
from .utils import outer_product_batch_str, zip_batch_strings
from .wires import Wires, ReprEnum

__all__ = ["Representation"]


class Representation:
    r"""
    A class for representations.

    A representation handles the underlying ansatz, wires and keeps track
    of each wire's representation.

    The dictionary to keep track of representations maps the indices of the wires
    to a tuple of the form ``(ReprEnum, parameter)``.

    Args:
        ansatz: An ansatz for this representation.
        wires: The wires of this representation.
    """

    def __init__(self, ansatz: Ansatz | None = None, wires: Wires | None = None) -> None:
        self._ansatz = ansatz
        self._wires = wires or Wires(set(), set(), set(), set())

    @property
    def adjoint(self) -> Representation:
        r"""
        The adjoint of this representation obtained by conjugating the ansatz and swapping
        the ket and bra wires.
        """
        bras = self.wires.bra.indices
        kets = self.wires.ket.indices
        ansatz = self.ansatz.reorder(kets + bras).conj if self.ansatz else None
        return Representation(ansatz, self.wires.adjoint)

    @property
    def ansatz(self) -> Ansatz | None:
        r"""
        The underlying ansatz of this representation.
        """
        return self._ansatz

    @property
    def dual(self) -> Representation:
        r"""
        The dual of this representation obtained by conjugating the ansatz and swapping
        the input and output wires.
        """
        ok = self.wires.ket.output.indices
        ik = self.wires.ket.input.indices
        ib = self.wires.bra.input.indices
        ob = self.wires.bra.output.indices
        ansatz = self.ansatz.reorder(ib + ob + ik + ok).conj if self.ansatz else None
        return Representation(ansatz, self.wires.dual)

    @property
    def wires(self) -> Wires | None:
        r"""
        The wires of this representation.
        """
        return self._wires

    def bargmann_triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The Bargmann parametrization of this representation, if available.
        It returns a triple (A, b, c) such that the Bargmann function of this is
        :math:`F(z) = c \exp\left(\frac{1}{2} z^T A z + b^T z\right).
        """
        try:
            return self.ansatz.triple
        except AttributeError as e:
            raise AttributeError("No Bargmann data for this component.") from e

    def contract(self, other: Representation, mode: str = "kron"):
        r"""
        Contracts two representations.

        Args:
            other: The other representation to contract with.
            mode: "zip" the batch dimensions, "kron" the batch dimensions
                or pass a custom batch string.
        """
        wires_result, perm = self.wires @ other.wires
        idx_z, idx_zconj = self.wires.contracted_indices(other.wires)

        if type(self.ansatz) is type(other.ansatz):
            self_ansatz = self.ansatz
            other_ansatz = other.ansatz
        else:
            self_ansatz = self.to_bargmann().ansatz
            other_ansatz = other.to_bargmann().ansatz

        if mode == "zip":
            eins_str = zip_batch_strings(
                len(self_ansatz.batch_shape), len(other_ansatz.batch_shape)
            )
        elif mode == "kron":
            eins_str = outer_product_batch_str(self_ansatz.batch_shape, other_ansatz.batch_shape)
        ansatz = self_ansatz.contract(other_ansatz, batch_str=eins_str, idx1=idx_z, idx2=idx_zconj)
        ansatz = ansatz.reorder(perm) if perm else ansatz
        return Representation(ansatz, wires_result)

    def fock_array(self, shape: int | Sequence[int]) -> ComplexTensor:
        r"""
        Returns an array of this representation in the Fock basis with the given shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component if it is
        available, otherwise it defaults to the value of ``AUTOSHAPE_MAX`` in the settings.

        Args:
            shape: The shape of the returned array, not including batch dimensions. If ``shape`` is
            given as an ``int``, it is broadcast to all the dimensions. If not given, it is estimated.
        Returns:
            array: The Fock array of this representation.
        """
        num_vars = (
            self.ansatz.num_CV_vars
            if isinstance(self.ansatz, PolyExpAnsatz)
            else self.ansatz.num_vars
        )
        if isinstance(shape, int):
            shape = (shape,) * num_vars
        shape = tuple(shape)
        if len(shape) != num_vars:
            raise ValueError(f"Expected Fock shape of length {num_vars}, got {len(shape)}")
        try:
            A, b, c = self.ansatz.triple

            As = (
                math.reshape(A, (-1, *A.shape[-2:])) if self.ansatz.batch_shape != () else A
            )  # tensorflow
            bs = (
                math.reshape(b, (-1, *A.shape[-1:])) if self.ansatz.batch_shape != () else b
            )  # tensorflow
            cs = (
                math.reshape(c, (-1, *c.shape[self.ansatz.batch_dims :]))
                if self.ansatz.batch_shape != ()  # tensorflow
                else c
            )

            batch = (self.ansatz.batch_size,) if self.ansatz.batch_shape != () else ()  # tensorflow

            if self.ansatz.batch_shape != ():  # tensorflow
                G = math.astensor(
                    [
                        math.hermite_renormalized(A, b, complex(1), shape=shape + cs.shape[1:])
                        for A, b in zip(As, bs)
                    ]
                )
            else:
                G = math.hermite_renormalized(As, bs, complex(1), shape=shape + cs.shape)

            G = math.reshape(G, batch + shape + (-1,))
            cs = math.reshape(cs, batch + (-1,))
            core_str = "".join(
                [chr(i) for i in range(97, 97 + len(G.shape[1:] if batch else G.shape))]
            )
            ret = math.einsum(f"...{core_str},...{core_str[-1]}->...{core_str[:-1]}", G, cs)
        except AttributeError:
            ret = self.ansatz.reduce(shape).array
        return math.reshape(ret, self.ansatz.batch_shape + shape)

    def to_bargmann(self) -> Representation:
        r"""
        Converts this representation to a Bargmann representation.
        """
        if isinstance(self.ansatz, PolyExpAnsatz):
            return self
        else:
            if self.ansatz._original_abc_data:
                A, b, c = self.ansatz._original_abc_data
            else:
                A, b, _ = identity_Abc(len(self.wires.quantum))
                c = self.ansatz.data
            bargmann = PolyExpAnsatz(A, b, c)
            for w in self.wires.quantum:
                w.repr = ReprEnum.BARGMANN
            return Representation(bargmann, self.wires)

    def to_fock(self, shape: int | Sequence[int]) -> Representation:
        r"""
        Converts this representation to a Fock representation.

        Args:
            shape: The shape of the returned representation. If ``shape``is given as
                an ``int``, it is broadcasted to all the dimensions. If ``None``, it
                defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        """
        fock = ArrayAnsatz(self.fock_array(shape), batch_dims=self.ansatz.batch_dims)
        try:
            if self.ansatz.num_derived_vars == 0:
                fock._original_abc_data = self.ansatz.triple
        except AttributeError:
            fock._original_abc_data = None
        wires = self.wires.copy()
        for w in wires.quantum_wires:
            w.repr = ReprEnum.FOCK
        return Representation(fock, wires)

    def __eq__(self, other):
        if isinstance(other, Representation):
            return self.ansatz == other.ansatz and self.wires == other.wires
        return False
