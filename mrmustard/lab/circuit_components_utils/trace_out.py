# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The class representing a trace out operation."""

from __future__ import annotations

from numpy.typing import ArrayLike

from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.utils.typing import Scalar

from ...physics.wires import ReprEnum, Wires
from ..circuit_components import CircuitComponent
from ..transformations.builtins import identity_gate

__all__ = ["TraceOut"]


class TraceOut(CircuitComponent):
    r"""A circuit component to perform trace-out operations.

    It has input wires on both the ket and bra sides, but no output wires. Its representation is
    the same as that of the identity channel.

    >>> from mrmustard.lab import *
    >>> from mrmustard import math
    >>> # initialize a multi-mode state
    >>> state = Coherent(0, alpha=1) >> Coherent(1, alpha=1) >> Coherent(2, alpha=1)
    >>> # trace out some of the modes
    >>> assert state >> TraceOut(0) == (Coherent(1, alpha=1) >> Coherent(2, alpha=1)).dm()
    >>> assert state >> TraceOut((1, 2)) == Coherent(0, alpha=1).dm()
    >>> # use the trace out to estimate expectation values of operators
    >>> op = Dgate(0, alpha=1)
    >>> expectation = state.dm().contract(op) >> TraceOut((0, 1, 2))
    >>> assert math.allclose(expectation, state.expectation(op))

    Args:
        modes: The modes to trace out.
    """

    def __init__(
        self,
        modes: int | tuple[int, ...],
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (identity_gate, ("n_modes", "lin_sup"))},
                n_modes=len(modes),
            ),
            wires=Wires(set(), set(modes), set(), set(modes)),
            name="Tr",
        )

    def __custom_rrshift__(
        self, other: CircuitComponent | Scalar | ArrayLike
    ) -> CircuitComponent | Scalar | ArrayLike:
        r"""A custom ``>>`` operator for the ``TraceOut`` component.
        It allows ``TraceOut`` to carry the method that processes ``other >> TraceOut``.
        We know that the trace in Bargmann is a Gaussian integral, and in
        Fock it's a trace (rather than an inner product with the identity).
        So we write two shortcuts here, and ``__rrshift__`` will be called first if
        present in the ``__rshift__`` method of the first object (``other`` here).
        """
        from ..states import DM, Ket  # noqa: PLC0415

        wires_to_trace = other.wires.output[self.wires.modes]
        if len(wires_to_trace) == 0:
            return other
        if not wires_to_trace.ket or not wires_to_trace.bra:
            try:
                wires_out, _ = (other.adjoint.wires + other.wires) @ self.wires
                ansatz = other.ansatz.conj.contract(
                    other.ansatz, (wires_to_trace.indices, wires_to_trace.indices)
                )
            except ValueError as e:
                raise ValueError("Component wires are incompatible with TraceOut.") from e
        else:
            ansatz = other.ansatz.trace(wires_to_trace.bra.indices, wires_to_trace.ket.indices)
            wires_out, _ = other.wires @ self.wires

        if len(wires_out) == 0:
            return ansatz.scalar

        if isinstance(other, (DM, Ket)):
            return DM._from_attributes(ansatz, wires_out)

        return CircuitComponent._from_attributes(ansatz, wires_out)
