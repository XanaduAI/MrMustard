from __future__ import annotations

from typing import Optional

from mrmustard.lab.abstract.circuitpart import CircuitPart, Wire
from mrmustard.lab.circuit import Circuit
from mrmustard.typing import Tensor
from mrmustard.utils.tagdispenser import TagDispenser


class Operation(CircuitPart):
    r"""A container for States, Transformations and Measurements that allows one to place them
    inside a circuit. It contains information about which modes in the circuit the operation
    is attached to via its wires. The Operation is an abstraction above the physics of the object
    that it contains. Its main purpose is to allow the user to easily construct circuits."""

    def __init__(
        self,
        modes_in: list[int],
        modes_out: list[int],
        has_dual: bool,
        name: str = None,
        **kwargs,
    ):
        self.name = name or self.__class__.__qualname__
        self.input_wire_at_mode: dict[int, Wire] = {
            m: Wire(
                end=self,
                L=TagDispenser().get_tag(),
                R=TagDispenser().get_tag() if has_dual else None,
            )
            for m in modes_in
        }
        self.output_wire_at_mode: dict[int, Wire] = {
            m: Wire(
                origin=self,
                L=TagDispenser().get_tag(),
                R=TagDispenser().get_tag() if has_dual else None,
            )
            for m in modes_out
        }
        self.has_dual = has_dual
        super().__init__(**kwargs)

    _repr_markdown_ = None

    def disconnect(self) -> Operation:
        "Re-issue new wires for this operation"
        for wire_dict in [self.input_wire_at_mode, self.output_wire_at_mode]:
            for m, wire in wire_dict.items():
                wire_dict[m] = Wire(
                    origin=None if wire.origin is not self else self,
                    end=None if wire.end is not self else self,
                    L=TagDispenser().get_tag(),
                    R=TagDispenser().get_tag() if self.has_dual else None,
                )
        return self

    @property
    def modes(self) -> Optional[list[int]]:
        "Returns the modes that this Operation is defined on"
        if self.modes_in == self.modes_out:
            return list(self.modes_in)
        elif len(self.modes_in) == 0:
            return list(self.modes_out)
        elif len(self.modes_out) == 0:
            return list(self.modes_in)
        else:
            raise ValueError("Operation is not defined on a single set of modes.")

    def enable_dual(self) -> None:
        "Enables the dual (R) part of all the wires of this Operation."
        for wire in self.all_wires:
            wire.enable_dual()

    def __hash__(self):  # is this needed?
        "hash function so that Operations can be used as keys in dictionaries."
        tags = tuple(tag for wire in self.all_wires for tag in [wire.L, wire.R] if tag is not None)
        return hash(tags)

    def __rshift__(self, other: CircuitPart) -> Circuit:
        other_parts = other.parts if isinstance(other, Circuit) else [other]
        dual = self.has_dual or other.has_dual
        if dual:
            self.enable_dual()
            other.enable_dual()
        return Circuit([self] + other_parts)

    def TN_tensor(self) -> Tensor:
        "Returns the TensorNetwork Tensor of this Operation."
        return self.TN_tensor()
