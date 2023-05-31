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
        dual_enabled: bool,
        name: str = None,
        **kwargs,
    ):
        self.name = name or self.__class__.__qualname__
        self.input_tag_at_mode: dict[int, tuple[int, Optional[int]]] = {
            m: (TagDispenser().get_tag(), TagDispenser().get_tag() if dual_enabled else None)
            for m in modes_in
        }
        self.output_tag_at_mode: dict[int, tuple[int, Optional[int]]] = {
            m: (TagDispenser().get_tag(), TagDispenser().get_tag() if dual_enabled else None)
            for m in modes_out
        }
        super().__init__(**kwargs)

    _repr_markdown_ = None

    @property
    def dual_enabled(self) -> bool:
        "Whether this Operation has a dual (R) part."
        return any(pair[1] is not None for pair in self.all_tags)

    @property
    def tensor_tags(self) -> list[int]:
        r"""Returns the tags of all the wires of this Operation in the
        tensor order: [out_L, out_R, in_L, in_R]
        """
        return self.tags_out + self.tags_in

    # def disconnect(self) -> Operation:
    #     "Re-issue new wires for this operation."
    #     for wire_dict in [self.input_wire_at_mode, self.output_wire_at_mode]:
    #         for m, wire in wire_dict.items():
    #             wire_dict[m] = Wire(
    #                 origin=None if wire.origin is not self else self,
    #                 end=None if wire.end is not self else self,
    #                 L=TagDispenser().get_tag(),
    #                 R=TagDispenser().get_tag() if self.dual_enabled else None,
    #             )
    #     return self

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
            raise ValueError(
                "Different input and output modes. Please refer to modes_in and modes_out."
            )

    def enable_dual(self) -> None:
        "Enables the dual (R) part of all the tags of this Operation."
        for tag in self.all_tags:
            tag[1] = TagDispenser().get_tag()

    # def __hash__(self):  # is this needed?
    #     "hash function so that Operations can be used as keys in dictionaries."
    #     tags = tuple(tag for wire in self.all_wires for tag in [wire.L, wire.R] if tag is not None)
    #     return hash(tags)

    # def __rshift__(self, other: CircuitPart) -> Circuit:
    #     "Delayed primal mode: self >> other becomes a circuit that will be evaluated later."
    #     if self.dual_enabled or other.dual_enabled:
    #         self.enable_dual()
    #         other.enable_dual()
    #     return Circuit([self] + (other if isinstance(other, Circuit) else [other]))

    # def __gt__(self, other: Operation) -> Operation:
    #     "Immediate primal mode: self > other is evaluated and then returned."
    #     return other.primal(self)  # Operation shouldn't need to know about primal

    # def TN_tensor(self) -> Tensor:
    #     "Returns the TensorNetwork Tensor of this Operation."
    #     return self.fock
