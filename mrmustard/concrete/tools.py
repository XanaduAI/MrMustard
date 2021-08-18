from __future__ import annotations

# TODO: figure out where the Circuit should be

__all__ = ["Circuit"]

from collections.abc import MutableSequence
from mrmustard import FockPlugin
from mrmustard._typing import *


class Circuit(MutableSequence):

    _fock = FockPlugin()

    def __init__(self, ops: Sequence[Op] = []):
        self._ops: List[Op] = [o for o in ops]

    def __call__(self, state: State) -> State:
        state_ = state  # NOTE: otherwise the next time we call the circuit, the state will be mutated
        for op in self._ops:
            state_ = op(state_)
        return state_

    def __getitem__(self, key):
        return self._ops.__getitem__(key)

    def __setitem__(self, key, value):
        return self._ops.__setitem__(key, value)

    def __delitem__(self, key):
        return self._ops.__delitem__(key)

    def __len__(self):
        return len(self._ops)

    def __repr__(self) -> str:
        return "\n".join([repr(g) for g in self._ops])

    def insert(self, index, object):
        return self._ops.insert(index, object)

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        r"""
        Returns the dictionary of trainable parameters
        """
        symp = [p for op in self._ops for p in op.trainable_parameters["symplectic"] if hasattr(op, "trainable_parameters")]
        orth = [p for op in self._ops for p in op.trainable_parameters["orthogonal"] if hasattr(op, "trainable_parameters")]
        eucl = [p for op in self._ops for p in op.trainable_parameters["euclidean"] if hasattr(op, "trainable_parameters")]
        return {"symplectic": symp, "orthogonal": orth, "euclidean": eucl}
