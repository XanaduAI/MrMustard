from __future__ import annotations

# TODO: figure out where the Circuit should be

__all__ = ["Circuit"]

from collections.abc import MutableSequence
from mrmustard._typing import *
from mrmustard.experimental import XPTensor


class Circuit(MutableSequence):
    def __init__(self, ops: Sequence[Op] = []):
        self.X = XPTensor(None, modes=[], additive=False)
        self.Y = XPTensor(None, modes=[], additive=True)
        self._ops: List[Op] = [o for o in ops]
        self._compiled = False

    def __call__(self, state: State) -> State:
        state_ = state  # NOTE: otherwise state will be mutated (is this true?)
        for op in self._ops:
            state_ = op(state_)
        return state_

    def __getitem__(self, key):
        return self._ops.__getitem__(key)

    def __setitem__(self, key, value):
        try:
            result = self._ops.__setitem__(key, value)
        except Exception as e:
            raise e
        self._compiled = False

    def __delitem__(self, key):
        try:
            result = self._ops.__delitem__(key)
        except Exception as e:
            raise e
        self._compiled = False

    def __len__(self):
        return len(self._ops)

    def __repr__(self) -> str:
        return f"Circuit | {len(self._ops)} ops | compiled = {self._compiled}"

    def insert(self, index, obj):
        try:
            result = self._ops.insert(index, obj)
        except Exception as e:
            raise e
        self._compiled = False

    def compile(self) -> None:
        for obj in self._ops:  # TODO: make this not redo the same thing
            self.update_channel(obj)
        self._compiled = True

    def recompile(self) -> None:
        self.X = XPTensor(None, modes=[], additive=False)
        self.Y = XPTensor(None, modes=[], additive=True)
        self.compile()
        self._compiled = True

    def update_channel(self, op):
        if hasattr(op, "X_matrix"):
            Xprime = XPTensor.from_xxpp(op.X_matrix(), op._modes, additive=False)
            Yprime = XPTensor.from_xxpp(op.Y_matrix(hbar=2.0), op._modes, additive=True)
            self.X = Xprime @ self.X
            self.Y = (Xprime @ self.Y) @ Xprime.T + Yprime

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        r"""
        Returns the dictionary of trainable parameters
        """
        symp = [p for op in self._ops for p in op.trainable_parameters["symplectic"] if hasattr(op, "trainable_parameters")]
        orth = [p for op in self._ops for p in op.trainable_parameters["orthogonal"] if hasattr(op, "trainable_parameters")]
        eucl = [p for op in self._ops for p in op.trainable_parameters["euclidean"] if hasattr(op, "trainable_parameters")]
        return {"symplectic": symp, "orthogonal": orth, "euclidean": eucl}
