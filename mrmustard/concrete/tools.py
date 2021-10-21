from __future__ import annotations

# TODO: figure out where the Circuit should be

__all__ = ["Circuit"]

from collections.abc import MutableSequence
from mrmustard._typing import *
from mrmustard.experimental import XPMatrix, XPVector
from mrmustard.abstract import Transformation

class Circuit(Transformation):
    def __init__(self, ops: Sequence[Op] = []):
        self._modes = []
        self._ops: List[Op] = [o for o in ops]
        self._compiled = False

    def X_matrix(self) -> Optional[Matrix]:
        X = XPMatrix(like_1=True)
        for op in self._ops:
            modes = [] if op._modes is None else op._modes
            X = XPMatrix.from_xxpp(op.X_matrix(), like_1=True, modes=(modes, modes)) @ X
        return X.to_xxpp()

    def Y_matrix(self, hbar: float) -> Optional[Matrix]:
        Y = XPMatrix(like_0=True)
        for op in self._ops:
            modes = [] if op._modes is None else op._modes
            opX = XPMatrix.from_xxpp(op.X_matrix(), modes=(modes, modes), like_1=True)
            opY = XPMatrix.from_xxpp(op.Y_matrix(hbar=hbar), modes=(modes, modes), like_0=True)
            Y = opX @ Y @ opX.T + opY
        return Y.to_xxpp()

    def d_vector(self, hbar: float) -> Optional[Vector]:
        d = XPVector()
        for op in self._ops:
            modes = [] if op._modes is None else op._modes
            opX = XPMatrix.from_xxpp(op.X_matrix(), modes=(modes, modes), like_1=True)
            opd = XPVector.from_xxpp(op.d_vector(hbar=hbar), modes=modes)
            d = opX @ d + opd
        return d.to_xxpp()

    def extend(self, obj):
        result = self._ops.extend(obj)
        self._compiled = False

    def append(self, obj):
        result = self._ops.append(obj)
        self._compiled = False

    def __len__(self):
        return len(self._ops)

    def __repr__(self) -> str:
        return f"Circuit | {len(self._ops)} ops | compiled = {self._compiled}"

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        r"""
        Returns the dictionary of trainable parameters
        """
        symp = [p for op in self._ops for p in op.trainable_parameters["symplectic"] if hasattr(op, "trainable_parameters")]
        orth = [p for op in self._ops for p in op.trainable_parameters["orthogonal"] if hasattr(op, "trainable_parameters")]
        eucl = [p for op in self._ops for p in op.trainable_parameters["euclidean"] if hasattr(op, "trainable_parameters")]
        return {"symplectic": symp, "orthogonal": orth, "euclidean": eucl}
