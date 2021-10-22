from __future__ import annotations

__all__ = ["Circuit"]

from collections.abc import MutableSequence
from mrmustard._typing import *
from mrmustard.experimental import XPMatrix, XPVector
from mrmustard.abstract import Transformation
from mrmustard.concrete import TMSV
from mrmustard import Backend

backend = Backend()

class Circuit(Transformation):
    def __init__(self, ops: Sequence = []):
        self._ops: List = [o for o in ops]
        self.reset()

    def reset(self):
        self._compiled: bool = False
        self._bell: Optional[State] = None
        self._modes: List[int] = []

    @property
    def bell(self):
        if self._bell is None:
            bell = bell_single = TMSV(r=2.5)
            for n in range(self.num_modes):
                bell = bell & bell_single
            order = tuple(range(0, 2*self.num_modes, 2)) + tuple(range(1, 2*self.num_modes, 2))
            self._bell = bell[order]
        return self._bell

    @property
    def num_modes(self) -> int:
        all_modes = set()
        for op in self._ops:
            all_modes = all_modes | set(op.modes)
        return len(all_modes)

    # NOTE: op.X_matrix() is called three times per op in the following methods, so circuits are composable but with an exponential cost.
    # TODO: Find a way around it?
    def X_matrix(self) -> Optional[Matrix]:
        X = XPMatrix(like_1=True)
        for op in self._ops:
            opX = XPMatrix.from_xxpp(op.X_matrix(), modes=(op.modes, op.modes), like_1=True)
            X = opX @ X
        return X.to_xxpp()
    
    def Y_matrix(self, hbar: float) -> Optional[Matrix]:
        Y = XPMatrix(like_0=True)
        for op in self._ops:
            opX = XPMatrix.from_xxpp(op.X_matrix(), modes=(op.modes, op.modes), like_1=True)
            opY = XPMatrix.from_xxpp(op.Y_matrix(hbar=hbar), modes=(op.modes, op.modes), like_0=True)
            Y = opX @ Y @ opX.T + opY
        return Y.to_xxpp()

    def d_vector(self, hbar: float) -> Optional[Vector]:
        d = XPVector()
        for op in self._ops:
            opX = XPMatrix.from_xxpp(op.X_matrix(), modes=(op.modes, op.modes), like_1=True)
            opd = XPVector.from_xxpp(op.d_vector(hbar=hbar), modes=op.modes)
            d = opX @ d + opd
        return d.to_xxpp()

    def extend(self, obj):
        result = self._ops.extend(obj)
        self.reset()

    def append(self, obj):
        result = self._ops.append(obj)
        self.reset()

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
