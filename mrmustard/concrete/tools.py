from __future__ import annotations
from collections.abc import MutableSequence
from typing import List, Callable, Sequence, Union

from mrmustard.core.backends import MathBackendInterface, TrainingBackendInterface
from mrmustard.core.baseclasses import Op, State, Detector, Tensor
from mrmustard.core.visual import Progressbar

from typing import TypeVar

Trainable = TypeVar("Trainable", Circuit, Op)

class Circuit(MutableSequence):

    def __init__(self, ops: Sequence[Op] = []):
        try:
            if all(isinstance(o, Op) for o in ops):
                self._ops: List[Op] = [o for o in ops]
        except TypeError:
            raise TypeError(f"not a sequence of ops")
        except Exception as e:
            raise e

    def __call__(self, state: State) -> State:
        state_ = state  # NOTE: otherwise the next time we call the circuit, the state will be mutated
        for op in self._ops:
            state_ = op(state_)
        return state_

    def __getitem__(self, key):
        return self._ops.__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Gate):
            raise ValueError(f"Item {type(value)} is not a gate")
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
    def symplectic_parameters(self) -> List:
        r"""
        Returns the list of symplectic parameters
        """
        return [par for op in self._ops for par in op.symplectic_parameters]

    @property
    def orthogonal_parameters(self) -> List:
        r"""
        Returns the list of orthogonal parameters
        """
        return [par for op in self._ops for par in op.orthogonal_parameters]

    @property
    def euclidean_parameters(self) -> List:
        r"""
        Returns the list of Euclidean parameters
        """
        return [par for op in self._ops for par in op.euclidean_parameters]



