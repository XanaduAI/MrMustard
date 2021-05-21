from collections.abc import MutableSequence
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Optional
from mrmustard._states import State
from mrmustard._opt import CircuitInterface
from mrmustard._backends import MathBackendInterface


class GateInterface(ABC):
    @abstractmethod
    def __call__(self, state: State) -> State:
        pass

    @abstractmethod
    def symplectic_matrix(self, hbar: float) -> Optional:
        pass

    @abstractmethod
    def displacement_vector(self, hbar: float) -> Optional:
        pass

    @abstractmethod
    def noise_matrix(self, hbar: float) -> Optional:
        pass

    @abstractproperty
    def euclidean_parameters(self) -> List:
        pass

    @abstractproperty
    def symplectic_parameters(self) -> List:
        pass


class Circuit(CircuitInterface, MutableSequence):
    _math_backend: MathBackendInterface

    def __init__(self):
        self._gates: List[GateInterface] = []

    def __call__(self, state: State) -> State:
        state_ = state
        for gate in self._gates:
            state_ = gate(state_)
        return state_

    def __getitem__(self, key):
        return self._gates.__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, GateInterface):
            raise ValueError(f"Item {type(value)} is not a gate")
        return self._gates.__setitem__(key, value)

    def __delitem__(self, key):
        return self._gates.__delitem__(key)

    def __len__(self):
        return len(self._gates)

    def __repr__(self) -> str:
        return "\n".join([repr(g) for g in self._gates])

    def insert(self, index, object):
        return self._gates.insert(index, object)

    @property
    def symplectic_parameters(self) -> List:
        r"""
        Returns the list of symplectic parameters
        """
        return [par for gate in self._gates for par in gate.symplectic_parameters]

    @property
    def euclidean_parameters(self) -> List:
        r"""
        Returns the list of Euclidean parameters
        """
        return [par for gate in self._gates for par in gate.euclidean_parameters]
