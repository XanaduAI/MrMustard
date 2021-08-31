from abc import ABC, abstractproperty
from mrmustard.plugins import train
from mrmustard._typing import *


class Parametrized(ABC):
    r"""
    Abstract base class for all parametrized objects (gates, detectors, etc...)

    Arguments (must be all called with keyword):
        For each trainable parameter:
        xxx (tensor): initial value
        xxx_bounds (float, float): numerical bounds. Default is (None, None) for unbounded parameters
        xxx_trainable (bool): whether the parameter `xxx` will be optimized
        yyy (any): other parameters
    """

    def __init__(self, **kwargs):
        self._trainable_parameters = []
        self._constant_parameters = []
        self.param_names = [key for key in kwargs if key + "_trainable" in kwargs]  # every parameter can be trainable! 🚀

        for name in self.param_names:
            self.__dict__["_" + name + "_trainable"] = kwargs[name + "_trainable"]  # making "is trainable" available as param._trainable
            if kwargs[name + "_trainable"]:
                var = train.new_variable(kwargs[name], kwargs[name + "_bounds"], name)
                self._trainable_parameters.append(var)
                self.__dict__[name] = var  # making parameters available as gate.parameter_name
            else:
                const = train.new_constant(kwargs[name], name)
                self._constant_parameters.append(const)
                self.__dict__[name] = const
        for key, val in kwargs.items():
            if not any(word in key for word in self.param_names):
                self.__dict__["_" + key] = val  # making other values available as gate._val_name

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:  # override as needed
        return {"symplectic": [], "orthogonal": [], "euclidean": self._trainable_parameters}
