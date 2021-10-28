# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractproperty
from mrmustard.core import train
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
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:  # override as needed in child classes
        return {"symplectic": [], "orthogonal": [], "euclidean": self._trainable_parameters}
