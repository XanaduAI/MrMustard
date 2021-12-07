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


from functools import reduce
from mrmustard.utils import training
from mrmustard.types import *
from mrmustard.math import Math

math = Math()


class Parametrized:
    r"""Abstract base class for all parametrized objects (gates, detectors, etc.)

    For each trainable parameter keyword arguments must be passed for the initial value ``xxx``
    (tensor), the numerical bounds ``xxx_bounds`` (float, float), whether the parameter ``xxx`` will
    be optimized ``xxx_trainable`` (bool), along with any other parameters.
    """

    def __init__(self, **kwargs):  # NOTE: only kwargs so that we can use the arg names
        self._trainable_parameters = []
        self._constant_parameters = []
        self._param_names = []
        owner = f"{self.__class__.__qualname__}"
        for name, value in kwargs.items():
            if math.from_backend(value):
                if math.is_trainable(value):
                    self._trainable_parameters.append(value)
                elif name + "_trainable" in kwargs and kwargs[name + "_trainable"]:
                    value = math.new_variable(value, kwargs[name + "_bounds"], owner + ":" + name)
                    self._trainable_parameters.append(value)
                else:
                    self._constant_parameters.append(value)
            elif name + "_trainable" in kwargs and kwargs[name + "_trainable"]:
                value = math.new_variable(value, kwargs[name + "_bounds"], owner + ":" + name)
                self._trainable_parameters.append(value)
            elif name + "_trainable" in kwargs and not kwargs[name + "_trainable"]:
                value = math.new_constant(value, owner + ":" + name)
                self._constant_parameters.append(value)
            else:
                name = "_" + name
            self.__dict__[name] = value
            self._param_names += [] if name.startswith("_") else [name]

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        r"""Returns the dictionary of trainable parameters, searching recursively in the object tree (for example, when in a Circuit)."""
        if hasattr(self, "_ops"):
            return {
                "symplectic": math.unique_tensors(
                    [p for item in self._ops for p in item.trainable_parameters["symplectic"]]
                ),
                "orthogonal": math.unique_tensors(
                    [p for item in self._ops for p in item.trainable_parameters["orthogonal"]]
                ),
                "euclidean": math.unique_tensors(
                    [p for item in self._ops for p in item.trainable_parameters["euclidean"]]
                ),
            }
        else:
            return {
                "symplectic": [],
                "orthogonal": [],
                "euclidean": self._trainable_parameters,
            }  # default

    @property
    def constant_parameters(self) -> Dict[str, List[Tensor]]:
        r"""Returns the dictionary of constant parameters, searching recursively in the object tree (for example, when in a Circuit)."""
        if hasattr(self, "_ops"):
            return {
                "symplectic": math.unique_tensors(
                    [p for item in self._ops for p in item.constant_parameters["symplectic"]]
                ),
                "orthogonal": math.unique_tensors(
                    [p for item in self._ops for p in item.constant_parameters["orthogonal"]]
                ),
                "euclidean": math.unique_tensors(
                    [p for item in self._ops for p in item.constant_parameters["euclidean"]]
                ),
            }
        else:
            return {
                "symplectic": [],
                "orthogonal": [],
                "euclidean": self._constant_parameters,
            }  # default
