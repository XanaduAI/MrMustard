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
from mrmustard.utils.types import *
from mrmustard.math import Math

math = Math()


class Parametrized:
    r"""
    Abstract base class for all parametrized objects (gates, detectors, etc...)

    Arguments (must be all called with keyword):
        For each trainable parameter:
        xxx (tensor): initial value
        xxx_bounds (float, float): numerical bounds. Default is (None, None) for unbounded parameters
        xxx_trainable (bool): whether the parameter `xxx` will be optimized
        yyy (any): other parameters
    """

    def __init__(self, **kwargs):  # NOTE: only kwargs so that we can use the arg names
        self._trainable_parameters = []
        self._constant_parameters = []
        # We can get a few types of arguments:
        # 1. trainable parameters native to the backend (e.g. tf.Variable)
        # 2. constant parameters native to the backend (e.g. tf.constant)
        # in these first two cases we just add the parameters to the two lists self._trainable_parameters  and self._constant_parameters.
        # 3. parameters that are not native to the backend but that are trainable if the there is another argument, boolean, with the same name and ending with _trainable that is True.
        # 4. arguments that are not native to the backend and that don't represent trainable parameters (e.g. modes, flags, etc...)
        # in these last two cases we either create a native parameter or add the arguments the __dict__ of the object preprending the name with _
        owner = f"{self.__class__.__qualname__}"
        for name, value in kwargs.items():
            if math.from_backend(value):
                if math.is_trainable(value):
                    self._trainable_parameters.append(value)
                elif name + "_trainable" in kwargs and kwargs[name + "_trainable"]:
                    value = training.new_variable(value, kwargs[name + "_bounds"], owner + ":" + name)
                    self._trainable_parameters.append(value)
                else:
                    self._constant_parameters.append(value)
            elif name + "_trainable" in kwargs and kwargs[name + "_trainable"]:
                value = training.new_variable(value, kwargs[name + "_bounds"], owner + ":" + name)
                self._trainable_parameters.append(value)
            elif name + "_trainable" in kwargs and not kwargs[name + "_trainable"]:
                value = training.new_constant(value, owner + ":" + name)
                self._constant_parameters.append(value)
            else:
                name = "_" + name
            self.__dict__[name] = value

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        r"""
        Returns the dictionary of trainable parameters, searching recursively in the object tree (e.g. when in a Circuit).
        """
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
            return {"symplectic": [], "orthogonal": [], "euclidean": self._trainable_parameters}  # default

    @property
    def constant_parameters(self) -> Dict[str, List[Tensor]]:
        r"""
        Returns the dictionary of constant parameters, searching recursively in the object tree (e.g. when in a Circuit).
        """
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
            return {"symplectic": [], "orthogonal": [], "euclidean": self._constant_parameters}  # default
