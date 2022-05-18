# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The classes in this module represent parameters passed to the
quantum operations represented by :class:`~.Parametrized` subclasses.

Parameter types
---------------
There are three basic types of parameters:
1. **Numerical parameters** (bound and fixed): An immediate, immutable numerical python objects
   (float, complex, int, numerical array). Implemented as-is, not encapsulated in a class.
   They are dynamically assigned to properties of the class.
2. **Trainable parameters** (bound but not fixed): These are parameters that are updated by
   the optimization procedure. There are three types of trainable parameters which define the
   optimization procedure: symplectic, euclidian and orthogonal.
3. **Constant parameters** (bound and fixed): This class of parameters belong to the autograd
   backend but remain fixed during the optimization procedure.
"""
# pylint: disable=super-init-not-called

from abc import ABC, abstractmethod
from mrmustard.math import Math

math = Math()


class Trainable(ABC):
    @abstractmethod
    def __init__(self, value, bounds, name, owner=None) -> None:
        pass

    @abstractmethod
    def update(self, cost_fn, learning_rate) -> None:
        pass

    @staticmethod
    def grad(cost_fn, value):
        cost, grad = math.value_and_gradients(cost_fn, value)
        return cost, grad


class Symplectic(Trainable):
    def __init__(self, value, bounds, name, owner=None) -> None:
        self.S = value if math.from_backend(value) else math.new_variable(value, bounds, name)
        self.name = name
        self.owner = owner

    def update(self, cost_fn, learning_rate):
        _, grad = self.grad(cost_fn, self.S)
        self._update_symplectic(grad, learning_rate)

    def _update_symplectic(self, dS_euclidean, symplectic_lr):
        Y = math.euclidean_to_symplectic(self.S, dS_euclidean)
        YT = math.transpose(Y)
        new_value = math.matmul(
            self.S, math.expm(-symplectic_lr * YT) @ math.expm(-symplectic_lr * (Y - YT))
        )
        math.assign(self.S, new_value)


class Euclidian(Trainable):
    def __init__(self, value, bounds, name, owner=None) -> None:
        self.E = value if math.from_backend(value) else math.new_variable(value, bounds, name)
        self.name = name
        self.owner = owner

    def update(self, cost_fn, learning_rate):
        _, grad = self.grad(cost_fn, self.E)
        self._update_euclidian(grad, learning_rate)

    def _update_euclidian(self, euclidean_grad, euclidean_lr):
        math.euclidean_opt.lr = euclidean_lr
        math.euclidean_opt.apply_gradients((euclidean_grad, self.E))


class Orthogonal(Trainable):
    def __init__(self, value, bounds, name, owner=None) -> None:
        self.O = value if math.from_backend(value) else math.new_variable(value, bounds, name)
        self.name = name
        self.owner = owner

    def update(self, cost_fn, learning_rate):
        _, grad = self.grad(cost_fn, self.O)
        self._update_orthogonal(grad, learning_rate)

    def _update_euclidian(self, dO_euclidean, orthogonal_lr):
        dO_orthogonal = 0.5 * (
            dO_euclidean - math.matmul(math.matmul(self.O, math.transpose(dO_euclidean)), self.O)
        )
        new_value = math.matmul(
            self.O, math.expm(orthogonal_lr * math.matmul(math.transpose(dO_orthogonal), self.O))
        )
        math.assign(self.O, new_value)


class Constant:
    def __init__(self, value, name, owner=None) -> None:
        self.O = value if math.from_backend(value) else math.new_constant(value, name)
        self.name = name
        self.owner = owner


def create_parameter(name, value, is_trainable=False, bounds=None, owner=None):
    if not is_trainable:
        return Constant(value, name, owner)

    trainable = _get_trainable_by_type(name)
    return trainable(value, bounds, name, owner)


def _get_trainable_by_type(name: str) -> Trainable:

    if name.startswith("symplectic"):
        return Symplectic

    if name.startswith("orthogonal"):
        return Orthogonal

    return Euclidian
