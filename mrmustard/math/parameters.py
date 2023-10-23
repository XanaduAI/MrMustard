# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the classes to describe constant and variable parameters used in Mr Mustard."""

from typing import Callable, Optional, Tuple, Sequence

from mrmustard.math import Math
from mrmustard.utils.typing import Tensor

math = Math()


__all__ = ["Constant", "Variable"]


# ~~~~~~~
# Classes
# ~~~~~~~


class ParameterBase:
    r"""
    A base class for Mr Mustard parameters.

    Args:
        value: The value of this parameter.
        name: The name of this parameter.
        dtype: The type of this parameter.
    """

    def __init__(self, value: any, name: str, dtype: any):
        self._value = value
        self._name = name
        self._dtype = dtype

    @property
    def dtype(self) -> str:
        r"""
        The type of this parameter.
        """
        return self._dtype

    @property
    def name(self) -> str:
        r"""
        The name of this parameter.
        """
        return self._name

    @property
    def value(self) -> any:
        r"""
        The value of this parameter.
        """
        return self._value


class Constant(ParameterBase):
    r"""
    A parameter with a constant, immutable value.

    Args:
        value: The value of this constant.
        name: The name of this constant.
        dtype: The type of this constant.
    """

    def __init__(self, value: any, name: str, dtype: any = math.float64):
        super().__init__(value, name, dtype)

    def __mul__(self, value):
        return type(self)(value=value * self.value, name=self.name, dtype=self.dtype)

    def __rmul__(self, value):
        return type(self)(value=self.value * value, name=self.name, dtype=self.dtype)


class Variable(ParameterBase):
    r"""
    A parameter whose value can change.

    Args:
        value: The value of this variable.
        name: The name of this variable.
        bounds: The numerical bounds of this variable.
        update_fn (optional): The function used to update this variable during
            training. If ``None``, training defaults to ``update_euclidean``.
    """

    def __init__(
        self,
        value: any,
        name: str,
        dtype: any = math.float64,
        bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        update_fn: Optional[Callable] = None,
    ):
        super().__init__(value, name, dtype)
        self._bounds = bounds
        self._update_fn = update_fn

    @property
    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        The numerical bounds of this variable.
        """
        return self._bounds

    @property
    def update_fn(self) -> Optional[Callable]:
        r"""
        The function used to update this variable during training.
        """
        return self._update_fn

    @property
    def value(self) -> any:
        r"""
        The value of this variable.
        """
        return self._value

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    @update_fn.setter
    def update_fn(self, value):
        self._update_fn = value

    @value.setter
    def value(self, value):
        self._value = value

    def __mul__(self, value):
        return type(self)(
            value=value * self.value,
            name=self.name,
            dtype=self.dtype,
            bounds=self.bounds,
            update_fn=self.update_fn,
        )

    def __rmul__(self, value):
        return type(self)(
            value=self.value * value,
            name=self.name,
            dtype=self.dtype,
            bounds=self.bounds,
            update_fn=self.update_fn,
        )


# ~~~~~~~~~
# Functions
# ~~~~~~~~~


def update_symplectic(grads_and_vars: Sequence[Tuple[Tensor, Variable]], symplectic_lr: float):
    r"""
    Updates the symplectic parameters using the given symplectic gradients.

    Implemented from:
        Wang J, Sun H, Fiori S. A Riemannian-steepest-descent approach
        for optimization on the real symplectic group.
        Mathematical Methods in the Applied Sciences. 2018 Jul 30;41(11):4273-86.
    """
    for dS_euclidean, S in grads_and_vars:
        Y = math.euclidean_to_symplectic(S, dS_euclidean)
        YT = math.transpose(Y)
        new_value = math.matmul(
            S, math.expm(-symplectic_lr * YT) @ math.expm(-symplectic_lr * (Y - YT))
        )
        math.assign(S, new_value)


def update_orthogonal(grads_and_vars: Sequence[Tuple[Tensor, Variable]], orthogonal_lr: float):
    r"""Updates the orthogonal parameters using the given orthogonal gradients.

    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.
    """
    for dO_euclidean, O in grads_and_vars:
        Y = math.euclidean_to_unitary(O, math.real(dO_euclidean))
        new_value = math.matmul(O, math.expm(-orthogonal_lr * Y))
        math.assign(O, new_value)


def update_unitary(grads_and_vars: Sequence[Tuple[Tensor, Variable]], unitary_lr: float):
    r"""Updates the unitary parameters using the given unitary gradients.

    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.
    """
    for dU_euclidean, U in grads_and_vars:
        Y = math.euclidean_to_unitary(U, dU_euclidean)
        new_value = math.matmul(U, math.expm(-unitary_lr * Y))
        math.assign(U, new_value)


def update_euclidean(grads_and_vars: Sequence[Tuple[Tensor, Variable]], euclidean_lr: float):
    """Updates the parameters using the euclidian gradients."""
    math.euclidean_opt.lr = euclidean_lr
    math.euclidean_opt.apply_gradients(grads_and_vars)
