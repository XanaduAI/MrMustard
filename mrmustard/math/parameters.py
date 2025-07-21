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

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mrmustard.math.backend_manager import BackendManager

math = BackendManager()


__all__ = ["Constant", "Variable"]


# ~~~~~~~~~
# Functions
# ~~~~~~~~~


def update_symplectic(grads_and_vars, symplectic_lr: float):
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
            S,
            math.expm(-symplectic_lr * YT) @ math.expm(-symplectic_lr * (Y - YT)),
        )
        math.assign(S, new_value)


def update_orthogonal(grads_and_vars, orthogonal_lr: float):
    r"""Updates the orthogonal parameters using the given orthogonal gradients.

    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.
    """
    for dO_euclidean, O in grads_and_vars:
        Y = math.euclidean_to_unitary(O, math.real(dO_euclidean))
        new_value = math.matmul(O, math.expm(-orthogonal_lr * Y))
        math.assign(O, new_value)


def update_unitary(grads_and_vars, unitary_lr: float):
    r"""Updates the unitary parameters using the given unitary gradients.

    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.
    """
    for dU_euclidean, U in grads_and_vars:
        Y = math.euclidean_to_unitary(U, dU_euclidean)
        new_value = math.matmul(U, math.expm(-unitary_lr * Y))
        math.assign(U, new_value)


def update_euclidean(grads_and_vars, euclidean_lr: float):
    """Updates the parameters using the euclidian gradients."""
    math.euclidean_opt.lr = euclidean_lr
    math.euclidean_opt.apply_gradients(grads_and_vars)


# ~~~~~~~
# Classes
# ~~~~~~~


class Constant:
    r"""
    A parameter with a constant, immutable value.

    .. code::

      my_const = Constant(1, "my_const")

    Args:
        value: The value of this constant.
        name: The name of this constant.
        dtype: The dtype of this constant.
    """

    def __init__(self, value: Any, name: str, dtype: Any = None):
        if math.from_backend(value) and not math.is_trainable(value):
            self._value = value
        elif hasattr(value, "dtype"):
            self._value = math.new_constant(value, name, value.dtype)
        else:
            self._value = math.new_constant(value, name, dtype)
        self._name = name

    @property
    def name(self) -> str:
        r"""
        The name of this constant.
        """
        return self._name

    @property
    def value(self) -> Any:
        r"""
        The value of this constant.
        """
        return self._value

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        ret._value, ret._name = aux_data
        return ret

    def _tree_flatten(self):  # pragma: no cover
        children = ()
        aux_data = (self.value, self.name)
        return (children, aux_data)

    def __mul__(self, value):
        return type(self)(value=value * self.value, name=self.name)

    def __rmul__(self, value):
        return type(self)(value=self.value * value, name=self.name)


class Variable:
    r"""
    A parameter whose value can change.

    .. code::

      my_var = Variable(1, "my_var")

    Args:
        value: The value of this variable.
        name: The name of this variable.
        bounds: The numerical bounds of this variable.
        update_fn: The function used to update this variable during training.
        dtype: The dtype of this variable.
    """

    def __init__(
        self,
        value: Any,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        update_fn: Callable = update_euclidean,
        dtype: Any = None,
    ):
        self._value = self._get_value(value, bounds, name, dtype)
        self._name = name
        self._bounds = bounds
        self._update_fn = update_fn

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        r"""
        The numerical bounds of this variable.
        """
        return self._bounds

    @property
    def name(self) -> str:
        r"""
        The name of this variable.
        """
        return self._name

    @property
    def update_fn(self) -> Callable | None:
        r"""
        The function used to update this variable during training.
        """
        return self._update_fn

    @update_fn.setter
    def update_fn(self, value):
        self._update_fn = value

    @property
    def value(self) -> Any:
        r"""
        The value of this variable.
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = self._get_value(value, self.bounds, self.name)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        ret._value = children[0]
        ret._name, ret._bounds, ret._update_fn = aux_data
        return ret

    @staticmethod
    def orthogonal(
        value: Any | None,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        N: int = 1,
    ):
        r"""
        Initializes a variable with ``update_fn`` for orthogonal optimization.

        Args:
            value: The value of the returned variable. If ``None``, a random orthogonal
                matrix of dimension ``N`` is initialized.
            name: The name of the returned variable.
            bounds: The numerical bounds of the returned variable.
            N: The dimension of the random orthogonal matrix. This value is ignored if
                ``value`` is not ``None``.

        Returns:
            A variable with ``update_fn`` for orthogonal optimization.
        """
        value = value or math.random_orthogonal(N)
        return Variable(value, name, bounds, update_orthogonal)

    @staticmethod
    def symplectic(
        value: Any,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        N: int = 1,
    ):
        r"""
        Initializes a variable with ``update_fn`` for simplectic optimization.

        Args:
            value: The value of the returned variable. If ``None``, a random symplectic
                matrix of dimension ``N`` is initialized.
            name: The name of the returned variable.
            bounds: The numerical bounds of the returned variable.
            N: The dimension of the random symplectic matrix. This value is ignored if
                ``value`` is not ``None``.

        Returns:
            A variable with ``update_fn`` for simplectic optimization.
        """
        value = value or math.random_symplectic(N)
        return Variable(value, name, bounds, update_symplectic)

    @staticmethod
    def unitary(
        value: Any,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        N: int = 1,
    ):
        r"""
        Initializes a variable with ``update_fn`` for unitary optimization.

        Args:
            value: The value of the returned variable. If ``None``, a random unitary
                matrix of dimension ``N`` is initialized.
            name: The name of the returned variable.
            bounds: The numerical bounds of the returned variable.
            N: The dimension of the random unitary matrix. This value is ignored if
                ``value`` is not ``None``.

        Returns:
            A variable with ``update_fn`` for unitary optimization.
        """
        value = value or math.random_unitary(N)
        return Variable(value, name, bounds, update_unitary)

    def _get_value(self, value, bounds, name, dtype=None):
        r"""
        Returns a variable from given ``value``, ``bounds``, and ``name``.
        """
        if math.from_backend(value) and math.is_trainable(value):
            return value
        if hasattr(value, "dtype"):
            return math.new_variable(value, bounds, name, value.dtype)
        return math.new_variable(value, bounds, name, dtype)

    def _tree_flatten(self):  # pragma: no cover
        children = (self.value,)
        aux_data = (self.name, self.bounds, self.update_fn)
        return (children, aux_data)

    def __mul__(self, value):
        return type(self)(
            value=value * self.value,
            name=self.name,
            bounds=self.bounds,
            update_fn=self.update_fn,
        )

    def __rmul__(self, value):
        return type(self)(
            value=self.value * value,
            name=self.name,
            bounds=self.bounds,
            update_fn=self.update_fn,
        )
