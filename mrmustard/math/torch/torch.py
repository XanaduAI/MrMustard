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

"""This module contains the Pytorch implementation of the :class:`Math` interface."""

import numpy as np
import torch
from .pytorch_utils import sqrtm as sqrtm_utils

from mrmustard.types import (
    List,
    Tensor,
    Sequence,
    Tuple,
    Optional,
    Dict,
    Trainable,
    Callable,
    Union,
)
from mrmustard.math.math_interface import MathInterface

# pylint: disable=too-many-public-methods,no-self-use


class TorchMath(MathInterface):
    r"""Torch implemantion of the :class:`Math` interface."""

    float64 = torch.float64
    float32 = torch.float32
    complex64 = torch.complex64
    complex128 = torch.complex128
    dtypes_dict = {
        "float32": float32,
        "float64": float64,
        "complex64": complex64,
        "complex128": complex128,
    }

    def __getattr__(self, name):
        return getattr(torch, name)

    # ~~~~~~~~~
    # Basic ops
    # ~~~~~~~~~

    def abs(self, array: torch.Tensor) -> torch.Tensor:
        return torch.abs(array)

    def any(self, array: torch.Tensor) -> torch.Tensor:
        return torch.any(array)

    def arange(
        self, start: int, limit: int = None, delta: int = 1, dtype=torch.float64
    ) -> torch.Tensor:
        return torch.arange(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: torch.Tensor) -> Tensor:
        return tensor.detach().cpu().numpy()

    def assign(self, tensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # Is this a TF-only feature?

    def astensor(self, array: Union[np.ndarray, torch.Tensor], dtype=None) -> torch.Tensor:
        return self.cast(torch.tensor(array), dtype)

    def atleast_1d(self, array: torch.Tensor, dtype=None) -> torch.Tensor:
        return self.cast(torch.reshape(self.astensor(array), [-1]), dtype)

    def cast(self, array: torch.Tensor, dtype: str = None) -> torch.Tensor:
        if isinstance(dtype, str):
            dtype = self.dtypes_dict[dtype]
        if dtype is None:
            return array

        return array.to(dtype)

    def clip(self, array, a_min, a_max) -> torch.Tensor:
        return torch.clamp(array, a_min, a_max)

    def concat(self, values: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.cat(values, axis)

    def conj(self, array: torch.Tensor) -> torch.Tensor:
        return torch.conj(array)

    def constraint_func(
        self, bounds: Tuple[Optional[float], Optional[float]]
    ) -> Optional[Callable]:
        bounds = (
            -np.inf if bounds[0] is None else bounds[0],
            np.inf if bounds[1] is None else bounds[1],
        )
        if bounds != (-np.inf, np.inf):
            constraint: Optional[Callable] = lambda x: torch.clamp(x, min=bounds[0], max=bounds[1])
        else:
            constraint = None
        return constraint

    def convolution(
        self,
        array: torch.Tensor,
        filters: torch.Tensor,
        strides: Optional[List[int]] = None,
        padding="VALID",
        data_format="NWC",
        dilations: Optional[List[int]] = None,
    ) -> torch.Tensor:  # TODO: implement this...
        r"""Wrapper for ``torch.nn.Conv1d`` and ``torch.nn.Conv2d``.

        Args:
            1D convolution: Tensor of shape [batch_size, input_channels, signal_length].
            2D convolution: [batch_size, input_channels, input_height, input_width]

        Returns:
        """

        batch_size = array.shape[0]
        input_channels = array.shape[1]
        output_channels = ...  # TODO: unsure of how to get output channels

        if array.dim() == 3:  # 1D case
            signal_length = array.shape[2]

            m = torch.nn.Conv1d(
                input_channels,
                output_channels,
                filters,
                stride=strides,
                padding=padding,
                dtype=data_format,
                dilation=dilations,
            )
            return m(array)

        if array.dim() == 4:  # 2D case
            input_height = array.shape[2]
            input_width = array.shape[3]

            m = torch.nn.Conv2d(
                input_channels,
                output_channels,
                filters,
                stride=strides,
                padding=padding,
                dtype=data_format,
                dilation=dilations,
            )
            return m(array)

        raise NotImplementedError

    def cos(self, array: torch.Tensor) -> torch.Tensor:
        return torch.cos(array)

    def cosh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.cosh(array)

    def det(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.det(matrix)

    def diag(self, array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.diag(array, k=k)

    def diag_part(self, array: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(array)

    def einsum(self, string: str, *tensors) -> torch.Tensor:
        return torch.einsum(string, *tensors)

    def exp(self, array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    def expand_dims(self, array: torch.Tensor, axis: int) -> torch.Tensor:
        raise NotImplementedError  # TODO: implement

    def expm(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(matrix)

    def eye(self, size: int, dtype=torch.float64) -> torch.Tensor:
        return torch.eye(size, dtype=dtype)

    def from_backend(self, value) -> bool:
        return isinstance(value, (torch.Tensor))  # TODO: check if exists torch.Variable

    def gather(self, array: torch.Tensor, indices: torch.Tensor, axis: int = None) -> torch.Tensor:
        # TODO: gather works differently in Pytorch vs Tensorflow.
        return torch.gather(array, axis, indices)

    def hash_tensor(self, tensor: torch.Tensor) -> str:
        return hash(tensor)

    def imag(self, array: torch.Tensor) -> torch.Tensor:
        return torch.imag(array)

    def inv(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.inverse(tensor)

    def is_trainable(self, tensor: torch.Tensor) -> bool:
        raise NotImplementedError  # TODO: check implementation

    def lgamma(self, x: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(x)

    def log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)

    def matmul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
    ) -> torch.Tensor:
        return torch.matmul(a, b)

    def matvec(
        self, a: torch.Tensor, b: torch.Tensor, transpose_a=False, adjoint_a=False
    ) -> torch.Tensor:
        return torch.mv(a, b)

    def maximum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)

    def minimum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.minimum(a, b)

    def new_variable(
        self, value, bounds: Tuple[Optional[float], Optional[float]], name: str, dtype=torch.float64
    ):
        return torch.tensor(value, dtype=dtype, requires_grad=True)

    def new_constant(self, value, name: str, dtype=torch.float64):
        return torch.tensor(value, dtype=dtype)

    def norm(self, array: torch.Tensor) -> torch.Tensor:
        """Note that the norm preserves the type of array."""
        return torch.norm(array)

    def ones(self, shape: Sequence[int], dtype=torch.float64) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype)

    def ones_like(self, array: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(array)

    def outer(self, array1: torch.Tensor, array2: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(array1, array2, [[], []])

    def pad(
        self,
        array: torch.Tensor,
        paddings: Sequence[Tuple[int, int]],
        mode="constant",
        constant_values=0,
    ) -> torch.Tensor:
        return torch.nn.functional.pad(array, paddings, mode=mode, value=constant_values)

    def pinv(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.pinverse(matrix)

    def pow(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # TODO: implement

    def real(self, array: torch.Tensor) -> torch.Tensor:
        return torch.real(array)

    def reshape(self, array: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        return torch.reshape(array, shape)

    def sin(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    def sinh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sinh(array)

    def sqrt(self, x: torch.Tensor, dtype=None) -> torch.Tensor:
        return self.cast(torch.sqrt(x), dtype)

    def sum(self, array: torch.Tensor, axes: Sequence[int] = None):
        if axes:
            return torch.sum(array, axes)
        else:
            return torch.sum(array)

    def tensordot(self, a: torch.Tensor, b: torch.Tensor, axes: List[int]) -> torch.Tensor:
        return torch.tensordot(a, b, axes)

    def tile(self, array: torch.Tensor, repeats: Sequence[int]) -> torch.Tensor:
        return torch.tile(array, repeats)

    def trace(self, array: torch.Tensor, dtype=None) -> torch.Tensor:
        return self.cast(torch.trace(array), dtype)

    def transpose(self, a: torch.Tensor, perm: List[int] = (0, 1)) -> torch.Tensor:
        return torch.t(a)

    def unique_tensors(self, lst: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError  # TODO: implement

    def update_tensor(
        self, tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dims: int = 0
    ):
        # TODO: dims need to be an argument, or should be interpreted from the other data
        return tensor.scatter_(dims, indices, values)

    def update_add_tensor(
        self, tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dims: int = 0
    ):
        # TODO: dims need to be an argument, or should be interpreted from the other data
        return tensor.scatter_add_(dims, indices, values)

    def value_and_gradients(
        self, cost_fn: Callable, parameters: Dict[str, List[Trainable]]
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        r"""Computes the loss and gradients of the given cost function.

        Args:
            cost_fn (Callable): The cost function. Takes in two arguments:
                - Output: The output tensor of the model.
            parameters (Dict): The parameters to optimize in three kinds:
                symplectic, orthogonal and euclidean.
            optimizer: The optimizer to be used by the math backend.

        Returns:
            The loss and the gradients.
        """
        self.optimizer.zero_grad()
        loss = (
            cost_fn()
        )  # TODO: I think this should be cost_fn(params), but if it works I think it is fine.
        loss.backward()
        self.optimizer.step()

        grads = [p.grad for p in parameters]

        return loss, grads

    def zeros(self, shape: Sequence[int], dtype=torch.float64) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    def zeros_like(self, array: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(array)

    def hermite_renormalized(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, shape: Tuple[int]
    ) -> torch.Tensor:  # TODO this is not ready
        r"""Renormalized multidimensional Hermite polynomial.

        This is given by the "exponential" Taylor series of :math:`exp(Ax^2 + Bx + C)` at zero,
        where the series has :math:`sqrt(n!)` at the denominator rather than `n!`.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The shape of the final tensor.

        Returns:
            The renormalized Hermite polynomial of given shape.
        """
        raise NotImplementedError

    def DefaultEuclideanOptimizer(self, params) -> torch.optim.Optimizer:
        r"""Default optimizer for the Euclidean parameters."""
        self.optimizer = torch.optim.Adam(params, lr=0.001)
        return self.optimizer

    def eigvals(self, tensor: torch.Tensor) -> Tensor:
        """Returns the eigenvalues of a matrix."""
        return torch.linalg.eigvals(tensor)

    def eigvalsh(self, tensor: torch.Tensor) -> Tensor:
        """Returns the eigenvalues of a Real Symmetric or Hermitian matrix."""
        return torch.linalg.eigvalsh(tensor)

    def svd(self, tensor: torch.Tensor) -> Tensor:
        """Returns the Singular Value Decomposition of a matrix."""
        return torch.linalg.svd(tensor)

    def xlogy(self, x: torch.Tensor, y: torch.Tensor) -> Tensor:
        """Returns 0 if ``x == 0``, and ``x * log(y)`` otherwise, elementwise."""
        return torch.xlogy(x, y)

    def sqrtm(self, tensor: torch.Tensor) -> Tensor:
        return sqrtm_utils(tensor)

    def boolean_mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> Tensor:
        """Returns a new 1-D tensor which indexes the `input` tensor according to the boolean mask `mask`."""
        return torch.masked_select(tensor, mask)

