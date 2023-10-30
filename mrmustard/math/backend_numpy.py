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

import numpy as np

from copy import deepcopy
from math import lgamma as mlgamma
from numpy.random import default_rng
from typing import Callable, List, Optional, Sequence, Tuple, Union
import scipy
from scipy.linalg import expm as scipy_expm
from scipy.special import xlogy as scipy_xlogy
from scipy.stats import multivariate_normal

from .autocast import Autocast
from .backend_base import BackendBase
from ..utils.settings import settings
from ..utils.typing import Trainable
from .lattice.strategies import binomial, vanilla
from .lattice.strategies.compactFock.inputValidation import (
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
)


class BackendNumpy(BackendBase):
    r"""
    A numpy backend.
    """

    int32 = np.int32
    float64 = np.float64
    float32 = np.float32
    complex64 = np.complex64
    complex128 = np.complex128

    def __init__(self):
        super().__init__(name="numpy")

    def abs(self, array: np.array) -> np.array:
        return np.abs(array)

    def any(self, array: np.array) -> np.array:
        return np.any(array)

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=np.float64) -> np.array:
        return np.arange(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: np.array) -> np.array:
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)

    def assign(self, tensor: np.array, value: np.array) -> np.array:
        tensor = value
        return tensor

    def astensor(self, array: Union[np.ndarray, np.array], dtype=None) -> np.array:
        return np.array(array)

    def atleast_1d(self, array: np.array, dtype=None) -> np.array:
        return self.cast(np.atleast_1d(array), dtype=dtype)

    def block(self, blocks: List[List[np.array]], axes=(-2, -1)) -> np.array:
        rows = [self.concat(row, axis=axes[1]) for row in blocks]
        return self.concat(rows, axis=axes[0])

    def boolean_mask(self, tensor: np.array, mask: np.array) -> np.ndarray:
        return np.array([t for i, t in enumerate(tensor) if mask[i]])

    def cast(self, array: np.array, dtype=None) -> np.array:
        if dtype is None:
            return array

        return np.array(array, dtype=dtype)  # gotta fix the warning

    def custom_gradient(self, func, args, kwargs):
        def trivial_decorator(*args, **kwargs):
            r"""
            Returns a trivial decorator that does nothing.
            """
            return func(*args, **kwargs)

        return trivial_decorator

    def clip(self, array, a_min, a_max) -> np.array:
        return np.clip(array, a_min, a_max)

    def concat(self, values: List[np.array], axis: int) -> np.array:
        # tf.concat can concatenate lists of scalars, while np.concatenate errors
        try:
            return np.concatenate(values, axis)
        except ValueError:
            return np.array(values)

    def conj(self, array: np.array) -> np.array:
        return np.conj(array)

    def cos(self, array: np.array) -> np.array:
        return np.cos(array)

    def cosh(self, array: np.array) -> np.array:
        return np.cosh(array)

    def atan2(self, y: np.array, x: np.array) -> np.array:
        return np.arctan(y, x)

    def make_complex(self, real: np.array, imag: np.array) -> np.array:
        return real + 1j * imag

    def det(self, matrix: np.array) -> np.array:
        return np.linalg.det(matrix)

    def diag(self, array: np.array, k: int = 0) -> np.array:
        if len(array.shape) == 1:
            return np.diag(array, k=k)
        elif len(array.shape) == 2:
            return np.array([np.diag(l, k=k).tolist() for l in array])
        else:
            # fallback into more complex algorithm
            original_sh = array.shape

            ravelled_sh = (np.prod(original_sh[:-1]), original_sh[-1])
            array = array.ravel().reshape(*ravelled_sh)

            ret = []
            for line in array:
                ret.append(np.diag(line, k))

            ret = np.array(ret)
            inner_shape = (
                original_sh[-1] + abs(k),
                original_sh[-1] + abs(k),
            )
            return ret.reshape(original_sh[:-1] + inner_shape)

    def diag_part(self, array: np.array, k: int) -> np.array:
        ret = np.diagonal(array, offset=k, axis1=-2, axis2=-1)
        ret.flags.writeable = True
        return ret

    def set_diag(self, array: np.array, diag: np.array, k: int) -> np.array:
        i = np.arange(0, array.shape[-2] - abs(k))
        if k < 0:
            i -= array.shape[-2] - abs(k)

        j = np.arange(abs(k), array.shape[-1])
        if k < 0:
            j -= abs(k)

        array[..., i, j] = diag

        return array

    def einsum(self, string: str, *tensors) -> np.array:
        if type(string) is str:
            return np.einsum(string, *tensors)
        return None  # provide same functionality as numpy.einsum or upgrade to opt_einsum

    def exp(self, array: np.array) -> np.array:
        return np.exp(array)

    def expand_dims(self, array: np.array, axis: int) -> np.array:
        return np.expand_dims(array, axis)

    def expm(self, matrix: np.array) -> np.array:
        return scipy_expm(matrix)

    def eye(self, size: int, dtype=np.float64) -> np.array:
        return np.eye(size, dtype=dtype)

    def eye_like(self, array: np.array) -> np.ndarray:
        return np.eye(array.shape[-1], dtype=array.dtype)

    def from_backend(self, value) -> bool:
        return isinstance(value, np.ndarray)

    def gather(self, array: np.array, indices: np.array, axis: int = 0) -> np.array:
        return np.take(array, indices, axis=axis)

    def imag(self, array: np.array) -> np.array:
        return np.imag(array)

    def inv(self, tensor: np.array) -> np.array:
        return np.linalg.inv(tensor)

    def is_trainable(self, tensor: np.array) -> bool:
        return False

    def lgamma(self, x: np.array) -> np.array:
        return np.array([mlgamma(v) for v in x])

    def log(self, x: np.array) -> np.array:
        return np.log(x)

    @Autocast()
    def matmul(
        self,
        a: np.array,
        b: np.array,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
    ) -> np.array:
        a = a.T if transpose_a else a
        b = b.T if transpose_b else b
        a = np.conj(a) if adjoint_a else a
        b = np.conj(b) if adjoint_b else b
        return np.matmul(a, b)

    @Autocast()
    def matvec(self, a: np.array, b: np.array, transpose_a=False, adjoint_a=False) -> np.array:
        return self.matmul(a, b, transpose_a, adjoint_a)

    @Autocast()
    def maximum(self, a: np.array, b: np.array) -> np.array:
        return np.maximum(a, b)

    @Autocast()
    def minimum(self, a: np.array, b: np.array) -> np.array:
        return np.minimum(a, b)

    def new_variable(
        self,
        value,
        bounds: Union[Tuple[Optional[float], Optional[float]], None],
        name: str,
        dtype=np.float64,
    ):
        return np.array(value, dtype=dtype)

    def new_constant(self, value, name: str, dtype=np.float64):
        return np.array(value, dtype=dtype)

    def norm(self, array: np.array) -> np.array:
        return np.linalg.norm(array)

    def ones(self, shape: Sequence[int], dtype=np.float64) -> np.array:
        return np.ones(shape, dtype=dtype)

    def ones_like(self, array: np.array) -> np.array:
        return np.ones(array.shape, dtype=array.dtype)

    @Autocast()
    def outer(self, array1: np.array, array2: np.array) -> np.array:
        return np.tensordot(array1, array2, [[], []])

    def pad(
        self,
        array: np.array,
        paddings: Sequence[Tuple[int, int]],
        mode="CONSTANT",
        constant_values=0,
    ) -> np.array:
        if mode == "CONSTANT":
            mode = "constant"
        return np.pad(array, paddings, mode, constant_values=constant_values)

    @staticmethod
    def pinv(matrix: np.array) -> np.array:
        return np.linalg.pinv(matrix)

    @Autocast()
    def pow(self, x: np.array, y: float) -> np.array:
        return np.power(x, y)

    def real(self, array: np.array) -> np.array:
        return np.real(array)

    def reshape(self, array: np.array, shape: Sequence[int]) -> np.array:
        return np.reshape(array, shape)

    def sin(self, array: np.array) -> np.array:
        return np.sin(array)

    def sinh(self, array: np.array) -> np.array:
        return np.sinh(array)

    def solve(self, matrix: np.array, rhs: np.array) -> np.array:
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = np.expand_dims(rhs, -1)
            return np.linalg.solve(matrix, rhs)[..., 0]
        return np.linalg.solve(matrix, rhs)

    def sqrt(self, x: np.array, dtype=None) -> np.array:
        return np.sqrt(self.cast(x, dtype))

    def sum(self, array: np.array, axes: Sequence[int] = None):
        if axes is None:
            return np.sum(array)

        ret = array
        for axis in axes:
            ret = np.sum(ret, axis=axis)
        return ret

    @Autocast()
    def tensordot(self, a: np.array, b: np.array, axes: List[int]) -> np.array:
        return np.tensordot(a, b, axes)

    def tile(self, array: np.array, repeats: Sequence[int]) -> np.array:
        return np.tile(array, repeats)

    def trace(self, array: np.array, dtype=None) -> np.array:
        return self.cast(np.trace(array), dtype)

    def transpose(self, a: np.array, perm: Sequence[int] = None) -> np.array:
        if a is None:
            return None  # TODO: remove and address None inputs where tranpose is used
        return np.transpose(a, axes=perm)

    @Autocast()
    def update_tensor(self, tensor: np.array, indices: np.array, values: np.array):
        ret = deepcopy(tensor)
        for n_index, index in enumerate(indices):
            ret[tuple(index)] = values[n_index]
        return ret

    @Autocast()
    def update_add_tensor(self, tensor: np.array, indices: np.array, values: np.array):
        # https://stackoverflow.com/questions/65734836/numpy-equivalent-to-tf-tensor-scatter-nd-add-method
        indices = np.array(indices)  # figure out why we need this
        indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
        ret = deepcopy(tensor)
        np.add.at(ret, indices, values)
        return ret

    def zeros(self, shape: Sequence[int], dtype=np.float64) -> np.array:
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, array: np.array) -> np.array:
        return np.zeros(np.array(array).shape, dtype=array.dtype)

    def map_fn(self, func, elements):
        # Is this done like this?
        return np.array([func(e) for e in elements])

    def squeeze(self, tensor, axis=None):
        return np.squeeze(tensor, axis=axis)

    def cholesky(self, input: np.ndarray):
        return np.linalg.cholesky(input)

    def Categorical(self, probs: np.ndarray, name: str):
        class Generator:
            def __init__(self, probs):
                self._probs = probs

            def sample(self):
                array = np.random.multinomial(1, pvals=probs)
                return np.where(array == 1)[0][0]

        return Generator(probs)

    def MultivariateNormalTriL(self, loc: np.ndarray, scale_tril: np.ndarray):
        class Generator:
            def __init__(self, mean, cov):
                self._mean = mean
                self._cov = cov

            def sample(self, dtype=None):
                fn = default_rng().multivariate_normal
                ret = fn(self._mean, self._cov)
                return ret

            def prob(self, x):
                return multivariate_normal.pdf(x, mean=self._mean, cov=self._cov)

        scale_tril = scale_tril @ np.transpose(scale_tril)
        return Generator(loc, scale_tril)

    @staticmethod
    def eigvals(tensor: np.array) -> np.ndarray:
        return np.linalg.eigvals(tensor)

    @staticmethod
    def xlogy(x: np.array, y: np.array) -> np.ndarray:
        return scipy_xlogy(x, y)

    @staticmethod
    def eigh(tensor: np.array) -> np.ndarray:
        return np.linalg.eigh(tensor)

    def sqrtm(self, tensor: np.array, rtol=1e-05, atol=1e-08) -> np.ndarray:
        if np.allclose(tensor, 0, rtol=rtol, atol=atol):
            return self.zeros_like(tensor)
        return scipy.linalg.sqrtm(tensor)

    # ~~~~~~~~~~~~~~~~~
    # Special functions
    # ~~~~~~~~~~~~~~~~~

    @staticmethod
    def DefaultEuclideanOptimizer() -> None:
        return None

    def hermite_renormalized(
        self, A: np.array, B: np.array, C: np.array, shape: Tuple[int]
    ) -> np.array:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx + 1/2*Ax^2)` at zero, where the series has :math:`sqrt(n!)`
        at the denominator rather than :math:`n!`. It computes all the amplitudes within the
        tensor of given shape.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The shape of the final tensor.

        Returns:
            The renormalized Hermite polynomial of given shape.
        """

        precision_bits = settings.PRECISION_BITS_HERMITE_POLY

        _A, _B, _C = self.asnumpy(A), self.asnumpy(B), self.asnumpy(C)

        if precision_bits == 128:  # numba
            G = vanilla(tuple(shape), _A, _B, _C)
        else:  # julia (with precision_bits = 512)
            # The following import must come after running "jl = Julia(compiled_modules=False)" in settings.py
            from julia import Main as Main_julia  # pylint: disable=import-outside-toplevel

            _A, _B, _C = (
                _A.astype(np.complex128),
                _B.astype(np.complex128),
                _C.astype(np.complex128),
            )
            G = Main_julia.Vanilla.vanilla(
                _A, _B, _C.item(), np.array(shape, dtype=np.int64), precision_bits
            )

        return G

    def hermite_renormalized_binomial(
        self,
        A: np.array,
        B: np.array,
        C: np.array,
        shape: Tuple[int],
        max_l2: Optional[float],
        global_cutoff: Optional[int],
    ) -> np.array:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx + 1/2*Ax^2)` at zero, where the series has :math:`sqrt(n!)`
        at the denominator rather than :math:`n!`. The computation fills a tensor of given shape
        up to a given L2 norm or global cutoff, whichever applies first. The max_l2 value, if
        not provided, is set to the default value of the AUTOCUTOFF_PROBABILITY setting.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The shape of the final tensor (local cutoffs).
            max_l2 (float): The maximum squared L2 norm of the tensor.
            global_cutoff (optional int): The global cutoff.

        Returns:
            The renormalized Hermite polynomial of given shape.
        """
        G, _ = binomial(
            tuple(shape),
            A,
            B,
            C,
            max_l2=max_l2 or settings.AUTOCUTOFF_PROBABILITY,
            global_cutoff=global_cutoff or sum(shape) - len(shape) + 1,
        )

        return G

    def reorder_AB_bargmann(self, A: np.array, B: np.array) -> Tuple[np.array, np.array]:
        r"""In mrmustard.math.numba.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = np.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        A = self.gather(A, ordering, axis=1)
        A = self.gather(A, ordering)
        B = self.gather(B, ordering)
        return A, B

    def hermite_renormalized_diagonal(
        self, A: np.array, B: np.array, C: np.array, cutoffs: Tuple[int]
    ) -> np.array:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB(A, B, C, cutoffs=cutoffs)

    def hermite_renormalized_diagonal_reorderedAB(
        self, A: np.array, B: np.array, C: np.array, cutoffs: Tuple[int]
    ) -> np.array:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx - Ax^2)` at zero, where the series has :math:`sqrt(n!)` at the
        denominator rather than :math:`n!`. Note the minus sign in front of ``A``.

        Calculates the diagonal of the Fock representation (i.e. the PNR detection probabilities of all modes)
        by applying the recursion relation in a selective manner.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            cutoffs: upper boundary of photon numbers in each mode

        Returns:
            The renormalized Hermite polynomial.
        """
        poly0, _, _, _, _ = hermite_multidimensional_diagonal(A, B, C, cutoffs)

        return poly0

    def hermite_renormalized_1leftoverMode(
        self, A: np.array, B: np.array, C: np.array, cutoffs: Tuple[int]
    ) -> np.array:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, B, C, cutoffs=cutoffs)

    def hermite_renormalized_1leftoverMode_reorderedAB(
        self, A: np.array, B: np.array, C: np.array, cutoffs: Tuple[int]
    ) -> np.array:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx - Ax^2)` at zero, where the series has :math:`sqrt(n!)` at the
        denominator rather than :math:`n!`. Note the minus sign in front of ``A``.

        Calculates all possible Fock representations of mode 0,
        where all other modes are PNR detected.
        This is done by applying the recursion relation in a selective manner.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            cutoffs: upper boundary of photon numbers in each mode

        Returns:
            The renormalized Hermite polynomial.
        """
        poly0, _, _, _, _ = hermite_multidimensional_1leftoverMode(A, B, C, cutoffs)
        return poly0

    def getitem(tensor, *, key):
        value = np.array(tensor)[key]
        return value

    def setitem(tensor, value, *, key):
        _tensor = np.array(tensor)
        value = np.array(value)
        _tensor[key] = value

        return _tensor
