import numpy as np
import tensorflow as tf
from scipy.special import binom
from scipy.stats import unitary_group
from typing import List, Tuple, Callable, Optional, Sequence, Union
from itertools import product
from mrmustard.backends import BackendInterface
from thewalrus.fock_gradients import hermite_numba, hermite_numba_gradient

#  The reason why we have a class with methods and not a namespace with functions
#  is that we want to enforce the interface, to ensure compatibility of a new backend
#  with the rest of the codebase. This is dependency inversion.


class Backend(BackendInterface):

    dtype_order = (tf.float16, float, tf.float32, tf.float64, tf.complex64, tf.complex128)
    no_cast = (tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32, tf.uint32, tf.int64, tf.uint64)

    def autocast(self, *args, **kwargs):
        'A method that casts to the highest numerical type'
        args_dtypes = [arg.dtype for arg in args if hasattr(arg, 'dtype') and arg.dtype not in self.no_cast]
        kwargs_dtypes = {k: v.dtype for k, v in kwargs.items() if hasattr(v, 'dtype') and v.dtype not in self.no_cast}
        dtypes = args_dtypes + list(kwargs_dtypes.values())
        if len(dtypes) == 0:
            return args, kwargs
        dtype = max(dtypes, key=lambda d: self.dtype_order.index(d))
        for arg in args:
            if hasattr(arg, 'dtype') and arg.dtype not in self.no_cast:
                arg = tf.cast(arg, dtype)
        for k, v in kwargs.items():
            if hasattr(v, 'dtype') and v.dtype not in self.no_cast:
                kwargs[k] = tf.cast(v, dtype)
        return args, kwargs

    def __getattr__(self, name: str) -> Callable:
        r"""we wrap the methods when they are called, to ensure that they are called with the correct dtypes"""
        try:
            func = self.__dict__[name]
        except KeyError:
            raise AttributeError("Backend '{}' has no method '{}'".format(self.__class__.__name__, name))

        def wrapper(*args, **kwargs):
            args, kwargs = self.autocast(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper

    # ~~~~~~~~~
    # Basic ops
    # ~~~~~~~~~

    def astensor(self, array: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        return tf.convert_to_tensor(array)

    def conj(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    def tile(self, array: tf.Tensor, repeats: Sequence[int]) -> tf.Tensor:
        return tf.tile(array, repeats)

    def einsum(self, string: str, *tensors) -> tf.Tensor:
        return tf.einsum(string, *tensors)

    def diag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.diag(array)

    def diag_part(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.diag_part(array)

    def reshape(self, array, shape) -> tf.Tensor:
        return tf.reshape(array, shape)

    def sum(self, array, axis=None):
        return tf.reduce_sum(array, axis)

    def arange(self, start: int, limit: int = None, delta: int = 1) -> tf.Tensor:
        return tf.range(start, limit, delta, dtype=tf.float64)

    def outer(self, array1: tf.Tensor, array2: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(array1, array2, [[], []])

    def eye(self, size: int, dtype=tf.float64) -> tf.Tensor:
        return tf.eye(size, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
        return tf.zeros(shape, dtype=dtype)

    def zeros_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(array)

    def ones(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
        return tf.ones(shape, dtype=dtype)

    def ones_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(array)

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def trace(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.trace(array)

    def tensordot(self, a: tf.Tensor, b: tf.Tensor, axes: List[int]) -> tf.Tensor:
        return tf.tensordot(a, b, axes)

    def transpose(self, a: tf.Tensor, perm: List[int]):
        return tf.transpose(a, perm)

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def exp(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(array)

    def norm(self, array: tf.Tensor) -> tf.Tensor:
        'Note that the norm preserves the type of array'
        return tf.linalg.norm(array)

    def unitary_to_orthogonal(self, U):
        r"""Unitary to orthogonal mapping.
        Args:
            U (array): unitary matrix in U(n)
        Returns:
            array: Orthogonal matrix in O(2n)
        """
        X = tf.math.real(U)
        Y = tf.math.imag(U)
        return self.block([[X, -Y], [Y, X]])

    def random_symplectic(self, dim: int = 1) -> tf.Tensor:
        'a random symplectic matrix in Sp(2*dim)'
        if dim == 1:
            W = np.exp(1j * np.random.uniform(size=(1, 1)))
            V = np.exp(1j * np.random.uniform(size=(1, 1)))
        else:
            W = unitary_group.rvs(dim=dim)
            V = unitary_group.rvs(dim=dim)
        r = np.random.uniform(size=dim)
        OW = self.unitary_to_orthogonal(W)
        OV = self.unitary_to_orthogonal(V)
        dd = self.concat([self.exp(-r), np.exp(r)], axis=0)
        return self.einsum("ij,j,jk->ik", OW, dd, OV)

    def random_orthogonal(self, dim: int = 1) -> tf.Tensor:
        'a random orthogonal matrix in O(2*dim)'
        if dim == 1:
            W = self.exp(1j * np.random.uniform(size=(1, 1)))
        else:
            W = unitary_group.rvs(dim=dim)
        return self.unitary_to_orthogonal(W)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Non-tf methods (will be refactored out of the backend)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def block(self, blocks: List[List[tf.Tensor]]) -> tf.Tensor:
        rows = [self.concat(row, axis=-1) for row in blocks]
        return self.concat(rows, axis=-2)

    def tile_vec(self, vec, num_modes: int):
        if vec is None:
            return None
        if vec.shape[-1] != 2 * num_modes:
            x, y = vec
            vec = self.concat([self.tile([x], [num_modes]), self.tile([y], [num_modes])], axis=-1)
        return vec

    def tile_mat(self, mat, num_modes):
        if mat is None:
            return None
        if mat.shape[-1] != 2 * num_modes:
            b = mat[..., 1, 0]
            c = mat[..., 0, 1]
            mat = (
                self.diag(self.tile_vec(self.diag_part(mat), num_modes))
                + self.diag(tf.tile([b], [num_modes]), k=num_modes)
                + self.diag(tf.tile([c], [num_modes]), k=-num_modes)
            )
        return mat

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Specialized methods for phase space
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add(self, old: tf.Tensor, new: Optional[tf.Tensor], modes: List[int]) -> tf.Tensor:
        'adds two phase-space tensors (cov matrices, displacement vectors, etc..) in the specified modes'
        if new is None:
            return old
        N = old.shape[-1] // 2
        indices = modes + [m + N for m in modes]
        return tf.tensor_scatter_nd_add(old, list(product(*[indices] * len(new.shape))), tf.reshape(new, -1))

    def matmul(self, a, b, modes: List[int]):
        'matrix multiplication of two phase-space matrices (cov, symplectic) in the specified modes'
        assert a.shape[-1] == b.shape[0]  # NOTE: this will change for batching
        N = a.shape[-1] // 2
        indices = tf.convert_to_tensor(modes + [m + N for m in modes])
        rows = tf.matmul(a, tf.gather(b, indices))
        return tf.tensor_scatter_nd_update(b, indices[:, None], rows)

    def matvec(self, mat: Optional[tf.Tensor], vec: tf.Tensor, modes: List[int]) -> tf.Tensor:
        'matrix-vector multiplication between a phase-space matrix and a vector in the specified modes'
        if mat is None:
            return vec
        N = vec.shape[-1] // 2
        indices = tf.convert_to_tensor(modes + [m + N for m in modes])
        updates = tf.linalg.matvec(mat, tf.gather(vec, indices))
        return tf.tensor_scatter_nd_update(vec, indices[:, None], updates)

    def all_diagonals(self, rho: tf.Tensor, real: bool) -> tf.Tensor:
        cutoffs = rho.shape[: rho.ndim // 2]
        rho = tf.reshape(rho, (np.prod(cutoffs), np.prod(cutoffs)))
        diag = tf.linalg.diag_part(rho)
        if real:
            return tf.math.real(tf.reshape(diag, cutoffs))
        else:
            return tf.reshape(diag, cutoffs)

    def constraint_func(self, bounds: Tuple[Optional[float], Optional[float]]) -> Optional[Callable]:
        bounds = (-np.inf if bounds[0] is None else bounds[0], np.inf if bounds[1] is None else bounds[1])
        if not bounds == (-np.inf, np.inf):
            constraint: Optional[Callable] = lambda x: tf.clip_by_value(x, bounds[0], bounds[1])
        else:
            constraint = None
        return constraint

    def new_variable(self, value, bounds: Tuple[Optional[float], Optional[float]], name: str):
        return tf.Variable(value, dtype=tf.float64, name=name, constraint=self.constraint_func(bounds))

    def new_constant(self, value, name: str):
        return tf.constant(value, dtype=tf.float64, name=name)

    def poisson(self, max_k: int, rate: tf.Tensor):
        "poisson distribution up to max_k"
        k = tf.range(max_k, dtype=tf.float64)
        rate = tf.cast(rate, tf.float64)
        return tf.math.exp(k * tf.math.log(rate + 1e-9) - rate - tf.math.lgamma(k + 1.0))

    def binomial_conditional_prob(self, success_prob: tf.Tensor, dim_out: int, dim_in: int):
        "P(out|in) = binom(in, out) * (1-success_prob)**(in-out) * success_prob**out"
        in_ = tf.range(dim_in, dtype=tf.float64)[None, :]
        out_ = tf.range(dim_out, dtype=tf.float64)[:, None]
        return tf.cast(binom(in_, out_), tf.float64) * success_prob ** out_ * (1.0 - success_prob) ** tf.math.maximum(in_ - out_, 0.0)

    def convolve_probs_1d(self, prob: tf.Tensor, other_probs: List[tf.Tensor]) -> tf.Tensor:
        "Convolution of a joint probability with a list of single-index probabilities"

        if prob.ndim > 3 or len(other_probs) > 3:
            raise ValueError("cannot convolve arrays with more than 3 axes")
        if not all([q.ndim == 1 for q in other_probs]):
            raise ValueError("other_probs must contain 1d arrays")
        if not all([len(q) == s for q, s in zip(other_probs, prob.shape)]):
            raise ValueError("The length of the 1d prob vectors must match shape of prob")

        q = other_probs[0]
        for q_ in other_probs[1:]:
            q = q[..., None] * q_[(None,) * q.ndim + (slice(None),)]

        return self.convolve_probs(prob, q)

    def convolve_probs(self, prob: tf.Tensor, other: tf.Tensor) -> tf.Tensor:
        r"""Convolve two probability distributions.
        Note that the output is not a complete joint probability,
        as it's computed only up to the dimension of the base probs."""

        if prob.ndim > 3 or other.ndim > 3:
            raise ValueError("cannot convolve arrays with more than 3 axes")
        if not prob.shape == other.shape:
            raise ValueError("prob and other must have the same shape")

        prob_padded = tf.pad(prob, [(s - 1, 0) for s in other.shape])
        other_reversed = other[(slice(None, None, -1),) * other.ndim]
        return tf.nn.convolution(
            prob_padded[None, ..., None],
            other_reversed[..., None, None],
            padding="VALID",
            data_format="N" + ("HD"[: other.ndim - 1])[::-1] + "WC",
        )[0, ..., 0]

    @staticmethod
    @tf.custom_gradient
    def hermite_renormalized(A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, shape: Sequence[int]) -> tf.Tensor:
        r"""
        Renormalized multidimensional Hermite polynomial given by the Taylor series of exp(Ax^2 + Bx + C) at zero.
            Args:
                A: The A matrix.
                B: The B vector.
                C: The C scalar.
                shape: The shape of the final tensor.
            Returns:
                The Fock state.
        """
        poly = hermite_numba(A, B, C, shape)

        def grad(dy):
            return hermite_numba_gradient(dy, poly, A, B, C, shape)

        return poly, grad

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Getting ready for Fock space
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def hermite_parameters(self, cov: Tensor, means: Tensor, mixed: bool, hbar: float) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Returns the A matrix, B vector and C scalar given a Wigner covariance matrix and a means vector.
        The A, B, C triple is needed to compute the Fock representation of the state.
        Args:
            cov: The Wigner covariance matrix.
            means: The Wigner means vector.
            mixed: Whether the state vector is mixed or not.
            hbar: The value of the Plank constant.
        Returns:
            The A matrix, B vector and C scalar.
        """
        num_modes = means.shape[-1] // 2

        # cov and means in the amplitude basis
        R = self.math.rotmat(num_modes)
        sigma = self.math.matmul(R,  cov/hbar, self.math.dagger(R))
        beta = self.math.matvec(R, means/self.math.sqrt(hbar))

        sQ = sigma + 0.5 * self.math.eye(2 * num_modes)
        sQinv = self.math.inv(sQ)
        X = self.math.Xmat(num_modes)
        A = self.math.matmul(X, self.math.eye(2 * num_modes) - sQinv)
        B = self.math.matvec(self.math.transpose(sQinv), self.math.conj(beta))
        exponent = -0.5 * self.math.sum(self.math.conj(beta)[:, None] * sQinv * beta[None, :])
        T = self.math.exp(exponent) / self.math.sqrt(self.math.det(sQ))
        N = num_modes - num_modes * mixed
        return (
            A[N:, N:],
            B[N:],
            T ** (0.5 + 0.5 * mixed),
        )  # will be off by global phase because T is real even for pure states