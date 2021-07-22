import numpy as np
import tensorflow as tf
from scipy.stats import unitary_group
from scipy.special import binom
from typing import List, Tuple, Callable, Optional, Union
from itertools import product

from mrmustard.backends import MathBackendInterface


class MathBackend(MathBackendInterface):
    def conj(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    def diag(self, array: tf.Tensor) -> tf.Tensor:
        if array.ndim == 1:
            return tf.linalg.diag(array)
        else:
            return tf.linalg.diag_part(array)

    def reshape(self, array, shape) -> tf.Tensor:
        return tf.reshape(array, shape)

    def sum(self, array, axis=None):
        return tf.reduce_sum(array, axis)

    def arange(self, start, limit=None, delta=1) -> tf.Tensor:
        return tf.range(start, limit, delta, dtype=tf.float64)

    def outer(self, arr1: tf.Tensor, arr2: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(arr1, arr2, [[], []])

    def identity(self, size: int) -> tf.Tensor:
        return tf.eye(size, dtype=tf.float64)

    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype=tf.float64) -> tf.Tensor:
        return tf.zeros(shape, dtype=dtype)

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def trace(self, array: tf.Tensor) -> tf.Tensor:
        cutoffs = array.shape[: array.ndim // 2]
        array = tf.reshape(array, (np.prod(cutoffs), np.prod(cutoffs)))
        return tf.linalg.trace(array)

    def tensordot(self, a, b, axes, dtype=None):
        if dtype is not None:
            a = tf.cast(a, dtype)
            b = tf.cast(b, dtype)
        return tf.tensordot(a, b, axes)

    def transpose(self, a, perm):
        return tf.transpose(a, perm)

    def block(self, blocks: List[List]):
        rows = [tf.concat(row, axis=-1) for row in blocks]
        return tf.concat(rows, axis=-2)

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def norm(self, array):
        return tf.linalg.norm(array)

    def add(self, old: tf.Tensor, new: Optional[tf.Tensor], modes: List[int]) -> tf.Tensor:
        if new is None:
            return old
        N = old.shape[-1] // 2
        indices = modes + [m + N for m in modes]
        return tf.tensor_scatter_nd_add(old, list(product(*[indices] * len(new.shape))), tf.reshape(new, -1))

    def sandwich(self, bread: Optional[tf.Tensor], filling: tf.Tensor, modes: List[int]) -> tf.Tensor:
        if bread is None:
            return filling
        N = filling.shape[-1] // 2
        indices = tf.convert_to_tensor(modes + [m + N for m in modes])
        rows = tf.matmul(bread, tf.gather(filling, indices))
        filling = tf.tensor_scatter_nd_update(filling, indices[:, None], rows)
        columns = bread @ tf.gather(tf.transpose(filling), indices)
        return tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(filling), indices[:, None], columns))

    def matvec(self, mat: Optional[tf.Tensor], vec: tf.Tensor, modes: List[int]) -> tf.Tensor:
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

    def new_symplectic_parameter(self, num_modes: int = 1) -> tf.Tensor:
        if num_modes == 1:
            W = np.exp(1j * np.random.uniform(size=(1, 1)))
            V = np.exp(1j * np.random.uniform(size=(1, 1)))
        else:
            W = unitary_group.rvs(dim=num_modes)
            V = unitary_group.rvs(dim=num_modes)
        r = np.random.uniform(size=num_modes)
        OW = self.unitary_to_orthogonal(W)
        OV = self.unitary_to_orthogonal(V)
        dd = tf.concat([tf.math.exp(-r), tf.math.exp(r)], axis=0)
        return tf.einsum("ij,j,jk->ik", OW, dd, OV)

    def new_orthogonal_parameter(self, num_modes: int = 1) -> tf.Tensor:
        if num_modes == 1:
            W = np.exp(1j * np.random.uniform(size=(1, 1)))
        else:
            W = unitary_group.rvs(dim=num_modes)
        return self.unitary_to_orthogonal(W)

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

    def make_constraint(self, bounds: Tuple[Optional[float], Optional[float]]) -> Optional[Callable]:
        bounds = (-np.inf if bounds[0] is None else bounds[0], np.inf if bounds[1] is None else bounds[1])
        if not bounds == (-np.inf, np.inf):
            constraint: Optional[Callable] = lambda x: tf.clip_by_value(x, bounds[0], bounds[1])
        else:
            constraint = None
        return constraint

    def new_variable(self, value, bounds: Tuple[Optional[float], Optional[float]], name: str):
        return tf.Variable(value, dtype=tf.float64, name=name, constraint=self.make_constraint(bounds))

    def new_constant(self, value, name: str):
        return tf.constant(value, dtype=tf.float64, name=name)

    def tile_vec(self, vec, num_modes: int):
        if vec is None:
            return None
        if vec.shape[-1] != 2 * num_modes:
            x, y = vec
            vec = tf.concat([tf.tile([x], [num_modes]), tf.tile([y], [num_modes])], axis=-1)
        return vec

    def tile_mat(self, mat, num_modes):
        if mat is None:
            return None
        if mat.shape[-1] != 2 * num_modes:
            b = mat[..., 1, 0]
            c = mat[..., 0, 1]
            mat = (
                tf.linalg.diag(self.tile_vec(tf.linalg.diag_part(mat), num_modes))
                + tf.linalg.diag(tf.tile([b], [num_modes]), k=num_modes)
                + tf.linalg.diag(tf.tile([c], [num_modes]), k=-num_modes)
            )
        return mat

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
