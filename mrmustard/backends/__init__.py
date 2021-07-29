from abc import ABC
from mrmustard.typing import *
import numpy as np
from functools import lru_cache


class BackendInterface(ABC):
    r"""
    The interface that all backends must implement.
    All methods are pure (no side effects) and are be used by the plugins.
    """

    dtype_order: Tuple
    no_cast: Tuple

    def autocast(self, *args, **kwargs):
        'A method that casts to the highest numerical type according to dtype_order'
        args_dtypes = [arg.dtype for arg in args if (hasattr(arg, 'dtype') and arg.dtype not in self.no_cast)]
        kwargs_dtypes = {k: v.dtype for k, v in kwargs.items() if (hasattr(v, 'dtype') and v.dtype not in self.no_cast)}
        dtypes = args_dtypes + list(kwargs_dtypes.values())
        if len(dtypes) == 0:
            return args, kwargs
        dtype = max(dtypes, key=lambda d: self.dtype_order.index(d))
        for arg in args:
            if hasattr(arg, 'dtype') and arg.dtype not in self.no_cast:
                arg = self.cast(arg, dtype)
        for k, v in kwargs.items():
            if hasattr(v, 'dtype') and v.dtype not in self.no_cast:
                kwargs[k] = self.cast(v, dtype)
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

    def astensor(self, array: Tensor) -> Tensor: ...

    def conj(self, array: Tensor) -> Tensor: ...

    def real(self, array: Tensor) -> Tensor: ...

    def imag(self, array: Tensor) -> Tensor: ...

    def exp(self, array: Tensor) -> Tensor: ...

    def lgamma(self, x: Tensor) -> Tensor: ...

    def log(self, x: Tensor) -> Tensor: ...

    def maximum(self, a: Tensor, b: Tensor) -> Tensor: ...

    def minimum(self, a: Tensor, b: Tensor) -> Tensor: ...

    def abs(self, array: Tensor) -> Tensor: ...

    def norm(self, array: Tensor) -> Tensor: ...

    def matmul(self, a: Tensor, b: Tensor, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False)  -> Tensor: ...

    def matvec(self, a: Tensor, b: Tensor, transpose_a=False, adjoint_a=False) -> Tensor: ...

    def tensordot(self, a: Tensor, b: Tensor, axes: List[int]) -> Tensor: ...

    def einsum(self, string: str, *tensors) -> tTesor: ...

    def inv(self, tensor: Tensor) -> Tensor: ...

    def pinv(self, matrix: Tensor) -> Tensor: ...

    def det(self, array: Tensor) -> Tensor: ...

    def tile(self, array: Tensor, repeats: Sequence[int]) -> Tensor: ...

    def diag(self, array: Tensor) -> Tensor: ...

    def diag_part(self, array: Tensor) -> Tensor: ...

    def pad(self, array: Tensor, paddings: Sequence[Tuple[int, int]], mode='CONSTANT', constant_values=0) -> Tensor: ...

    def convolution(self, array: Tensor, filters: Tensor, strides: List[int], padding='VALID', data_format='NWC', dilations: Optional[List[int]] = None) -> Tensor: ...

    def transpose(self, a: Tensor, perm: Sequence[int]): ...

    def reshape(self, array: Tensor, shape: Sequence[int]) -> Tensor: ...

    def sum(self, array: Tensor, axes: Sequence[int]=None): ...

    def arange(self, start: int, limit: int = None, delta: int = 1) -> Tensor: ...  # NOTE: should be float64 by default

    def outer(self, array1: Tensor, array2: Tensor) -> Tensor: ...

    def eye(self, size: int, dtype=float64) -> Tensor: ...

    def zeros(self, shape: Sequence[int], dtype) -> Tensor: # NOTE: should be float64 by default

    def zeros_like(self, array: Tensor) -> Tensor: ...

    def ones(self, shape: Sequence[int], dtype) -> Tensor: ... # NOTE: should be float64 by default

    def ones_like(self, array: Tensor) -> Tensor: ...

    def gather(self, array: Tensor, indices: Tensor, axis: Tensor) -> Tensor: ...

    def trace(self, array: Tensor) -> Tensor: ...

    def concat(self, values: Sequence[Tensor], axis: int) -> Tensor: ...

    def update_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor: ...

    def update_add_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor: ...

    def constraint_func(self, bounds: Tuple[Optional[float], Optional[float]]) -> Optional[Callable]: ...

    def new_variable(self, value: Tensor, bounds: Tuple[Optional[float], Optional[float]], name: str) -> Tensor: ...

    def new_constant(self, value: Tensor, name: str) -> Tensor: ...

    def hermite_renormalized(self, A: Tensor, B: Tensor, C: Tensor, shape: Sequence[int]) -> Tensor: ...

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Methods that build on the basic ops and don't need to be overridden in the backend implementation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def dagger(self, array: Tensor) -> Tensor:
        return self.conj(self.transpose(array))

    def block(self, blocks: List[List[Tensor]]) -> Tensor:
        rows = [self.concat(row, axis=-1) for row in blocks]
        return self.concat(rows, axis=-2)

    def unitary_to_orthogonal(self, U):
        r"""Unitary to orthogonal mapping.
        Args:
            U (array): unitary matrix in U(n)
        Returns:
            array: Orthogonal matrix in O(2n)
        """
        X = self.real(U)
        Y = self.imag(U)
        return self.block([[X, -Y], [Y, X]])

    def random_symplectic(self, dim: int = 1) -> Tensor:
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

    def random_orthogonal(self, dim: int = 1) -> Tensor:
        'a random orthogonal matrix in O(2*dim)'
        if dim == 1:
            W = self.exp(1j * np.random.uniform(size=(1, 1)))
        else:
            W = unitary_group.rvs(dim=dim)
        return self.unitary_to_orthogonal(W)

    def single_mode_to_multimode_vec(self, vec, num_modes: int):
        r"""
        Expand a 2-vector (i.e. single-mode) to a larger number of modes.
        """
        if vec.shape[-1] != 2:
            raise ValueError("vec must be 2-dimensional (i.e. single-mode)")
        x, y = vec
        vec = self.concat([self.tile([x], [num_modes]), self.tile([y], [num_modes])], axis=-1)
        return vec

    def single_mode_to_multimode_mat(self, mat: Tensor, num_modes: int):
        r"""
        Expand a 2x2 matrix (i.e. single-mode) to a larger number of modes.
        """
        if mat.shape[-2:] != (2, 2):
            raise ValueError("mat must be a single-mode (2x2) matrix")
        b = mat[..., 1, 0]
        c = mat[..., 0, 1]
        mat = (
            self.diag(self.tile_vec(self.diag_part(mat), num_modes))
            + self.diag(self.tile([b], [num_modes]), k=num_modes)
            + self.diag(self.tile([c], [num_modes]), k=-num_modes)
        )
        return mat

    @lru_cache()
    def Xmat(num_modes: int):
        r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`
        Args:
            num_modes (int): positive integer
        Returns:
            array: :math:`2N\times 2N` array
        """
        I = np.identity(num_modes)
        O = np.zeros((num_modes, num_modes))
        return np.block([[O, I], [I, O]])


    @lru_cache()
    def rotmat(num_modes: int):
        "Rotation matrix from quadratures to complex amplitudes"
        idl = np.identity(num_modes)
        return np.sqrt(0.5) * np.block([[idl, 1j * idl], [idl, -1j * idl]])


    @lru_cache()
    def J(num_modes: int):
        "Symplectic form"
        I = np.identity(num_modes)
        O = np.zeros_like(I)
        return np.block([[O, I], [-I, O]])

    def add_at_modes(self, old: Tensor, new: Optional[Tensor], modes: List[int]) -> Tensor:
        'adds two phase-space tensors (cov matrices, displacement vectors, etc..) on the specified modes'
        if new is None:
            return old
        N = old.shape[-1] // 2
        indices = modes + [m + N for m in modes]
        return self.update_add_tensor(old, list(product(*[indices] * len(new.shape))), self.reshape(new, -1))

    def left_matmul_at_modes(self, a_partial: Tensor, b_full: Tensor, modes: List[int]) -> Tensor:
        r"""
        Left Matrix multiplication of a restricted matrix and a full matrix on the specified modes.
        It assumes that that `a_partial` is a matrix operating on M modes and that `modes` is a list of M integers,
        i.e. it will apply a_partial on the corresponding M modes of `b_full`.

        Args:
            a_partial (array): :math:`2M\times 2M` array
            b_full (array): :math:`2N\times 2N` array
            modes (list): list of `M` modes to perform the multiplication on
        Returns:
            array: :math:`2N\times 2N` array
        """
        if a_partial is None:
            return b_full
        N = b_full.shape[-1] // 2
        indices = self.astensor(modes + [m + N for m in modes])
        b_rows = self.gather(b_full, indices, axis=0)
        b_rows = self.matmul(a_partial, b_rows)
        return self.update_tensor(b_full, indices[:, None], b_rows)

    def right_matmul_at_modes(self, a_full: Tensor, b_partial: Tensor, modes: List[int]) -> Tensor:
        r"""
        Right Matrix multiplication of a full matrix and a restricted matrix on the specified modes.
        It assumes that that `b_partial` is a matrix operating on M modes and that `modes` is a list of M integers,
        i.e. it will apply b_partial on the corresponding M modes of `a_full`.

        Args:
            a_full (array): :math:`2N\times 2N` array
            b_partial (array): :math:`2M\times 2M` array
            modes (list): list of `M` modes to perform the multiplication on
        Returns:
            array: :math:`2N\times 2N` array
        """
        if b_partial is None:
            return a
        N = a_full.shape[-1] // 2
        indices = self.astensor(modes + [m + N for m in modes])
        a_columns = self.gather(a_full, indices, axis=1)
        a_columns = self.matmul(a_columns, b_partial)
        return self.update_tensor(a_full, indices[None, :], a_columns)

    def matvec_at_modes(self, mat: Optional[Tensor], vec: Tensor, modes: List[int]) -> Tensor:
        'matrix-vector multiplication between a phase-space matrix and a vector in the specified modes'
        if mat is None:
            return vec
        N = vec.shape[-1] // 2
        indices = self.astensor(modes + [m + N for m in modes])
        updates = self.matvec(mat, self.gather(vec, indices))
        return self.update_tensor(vec, indices[:, None], updates)

    def all_diagonals(self, rho: Tensor, real: bool) -> Tensor:
        cutoffs = rho.shape[: rho.ndim // 2]
        rho = self.reshape(rho, (np.prod(cutoffs), np.prod(cutoffs)))
        diag = self.diag_part(rho)
        if real:
            return self.real(self.reshape(diag, cutoffs))
        else:
            return self.reshape(diag, cutoffs)

    def poisson(self, max_k: int, rate: Tensor) -> Tensor:
        "poisson distribution up to max_k"
        k = self.arange(max_k)
        rate = self.cast(rate, k.dtype)
        return self.exp(k * self.log(rate + 1e-9) - rate - self.lgamma(k + 1.0))

    def binomial_conditional_prob(self, success_prob: Tensor, dim_out: int, dim_in: int):
        "P(out|in) = binom(in, out) * (1-success_prob)**(in-out) * success_prob**out"
        in_ = self.arange(dim_in)[None, :]
        out_ = self.arange(dim_out)[:, None]
        return self.cast(binom(in_, out_), in_.dtype) * success_prob ** out_ * (1.0 - success_prob) ** self.maximum(in_ - out_, 0.0)

    def convolve_probs_1d(self, prob: Tensor, other_probs: List[Tensor]) -> Tensor:
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

    def convolve_probs(self, prob: Tensor, other: Tensor) -> Tensor:
        r"""Convolve two probability distributions (up to 3D) with the same shape.
        Note that the output is not guaranteed to be a complete joint probability,
        as it's computed only up to the dimension of the base probs."""

        if prob.ndim > 3 or other.ndim > 3:
            raise ValueError("cannot convolve arrays with more than 3 axes")
        if not prob.shape == other.shape:
            raise ValueError("prob and other must have the same shape")

        prob_padded = self.pad(prob, [(s - 1, 0) for s in other.shape])
        other_reversed = other[(slice(None, None, -1),) * other.ndim]
        return self.convolution(
            prob_padded[None, ..., None],
            other_reversed[..., None, None],
            padding="VALID",
            data_format="N" + ("HD"[: other.ndim - 1])[::-1] + "WC",
        )[0, ..., 0]