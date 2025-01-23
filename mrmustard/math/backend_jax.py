"""This module contains the JAX backend."""

from __future__ import annotations
from typing import Callable, Sequence
import os
import jax

jax.config.update('jax_default_device', jax.devices('cpu')[0])  # comment this line to run on GPU
jax.config.update("jax_enable_x64", True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # comment this line to run on GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # comment this line to run on GPU

import jax.numpy as jnp
import jax.scipy as jsp
from jax.experimental import host_callback
import numpy as np

from ..utils.settings import settings
from ..utils.typing import Tensor, Trainable
from .autocast import Autocast
from .backend_base import BackendBase
from .lattice import strategies
from functools import partial

from mrmustard.math.lattice.strategies.compactFock.inputValidation import (
    grad_hermite_multidimensional_1leftoverMode,
    grad_hermite_multidimensional_diagonal,
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
    hermite_multidimensional_diagonal_batch,
)


# pylint: disable=too-many-public-methods
class BackendJax(BackendBase):
    """A JAX backend implementation."""

    int32 = jnp.int32
    float32 = jnp.float32
    float64 = jnp.float64
    complex64 = jnp.complex64
    complex128 = jnp.complex128

    def __init__(self):
        super().__init__(name="jax")

    def __repr__(self) -> str:
        return "BackendJax()"

    @partial(jax.jit, static_argnames=['self'])
    def abs(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(array)

    @partial(jax.jit, static_argnames=['self'])
    def any(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.any(array)

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.arange(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: jnp.ndarray) -> np.ndarray:
        return np.array(tensor)


    #@partial(jax.jit, static_argnames=['self', 'axes'])
    def block(self, blocks: list[list[jnp.ndarray]], axes=(-2, -1)) -> jnp.ndarray:
        #jax.debug.print('blocks={blocks}, axes={axes}, shape={shape1}, {shape2}, {shape3}, {shape4}', blocks=blocks, axes=axes, shape1=blocks[0][0].shape, shape2=blocks[0][1].shape, shape3=blocks[1][0].shape, shape4=blocks[1][1].shape)
        rows = [self.concat(row, axis=axes[1]) for row in blocks]
        return self.concat(rows, axis=axes[0])

    @partial(jax.jit, static_argnames=['self', 'axis'])
    def prod(self, x: jnp.ndarray, axis: int | None):
        return jnp.prod(x, axis=axis)

    def assign(self, tensor: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        return value  # JAX arrays are immutable, so we just return the new value

    def astensor(self, array: np.ndarray | jnp.ndarray, dtype=None) -> jnp.ndarray:
        dtype = dtype or (array.dtype.name if hasattr(array, 'dtype') else None)
        return jnp.asarray(array, dtype=dtype)

    #@partial(jax.jit, static_argnames=['self', 'dtype'])
    def atleast_1d(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.atleast_1d(array)

    #@partial(jax.jit, static_argnames=['self', 'dtype'])
    def atleast_2d(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.atleast_2d(array)
    
    @partial(jax.jit, static_argnames=['self'])
    def log(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(array)

    #@partial(jax.jit, static_argnames=['self', 'dtype'])
    def atleast_3d(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        if isinstance(array, tuple):
            raise ValueError("Tuple input is not supported for atleast_3d.")
        array = self.atleast_2d(self.atleast_1d(array))
        if len(array.shape) == 2:
            array = array[None, ...]
        return array

    @partial(jax.jit, static_argnames=['self'])
    def block_diag(self, mat1: jnp.ndarray, mat2: jnp.ndarray) -> jnp.ndarray:
        Za = self.zeros((mat1.shape[-2], mat2.shape[-1]), dtype=mat1.dtype)
        Zb = self.zeros((mat2.shape[-2], mat1.shape[-1]), dtype=mat1.dtype)
        return self.concat(
            [self.concat([mat1, Za], axis=-1), self.concat([Zb, mat2], axis=-1)],
            axis=-2,
        )

    def constraint_func(self, bounds: tuple[float | None, float | None]) -> Callable:
        lower = -jnp.inf if bounds[0] is None else bounds[0]
        upper = jnp.inf if bounds[1] is None else bounds[1]
        
        @jax.jit
        def constraint(x):
            return jnp.clip(x, lower, upper)
        return constraint

    @partial(jax.jit, static_argnames=['self', 'dtype'])
    def cast(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        if dtype is None:
            return array
        return jnp.asarray(array, dtype=dtype)

    @partial(jax.jit, static_argnames=['self', 'axis'])
    def concat(self, values: Sequence[jnp.ndarray], axis: int) -> jnp.ndarray:
        return jnp.concatenate(values, axis)
    
    @partial(jax.jit, static_argnames=['self'])
    def allclose(self, array1: jnp.ndarray, array2: jnp.ndarray, atol=1e-9) -> bool:
        return jnp.allclose(array1, array2, atol=atol)

    @partial(jax.jit, static_argnames=['self'])
    def conj(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.conj(array)

    @partial(jax.jit, static_argnames=['self', 'name', 'dtype', 'bounds'])
    def new_variable(
        self,
        value: jnp.ndarray,
        bounds: tuple[float | None, float | None] | None,
        name: str,  # pylint: disable=unused-argument
        dtype=None,
    ):
        bounds = bounds or (None, None)
        dtype = dtype or self.float64
        value = self.astensor(value, dtype)
        return self.constraint_func(bounds)(value)

    @partial(jax.jit, static_argnames=['self', 'name', 'dtype'])
    def new_constant(self, value, name: str, dtype=None):  # pylint: disable=unused-argument
        dtype = dtype or self.float64
        value = self.astensor(value, dtype)
        return value

    @Autocast()
    @partial(jax.jit, static_argnames=['self', 'data_format', 'padding'])
    def convolution(
        self,
        array: jnp.ndarray,
        filters: jnp.ndarray,
        padding: str | None = None,
        data_format="NWC",
    ) -> jnp.ndarray:
        padding = padding or "VALID"
        return jax.lax.conv(array, filters, (1, 1), padding)

    def tile(self, array: jnp.ndarray, repeats: Sequence[int]) -> jnp.ndarray:
        repeats = tuple(repeats)
        array = jnp.array(array)
        return jnp.tile(array, repeats)

    @Autocast()
    @partial(jax.jit, static_argnames=['self'])
    def update_tensor(self, tensor: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        indices = self.atleast_2d(indices)
        return tensor.at[tuple(indices.T)].set(values)
    
    @Autocast()
    @partial(jax.jit, static_argnames=['self'])
    def update_add_tensor(
        self, tensor: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray
    ) -> jnp.ndarray:
        indices = self.atleast_2d(indices)
        return tensor.at[tuple(indices.T)].add(values)

    @Autocast()
    @partial(jax.jit, static_argnames=['self'])
    def matvec(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    @partial(jax.jit, static_argnames=['self'])
    def cos(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(array)

    @partial(jax.jit, static_argnames=['self'])
    def cosh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.cosh(array)

    @partial(jax.jit, static_argnames=['self'])
    def det(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.det(matrix)

    @partial(jax.jit, static_argnames=['self', 'k'])
    def diag(self, array: jnp.ndarray, k: int = 0) -> jnp.ndarray:
        if array.ndim == 0:
            return array
        else:
            return jnp.diag(array, k=k)

            
    @partial(jax.jit, static_argnames=['self', 'k'])
    def diag_part(self, array: jnp.ndarray, k: int) -> jnp.ndarray:
        return jnp.diagonal(array, offset=k, axis1=-2, axis2=-1)

    @partial(jax.jit, static_argnames=['self', 'string'])
    def einsum(self, string: str, *tensors) -> jnp.ndarray:
        if isinstance(string, str):
            return jnp.einsum(string, *tensors)
        return None

    @partial(jax.jit, static_argnames=['self'])
    def exp(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(array)

    @partial(jax.jit, static_argnames=['self', 'axis'])
    def expand_dims(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.expand_dims(array, axis)

    @partial(jax.jit, static_argnames=['self'])
    def expm(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jsp.linalg.expm(matrix)

    def eye(self, size: int, dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.eye(size, dtype=dtype)

    @partial(jax.jit, static_argnames=['self'])
    def eye_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(array.shape[-1], dtype=array.dtype)

    #@partial(jax.jit, static_argnames=['self'])
    def from_backend(self, value) -> bool:
        return isinstance(value, jnp.ndarray)

    
    @partial(jax.jit, static_argnames=['self', 'repeats', 'axis'])
    def repeat(self, array: jnp.ndarray, repeats: int, axis: int = None) -> jnp.ndarray:
        return jnp.repeat(array, repeats, axis=axis)

    #@partial(jax.jit, static_argnames=['self', 'axis'])
    #def gather(self, array: jnp.ndarray, indices: jnp.ndarray, axis: int) -> jnp.ndarray:
    #    return jnp.take(array, indices, axis=axis)

    @partial(jax.jit, static_argnames=['self', 'indices', 'axis'])
    def gather(self, array: jnp.ndarray, indices: tuple[int], axis: int) -> jnp.ndarray:
        return jnp.take(array, np.array(indices), axis=axis)

    @partial(jax.jit, static_argnames=['self'])
    def imag(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.imag(array)

    @partial(jax.jit, static_argnames=['self'])
    def inv(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.inv(tensor)

    @partial(jax.jit, static_argnames=['self'])
    def is_trainable(self, tensor: jnp.ndarray) -> bool:
        return True  # JAX arrays are always immutable

    @partial(jax.jit, static_argnames=['self'])
    def make_complex(self, real: jnp.ndarray, imag: jnp.ndarray) -> jnp.ndarray:
        return real + 1j * imag

    @Autocast()
    @partial(jax.jit, static_argnames=['self'])
    def matmul(self, *matrices: jnp.ndarray) -> jnp.ndarray:
        mat = jnp.linalg.multi_dot(matrices)
        """
        mat = matrices[0]
        for matrix in matrices[1:]:
            mat = jnp.matmul(mat, matrix)
        """
        return mat

    @partial(jax.jit, static_argnames=['self', 'old', 'new'])
    def moveaxis(self, array: jnp.ndarray, old: int | Sequence[int], new: int | Sequence[int]) -> jnp.ndarray:
        return jnp.moveaxis(array, old, new)

    def ones(self, shape: Sequence[int], dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.ones(shape, dtype=dtype)

    @partial(jax.jit, static_argnames=['self'])
    def ones_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(array)

    def pad(self, array: jnp.ndarray, paddings: Sequence[tuple[int, int]], mode="constant", constant_values=0) -> jnp.ndarray:
        return jnp.pad(array, paddings, mode=mode.lower(), constant_values=constant_values)

    @partial(jax.jit, static_argnames=['self'])
    def pinv(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.pinv(matrix)

    @partial(jax.jit, static_argnames=['self'])
    def real(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(array)

    def reshape(self, array: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.reshape(array, shape)

    @partial(jax.jit, static_argnames=['self'])
    def sin(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(array)

    @partial(jax.jit, static_argnames=['self'])
    def sinh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sinh(array)

    @partial(jax.jit, static_argnames=['self'])
    def solve(self, matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = jnp.expand_dims(rhs, -1)
            return jnp.linalg.solve(matrix, rhs)[..., 0]
        return jnp.linalg.solve(matrix, rhs)

    @partial(jax.jit, static_argnames=['self', 'dtype'])
    def sqrt(self, x: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.sqrt(self.cast(x, dtype))

    @partial(jax.jit, static_argnames=['self', 'axes'])
    def sum(self, array: jnp.ndarray, axes: Sequence[int] = None):
        return jnp.sum(array, axis=axes)

    @Autocast()
    def tensordot(self, a: jnp.ndarray, b: jnp.ndarray, axes: Sequence[int]) -> jnp.ndarray:
        return jnp.tensordot(a, b, axes)

    @partial(jax.jit, static_argnames=['self', 'dtype'])
    def trace(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return self.cast(jnp.trace(array), dtype)

    def transpose(self, a: jnp.ndarray, perm: Sequence[int] = None) -> jnp.ndarray:
        if a is None:
            return None
        return jnp.transpose(a, perm)

    def zeros(self, shape: Sequence[int], dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.zeros(shape, dtype=dtype)

    @partial(jax.jit, static_argnames=['self'])
    def zeros_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(array)

    @partial(jax.jit, static_argnames=['self', 'axis'])
    def squeeze(self, tensor: jnp.ndarray, axis=None):
        return jnp.squeeze(tensor, axis=axis)

    @partial(jax.jit, static_argnames=['self'])
    def cholesky(self, input: jnp.ndarray):
        return jnp.linalg.cholesky(input)

    @staticmethod
    @jax.jit
    def eigh(tensor: jnp.ndarray) -> tuple:
        return jnp.linalg.eigh(tensor)
    
    @partial(jax.jit, static_argnames=['self'])
    def where(self, array: jnp.ndarray, array1: jnp.ndarray, array2: jnp.ndarray):
        return jnp.where(array, array1, array2)
    
    @property
    def inf(self):
        return jnp.inf

    @staticmethod
    @jax.jit
    def eigvals(tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.eigvals(tensor)

    @partial(jax.jit, static_argnames=['self'])
    def eqigendecomposition_sqrtm(self, tensor: jnp.ndarray) -> jnp.ndarray:
        eigvals, eigvecs = jnp.linalg.eigh(tensor)
        return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ jnp.conj(eigvecs.T)
    
    @partial(jax.jit, static_argnames=['self', 'dtype', 'rtol', 'atol'])
    def sqrtm(self, tensor: jnp.ndarray, dtype, rtol=1e-05, atol=1e-08) -> jnp.ndarray:
        ret = jax.lax.cond(
            jnp.allclose(tensor, 0, rtol=rtol, atol=atol),
            lambda _: self.zeros_like(tensor),
            lambda _: self.eqigendecomposition_sqrtm(tensor),
            None
        )

        if dtype is None:
            return self.cast(ret, self.complex128)
        return self.cast(ret, dtype)

    # Special functions for optimization
    def DefaultEuclideanOptimizer(self):
        return jax.experimental.optimizers.adam(learning_rate=0.001)


    #@partial(jax.jit, static_argnames=['self', 'cost_fn'])
    #def value_and_gradients(self, cost_fn: Callable, parameters: list[Trainable]) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    #    """Computes the loss and gradients of the given cost function."""
    #    loss, gradients  = jax.value_and_grad(cost_fn)(parameters)
    #    return loss, gradients
    
    @partial(jax.jit, static_argnames=['self', 'cost_fn'])
    def value_and_gradients(
        self, cost_fn: Callable, parameters: list[Trainable]
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        r"""Computes the loss and gradients of the given cost function.

        Args:
            cost_fn (Callable with no args): The cost function.
            parameters (List[Trainable]): The parameters to optimize.

        Returns:
            tuple(ndarray, List[ndarray]): the loss and the gradients
        """
        # Create a wrapper that takes parameters as input
        def wrapped_cost_fn(params):
            # Temporarily replace parameters with new values
            old_values = []
            for p, new_value in zip(parameters, params):
                old_values.append(p.value)
                p.value = new_value
            
            # Compute cost
            loss = cost_fn()
            
            # Restore original values
            for p, old_value in zip(parameters, old_values):
                p.value = old_value
                
            return loss
            
        # Get current parameter values
        param_values = [p.value for p in parameters]
        
        # Compute value and gradient using JAX
        loss, gradients = jax.value_and_grad(wrapped_cost_fn)(param_values)
        return loss, gradients


    @jax.custom_vjp
    @partial(jax.jit, static_argnames=['shape'])
    def hermite_renormalized(
        A: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, shape: tuple[int]
    ) -> jnp.ndarray:
        function = partial(strategies.vanilla, tuple(shape))
        G = jax.pure_callback(
            lambda A, b, c: function(np.array(A), np.array(b), np.array(c)),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            A, b, c,
        )
        return G

    @partial(jax.jit, static_argnames=['shape'])
    def hermite_renormalized_fwd(A, b, c, shape):
        function = partial(strategies.vanilla, tuple(shape))
        G = jax.pure_callback(
            lambda A, b, c: function(np.array(A), np.array(b), np.array(c)),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            A, b, c,
        )
        return G, (G, c, A, b)

    @jax.jit
    def hermite_renormalized_bwd(res, g):
        G, c, A, b = res
        dLdA, dLdB, dLdC = jax.pure_callback(
            lambda G, c, g: strategies.vanilla_vjp(np.array(G), np.array(c), np.conj(jax.lax.stop_gradient(g))),
            [jax.ShapeDtypeStruct(A.shape, jnp.complex128), jax.ShapeDtypeStruct(b.shape, jnp.complex128), jax.ShapeDtypeStruct(c.shape, jnp.complex128)],
            G, c, g
        )
        return (jnp.conj(dLdA), jnp.conj(dLdB), jnp.conj(dLdC), None)

    hermite_renormalized.defvjp(hermite_renormalized_fwd, hermite_renormalized_bwd)

    def hermite_renormalized_batch(self, A, b, c, shape):
        return strategies.vanilla_batch(tuple(shape), np.array(A), np.array(b), np.array(c))


    # Add other Hermite-related functions as needed, following the same pattern as the TensorFlow backend 
    @jax.custom_jvp
    def hermite_renormalized_binomial(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        shape: tuple[int],
        max_l2: float | None,
        global_cutoff: int | None,
    ) -> jnp.ndarray:
        """Renormalized multidimensional Hermite polynomial with binomial strategy.
        Includes custom gradient computation via JAX's custom_vjp.
        """
        _A, _B, _C = np.array(A), np.array(B), np.array(C)
        G, _ = strategies.binomial(
            tuple(shape),
            _A,
            _B,
            _C,
            max_l2=max_l2 or settings.AUTOSHAPE_PROBABILITY,
            global_cutoff=global_cutoff or sum(shape) - len(shape) + 1,
        )
        return jnp.array(G), (G, c, A, B)

    @hermite_renormalized_binomial.defjvp
    def hermite_renormalized_binomial_jvp(self, res, g):
        G, c, A, B = res
        dLdGconj = np.array(g)
        dLdA, dLdB, dLdC = strategies.vanilla_vjp(G, c, np.conj(dLdGconj))
        return (self.conj(dLdA), self.conj(dLdB), self.conj(dLdC), None)

    @jax.custom_vjp
    def hermite_renormalized_diagonal_reorderedAB(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        """Renormalized multidimensional Hermite polynomial for diagonal elements.
        Includes custom gradient computation via JAX's custom_vjp.
        """
        A, B, C = np.array(A), np.array(B), np.array(C)
        precision_bits = settings.PRECISION_BITS_HERMITE_POLY

        if precision_bits == 128:  # numba (complex128)
            poly0, poly2, poly1010, poly1001, poly1 = hermite_multidimensional_diagonal(
                A, B, C, cutoffs
            )
        else:  # julia (higher precision than complex128)
            from juliacall import Main as jl
            polys = jl.DiagonalAmps.fock_diagonal_amps(
                A, B, C.item(), tuple(cutoffs), precision_bits
            )
            poly0, poly2, poly1010, poly1001, poly1 = [jnp.array(p) for p in polys]

        return poly0

    def hermite_renormalized_diagonal_reorderedAB_fwd(self, A, B, C, cutoffs):
        A, B, C = np.array(A), np.array(B), np.array(C)
        precision_bits = settings.PRECISION_BITS_HERMITE_POLY

        if precision_bits == 128:
            polys = hermite_multidimensional_diagonal(A, B, C, cutoffs)
        else:
            from juliacall import Main as jl
            polys = jl.DiagonalAmps.fock_diagonal_amps(
                A, B, C.item(), tuple(cutoffs), precision_bits
            )
        poly0, poly2, poly1010, poly1001, poly1 = [np.array(p) for p in polys]
        
        return jnp.array(poly0), (A, B, C, poly0, poly2, poly1010, poly1001, poly1)

    def hermite_renormalized_diagonal_reorderedAB_bwd(self, res, g):
        A, B, C, poly0, poly2, poly1010, poly1001, poly1 = res
        dLdpoly = np.array(g)
        
        precision_bits = settings.PRECISION_BITS_HERMITE_POLY
        if precision_bits == 128:
            dpoly_dC, dpoly_dA, dpoly_dB = grad_hermite_multidimensional_diagonal(
                A, B, C.item(), poly0, poly2, poly1010, poly1001, poly1
            )
        else:
            from juliacall import Main as jl
            dpoly_dC = poly0 / C.item()
            dpoly_dA, dpoly_dB = jl.DiagonalGrad.fock_diagonal_grad(
                A, B, poly0, poly2, poly1010, poly1001, poly1, precision_bits
            )

        ax = tuple(range(dLdpoly.ndim))
        dLdA = jnp.sum(dLdpoly[..., None, None] * jnp.conj(dpoly_dA), axis=ax)
        dLdB = jnp.sum(dLdpoly[..., None] * jnp.conj(dpoly_dB), axis=ax)
        dLdC = jnp.sum(dLdpoly * jnp.conj(dpoly_dC), axis=ax)
        
        return dLdA, dLdB, dLdC, None

    hermite_renormalized_diagonal_reorderedAB.defvjp(
        hermite_renormalized_diagonal_reorderedAB_fwd,
        hermite_renormalized_diagonal_reorderedAB_bwd
    )

    @jax.custom_vjp
    def hermite_renormalized_1leftoverMode_reorderedAB(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        """Renormalized multidimensional Hermite polynomial for one leftover mode.
        Includes custom gradient computation via JAX's custom_vjp.
        """
        A, B, C = np.array(A), np.array(B), np.array(C)
        precision_bits = settings.PRECISION_BITS_HERMITE_POLY

        if precision_bits == 128:
            polys = hermite_multidimensional_1leftoverMode(A, B, C.item(), cutoffs)
        else:
            from juliacall import Main as jl
            polys = jl.LeftoverModeAmps.fock_1leftoverMode_amps(
                A, B, C.item(), tuple(cutoffs), precision_bits
            )
        poly0, poly2, poly1010, poly1001, poly1 = [np.array(p) for p in polys]
        
        return jnp.array(poly0)

    def hermite_renormalized_1leftoverMode_reorderedAB_fwd(self, A, B, C, cutoffs):
        A, B, C = np.array(A), np.array(B), np.array(C)
        precision_bits = settings.PRECISION_BITS_HERMITE_POLY

        if precision_bits == 128:
            polys = hermite_multidimensional_1leftoverMode(A, B, C.item(), cutoffs)
        else:
            from juliacall import Main as jl
            polys = jl.LeftoverModeAmps.fock_1leftoverMode_amps(
                A, B, C.item(), tuple(cutoffs), precision_bits
            )
        poly0, poly2, poly1010, poly1001, poly1 = [np.array(p) for p in polys]
        
        return jnp.array(poly0), (A, B, C, poly0, poly2, poly1010, poly1001, poly1)

    def hermite_renormalized_1leftoverMode_reorderedAB_bwd(self, res, g):
        A, B, C, poly0, poly2, poly1010, poly1001, poly1 = res
        dLdpoly = np.array(g)
        
        precision_bits = settings.PRECISION_BITS_HERMITE_POLY
        if precision_bits == 128:
            dpoly_dC, dpoly_dA, dpoly_dB = grad_hermite_multidimensional_1leftoverMode(
                A, B, C, poly0, poly2, poly1010, poly1001, poly1
            )
        else:
            from juliacall import Main as jl
            dpoly_dC = poly0 / C.item()
            dpoly_dA, dpoly_dB = jl.LeftoverModeGrad.fock_1leftoverMode_grad(
                A, B, poly0, poly2, poly1010, poly1001, poly1, precision_bits
            )

        ax = tuple(range(dLdpoly.ndim))
        dLdA = jnp.sum(dLdpoly[..., None, None] * jnp.conj(dpoly_dA), axis=ax)
        dLdB = jnp.sum(dLdpoly[..., None] * jnp.conj(dpoly_dB), axis=ax)
        dLdC = jnp.sum(dLdpoly * jnp.conj(dpoly_dC), axis=ax)
        
        return dLdA, dLdB, dLdC, None

    hermite_renormalized_1leftoverMode_reorderedAB.defvjp(
        hermite_renormalized_1leftoverMode_reorderedAB_fwd,
        hermite_renormalized_1leftoverMode_reorderedAB_bwd
    )