# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains jax-specific implementations of the hermite_renormalized functions."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..lattice import strategies
from ..lattice.strategies.compactFock.inputValidation import (
    grad_hermite_multidimensional_1leftoverMode,
    grad_hermite_multidimensional_diagonal,
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
)

__all__ = [
    "hermite_renormalized_1leftoverMode_jax",
    "hermite_renormalized_batched_jax",
    "hermite_renormalized_binomial_jax",
    "hermite_renormalized_diagonal_jax",
    "hermite_renormalized_jax",
]

# ~~~~~~~~~~~~~~~~~~~~
# hermite_renormalized
# ~~~~~~~~~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
@partial(jax.jit, static_argnums=(3, 4))
def hermite_renormalized_jax(
    A: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    shape: tuple[int],
    stable: bool,
) -> jnp.ndarray:
    r"""
    The jax custom gradient for hermite_renormalized.
    """
    if stable:
        G = jax.pure_callback(
            lambda A, b, c: strategies.stable_numba(shape, np.array(A), np.array(b), np.array(c)),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            A,
            b,
            c,
        )
    else:
        G = jax.pure_callback(
            lambda A, b, c: strategies.vanilla_numba(shape, np.array(A), np.array(b), np.array(c)),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            A,
            b,
            c,
        )
    return G


def hermite_renormalized_jax_fwd(A, b, c, shape, stable):
    r"""
    The jax forward pass for hermite_renormalized.
    """
    G = hermite_renormalized_jax(A, b, c, shape, stable)
    return (G, (G, A, b, c))


def hermite_renormalized_jax_bwd(shape, stable, res, g):
    r"""
    The jax backward pass for hermite_renormalized.
    """
    G, A, b, c = res
    dLdA, dLdB, dLdC = jax.pure_callback(
        lambda G, c, g: strategies.vanilla_vjp_numba(np.array(G), np.array(c), np.array(g)),
        (
            jax.ShapeDtypeStruct(A.shape, jnp.complex128),
            jax.ShapeDtypeStruct(b.shape, jnp.complex128),
            jax.ShapeDtypeStruct(c.shape, jnp.complex128),
        ),
        G,
        c,
        g,
    )
    return dLdA, dLdB, dLdC


hermite_renormalized_jax.defvjp(hermite_renormalized_jax_fwd, hermite_renormalized_jax_bwd)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hermite_renormalized_batched
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
@partial(jax.jit, static_argnums=(3, 4))
def hermite_renormalized_batched_jax(
    A: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    shape: tuple[int],
    stable: bool,
) -> jnp.ndarray:
    r"""
    The jax custom gradient for hermite_renormalized_batched.
    """
    batch_size = A.shape[0]
    output_shape = (batch_size, *shape)
    return jax.pure_callback(
        lambda A, b, c: strategies.vanilla_batch_numba(
            shape,
            np.asarray(A),
            np.asarray(b),
            np.asarray(c),
            stable,
            None,
        ),
        jax.ShapeDtypeStruct(output_shape, jnp.complex128),
        A,
        b,
        c,
    )


def hermite_renormalized_batched_jax_fwd(A, b, c, shape, stable):
    r"""
    The jax forward pass for hermite_renormalized_batched.
    """
    G = hermite_renormalized_batched_jax(A, b, c, shape, stable)
    return (G, (G, A, b, c))


def hermite_renormalized_batched_jax_bwd(shape, stable, res, g):
    r"""
    The jax backward pass for hermite_renormalized_batched.
    """
    G, A, b, c = res
    dLdA, dLdB, dLdC = jax.pure_callback(
        lambda G, c, g: strategies.vanilla_batch_vjp_numba(np.array(G), np.array(c), np.array(g)),
        (
            jax.ShapeDtypeStruct(A.shape, jnp.complex128),
            jax.ShapeDtypeStruct(b.shape, jnp.complex128),
            jax.ShapeDtypeStruct(c.shape, jnp.complex128),
        ),
        G,
        c,
        g,
    )
    return dLdA, dLdB, dLdC


hermite_renormalized_batched_jax.defvjp(
    hermite_renormalized_batched_jax_fwd,
    hermite_renormalized_batched_jax_bwd,
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hermite_renormalized_binomial
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
@partial(jax.jit, static_argnums=(3, 4, 5))
def hermite_renormalized_binomial_jax(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    shape: tuple[int],
    max_l2: float | None,
    global_cutoff: int | None,
) -> jnp.ndarray:
    r"""
    The jax custom gradient for hermite_renormalized_binomial.
    """
    function = partial(strategies.binomial, tuple(shape))
    return jax.pure_callback(
        lambda A, B, C, max_l2, global_cutoff: function(
            np.asarray(A),
            np.asarray(B),
            np.asarray(C),
            max_l2,
            global_cutoff,
        )[0],
        jax.ShapeDtypeStruct(shape, jnp.complex128),
        A,
        B,
        C,
        max_l2,
        global_cutoff,
    )


def hermite_renormalized_binomial_jax_fwd(A, b, c, shape, max_l2, global_cutoff):
    r"""
    The jax forward pass for hermite_renormalized_binomial.
    """
    G = hermite_renormalized_binomial_jax(A, b, c, shape, max_l2, global_cutoff)
    return (G, (G, A, b, c))


def hermite_renormalized_binomial_jax_bwd(shape, max_l2, global_cutoff, res, g):
    r"""
    The jax backward pass for hermite_renormalized_binomial.
    """
    G, A, b, c = res
    dLdA, dLdB, dLdC = jax.pure_callback(
        lambda G, c, g: strategies.vanilla_vjp(np.array(G), np.array(c), np.array(g)),
        (
            jax.ShapeDtypeStruct(A.shape, jnp.complex128),
            jax.ShapeDtypeStruct(b.shape, jnp.complex128),
            jax.ShapeDtypeStruct(c.shape, jnp.complex128),
        ),
        G,
        c,
        g,
    )
    return dLdA, dLdB, dLdC


hermite_renormalized_binomial_jax.defvjp(
    hermite_renormalized_binomial_jax_fwd,
    hermite_renormalized_binomial_jax_bwd,
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hermite_renormalized_diagonal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(3,))
@partial(jax.jit, static_argnums=(3,))
def hermite_renormalized_diagonal_jax(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    cutoffs: tuple[int],
) -> jnp.ndarray:
    r"""
    The jax custom gradient for hermite_renormalized_diagonal.
    """
    M = len(cutoffs)
    batch_shape = B.shape[-1:] if B.ndim == 2 else ()
    shape = (1, 1, 1, *batch_shape) if M == 1 else (M, M - 1, *cutoffs, *batch_shape)
    return jax.pure_callback(
        lambda A, B, C, cutoffs: hermite_multidimensional_diagonal(
            np.array(A), np.array(B), np.array(C), np.array(cutoffs)
        ),
        (
            jax.ShapeDtypeStruct((*cutoffs, *batch_shape), jnp.complex128),
            jax.ShapeDtypeStruct((M, *cutoffs, *batch_shape), jnp.complex128),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            jax.ShapeDtypeStruct((2 * M, *cutoffs, *batch_shape), jnp.complex128),
        ),
        A,
        B,
        C,
        cutoffs,
    )


def hermite_renormalized_diagonal_jax_fwd(A, b, c, cutoffs):
    r"""
    The jax forward pass for hermite_renormalized_diagonal.
    """
    primal_output = hermite_renormalized_diagonal_jax(A, b, c, cutoffs)
    return (primal_output, (*primal_output, A, b, c))


def hermite_renormalized_diagonal_jax_bwd(cutoffs, res, g):
    r"""
    The jax backward pass for hermite_renormalized_diagonal.
    """
    poly0, poly2, poly1010, poly1001, poly1, A, b, c = res
    if b.ndim > 1:
        raise ValueError("B batched")
    dpoly_dC, dpoly_dA, dpoly_dB = jax.pure_callback(
        lambda A, B, C, arr0, arr2, arr1010, arr1001, arr1: grad_hermite_multidimensional_diagonal(
            np.array(A),
            np.array(B),
            np.array(C),
            np.array(arr0),
            np.array(arr2),
            np.array(arr1010),
            np.array(arr1001),
            np.array(arr1),
        ),
        (
            jax.ShapeDtypeStruct(poly0.shape + c.shape, jnp.complex128),
            jax.ShapeDtypeStruct(poly0.shape + A.shape, jnp.complex128),
            jax.ShapeDtypeStruct(poly0.shape + b.shape, jnp.complex128),
        ),
        A,
        b,
        c,
        poly0,
        poly2,
        poly1010,
        poly1001,
        poly1,
    )
    dLdpoly = g[0]
    ax = tuple(range(dLdpoly.ndim))
    dLdA = jnp.sum(dLdpoly[..., None, None] * dpoly_dA, axis=ax)
    dLdB = jnp.sum(dLdpoly[..., None] * dpoly_dB, axis=ax)
    dLdC = jnp.sum(dLdpoly * dpoly_dC, axis=ax)
    return dLdA, dLdB, dLdC


hermite_renormalized_diagonal_jax.defvjp(
    hermite_renormalized_diagonal_jax_fwd,
    hermite_renormalized_diagonal_jax_bwd,
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hermite_renormalized_1leftoverMode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
@partial(jax.jit, static_argnums=(3, 4))
def hermite_renormalized_1leftoverMode_jax(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    output_cutoff: int,
    pnr_cutoffs: tuple[int, ...],
) -> jnp.ndarray:
    r"""
    The jax custom gradient for hermite_renormalized_1leftoverMode.
    """
    cutoffs = (output_cutoff + 1, *tuple(p + 1 for p in pnr_cutoffs))
    M = len(cutoffs)
    cutoff_leftoverMode = cutoffs[0]
    cutoffs_tail = tuple(cutoffs[1:])
    if M == 2:
        shape = (1, 1, 1, 1, 1)
    else:
        shape = (cutoff_leftoverMode, cutoff_leftoverMode, M - 1, M - 2, *cutoffs_tail)
    return jax.pure_callback(
        lambda A, B, C, cutoffs: hermite_multidimensional_1leftoverMode(
            np.array(A), np.array(B), np.array(C), np.array(cutoffs)
        ),
        (
            jax.ShapeDtypeStruct(
                (cutoff_leftoverMode, cutoff_leftoverMode, *cutoffs_tail), jnp.complex128
            ),
            jax.ShapeDtypeStruct(
                (cutoff_leftoverMode, cutoff_leftoverMode, M - 1, *cutoffs_tail), jnp.complex128
            ),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            jax.ShapeDtypeStruct(
                (cutoff_leftoverMode, cutoff_leftoverMode, 2 * (M - 1), *cutoffs_tail),
                jnp.complex128,
            ),
        ),
        A,
        B,
        C,
        cutoffs,
    )


def hermite_renormalized_1leftoverMode_jax_fwd(A, b, c, output_cutoff, pnr_cutoffs):
    r"""
    The jax forward pass for hermite_renormalized_reorderedAB.
    """
    primal_output = hermite_renormalized_1leftoverMode_jax(A, b, c, output_cutoff, pnr_cutoffs)
    return (primal_output, (*primal_output, A, b, c))


def hermite_renormalized_1leftoverMode_jax_bwd(output_cutoff, pnr_cutoffs, res, g):
    r"""
    The jax backward pass for hermite_renormalized_1leftoverMode.
    """
    poly0, poly2, poly1010, poly1001, poly1, A, b, c = res
    dpoly_dC, dpoly_dA, dpoly_dB = jax.pure_callback(
        lambda A,
        B,
        C,
        arr0,
        arr2,
        arr1010,
        arr1001,
        arr1: grad_hermite_multidimensional_1leftoverMode(
            np.array(A),
            np.array(B),
            np.array(C),
            np.array(arr0),
            np.array(arr2),
            np.array(arr1010),
            np.array(arr1001),
            np.array(arr1),
        ),
        (
            jax.ShapeDtypeStruct(poly0.shape + c.shape, jnp.complex128),
            jax.ShapeDtypeStruct(poly0.shape + A.shape, jnp.complex128),
            jax.ShapeDtypeStruct(poly0.shape + b.shape, jnp.complex128),
        ),
        A,
        b,
        c,
        poly0,
        poly2,
        poly1010,
        poly1001,
        poly1,
    )
    dLdpoly = g[0]
    ax = tuple(range(dLdpoly.ndim))
    dLdA = jnp.sum(dLdpoly[..., None, None] * dpoly_dA, axis=ax)
    dLdB = jnp.sum(dLdpoly[..., None] * dpoly_dB, axis=ax)
    dLdC = jnp.sum(dLdpoly * dpoly_dC, axis=ax)
    return dLdA, dLdB, dLdC


hermite_renormalized_1leftoverMode_jax.defvjp(
    hermite_renormalized_1leftoverMode_jax_fwd,
    hermite_renormalized_1leftoverMode_jax_bwd,
)
