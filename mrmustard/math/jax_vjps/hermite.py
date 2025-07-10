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

__all__ = ["hermite_renormalized_batched_jax", "hermite_renormalized_unbatched_jax"]

# ~~~~~~~~~~~~~~~~~
# hermite_renormalized_unbatched
# ~~~~~~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
@partial(jax.jit, static_argnums=(3, 4))
def hermite_renormalized_unbatched_jax(
    A: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    shape: tuple[int],
    stable: bool,
) -> jnp.ndarray:
    r"""
    The jax custom gradient for hermite_renormalized_unbatched.
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


def hermite_renormalized_unbatched_jax_fwd(A, b, c, shape, stable):
    r"""
    The jax forward pass for hermite_renormalized_unbatched.
    """
    G = hermite_renormalized_unbatched_jax(A, b, c, shape, stable)
    return (G, (G, A, b, c))


def hermite_renormalized_unbatched_jax_bwd(shape, stable, res, g):
    r"""
    The jax backward pass for hermite_renormalized_unbatched.
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


hermite_renormalized_unbatched_jax.defvjp(
    hermite_renormalized_unbatched_jax_fwd,
    hermite_renormalized_unbatched_jax_bwd,
)


# ~~~~~~~~~~~~~~~~~
# hermite_renormalized_batched
# ~~~~~~~~~~~~~~~~~


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
    The jax forward pass for hermite_renormalized_unbatched.
    """
    G = hermite_renormalized_batched_jax(A, b, c, shape, stable)
    return (G, (G, A, b, c))


def hermite_renormalized_batched_jax_bwd(shape, stable, res, g):
    r"""
    The jax backward pass for hermite_renormalized_unbatched.
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
