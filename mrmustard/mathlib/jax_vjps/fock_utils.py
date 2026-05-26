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

"""Custom vjps for fock utilities."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mrmustard.mathlib import cython_lattice
from mrmustard.mathlib.lattice import strategies

__all__ = [
    "beamsplitter_jax",
    "displacement_jax",
    "homodyne_projector_jax",
    "squeezed_jax",
    "squeezer_jax",
]

# ~~~~~~~~~~~~
# beamsplitter
# ~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
@partial(jax.jit, static_argnums=(2, 3))
def beamsplitter_jax(theta: float, phi: float, shape: tuple[int, ...], method: str) -> jnp.ndarray:
    r"""The jax custom gradient for the beamsplitter gate."""
    if method == "schwinger":
        bs_unitary = jax.pure_callback(
            lambda t, s: strategies.beamsplitter_schwinger(shape, np.asarray(t), np.asarray(s)),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            theta,
            phi,
        )
    else:
        stable = method == "stable"
        batch_shape = theta.shape
        if batch_shape == ():
            bs_unitary = jax.pure_callback(
                lambda t, s: cython_lattice.beamsplitter(
                    shape,
                    np.asarray(t, dtype=np.float64),
                    np.asarray(s, dtype=np.float64),
                    stable=stable,
                ),
                jax.ShapeDtypeStruct(shape, jnp.complex128),
                theta,
                phi,
            )
        else:
            theta_flattened = theta.reshape(-1)
            phi_flattened = phi.reshape(-1)
            batch_size = theta_flattened.shape[0]
            bs_unitary = jax.pure_callback(
                lambda t, s: cython_lattice.beamsplitter_batched(
                    shape,
                    np.asarray(t, dtype=np.float64),
                    np.asarray(s, dtype=np.float64),
                    stable=stable,
                ),
                jax.ShapeDtypeStruct((batch_size, *shape), jnp.complex128),
                theta_flattened,
                phi_flattened,
            )
            bs_unitary = bs_unitary.reshape((*batch_shape, *shape))
    return bs_unitary


def beamsplitter_jax_fwd(
    theta: float,
    phi: float,
    shape: tuple[int, ...],
    method: str,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, float, float]]:
    r"""The jax forward pass for the beamsplitter gate."""
    bs_unitary = beamsplitter_jax(theta, phi, shape, method)
    return bs_unitary, (bs_unitary, theta, phi)


def beamsplitter_jax_bwd(
    shape,
    method,
    res: tuple[jnp.ndarray, float, float],
    g: jnp.ndarray,
) -> tuple[float, float]:
    r"""The jax backward pass for the beamsplitter gate."""
    bs_unitary, theta, phi = res
    dtheta, dphi = jax.pure_callback(
        lambda bs_unitary, g, theta, phi: strategies.beamsplitter_vjp(
            np.asarray(bs_unitary),
            np.asarray(g),
            np.asarray(theta),
            np.asarray(phi),
        ),
        (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)),
        bs_unitary,
        g,
        theta,
        phi,
    )
    return dtheta, dphi


beamsplitter_jax.defvjp(beamsplitter_jax_fwd, beamsplitter_jax_bwd)


# ~~~~~~~~~~~~
# displacement
# ~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(1,))
@partial(jax.jit, static_argnums=(1,))
def displacement_jax(alpha: complex, shape: tuple[int, ...]) -> jnp.ndarray:
    r"""The jax custom gradient for the displacement gate."""
    batch_shape = alpha.shape
    if batch_shape == ():
        return jax.pure_callback(
            lambda alpha: cython_lattice.displacement(
                cutoffs=shape,
                alpha=np.asarray(alpha, dtype=np.complex128),
            ),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            alpha,
        )
    alpha_flattened = alpha.reshape(-1)
    batch_size = alpha_flattened.shape[0]
    ret = jax.pure_callback(
        lambda alpha: cython_lattice.displacement_batched(
            cutoffs=shape,
            alpha=np.asarray(alpha, dtype=np.complex128),
        ),
        jax.ShapeDtypeStruct((batch_size, *shape), jnp.complex128),
        alpha_flattened,
    )
    return ret.reshape((*batch_shape, *shape))


def displacement_jax_fwd(
    alpha: complex,
    shape: tuple[int, ...],
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, complex]]:
    r"""The jax forward pass for the displacement gate."""
    gate = displacement_jax(alpha, shape)
    return gate, (gate, alpha)


def displacement_jax_bwd(
    shape: tuple[int, ...],
    res: tuple[jnp.ndarray, complex],
    dL_dD: jnp.ndarray,
) -> tuple[jnp.ndarray]:
    r"""The jax backward pass for the displacement gate."""
    gate, alpha = res
    batch_shape = alpha.shape
    dD_da, dD_dac = jax.pure_callback(
        lambda gate, alpha: cython_lattice.jacobian_displacement(
            np.asarray(gate),
            np.asarray(alpha),
        ),
        (
            jax.ShapeDtypeStruct((*batch_shape, *shape), jnp.complex128),
            jax.ShapeDtypeStruct((*batch_shape, *shape), jnp.complex128),
        ),
        gate,
        alpha,
    )
    return (
        jnp.sum(dL_dD * dD_da, axis=[-1, -2]) + jnp.conj(jnp.sum(dL_dD * dD_dac, axis=[-1, -2])),
    )


displacement_jax.defvjp(displacement_jax_fwd, displacement_jax_bwd)


# ~~~~~~~~~~~~~~~~~~
# homodyne_projector
# ~~~~~~~~~~~~~~~~~~


@partial(jax.jit, static_argnums=(0,))
def homodyne_projector_jax(
    fock_dim: int, A: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray
) -> jnp.ndarray:
    r"""The jax custom gradient for the homodyne projector."""
    batch_shape = c.shape
    if batch_shape == ():
        return jax.pure_callback(
            lambda fock_dim, A, b, c: cython_lattice.homodyne_projector(
                fock_dim, np.asarray(A), np.asarray(b), np.asarray(c)
            ),
            jax.ShapeDtypeStruct((*batch_shape, fock_dim, fock_dim, fock_dim), jnp.complex128),
            fock_dim,
            A,
            b,
            c,
        )
    A_flattened = A.reshape(-1, *A.shape[-2:])
    b_flattened = b.reshape(-1, *b.shape[-1:])
    c_flattened = c.reshape(-1)
    batch_size = A_flattened.shape[0]
    ret = jax.pure_callback(
        cython_lattice.homodyne_projector_batched,
        jax.ShapeDtypeStruct((batch_size, fock_dim, fock_dim, fock_dim), jnp.complex128),
        fock_dim,
        A_flattened,
        b_flattened,
        c_flattened,
    )
    return ret.reshape((*batch_shape, fock_dim, fock_dim, fock_dim))


# ~~~~~~~~
# squeezed
# ~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(2,))
@partial(jax.jit, static_argnums=(2,))
def squeezed_jax(r: float, phi: float, shape: tuple[int]) -> jnp.ndarray:
    r"""The jax custom gradient for the squeezed state."""
    batch_shape = r.shape
    if batch_shape == ():
        return jax.pure_callback(
            lambda shape, r, phi: cython_lattice.squeezed(
                int(shape[0]), np.asarray(r), np.asarray(phi)
            ),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            shape,
            r,
            phi,
        )
    r_flattened = r.reshape(-1)
    phi_flattened = phi.reshape(-1)
    batch_size = r_flattened.shape[0]
    ret = jax.pure_callback(
        lambda shape, r, phi: cython_lattice.squeezed_batched(
            int(shape[0]), np.asarray(r), np.asarray(phi)
        ),
        jax.ShapeDtypeStruct((batch_size, shape[0]), jnp.complex128),
        shape,
        r_flattened,
        phi_flattened,
    )
    return ret.reshape((*batch_shape, shape[0]))


def squeezed_jax_fwd(r, phi, shape):
    r"""The jax forward pass for the squeezed state."""
    primal_output = squeezed_jax(r, phi, shape)
    return (primal_output, (primal_output, r, phi))


def squeezed_jax_bwd(shape, res, g):
    r"""The jax backward pass for the squeezed state."""
    sq_state, r, phi = res
    batch_shape = r.shape
    if batch_shape == ():
        return jax.pure_callback(
            lambda sq_state, g, r, phi: strategies.squeezed_vjp(
                np.asarray(sq_state),
                np.asarray(g),
                np.asarray(r),
                np.asarray(phi),
            ),
            (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)),
            sq_state,
            g,
            r,
            phi,
        )
    return jax.pure_callback(
        lambda sq_state, g, r, phi: strategies.squeezed_vjp_batched(
            np.asarray(sq_state),
            np.asarray(g),
            np.asarray(r),
            np.asarray(phi),
        ),
        (
            jax.ShapeDtypeStruct(batch_shape, jnp.float64),
            jax.ShapeDtypeStruct(batch_shape, jnp.float64),
        ),
        sq_state,
        g,
        r,
        phi,
    )


squeezed_jax.defvjp(squeezed_jax_fwd, squeezed_jax_bwd)


# ~~~~~~~~
# squeezer
# ~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(2,))
@partial(jax.jit, static_argnums=(2,))
def squeezer_jax(r: float, phi: float, shape: tuple[int, int]) -> jnp.ndarray:
    r"""The jax custom gradient for the squeezer gate."""
    batch_shape = r.shape
    if batch_shape == ():
        return jax.pure_callback(
            lambda shape, r, phi: cython_lattice.squeezer(
                tuple(int(s) for s in shape), np.asarray(r), np.asarray(phi)
            ),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            shape,
            r,
            phi,
        )
    r_flattened = r.reshape(-1)
    phi_flattened = phi.reshape(-1)
    batch_size = r_flattened.shape[0]
    ret = jax.pure_callback(
        lambda shape, r, phi: cython_lattice.squeezer_batched(
            tuple(int(s) for s in shape), np.asarray(r), np.asarray(phi)
        ),
        jax.ShapeDtypeStruct((batch_size, *shape), jnp.complex128),
        shape,
        r_flattened,
        phi_flattened,
    )
    return ret.reshape((*batch_shape, *shape))


def squeezer_jax_fwd(r, phi, shape):
    r"""The jax forward pass for the squeezer gate."""
    primal_output = squeezer_jax(r, phi, shape)
    return (primal_output, (primal_output, r, phi))


def squeezer_jax_bwd(shape, res, g):
    r"""The jax backward pass for the squeezer gate."""
    squeezer, r, phi = res
    batch_shape = r.shape
    if batch_shape == ():
        return jax.pure_callback(
            lambda squeezer, g, r, phi: strategies.squeezer_vjp(
                np.asarray(squeezer),
                np.asarray(g),
                np.asarray(r),
                np.asarray(phi),
            ),
            (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)),
            squeezer,
            g,
            r,
            phi,
        )
    return jax.pure_callback(
        lambda squeezer, g, r, phi: strategies.squeezer_vjp_batched(
            np.asarray(squeezer),
            np.asarray(g),
            np.asarray(r),
            np.asarray(phi),
        ),
        (
            jax.ShapeDtypeStruct(batch_shape, jnp.float64),
            jax.ShapeDtypeStruct(batch_shape, jnp.float64),
        ),
        squeezer,
        g,
        r,
        phi,
    )


squeezer_jax.defvjp(squeezer_jax_fwd, squeezer_jax_bwd)
