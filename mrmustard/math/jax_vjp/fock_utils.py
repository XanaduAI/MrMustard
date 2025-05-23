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

# pylint: disable=redefined-outer-name

"""
Custom vjps for fock utilities.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from mrmustard import math
from mrmustard.math.lattice import strategies

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ static functions ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
@partial(jax.jit, static_argnums=(2, 3))
def displacement(x, y, shape, tol=1e-15):
    r"""creates a single mode displacement matrix"""
    return math.conditional(
        math.sqrt(x * x + y * y) > tol,
        partial(temp_true_branch, shape),
        partial(temp_false_branch, shape),
        x,
        y,
    )


def displacement_fwd(x, y, shape, tol):
    gate = displacement(x, y, shape, tol)
    return gate, (gate, x, y)


def temp_true_branch(shape, x, y):
    return jax.pure_callback(
        lambda x, y: strategies.displacement(
            cutoffs=shape, alpha=math.asnumpy(x) + 1j * math.asnumpy(y), dtype=np.complex128
        ),
        jax.ShapeDtypeStruct(shape, jnp.complex128),
        x,
        y,
    )


def temp_false_branch(shape, x, y):
    return math.eye(max(shape), dtype="complex128")[: shape[0], : shape[1]]


def displacement_bwd(shape, tol, res, g):
    gate, x, y = res

    dD_da, dD_dac = jax.pure_callback(
        lambda gate, x, y: strategies.jacobian_displacement(
            math.asnumpy(gate), math.asnumpy(x) + 1j * math.asnumpy(y)
        ),
        (jax.ShapeDtypeStruct(shape, jnp.complex128), jax.ShapeDtypeStruct(shape, jnp.complex128)),
        gate,
        x,
        y,
    )

    dL_dac = math.sum(math.conj(g) * dD_dac + g * math.conj(dD_da))
    dLdx = 2 * math.real(dL_dac)
    dLdy = 2 * math.imag(dL_dac)
    return dLdx, dLdy


displacement.defvjp(displacement_fwd, displacement_bwd)


# @math.custom_gradient
# def beamsplitter(theta: float, phi: float, shape: Sequence[int], method: str):
#     r"""Creates a beamsplitter tensor with given cutoffs using a numba-based fock lattice strategy.

#     Args:
#         theta (float): transmittivity angle of the beamsplitter
#         phi (float): phase angle of the beamsplitter
#         cutoffs (int,int): cutoff dimensions of the two modes
#     """
#     if method == "vanilla":
#         bs_unitary = strategies.beamsplitter(shape, math.asnumpy(theta), math.asnumpy(phi))
#     elif method == "schwinger":
#         bs_unitary = strategies.beamsplitter_schwinger(
#             shape, math.asnumpy(theta), math.asnumpy(phi)
#         )
#     else:
#         raise ValueError(
#             f"Unknown beamsplitter method {method}. Options are 'vanilla' and 'schwinger'."
#         )

#     ret = math.astensor(bs_unitary, dtype=bs_unitary.dtype.name)
#     if math.backend_name in ["numpy", "jax"]:
#         return ret

#     def vjp(dLdGc):
#         dtheta, dphi = strategies.beamsplitter_vjp(
#             math.asnumpy(bs_unitary),
#             math.asnumpy(math.conj(dLdGc)),
#             math.asnumpy(theta),
#             math.asnumpy(phi),
#         )
#         return math.astensor(dtheta, dtype=theta.dtype), math.astensor(dphi, dtype=phi.dtype)

#     return ret, vjp


# @math.custom_gradient
# def squeezer(r, phi, shape):
#     r"""creates a single mode squeezer matrix using a numba-based fock lattice strategy"""
#     sq_unitary = strategies.squeezer(shape, math.asnumpy(r), math.asnumpy(phi))

#     ret = math.astensor(sq_unitary, dtype=sq_unitary.dtype.name)
#     if math.backend_name in ["numpy", "jax"]:
#         return ret

#     def vjp(dLdGc):
#         dr, dphi = strategies.squeezer_vjp(
#             math.asnumpy(sq_unitary),
#             math.asnumpy(math.conj(dLdGc)),
#             math.asnumpy(r),
#             math.asnumpy(phi),
#         )
#         return math.astensor(dr, dtype=r.dtype), math.astensor(dphi, phi.dtype)

#     return ret, vjp


# @math.custom_gradient
# def squeezed(r, phi, shape):
#     r"""creates a single mode squeezed state using a numba-based fock lattice strategy"""
#     sq_ket = strategies.squeezed(shape, math.asnumpy(r), math.asnumpy(phi))

#     ret = math.astensor(sq_ket, dtype=sq_ket.dtype.name)
#     if math.backend_name in ["numpy", "jax"]:  # pragma: no cover
#         return ret

#     def vjp(dLdGc):
#         dr, dphi = strategies.squeezed_vjp(
#             math.asnumpy(sq_ket),
#             math.asnumpy(math.conj(dLdGc)),
#             math.asnumpy(r),
#             math.asnumpy(phi),
#         )
#         return math.astensor(dr, dtype=r.dtype), math.astensor(dphi, phi.dtype)

#     return ret, vjp
