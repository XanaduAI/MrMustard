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

"""
Custom vjps for fock utilities.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from mrmustard import math
from mrmustard.math.lattice import strategies

__all__ = ["displacement_jax"]


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
@partial(jax.jit, static_argnums=(2, 3))
def displacement_jax(x, y, shape, tol):
    def true_branch(shape, x, y):
        return jax.pure_callback(
            lambda x, y: strategies.displacement(
                cutoffs=shape, alpha=math.asnumpy(x) + 1j * math.asnumpy(y), dtype=np.complex128
            ),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            x,
            y,
        )

    def false_branch(shape, x, y):
        return math.eye(max(shape), dtype="complex128")[: shape[0], : shape[1]]

    return math.conditional(
        math.sqrt(x * x + y * y) > tol,
        partial(true_branch, shape),
        partial(false_branch, shape),
        x,
        y,
    )


def displacement_jax_fwd(x, y, shape, tol):
    gate = displacement_jax(x, y, shape, tol)
    return gate, (gate, x, y)


def displacement_jax_bwd(shape, tol, res, g):
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


displacement_jax.defvjp(displacement_jax_fwd, displacement_jax_bwd)
