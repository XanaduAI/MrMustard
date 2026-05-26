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

"""JAX implementations of complex Gaussian integrals."""

import jax
import jax.numpy as jnp

__all__ = [
    "complex_gaussian_integral_1_jax",
    "complex_gaussian_integral_2_jax",
]


@jax.jit
def complex_gaussian_integral_1_jax(A, b, idx12):
    """JAX implementation of complex Gaussian integral for one Abc.

    This is the core implementation that operates on single (non-batched) arrays.
    Use jax.vmap for batched operations.

    Args:
        A: Complex matrix of shape (n, n)
        b: Complex vector of shape (n,)
        idx12: Integer array of indices to integrate over

    Returns:
        Tuple of (A_out, b_out, log_c_out)
    """
    # Get indices not in idx12 using boolean mask
    n = A.shape[-1]
    mask = jnp.ones(n, dtype=bool)
    mask = mask.at[idx12].set(False)
    # Need to specify size for JAX JIT compatibility
    output_size = n - idx12.shape[0]
    not_idx12 = jnp.where(mask, size=output_size)[0]

    # Extract blocks using advanced indexing
    B = A[jnp.ix_(idx12, idx12)]
    C = A[jnp.ix_(not_idx12, idx12)]
    D = A[jnp.ix_(not_idx12, not_idx12)]
    g = b[idx12]
    h = b[not_idx12]

    # Build X matrix (conjugate pair structure)
    n = B.shape[-1]  # this is 2*m where m is number of conjugate pairs
    m = n // 2
    X = jnp.zeros((n, n), dtype=B.dtype)
    # Set off-diagonal blocks to identity
    X = X.at[jnp.arange(m), m + jnp.arange(m)].set(1.0)
    X = X.at[m + jnp.arange(m), jnp.arange(m)].set(1.0)

    # Compute M and its determinant
    M = B - X
    det = jnp.linalg.det(1j * M)

    # Check for singular matrix
    is_singular = jnp.isclose(jnp.abs(det), 0.0, atol=1e-20)

    # Compute inverse and results (JAX will handle the singular case via where)
    Minv = jnp.linalg.inv(M)

    A_out = D - C @ Minv @ C.T
    b_out = h - C @ Minv @ g
    log_c_out = -0.5 * (g @ Minv @ g) - 0.5 * jnp.log(det)

    # Return inf for singular case
    inf_like_A = jnp.full_like(A_out, jnp.inf)
    inf_like_b = jnp.full_like(b_out, jnp.inf)
    inf_like_c = jnp.array(jnp.inf, dtype=log_c_out.dtype)

    A_out = jnp.where(is_singular, inf_like_A, A_out)
    b_out = jnp.where(is_singular, inf_like_b, b_out)
    log_c_out = jnp.where(is_singular, inf_like_c, log_c_out)

    return A_out, b_out, log_c_out


@jax.jit
def complex_gaussian_integral_2_jax(A1, b1, A2, b2, idx1, idx2):
    """JAX implementation of complex Gaussian integral for two Abc.

    This is the core implementation that operates on single (non-batched) arrays.
    Use jax.vmap for batched operations.

    Args:
        A1: Complex matrix of shape (n1, n1)
        b1: Complex vector of shape (n1,)
        A2: Complex matrix of shape (n2, n2)
        b2: Complex vector of shape (n2,)
        idx1: Integer array of indices for first Abc
        idx2: Integer array of indices for second Abc

    Returns:
        Tuple of (A_out, b_out, log_c_out)
    """
    # Get indices not in idx1 and idx2 using boolean masks
    n1 = A1.shape[-1]
    n2 = A2.shape[-1]
    mask1 = jnp.ones(n1, dtype=bool)
    mask1 = mask1.at[idx1].set(False)
    output_size1 = n1 - idx1.shape[0]
    not_idx1 = jnp.where(mask1, size=output_size1)[0]

    mask2 = jnp.ones(n2, dtype=bool)
    mask2 = mask2.at[idx2].set(False)
    output_size2 = n2 - idx2.shape[0]
    not_idx2 = jnp.where(mask2, size=output_size2)[0]

    # Extract blocks from A1 and b1
    A = A1[jnp.ix_(idx1, idx1)]
    C = A1[jnp.ix_(not_idx1, idx1)]
    B = A1[jnp.ix_(not_idx1, not_idx1)]
    g = b1[idx1]
    h = b1[not_idx1]

    # Extract blocks from A2 and b2
    D = A2[jnp.ix_(idx2, idx2)]
    F = A2[jnp.ix_(not_idx2, idx2)]
    E = A2[jnp.ix_(not_idx2, not_idx2)]
    i = b2[idx2]
    j = b2[not_idx2]

    # Compute L matrix
    m = A1.shape[-1] - len(idx1)
    invL = A @ D - jnp.eye(A.shape[0], dtype=A1.dtype)
    detinv = jnp.linalg.det(-invL)

    # Check for singular matrix
    is_singular = jnp.isclose(jnp.abs(detinv), 0.0, atol=1e-20)

    # Compute inverse and results
    L = jnp.linalg.inv(invL)

    # Build output matrix A_out
    LCT = L @ C.T
    output_size = m + len(not_idx2)
    A_out = jnp.zeros((output_size, output_size), dtype=A1.dtype)

    A_out = A_out.at[:m, :m].set(B - C @ D @ LCT)
    A_out = A_out.at[m:, :m].set(-F @ LCT)
    A_out = A_out.at[:m, m:].set((-F @ LCT).T)
    A_out = A_out.at[m:, m:].set(E - F @ L @ A @ F.T)

    # Build output vector b_out
    Lg = L @ g
    LTi = L.T @ i
    b_out = jnp.zeros((output_size,), dtype=A1.dtype)
    b_out = b_out.at[:m].set(h - C @ (D @ Lg) - C @ LTi)
    b_out = b_out.at[m:].set(j - F @ (A @ LTi) - F @ Lg)

    # Compute log_c_out
    log_c_out = (-0.5 * (g @ D @ Lg + 2 * g @ LTi + i @ A @ LTi)) - 0.5 * jnp.log(detinv)

    # Return inf for singular case
    inf_like_A = jnp.full_like(A_out, jnp.inf)
    inf_like_b = jnp.full_like(b_out, jnp.inf)
    inf_like_c = jnp.array(jnp.inf, dtype=log_c_out.dtype)

    A_out = jnp.where(is_singular, inf_like_A, A_out)
    b_out = jnp.where(is_singular, inf_like_b, b_out)
    log_c_out = jnp.where(is_singular, inf_like_c, log_c_out)

    return A_out, b_out, log_c_out
