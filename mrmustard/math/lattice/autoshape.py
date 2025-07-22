# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the autoshape function."""

import numpy as np
from numba import njit

SQRT = np.sqrt(np.arange(100000))


@njit(cache=True)
def autoshape_numba(A, b, c, max_prob, max_shape, min_shape) -> int:  # pragma: no cover
    r"""Strategy to compute the shape of the Fock representation of a Gaussian DM
    such that its trace is above a certain bound given as ``max_prob``.
    This is an adaptation of Robbe's diagonal strategy, with early stopping.
    Details in https://quantum-journal.org/papers/q-2023-08-29-1097/.

    Args:
        A (np.ndarray): 2Mx2M matrix of the Bargmann ansatz.
        b (np.ndarray): 2M-dim vector of the Bargmann ansatz.
        c (float): The vacuum amplitude.
        max_prob (float): The probability value to stop at.
        max_shape (int): The upper limit for clipping the shape.
        min_shape (int): The lower limit for clipping the shape.

    **Details:**

    Here's how it works. First we get the reduced density matrix at the given mode. Then we
    maintain two buffers that contain values around the diagonal: buf3 and buf2, of size 2x3
    and 2x2 respectively. The buffers contain the following elements of the density matrix:

    .. code-block::

                     ┌────┐                           ┌────┐
                     │buf3│                           │buf2│
                     └────┘                           └────┘
            ┌────┬────┬────┬────┬────┐      ┌────┬────┬────┬────┬────┐
            │  c │    │ b3 │    │    │      │    │ b2 │    │    │    │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │    │ b3 │    │ b3 │    │      │ b2 │    │ b2 │    │    │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │ b3 │    │ b3 │    │ b3 │      │    │ b2 │    │ b2 │    │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │    │ b3 │    │ b3 │    │      │    │    │ b2 │    │ b2 │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │    │    │ b3 │    │etc │      │    │    │    │ b2 │etc │
            └────┴────┴────┴────┴────┘      └────┴────┴────┴────┴────┘

    Note that the buffers don't have shape (n,3) and (n,2) because they only need to keep
    two consecutive groups of elements, because the recursion only needs the previous two groups.
    By using indices mod 2, the data needed at each iteration is in the columns not being updated.

    The updates roughly look like this:

    .. code-block::

                    ┌───────┐                     ┌───────┐
                    │k even │                     │ k odd │
                    └───────┘                     └───────┘
                 ┌──────┬──────┐               ┌──────┬──────┐
                 │  A───▶      │               │      ◀───A  │
        ┌────┐   │      │      │               │      │      │
        │buf2│   ├──────┼──────┤               ├──────┼──────┤
        └────┘   │  A───▶      │               │      ◀───A  │
                 │      │      │               │      │      │
                 └───┬──┴───▲──┘               └───▲──┴───┬──┘
                     └──b──┐│                     ┌┼──b───┘
                           ││                     ││
                     ┌──b──┼┘                     │└──b───┐
                 ┌───┴──┬──▼───┐               ┌──▼───┬───┴──┐
        ┌────┐   │  A───▶      │               │      ◀───A  │
        │buf3│   │      │      │               │      │      │
        └────┘   ├──────┼──────┤               ├──────┼──────┤
                 │  A───▶      │               │      ◀───A  │
                 │      │      │               │      │      │
                 ├──────┼──────┤               ├──────┼──────┤
                 │  A───▶      │               │      ◀───A  │
                 │      │      │               │      │      │
                 └──────┴──────┘               └──────┴──────┘

    A and b mean that the contribution is multiplied by some element of A or b.
    There are also diagonal A-arrows between the columns of the same buffer, but they are not shown here.
    For pivot in (k,k) buf2 is updated, for pivots in (k+1,k) and (k,k+1) buf3 is updated.

    The rules for updating are in https://quantum-journal.org/papers/q-2023-08-29-1097/
    """
    # reduced DMs
    M = len(b) // 2
    shape = np.ones(M, dtype=np.int64)
    A = A.reshape((2, M, 2, M)).transpose((1, 3, 0, 2))  # (M,M,2,2)
    b = b.reshape((2, M)).transpose()  # (M,2)
    zero = np.zeros((M - 1, M - 1), dtype=np.complex128)
    id = np.eye(M - 1, dtype=np.complex128)  # noqa: A001
    X = np.vstack((np.hstack((zero, id)), np.hstack((id, zero))))
    for m in range(M):
        idx_m = np.array([m])
        idx_n = np.delete(np.arange(M), m)
        A_mm = np.ascontiguousarray(A[idx_m, :][:, idx_m].transpose((2, 0, 3, 1))).reshape((2, 2))
        A_nn = np.ascontiguousarray(A[idx_n, :][:, idx_n].transpose((2, 0, 3, 1))).reshape(
            (2 * M - 2, 2 * M - 2),
        )
        A_mn = np.ascontiguousarray(A[idx_m, :][:, idx_n].transpose((2, 0, 3, 1))).reshape(
            (2, 2 * M - 2),
        )
        A_nm = np.transpose(A_mn)
        b_m = np.ascontiguousarray(b[idx_m].transpose()).reshape((2,))
        b_n = np.ascontiguousarray(b[idx_n].transpose()).reshape((2 * M - 2,))
        # single-mode A,b,c
        A_ = A_mm - A_mn @ np.linalg.inv(A_nn - X) @ A_nm
        b_ = b_m - A_mn @ np.linalg.inv(A_nn - X) @ b_n
        c_ = (
            c
            * np.exp(-0.5 * b_n @ np.linalg.inv(A_nn - X) @ b_n)
            / np.sqrt(np.linalg.det(A_nn - X))
        )
        # buffers are transposed with respect to the diagram
        buf2 = np.zeros((2, 2), dtype=np.complex128)
        buf3 = np.zeros((2, 3), dtype=np.complex128)
        buf3[0, 1] = c_  # vacuum probability at (0,0)
        norm = np.abs(c_)
        k = 0
        while norm < max_prob and k < max_shape:
            buf2[(k + 1) % 2] = (b_ * buf3[k % 2, 1] + A_ @ buf2[k % 2] * SQRT[k]) / SQRT[k + 1]
            buf3[(k + 1) % 2, 0] = (
                b_[0] * buf2[(k + 1) % 2, 0]
                + A_[0, 0] * buf3[k % 2, 1] * SQRT[k + 1]
                + A_[0, 1] * buf3[k % 2, 0] * SQRT[k]
            ) / SQRT[k + 2]
            buf3[(k + 1) % 2, 1] = (
                b_[1] * buf2[(k + 1) % 2, 0]
                + A_[1, 0] * buf3[k % 2, 1] * SQRT[k + 1]
                + A_[1, 1] * buf3[k % 2, 0] * SQRT[k]
            ) / SQRT[k + 1]
            buf3[(k + 1) % 2, 2] = (
                b_[1] * buf2[(k + 1) % 2, 1]
                + A_[1, 0] * buf3[k % 2, 2] * SQRT[k]
                + A_[1, 1] * buf3[k % 2, 1] * SQRT[k + 1]
            ) / SQRT[k + 2]
            norm += np.abs(buf3[(k + 1) % 2, 1])
            k += 1
        shape[m] = k
    return np.clip(shape, min_shape, max_shape)
