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
Custom optax ``GradientTransformation``s for non-euclidean parameter updates.
"""

import jax
import optax

from mrmustard import math


def update_orthogonal(orthogonal_lr: float):
    r"""
    Creates an optax GradientTransformation for orthogonal parameter updates.

    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.

    Args:
        orthogonal_lr: The learning rate for orthogonal updates.

    Returns:
        An optax.GradientTransformation for orthogonal updates.
    """

    def init_fn(params):
        return None

    def update_fn(grads, state, params):
        def update_single(dO_euclidean, O):
            Y = math.euclidean_to_unitary(O, math.real(dO_euclidean))
            new_value = math.matmul(O, math.expm(-orthogonal_lr * Y))
            return new_value - O

        updates = jax.tree_util.tree_map(update_single, grads, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def update_symplectic(symplectic_lr: float):
    r"""Creates an optax GradientTransformation for symplectic parameter updates.

    Implemented from:
        Wang J, Sun H, Fiori S. A Riemannian-steepest-descent approach
        for optimization on the real symplectic group.
        Mathematical Methods in the Applied Sciences. 2018 Jul 30;41(11):4273-86.

    Args:
        symplectic_lr: The learning rate for symplectic updates.

    Returns:
        An optax.GradientTransformation for symplectic updates.
    """

    def init_fn(params):
        return None

    def update_fn(grads, state, params):
        def update_single(dS_euclidean, S):
            Y = math.euclidean_to_symplectic(S, dS_euclidean)
            YT = math.transpose(Y)
            new_value = math.matmul(
                S,
                math.expm(-symplectic_lr * YT) @ math.expm(-symplectic_lr * (Y - YT)),
            )
            return new_value - S

        updates = jax.tree_util.tree_map(update_single, grads, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def update_unitary(unitary_lr: float):
    r"""
    Creates an optax GradientTransformation for unitary parameter updates.

    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.

    Args:
        unitary_lr: The learning rate for unitary updates.

    Returns:
        An optax.GradientTransformation for unitary updates.
    """

    def init_fn(params):
        return None

    def update_fn(grads, state, params):
        def update_single(dU_euclidean, U):
            Y = math.euclidean_to_unitary(U, math.conj(dU_euclidean))
            new_value = math.matmul(U, math.expm(-unitary_lr * Y))
            return new_value - U

        updates = jax.tree_util.tree_map(update_single, grads, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
