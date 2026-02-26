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

r"""
Custom optax ``GradientTransformation``\ s for non-euclidean parameter updates.
"""

from collections.abc import Callable

import jax
import optax

from mrmustard import math

__all__ = [
    "riemannian_gradient",
    "riemannian_retraction",
    "update_orthogonal",
    "update_symplectic",
    "update_unitary",
]


def riemannian_gradient(method: str):
    r"""
    Transforms Euclidean gradients to Riemannian gradients (Lie Algebra elements).
    """

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(grads, state, params):
        if params is None:
            return grads, state

        if method == "symplectic":
            updates = jax.tree_util.tree_map(
                lambda p, g: math.euclidean_to_symplectic(p, g),
                params,
                grads,
            )
        elif method == "unitary":
            updates = jax.tree_util.tree_map(
                lambda p, g: math.euclidean_to_unitary(p, g),
                params,
                grads,
            )
        elif method == "orthogonal":
            updates = jax.tree_util.tree_map(
                lambda p, g: math.euclidean_to_unitary(p, math.real(g)),
                params,
                grads,
            )
        else:
            updates = grads

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def riemannian_retraction():
    r"""
    Applies the retraction step: S_new = S @ expm(update).
    Returns S_new - S (the additive update expected by optax).
    """

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params):
        # updates is the step in Lie Algebra (e.g. -lr * m / v from AdaBelief)
        # params is the current point on the manifold

        def apply_step(p, u):
            p_new = math.matmul(p, math.expm(u))
            # Return additive difference (expected by optax)
            return p_new - p

        new_updates = jax.tree_util.tree_map(apply_step, params, updates)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def update_orthogonal(
    orthogonal_lr: float | Callable[[int], float], optimizer_cls: Callable = optax.adabelief
):
    r"""
    Creates an optax GradientTransformation for orthogonal parameter updates using Riemannian optimization.

    Args:
        orthogonal_lr: The learning rate for orthogonal updates.
        optimizer_cls: The optimizer class to use (default: optax.adabelief).

    Returns:
        An optax.GradientTransformation for orthogonal updates.
    """
    return optax.chain(
        riemannian_gradient("orthogonal"),
        optimizer_cls(learning_rate=orthogonal_lr),
        riemannian_retraction(),
    )


def update_symplectic(
    symplectic_lr: float | Callable[[int], float], optimizer_cls: Callable = optax.adabelief
):
    r"""
    Creates an optax GradientTransformation for symplectic parameter updates using Riemannian optimization.

    Args:
        symplectic_lr: The learning rate for symplectic updates.
        optimizer_cls: The optimizer class to use (default: optax.adabelief).

    Returns:
        An optax.GradientTransformation for symplectic updates.
    """
    return optax.chain(
        riemannian_gradient("symplectic"),
        optimizer_cls(learning_rate=symplectic_lr),
        riemannian_retraction(),
    )


def update_unitary(
    unitary_lr: float | Callable[[int], float], optimizer_cls: Callable = optax.adabelief
):
    r"""
    Creates an optax GradientTransformation for unitary parameter updates using Riemannian optimization.

    Args:
        unitary_lr: The learning rate for unitary updates.
        optimizer_cls: The optimizer class to use (default: optax.adabelief).

    Returns:
        An optax.GradientTransformation for unitary updates.
    """
    return optax.chain(
        riemannian_gradient("unitary"),
        optimizer_cls(learning_rate=unitary_lr),
        riemannian_retraction(),
    )
