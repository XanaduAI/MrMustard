# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from mrmustard import math
from mrmustard.physics.ansatz import Ansatz, ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.triples import identity_Abc

__all__ = [
    "bargmann_to_fock",
    "fock_to_bargmann",
    "to_bargmann",
    "to_fock",
]


def bargmann_to_fock(func: Callable[..., PolyExpAnsatz]) -> Callable[..., ArrayAnsatz]:
    r"""Decorator that wraps a function returning a ``PolyExpAnsatz``
    and returns an ``ArrayAnsatz`` instead using ``to_fock``.

    Args:
        func: A callable that returns a ``PolyExpAnsatz``.

    Returns:
        A wrapped function that returns an ``ArrayAnsatz`` constructed
        from the Bargmann triple (A, b, c) using ``to_fock``.
        The wrapped function has the same signature as the original
        function with an additional optional ``shape`` parameter.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        shape = kwargs.pop("shape", None)
        ansatz = func(*args, **kwargs)
        return to_fock(ansatz, shape=shape)

    return wrapper


def fock_to_bargmann(func: Callable[..., ArrayAnsatz]) -> Callable[..., PolyExpAnsatz]:
    r"""Decorator that wraps a function returning an ``ArrayAnsatz``
    and returns a ``PolyExpAnsatz`` instead using ``to_bargmann``.

    Args:
        func: A callable that returns an ``ArrayAnsatz``.

    Returns:
        A wrapped function that returns a ``PolyExpAnsatz`` constructed
        from the ``ArrayAnsatz`` using ``to_bargmann``.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        ansatz = func(*args, **kwargs)
        return to_bargmann(ansatz)

    return wrapper


def to_fock(
    ansatz: Ansatz,
    shape: tuple[int, ...],
    stable: bool | None = None,
    preserve_lin_sup: bool = False,
) -> ArrayAnsatz:
    r"""Converts a PolyExpAnsatz to an ArrayAnsatz.
    If the ansatz is already an ArrayAnsatz, it reduces the shape to the given shape.

    Args:
        ansatz: The ansatz to convert.
        shape: The shape of the ArrayAnsatz.
        stable: Whether to use the stable version of the hermite_renormalized function.
        preserve_lin_sup: If True, do not sum over the linear superposition dimension.

    Returns:
        ArrayAnsatz: The converted ArrayAnsatz.
    """
    if 0 in shape:
        raise ValueError("Fock space dimension is 0.")
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)

    sum_lin_sup = ansatz._lin_sup and not preserve_lin_sup
    batch_dims = ansatz.batch_dims - int(sum_lin_sup)

    if len(shape) == 0:
        array = ansatz.scalar if sum_lin_sup else ansatz.c
    else:
        A, b, c = ansatz.triple
        # TODO: make hermite_renormalized work with num_derived_vars > 0 in sc-97587
        if ansatz.num_derived_vars == 0:
            array = math.hermite_renormalized(
                A,
                b,
                c,
                shape=shape,
                stable=stable,
            )
        else:
            G = math.hermite_renormalized(
                A,
                b,
                math.ones(ansatz.batch_shape, dtype=math.complex128),
                shape=shape + ansatz.shape_derived_vars,
                stable=stable,
            )
            G = math.reshape(G, ansatz.batch_shape + shape + (-1,))
            cs = math.reshape(c, (*ansatz.batch_shape, -1))
            core_str = "".join(
                [chr(i) for i in range(97, 97 + len(G.shape[ansatz.batch_dims :]))],
            )
            array = math.einsum(f"...{core_str},...{core_str[-1]}->...{core_str[:-1]}", G, cs)

    if sum_lin_sup and len(shape) > 0:
        array = math.sum(array, axis=ansatz.batch_dims - 1)

    return ArrayAnsatz(array, batch_dims)


def to_bargmann(ansatz: Ansatz) -> PolyExpAnsatz:
    r"""Converts an ArrayAnsatz to a PolyExpAnsatz.
    If the ansatz is already a PolyExpAnsatz, it returns the ansatz unchanged.

    Args:
        ansatz: The ansatz to convert.

    Returns:
        PolyExpAnsatz: The converted PolyExpAnsatz.
    """
    if isinstance(ansatz, PolyExpAnsatz):
        return ansatz
    try:
        A, b, c = ansatz._original_abc_data
    except (AttributeError, TypeError):
        A, b, _ = identity_Abc(ansatz.core_dims)
        A = math.broadcast_to(A, (*ansatz.batch_shape, 2 * ansatz.core_dims, 2 * ansatz.core_dims))
        b = math.broadcast_to(b, (*ansatz.batch_shape, 2 * ansatz.core_dims))
        c = ansatz.array
    return PolyExpAnsatz(A, b, c)
