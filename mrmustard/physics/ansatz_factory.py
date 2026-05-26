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

"""This module contains the ``AnsatzFactory`` class."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from mrmustard import math

from .ansatz import Ansatz, PolyExpAnsatz
from .wires import ReprEnum

__all__ = ["AnsatzFactory"]


class AnsatzFactory:
    r"""A base class for ansatz factories.
    Takes a function that represents the ansätze and generates new ansätze given parameters.
    Caches the ansätze for given parameters and representation.

    Args:
        ansatz_dict: A dictionary mapping ``ReprEnum`` to a tuple containing a function that generates the respective
            ansätze and a list of its parameter names.
        **kwargs: Additional keyword arguments to pass to the ansatz functions.

    Raises:
        ValueError: If the AnsatzFactory is initialized with an empty dictionary.
    """

    def __init__(
        self,
        ansatz_dict: Mapping[ReprEnum, tuple[Callable[..., Ansatz], tuple[str, ...]]],
        **kwargs,
    ) -> None:
        if not ansatz_dict:
            raise ValueError(
                "The AnsatzFactory must be initialized with at least one Ansatz function!"
            )
        self._additional_args = kwargs
        self._ansatz_cache = {}
        self._ansatz_dict = ansatz_dict

    @property
    def ansatz_dict(self) -> Mapping[ReprEnum, tuple[Callable[..., Ansatz], tuple[str, ...]]]:
        r"""The ansatz generating functions of this ansatz factory."""
        return self._ansatz_dict

    @property
    def additional_args(self) -> dict[str, Any]:
        r"""The additional arguments to pass into the ansatz functions."""
        return self._additional_args

    @classmethod
    def from_ansatz(
        cls, ansatz: Ansatz, representation: ReprEnum | None = None
    ) -> tuple[AnsatzFactory, ReprEnum]:
        r"""Creates an AnsatzFactory from an Ansatz.

        Args:
            ansatz: The ansatz to create the AnsatzFactory from.
            representation: The representation of the ansatz. If None, it will be inferred from the ansatz.

        Returns:
            A tuple containing an AnsatzFactory with the given ansatz and the representation of the ansatz.
        """
        if representation is None:
            representation = (
                ReprEnum.BARGMANN if isinstance(ansatz, PolyExpAnsatz) else ReprEnum.FOCK
            )
        return cls(ansatz_dict={representation: (lambda **kwargs: ansatz, ())}), representation

    def get_cached_ansatz(self, representation: ReprEnum, **kwargs: Any) -> Ansatz | None:
        r"""Retrieves the cached ansatz for the given kwargs.
        Returns None if caching should be skipped.

        Args:
            representation: The representation of the ansatz.
            **kwargs: The arguments used to generate the ansatz.

        Returns:
            The cached ansatz for the given kwargs.
        """
        hashed = self._hash_kwargs(**kwargs)
        cached = self._ansatz_cache.get(representation, None)
        if cached is None:
            return None
        return cached.get(hashed, None)

    def _hash_kwargs(self, **kwargs: Any) -> str:
        r"""Efficiently hash keyword arguments for caching.

        Args:
            **kwargs: Keyword arguments to hash.

        Returns:
            A SHA256 hexadecimal hash string.
        """
        json_data = json.dumps(self._prepare_hashable(kwargs), sort_keys=True)
        return hashlib.sha256(json_data.encode("utf-8")).hexdigest()

    def _prepare_hashable(self, obj: Any) -> Any:  # noqa: PLR0911
        r"""Recursively converts unhashable types into hashable/serializable types.
        Note: if the backend is JAX, ``Tracer`` objects are skipped.

        Args:
            obj: The object to convert.

        Returns:
            The converted object.
        """
        if obj is None:
            return None
        if math.backend_name == "jax":
            from jax.core import Tracer  # noqa: PLC0415

            if isinstance(obj, Tracer):
                return None
        if isinstance(obj, dict):
            return {k: self._prepare_hashable(obj[k]) for k in sorted(obj.keys())}
        if isinstance(obj, (list, tuple)):
            return [self._prepare_hashable(i) for i in obj]
        if isinstance(obj, complex):
            return ["__complex__", obj.real, obj.imag]
        if hasattr(obj, "tolist"):
            return self._prepare_hashable(obj.tolist())
        if hasattr(obj, "dtype"):
            return self._prepare_hashable(obj.item())
        return obj

    def _set_cached_ansatz(self, ansatz: Ansatz, representation: ReprEnum, **kwargs: Any) -> None:
        r"""Caches the ansatz for the given kwargs.
        Skips caching if hashing returns None.

        Args:
            ansatz: The ansatz to cache.
            representation: The representation of the ansatz.
            **kwargs: The kwargs used to generate the ansatz.
        """
        hashed = self._hash_kwargs(**kwargs)
        if hashed is None:
            return
        cached_repr = self._ansatz_cache.get(representation, None)
        if cached_repr is None:
            self._ansatz_cache[representation] = {}
        cached_ansatz = self._ansatz_cache[representation].get(hashed, None)
        if cached_ansatz is None:
            self._ansatz_cache[representation][hashed] = ansatz

    def __call__(self, representation: ReprEnum, **kwargs: Any) -> Ansatz:
        r"""Generates the ansatz of this ansatz factory.

        Args:
            representation: The representation to compute the ansatz for.
            **kwargs: The arguments to pass into the respective ansatz function.
                Note: collisions with additional arguments are possible and
                will take precedence over the additional arguments.

        Returns:
            The ansatz generated by the ansatz factory.

        Raises:
            NotImplementedError: If no ansatz function is found for the specified representation.
        """
        ansatz_fn_params = self._ansatz_dict.get(representation, None)

        if ansatz_fn_params is None:
            raise NotImplementedError(
                f"No ansatz function found for representation {representation}!"
            )

        ansatz_fn, params = ansatz_fn_params
        # parse kwargs
        ansatz_kwargs = self._additional_args.copy()
        for param_name in params:
            if param_name in kwargs:
                param = kwargs.pop(param_name)
                try:  # noqa: SIM105
                    param = param.value
                except AttributeError:
                    pass
                ansatz_kwargs[param_name] = param

        if (
            cached_ansatz := self.get_cached_ansatz(representation=representation, **ansatz_kwargs)
        ) is not None:
            return cached_ansatz

        ret = ansatz_fn(**ansatz_kwargs)
        self._set_cached_ansatz(ansatz=ret, representation=representation, **ansatz_kwargs)

        return ret
