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

"""This module contains tests for ``AnsatzFactory`` objects."""

import hashlib
import json

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab.states.builtins import number_state
from mrmustard.lab.transformations.builtins import amplifier_channel
from mrmustard.physics import triples
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.fock_utils import fock_state
from mrmustard.physics.mm_einsum import bargmann_to_fock, fock_to_bargmann
from mrmustard.physics.mm_einsum.conversions import to_bargmann, to_fock
from mrmustard.physics.wires import ReprEnum


@pytest.fixture
def amplifier_ansatz_dict():
    return {
        ReprEnum.BARGMANN: (amplifier_channel, ("gain", "lin_sup")),
        ReprEnum.FOCK: (bargmann_to_fock(amplifier_channel), ("gain", "lin_sup", "shape")),
    }


@pytest.fixture
def number_ansatz_dict():
    return {
        ReprEnum.BARGMANN: (fock_to_bargmann(number_state), ("n", "shape")),
        ReprEnum.FOCK: (number_state, ("n", "shape")),
    }


class TestAnsatzFactory:
    r"""Tests the ansatz factory."""

    def test_init(self, amplifier_ansatz_dict):
        ansatz_factory = AnsatzFactory(ansatz_dict=amplifier_ansatz_dict, foo="bar")
        assert isinstance(ansatz_factory, AnsatzFactory)
        assert ansatz_factory.ansatz_dict == amplifier_ansatz_dict
        assert ansatz_factory.additional_args == {"foo": "bar"}

        with pytest.raises(ValueError, match="at least one Ansatz function"):
            AnsatzFactory(ansatz_dict={})

    def test_bargmann_to_fock_keyword_only_params(self):
        r"""Test that bargmann_to_fock decorator correctly handles keyword-only parameters."""

        # Create a function with keyword-only parameters
        def test_func(x, *, y=10, z=20):
            # Return a simple ansatz-like object that we can verify
            return PolyExpAnsatz(*triples.amplifier_Abc(gain=x + y + z))

        # Decorate it
        decorated = bargmann_to_fock(test_func)

        # Test calling with keyword-only parameters
        result = decorated(1.0, y=2.0, z=3.0, shape=(5, 5, 5, 5))
        assert isinstance(result, ArrayAnsatz)

        # Verify the function received the correct parameters by checking the result
        # The gain should be 1.0 + 2.0 + 3.0 = 6.0
        expected = to_fock(PolyExpAnsatz(*triples.amplifier_Abc(gain=6.0)), shape=(5, 5, 5, 5))
        assert result == expected

        # Test with default values
        result2 = decorated(1.0, shape=(5, 5, 5, 5))
        expected2 = to_fock(PolyExpAnsatz(*triples.amplifier_Abc(gain=31.0)), shape=(5, 5, 5, 5))
        assert result2 == expected2

    def test_call_bargmann_to_fock(self, amplifier_ansatz_dict):
        gain = 1.2
        shape = (10, 10, 10, 10)

        ansatz_factory = AnsatzFactory(ansatz_dict=amplifier_ansatz_dict)
        expected_ansatz = PolyExpAnsatz(*triples.amplifier_Abc(gain=gain))

        # test Bargmann representation
        ansatz = ansatz_factory(representation=ReprEnum.BARGMANN, gain=gain)
        assert isinstance(ansatz, PolyExpAnsatz)
        assert ansatz == expected_ansatz

        # test cached ansatz
        assert ansatz_factory(representation=ReprEnum.BARGMANN, gain=gain) is ansatz

        # test Fock representation
        fock_ansatz = ansatz_factory(representation=ReprEnum.FOCK, gain=gain, shape=shape)
        assert isinstance(fock_ansatz, ArrayAnsatz)
        assert fock_ansatz == to_fock(expected_ansatz, shape=shape)

        # test cached ansatz
        assert ansatz_factory(representation=ReprEnum.FOCK, gain=gain, shape=shape) is fock_ansatz

        # test error for unknown representation
        with pytest.raises(
            NotImplementedError, match="No ansatz function found for representation QUADRATURE!"
        ):
            ansatz_factory(representation=ReprEnum.QUADRATURE, gain=gain)

    def test_call_fock_to_bargmann(self, number_ansatz_dict):
        n = 10
        cutoff = 20
        shape = (cutoff + 1,)

        ansatz_factory = AnsatzFactory(ansatz_dict=number_ansatz_dict)
        expected_ansatz = ArrayAnsatz(fock_state(n, cutoff=cutoff), batch_dims=len(math.shape(n)))

        # test Fock representation
        ansatz = ansatz_factory(representation=ReprEnum.FOCK, n=n, cutoff=cutoff, shape=shape)
        assert isinstance(ansatz, ArrayAnsatz)
        assert ansatz == expected_ansatz

        # test cached ansatz
        assert (
            ansatz_factory(representation=ReprEnum.FOCK, n=n, cutoff=cutoff, shape=shape) is ansatz
        )

        # test Bargmann representation
        bargmann_ansatz = ansatz_factory(
            representation=ReprEnum.BARGMANN, n=n, cutoff=cutoff, shape=shape
        )
        assert isinstance(bargmann_ansatz, PolyExpAnsatz)
        assert bargmann_ansatz == to_bargmann(expected_ansatz)

        # test cached ansatz
        assert (
            ansatz_factory(representation=ReprEnum.BARGMANN, n=n, cutoff=cutoff, shape=shape)
            is bargmann_ansatz
        )

        # test error for unknown representation
        with pytest.raises(
            NotImplementedError, match="No ansatz function found for representation QUADRATURE!"
        ):
            ansatz_factory(representation=ReprEnum.QUADRATURE, n=n, cutoff=cutoff)

    def test_from_ansatz(self):
        ansatz = PolyExpAnsatz(*triples.amplifier_Abc(gain=1.2))
        ansatz_factory, representation = AnsatzFactory.from_ansatz(ansatz)
        assert isinstance(ansatz_factory, AnsatzFactory)
        assert representation == ReprEnum.BARGMANN

        ansatz = to_fock(ansatz, shape=(10, 10, 10, 10))
        ansatz_factory, representation = AnsatzFactory.from_ansatz(ansatz)
        assert isinstance(ansatz_factory, AnsatzFactory)
        assert representation == ReprEnum.FOCK

    def test_get_cached_ansatz(self, amplifier_ansatz_dict):
        ansatz_factory = AnsatzFactory(ansatz_dict=amplifier_ansatz_dict)
        ansatz = ansatz_factory(representation=ReprEnum.BARGMANN, gain=1.2)

        assert ansatz is not None
        assert (
            ansatz_factory.get_cached_ansatz(representation=ReprEnum.BARGMANN, gain=1.2) == ansatz
        )
        assert ansatz_factory.get_cached_ansatz(representation=ReprEnum.FOCK, gain=1.2) is None
        assert ansatz_factory.get_cached_ansatz(representation=ReprEnum.BARGMANN, gain=1.5) is None

    def test_hash_kwargs(self, amplifier_ansatz_dict):
        r"""Test the _hash_kwargs method for various input types."""
        ansatz_factory = AnsatzFactory(ansatz_dict=amplifier_ansatz_dict)

        # Test empty kwargs
        empty_hash = ansatz_factory._hash_kwargs()
        expected_empty_hash = hashlib.sha256(
            json.dumps({}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        assert empty_hash == expected_empty_hash
        assert isinstance(empty_hash, str)
        assert len(empty_hash) == 64  # SHA256 hex digest length

        # Test simple types
        int_hash = ansatz_factory._hash_kwargs(x=42)
        assert isinstance(int_hash, str)
        assert len(int_hash) == 64

        float_hash = ansatz_factory._hash_kwargs(x=3.14)
        assert isinstance(float_hash, str)

        complex_hash = ansatz_factory._hash_kwargs(x=1.0 + 2.0j)
        assert isinstance(complex_hash, str)

        bool_hash = ansatz_factory._hash_kwargs(x=True)
        assert isinstance(bool_hash, str)

        str_hash = ansatz_factory._hash_kwargs(x="test")
        assert isinstance(str_hash, str)

        none_hash = ansatz_factory._hash_kwargs(x=None)
        assert isinstance(none_hash, str)

        # Test order independence: same kwargs in different order should produce same hash
        hash_order1 = ansatz_factory._hash_kwargs(x=42, y="test", z=3.14)
        hash_order2 = ansatz_factory._hash_kwargs(z=3.14, x=42, y="test")
        assert hash_order1 == hash_order2

        # Test different kwargs produce different hashes
        hash_diff1 = ansatz_factory._hash_kwargs(x=42)
        assert isinstance(hash_diff1, str)
        hash_diff2 = ansatz_factory._hash_kwargs(x=42)
        assert hash_diff1 == hash_diff2
        hash_diff3 = ansatz_factory._hash_kwargs(x=43)
        assert hash_diff1 != hash_diff3

        # Test lists
        list_hash = ansatz_factory._hash_kwargs(x=[1, 2, 3])
        assert isinstance(list_hash, str)
        list_hash2 = ansatz_factory._hash_kwargs(x=[1, 2, 3])
        assert list_hash == list_hash2
        list_hash3 = ansatz_factory._hash_kwargs(x=[1, 2, 4])
        assert list_hash != list_hash3

        # Test tuples
        tuple_hash = ansatz_factory._hash_kwargs(x=(1, 2, 3))
        assert isinstance(tuple_hash, str)
        tuple_hash2 = ansatz_factory._hash_kwargs(x=(1, 2, 3))
        assert tuple_hash == tuple_hash2
        tuple_hash3 = ansatz_factory._hash_kwargs(x=(1, 2, 4))
        assert tuple_hash != tuple_hash3

        # Test dicts
        dict_hash = ansatz_factory._hash_kwargs(x={"a": 1, "b": 2})
        assert isinstance(dict_hash, str)
        dict_hash2 = ansatz_factory._hash_kwargs(x={"a": 1, "b": 2})
        assert dict_hash == dict_hash2
        dict_hash3 = ansatz_factory._hash_kwargs(x={"c": 3, "d": 4})
        assert dict_hash != dict_hash3

        # Test arrays
        arr1 = math.astensor([1, 2, 3])
        arr_hash1 = ansatz_factory._hash_kwargs(x=arr1)
        assert isinstance(arr_hash1, str)
        arr_hash2 = ansatz_factory._hash_kwargs(x=arr1)
        assert arr_hash1 == arr_hash2
        arr2 = math.astensor([4, 5, 6])
        arr_hash3 = ansatz_factory._hash_kwargs(x=arr2)
        assert arr_hash1 != arr_hash3

        # Test numpy scalar types
        np_int_hash = ansatz_factory._hash_kwargs(x=np.int64(42))
        assert isinstance(np_int_hash, str)

        # Test nested structures
        nested_hash = ansatz_factory._hash_kwargs(
            x=[1, 2, {"a": math.astensor([1, 2]), "b": (3, 4)}]
        )
        assert isinstance(nested_hash, str)
        nested_hash2 = ansatz_factory._hash_kwargs(
            x=[1, 2, {"a": math.astensor([1, 2]), "b": (3, 4)}]
        )
        assert nested_hash == nested_hash2
        nested_hash3 = ansatz_factory._hash_kwargs(
            x=[1, 2, {"c": math.astensor([5, 6]), "d": (7, 8)}]
        )
        assert nested_hash != nested_hash3

        # Test collision cases
        hash1 = ansatz_factory._hash_kwargs(a=1, b=2)
        hash2 = ansatz_factory._hash_kwargs(a1="b2")
        assert hash1 != hash2

        hash3 = ansatz_factory._hash_kwargs(n=10)
        hash4 = ansatz_factory._hash_kwargs(n1=0)
        assert hash3 != hash4

        hash5 = ansatz_factory._hash_kwargs(x=12, y=3)
        hash6 = ansatz_factory._hash_kwargs(x1=2, y=3)
        assert hash5 != hash6

        hash7 = ansatz_factory._hash_kwargs(ab="cd")
        hash8 = ansatz_factory._hash_kwargs(a="b", c="d")
        assert hash7 != hash8

        hash9 = ansatz_factory._hash_kwargs(x=1.0 + 2.0j)
        hash10 = ansatz_factory._hash_kwargs(x=[1.0, 2.0])
        assert hash9 != hash10
