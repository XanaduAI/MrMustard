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

"""Test the MrMustard serialization library."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pytest
import tensorflow as tf

from mrmustard import math
from mrmustard.lab_dev import Circuit, Coherent, Dgate
from mrmustard.lab_dev.circuit_components import AdjointView
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.utils.serialize import save, load, get_zipfile, cache_subdir

from ..conftest import skip_np
from ..random import Abc_triple


@dataclass
class Dummy:
    """A dummy class for testing."""

    val: int
    word: str

    @classmethod
    def deserialize(cls, data):
        """Basic deserializer method."""
        return cls(**data)


@dataclass
class DummyOneNP:
    """A dummy class with numpy data."""

    name: str
    array: np.ndarray

    @classmethod
    def deserialize(cls, data):
        """Basic deserializer method."""
        return cls(**data)


@dataclass
class DummyTwoNP:
    """Another dummy class with more numpy data."""

    name: str
    array1: np.ndarray
    array2: np.ndarray

    @classmethod
    def deserialize(cls, data):
        """Basic deserializer method."""
        return cls(**data)


@pytest.fixture(name="cache_dir")
def fixture_cache_dir(tmpdir, monkeypatch):
    """Mock the serialization cache using tmpdir, and return the new cache folder."""
    cache = Path(tmpdir)
    monkeypatch.setattr("mrmustard.utils.serialize.CACHE", cache)
    return cache


class TestSerialize:
    """Test the serialize module."""

    def test_basic(self, cache_dir):
        """Test basic save and load functionality."""
        path = save(Dummy, val=5, word="hello")
        assert path.exists() and path.parent == cache_dir
        assert load(path) == Dummy(val=5, word="hello")
        assert not list(cache_dir.glob("*"))  # removed by load

    def test_one_numpy_obj(self, cache_dir):
        """Test save and load functionality with numpy data."""
        path = save(DummyOneNP, name="myname", arrays={"array": np.array([1.1, 2.2])})
        assert path.exists() and path.with_suffix(".npz").exists()
        loaded = load(path)

        assert isinstance(loaded, DummyOneNP)
        assert loaded.name == "myname"
        assert np.array_equal(loaded.array, np.array([1.1, 2.2]))
        assert not list(cache_dir.glob("*"))

    def test_two_numpy_obj(self, cache_dir):
        """Test save and load functionality with more numpy data."""
        a1 = np.array([1.1, 2.2])
        a2 = np.array([3.3 + 4.4j, 5.5 + 6.6j])
        path = save(DummyTwoNP, name="myname", arrays={"array1": a1, "array2": a2})
        assert path.exists() and path.with_suffix(".npz").exists()
        loaded = load(path)

        assert isinstance(loaded, DummyTwoNP)
        assert loaded.name == "myname"
        assert np.array_equal(loaded.array1, a1)
        assert np.array_equal(loaded.array2, a2)
        assert not list(cache_dir.glob("*"))

    def test_overlap_forbidden(self):
        """Test that array names must be distinct from non-array names."""
        with pytest.raises(
            ValueError, match=r"Arrays cannot have the same name as generic data: {'val'}"
        ):
            save(Dummy, arrays={"val": [1]}, val=2)

    def test_tensorflow_support(self, cache_dir):
        """Test that TensorFlow data is supported."""
        skip_np()
        x = math.astensor([1.1, 2.2])
        loaded = load(save(DummyOneNP, name="myname", arrays={"array": x}))
        assert tf.is_tensor(loaded.array)
        assert np.array_equal(loaded.array, x)
        assert not list(cache_dir.glob("*"))

    def test_backend_change_error(self, cache_dir, monkeypatch):
        """Test that data must be deserialized with the same backend."""
        skip_np()
        x = math.astensor([1.1, 2.2])
        path = save(DummyOneNP, name="myname", arrays={"array": x})
        # can be thought of as restarting python and not changing to tensorflow
        monkeypatch.setattr("mrmustard.math._backend._name", "numpy")
        with pytest.raises(
            TypeError,
            match="Data serialized with tensorflow backend, cannot deserialize to the currently active numpy backend",
        ):
            load(path)
        assert sorted(cache_dir.glob("*")) == [path, path.with_suffix(".npz")]


class TestHelpers:
    """Test various helper functions from the serialize module."""

    def test_get_zipfile(self, cache_dir):
        """Test the get_zipfile helper."""
        result = get_zipfile()
        assert not result.exists()  # it doesn't make the file
        assert result.parent == cache_dir
        assert re.match(r"^collection_[a-f0-9]{32}\.zip$", result.name)

        assert get_zipfile("myfile.zip") == cache_dir / "myfile.zip"

    def test_cache_subdir_context(self, cache_dir):
        """Test the cache_subdir context manager."""
        with cache_subdir() as subdir:
            # something that uses CACHE internally
            assert get_zipfile().parent == subdir

        assert get_zipfile().parent == cache_dir
        assert subdir.parent == cache_dir
        assert subdir.exists()


@pytest.mark.parametrize(
    "obj",
    [
        lambda: Circuit([Coherent([0], x=1.0), Dgate([0], 0.1)]),
        lambda: AdjointView(Dgate([1], x=0.1, y=0.1)),
        lambda: Fock(np.random.random((5, 7, 8)), batched=False),
        lambda: Fock(np.random.random((1, 5, 7, 8)), batched=True),
        lambda: Bargmann(*Abc_triple(2)),
    ],
)
def test_actual_objects(obj, cache_dir):
    r"""
    Test that serializing then deserializing a MrMustard object creates an equivalent instance.
    """
    obj = obj()
    assert load(obj.serialize()) == obj
    assert not list(cache_dir.glob("*"))
