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
import json
import re

import numpy as np
import pytest
import tensorflow as tf

from mrmustard import math, settings, __version__
from mrmustard.lab_dev import Circuit, Coherent, Dgate
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


@pytest.fixture(autouse=True)
def cache_dir(tmpdir):
    """Set the serialization cache using tmpdir."""
    settings.CACHE_DIR = tmpdir


class TestSerialize:
    """Test the serialize module."""

    @pytest.mark.parametrize("remove_after", [False, True])
    def test_basic(self, remove_after):
        """Test basic save and load functionality."""
        path = save(Dummy, val=5, word="hello")
        assert path.exists() and path.parent == settings.CACHE_DIR
        with path.open() as f:
            assert json.load(f) == {
                "class": f"{Dummy.__module__}.{Dummy.__qualname__}",
                "version": __version__,
                "val": 5,
                "word": "hello",
            }
        assert load(path, remove_after=remove_after) == Dummy(val=5, word="hello")
        cached_files = list(settings.CACHE_DIR.glob("*"))
        assert (not cached_files) if remove_after else (cached_files == [path])

    def test_one_numpy_obj(self):
        """Test save and load functionality with numpy data."""
        path = save(DummyOneNP, name="myname", arrays={"array": np.array([1.1, 2.2])})
        assert path.exists() and path.with_suffix(".npz").exists()
        loaded = load(path)

        assert isinstance(loaded, DummyOneNP)
        assert loaded.name == "myname"
        assert np.array_equal(loaded.array, np.array([1.1, 2.2]))
        assert sorted(settings.CACHE_DIR.glob("*")) == [path, path.with_suffix(".npz")]

    def test_two_numpy_obj(self):
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

    def test_overlap_forbidden(self):
        """Test that array names must be distinct from non-array names."""
        with pytest.raises(
            ValueError, match=r"Arrays cannot have the same name as generic data: {'val'}"
        ):
            save(Dummy, arrays={"val": [1]}, val=2)

    def test_tensorflow_support(self):
        """Test that TensorFlow data is supported."""
        skip_np()
        x = math.astensor([1.1, 2.2])
        loaded = load(save(DummyOneNP, name="myname", arrays={"array": x}))
        assert tf.is_tensor(loaded.array)
        assert np.array_equal(loaded.array, x)

    def test_backend_change_error(self, monkeypatch):
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
        assert sorted(settings.CACHE_DIR.glob("*")) == [path, path.with_suffix(".npz")]


class TestHelpers:
    """Test various helper functions from the serialize module."""

    def test_get_zipfile(self):
        """Test the get_zipfile helper."""
        result = get_zipfile()
        assert not result.exists()  # it doesn't make the file
        assert result.parent == settings.CACHE_DIR
        assert re.match(r"^collection_[a-f0-9]{32}\.zip$", result.name)

        assert get_zipfile("myfile.zip") == settings.CACHE_DIR / "myfile.zip"

    def test_cache_subdir_context(self):
        """Test the cache_subdir context manager."""
        with cache_subdir() as subdir:
            # something that uses CACHE internally
            assert get_zipfile().parent == subdir

        assert get_zipfile().parent == settings.CACHE_DIR
        assert subdir.parent == settings.CACHE_DIR
        assert subdir.exists()


@pytest.mark.parametrize(
    "obj",
    [
        lambda: Circuit([Coherent([0], x=1.0), Dgate([0], 0.1)]),
        lambda: Dgate([1], x=0.1, y=0.1),
        lambda: Fock(np.random.random((5, 7, 8)), batched=False),
        lambda: Fock(np.random.random((1, 5, 7, 8)), batched=True),
        lambda: Bargmann(*Abc_triple(2)),
    ],
)
def test_actual_objects(obj):
    r"""
    Test that serializing then deserializing a MrMustard object creates an equivalent instance.
    """
    obj = obj()
    assert load(obj.serialize()) == obj
