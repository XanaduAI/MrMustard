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

import json
from dataclasses import dataclass

import numpy as np
import pytest
import tensorflow as tf

from mrmustard import __version__, math, settings
from mrmustard.lab import (
    Amplifier,
    Attenuator,
    BSgate,
    BtoChar,
    BtoQ,
    Circuit,
    Coherent,
    Dgate,
    DisplacedSqueezed,
    FockDamping,
    Identity,
    Number,
    QuadratureEigenstate,
    Rgate,
    S2gate,
    Sgate,
    SqueezedVacuum,
    Thermal,
    TraceOut,
    TwoModeSqueezedVacuum,
    Vacuum,
)
from mrmustard.utils.serialize import load, save


class Deserialize:
    """Base class with a simple deserialization implementation."""

    @classmethod
    def deserialize(cls, data):
        """Basic deserializer method."""
        return cls(**data)


@dataclass
class Dummy(Deserialize):
    """A dummy class for testing."""

    val: int
    word: str


@dataclass
class DummyOneNP(Deserialize):
    """A dummy class with numpy data."""

    name: str
    array: np.ndarray


@dataclass
class DummyTwoNP(Deserialize):
    """Another dummy class with more numpy data."""

    name: str
    array1: np.ndarray
    array2: np.ndarray


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
        with path.open("r", encoding="utf-8") as f:
            assert json.load(f) == {
                "class": f"{Dummy.__module__}.{Dummy.__qualname__}",
                "version": __version__,
                "arrays": False,
                "val": 5,
                "word": "hello",
            }
        assert load(path, remove_after=remove_after) == Dummy(val=5, word="hello")
        cached_files = list(settings.CACHE_DIR.glob("*"))
        assert (not cached_files) if remove_after else (cached_files == [path])

    def test_one_numpy_obj(self):
        """Test save and load functionality with numpy data."""
        path = save(DummyOneNP, name="myname", arrays={"array": np.array([1.1, 2.2])})
        assert path.exists() and path.suffix == ".zip"
        loaded = load(path)

        assert isinstance(loaded, DummyOneNP)
        assert loaded.name == "myname"
        assert np.array_equal(loaded.array, np.array([1.1, 2.2]))
        assert list(settings.CACHE_DIR.glob("*")) == [path]

    def test_two_numpy_obj(self):
        """Test save and load functionality with more numpy data."""
        a1 = np.array([1.1, 2.2])
        a2 = np.array([3.3 + 4.4j, 5.5 + 6.6j])
        path = save(DummyTwoNP, name="myname", arrays={"array1": a1, "array2": a2})
        assert path.exists() and path.suffix == ".zip"
        loaded = load(path)

        assert isinstance(loaded, DummyTwoNP)
        assert loaded.name == "myname"
        assert np.array_equal(loaded.array1, a1)
        assert np.array_equal(loaded.array2, a2)

    def test_overlap_forbidden(self):
        """Test that array names must be distinct from non-array names."""
        with pytest.raises(
            ValueError,
            match=r"Arrays cannot have the same name as generic data: {'val'}",
        ):
            save(Dummy, arrays={"val": [1]}, val=2)

    @pytest.mark.requires_backend("tensorflow")
    def test_tensorflow_support(self):
        """Test that TensorFlow data is supported."""
        x = math.astensor([1.1, 2.2])
        loaded = load(save(DummyOneNP, name="myname", arrays={"array": x}))
        assert tf.is_tensor(loaded.array)
        assert np.array_equal(loaded.array, x)

    @pytest.mark.requires_backend("tensorflow")
    def test_backend_change_error(self, monkeypatch):
        """Test that data must be deserialized with the same backend."""
        x = math.astensor([1.1, 2.2])
        path = save(DummyOneNP, name="myname", arrays={"array": x})
        # can be thought of as restarting python and not changing to tensorflow
        monkeypatch.setattr("mrmustard.math._backend._name", "numpy")
        with pytest.raises(
            TypeError,
            match="Data serialized with tensorflow backend, cannot deserialize to the currently active numpy backend",
        ):
            load(path)
        assert sorted(settings.CACHE_DIR.glob("*")) == [path]

    def test_zip_remove_after(self):
        """Test that remove_after works with zip files."""
        path = save(DummyOneNP, name="myname", arrays={"array": np.array([1.1, 2.2])})
        assert path.exists() and path.suffix == ".zip"
        load(path, remove_after=True)
        assert not list(settings.CACHE_DIR.glob("*"))

    def test_all_components_serializable(self):
        """Test that all circuit components are serializable."""
        circ = Circuit(
            [
                Coherent(0, x=1.0),
                Dgate(0, 0.1),
                BSgate((1, 2), theta=0.1, theta_trainable=True, theta_bounds=(-0.5, 0.5)),
                Dgate(0, x=1.1, y=2.2),
                Identity((1, 2)),
                Rgate(1, theta=0.1),
                S2gate((0, 1), 1, 1),
                Sgate(0, 0.1, 0.2, r_trainable=True),
                FockDamping(0, damping=0.1),
                BtoQ(0, np.pi / 2),
                Amplifier(0, gain=4),
                Attenuator(1, transmissivity=0.1),
                BtoChar(0, s=1),
                TraceOut((0, 1)),
                Thermal(0, nbar=3),
                Coherent(0, x=0.3, y=0.2, y_trainable=True, y_bounds=(-0.5, 0.5)).dual,
                DisplacedSqueezed(0, 1, 2, 3, 4, x_bounds=(-1.5, 1.5), x_trainable=True),
                Number(1, n=20),
                QuadratureEigenstate(2, x=1, phi=0, phi_trainable=True, phi_bounds=(-1, 1)).dual,
                SqueezedVacuum(3, r=0.4, phi=0.2),
                TwoModeSqueezedVacuum((0, 1), r=0.3, phi=0.2).dual,
                Vacuum(4).dual,
            ],
        )
        path = circ.serialize()
        assert list(path.parent.glob("*")) == [path]
        assert path.suffix == ".zip"

        loaded = load(path)
        assert loaded == circ
        assert all(type(a) is type(b) for a, b in zip(circ.components, loaded.components))
        assert list(path.parent.glob("*")) == [path]
