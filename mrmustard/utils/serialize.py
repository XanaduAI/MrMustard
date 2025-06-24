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

"""A serialization library for MrMustard."""

import json
from pathlib import Path
from pydoc import locate
from time import strftime
from uuid import uuid4
from zipfile import ZipFile

import numpy as np

from mrmustard import __version__, math, settings


def save(cls: type, filename=None, do_zip=True, arrays=None, **data) -> Path:
    r"""
    Save a serialized set of data to file for later deserialization.

    This function can serialize any object that implements the MrMustard
    serialization interface, which is the following:

    .. code-block:: python

        @classmethod
        def deserialize(cls, data: dict) -> cls

    Though not required to satisfy the interface, it is conventional for a
    class to implement an instance method called ``serialize`` that takes no
    arguments, calls this function with its data, and returns the Path result.

    Args:
        cls (type): The object type being serialized

    Keyword Args:
        filename (str): A custom filename to save the data to
        do_zip (bool): If arrays are provided, zip results
        arrays (Dict[str, NDArray]): A dict of array-like objects to be serialized
        data (Dict[Any, Any]): A JSON-serializable dict of arbitrary data to save

    Returns:
        Path: the path to the saved object, to be retrieved later with ``load``
    """
    if not filename:
        filename = f"{cls.__qualname__}_{strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.json"
    elif filename[-5:] != ".json":
        filename = f"{filename}.json"

    file = settings.CACHE_DIR / filename
    data["class"] = f"{cls.__module__}.{cls.__qualname__}"
    data["version"] = __version__

    if arrays:
        if overlap := set(arrays).intersection(set(data)):
            raise ValueError(f"Arrays cannot have the same name as generic data: {overlap}")
        np.savez(file.with_suffix(".npz"), **{k: math.asnumpy(v) for k, v in arrays.items()})
        data["arrays"] = True
        data["backend"] = math.backend_name
    else:
        data["arrays"] = False

    with file.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    if arrays and do_zip:
        npz = file.with_suffix(".npz")
        with ZipFile(file.with_suffix(".zip"), "w") as zipf:
            zipf.write(file, file.name)
            zipf.write(npz, npz.name)

        file.unlink()
        npz.unlink()
        return Path(zipf.filename)

    return file


def load(file: Path, remove_after=False):
    r"""
    The deserializer entrypoint for objects saved with the ``save`` method.

    Args:
        file (Path): The json file to load from
        remove_after (Optional[bool]): Once load is complete, delete the saved
            file. Default is False
    """
    was_zipped = False
    file = Path(file)
    if file.suffix == ".zip":
        was_zipped = True
        with ZipFile(file) as zipf:
            zipf.extractall(file.parent)
        file = file.with_suffix(".json")

    with file.open("r", encoding="utf-8") as f:
        data: dict = json.load(f)

    cls = locate(data.pop("class"))
    _ = data.pop("version")

    if data.pop("arrays"):
        npz_file = file.with_suffix(".npz")
        if (backend := data.pop("backend")) != math.backend_name:
            if was_zipped:
                npz_file.unlink()
                file.unlink()
            raise TypeError(
                f"Data serialized with {backend} backend, cannot deserialize to the currently active {math.backend_name} backend",
            )
        data.update(**{k: math.astensor(v) for k, v in np.load(npz_file).items()})
        if remove_after or was_zipped:
            npz_file.unlink()

    if remove_after or was_zipped:
        file.unlink()
        if remove_after and was_zipped:
            file.with_suffix(".zip").unlink()

    return cls.deserialize(data)
