# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys

# ~~~~~~~
# Helpers
# ~~~~~~~


def lazy_import(filename):
    try:
        return sys.modules[filename]
    except KeyError:
        spec = importlib.util.find_spec(filename)
        module = importlib.util.module_from_spec(spec)
        loader = importlib.util.LazyLoader(spec.loader)
        return module, loader


filename_np = "mrmustard.backend.backend_numpy"
module_np, loader_np = lazy_import(filename_np)
filename_tf = "mrmustard.backend.backend_tensorflow"
module_tf, loader_tf = lazy_import(filename_tf)


class Backend:
    _backend = None

    @property
    def backend(cls):
        return cls._backend

    def change_backend(cls, new_backend):
        if new_backend == "numpy":
            loader_np.exec_module(module_np)
            cls._backend = module_np.BackendNumpy()
        elif new_backend == "tensorflow":
            loader_tf.exec_module(module_tf)
            cls._backend = module_tf.BackendTensorflow()
        else:
            msg = f"Backend {new_backend} not supported"
            raise ValueError(msg)

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Backend, cls).__new__(cls)
        return cls.instance

    ### Functions
    def _apply(self, fn):
        return getattr(self.backend, fn)
        try:
            return getattr(self.backend, fn)
        except:
            raise NotImplemented

    def hello(self):
        r"""A function to say hello."""
        self._apply("hello")()

    def sum(self, x, y):
        r"""A function to sum two numbers."""
        return self._apply("sum")(x, y)
