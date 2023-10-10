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

# from .backend_numpy import BackendNumpy


class Backend:
    _backend = None

    @property
    def backend(cls):
        return cls._backend

    def change_backend(cls, new_backend):
        if new_backend == "numpy":
            from .backend_numpy import BackendNumpy

            cls._backend = BackendNumpy()
        elif new_backend == "tensorflow":
            from .backend_tensorflow import BackendTensorflow

            cls._backend = BackendTensorflow()
        else:
            msg = f"Backend {new_backend} not supported"
            raise ValueError(msg)

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Backend, cls).__new__(cls)
        return cls.instance

    ### Functions
    def _apply(self, fn, args=()):
        try:
            return getattr(self.backend, fn)(*args)
        except:
            raise NotImplemented

    def hello(self):
        r"""A function to say hello."""
        return self._apply("hello")

    def sum(self, x, y):
        r"""A function to sum two numbers."""
        return self._apply("sum", (x, y))
