# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from rich.pretty import install

install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"


class Settings:
    def __init__(self):
        self._backend = "tensorflow"
        self.HBAR = 2.0
        self.TMSV_DEFAULT_R = 3.0

    @property
    def backend(self):
        return self._backend

    # property setter for backend
    @backend.setter
    def backend(self, backend_name: str):
        if backend_name not in ["tensorflow", "pytorch"]:
            raise ValueError("Backend must be either 'tensorflow' or 'pytorch'")
        self._backend = backend_name
        self.__activate_backend()

    def __activate_backend(self):
        "Activates the math backend in the modules where it is used"
        from mrmustard.physics import fock, gaussian
        from mrmustard.utils import training, xptensor

        fock._set_backend(self.backend)
        gaussian._set_backend(self.backend)
        training._set_backend(self.backend)
        xptensor._set_backend(self.backend)


settings = Settings()
settings.backend = "tensorflow"
