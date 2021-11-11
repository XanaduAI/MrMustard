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


from mrmustard import settings
import importlib

if importlib.util.find_spec("tensorflow"):
    from mrmustard.math.tensorflow import TFMath
if importlib.util.find_spec("torch"):
    from mrmustard.math.torch import TorchMath
class Math:
    r"""
    This class provides a unified interface for performing math operations.
    """
    def __getattribute__(self, name):
        if settings.backend == "tensorflow":
            return object.__getattribute__(TFMath(), name)
        elif settings.backend == "torch":
            return object.__getattribute__(TorchMath(), name)
            
    
