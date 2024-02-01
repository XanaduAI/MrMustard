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

"""
This module contains a class to filter undesired warnings.
"""

import logging


class WarningFilters(logging.Filter):
    r"""
    A custom logging filter to selectively allow log records based on specific warnings.

    Args:
        warnings: A list of warning messages that must be filtered.
    """

    def __init__(self, warnings: list[str]):
        super().__init__()
        self.warnings = warnings

    def filter(self, record) -> bool:
        r"""
        Determine if the log record should be allowed based on specific warnings.

        Args:
            record: The ``LogRecord`` to be filtered.

        Returns:
            ``True`` if the log record should be allowed, ``False`` otherwise.
        """
        return any(w in record.getMessage() for w in self.warnings)


# ComplexWarning filter for tensorflow.
msg = "WARNING:tensorflow:You are casting an input of type complex128 to an incompatible dtype float64."
msg += "  This will discard the imaginary part and may not be what you intended."
complex_warninig_filter = WarningFilters([msg])


def add_complex_warning_filter():
    r"""
    Adds the filter for tensorflow's ComplexWarning, or does nothing if the filter is already in place.
    """
    logger = logging.getLogger("tensorflow")
    logger.addFilter(complex_warninig_filter)


def remove_complex_warning_filter():
    r"""
    Removes the filter for tensorflow's ComplexWarning, or does nothing if no such filter is present.
    """
    logger = logging.getLogger("tensorflow")
    logger.removeFilter(complex_warninig_filter)
