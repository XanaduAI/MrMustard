# Copyright 2010 Pallets

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Unit tests for the logging mechanism used in Mr Mustard.

The implementation of these unit tests is based on the solution used for testing logging in
the Flask web application framework:
https://github.com/pallets/flask/blob/master/tests/test_logging.py
"""

import logging
import sys

import pytest

from mrmustard.training import optimizer
from mrmustard.utils.logger import create_logger, default_handler, logging_handler_defined

modules_contain_logging = [optimizer]


@pytest.fixture(autouse=True)
def reset_logging(pytestconfig):
    """Reset the logging specific configurations such as handlers or levels as
    well as manage pytest's LoggingPlugin."""
    root_handlers = logging.root.handlers[:]
    logging.root.handlers = []
    root_level = logging.root.level

    # Reset the logging specific configurations before test
    for module in modules_contain_logging:
        logger = logging.getLogger(module.__name__)
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    logging_plugin = pytestconfig.pluginmanager.unregister(name="logging-plugin")

    yield

    # Reset the logging specific configurations after test
    for module in modules_contain_logging:
        logger = logging.getLogger(module.__name__)
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    logging.root.handlers[:] = root_handlers
    logging.root.setLevel(root_level)

    if logging_plugin:
        pytestconfig.pluginmanager.register(logging_plugin, "logging-plugin")


@pytest.mark.parametrize("module", modules_contain_logging)
class TestLogger:
    """Tests for the functions that are used to create a logger"""

    def test_logging_handler_defined(self, module):
        """Tests that the logging_handler_defined function works correctly in
        the following cases:

        1. When a custom logger was just created and by default has no level
           handler
        2. Adding a handler to the root logger affects the custom logger
        3. When propagation is set to False, only the handlers of the custom
        logger are checked"""
        logger = logging.getLogger(module.__name__)
        assert not logging_handler_defined(logger)

        handler = logging.StreamHandler()
        logging.root.addHandler(handler)
        assert logging_handler_defined(logger)

        logger.propagate = False
        assert not logging_handler_defined(logger)

    def test_create_logger(self, module):
        """Tests that the create_logger function returns a logger with the
        default configuration set for an MM logger"""
        logger = create_logger(module.__name__)
        assert logger.level == logging.INFO
        assert logging_handler_defined(logger)
        assert logger.handlers[0] == default_handler


class TestLoggerIntegration:
    """Tests that the MM logger integrates well with user defined logging
    configurations."""

    def test_custom_configuration_without_mm_logger(self, tmpdir):
        """Tests that if there was no MM logger created, custom logging
        configurations work as expected and no configuration details were set
        incorrectly."""

        level = logging.DEBUG

        test_file = tmpdir.join("test_file")
        logging.basicConfig(filename=test_file, level=level)
        logging.debug("A log entry.")

        assert "A log entry." in test_file.read()

    @pytest.mark.parametrize("module", modules_contain_logging)
    def test_default_mm_logger(self, module):
        """Tests that stderr is set for the MM logger by default as stream if
        there were not other configurations made."""

        logger = create_logger(module.__name__)
        assert len(logger.handlers) == 1
        # Check if the stream of the first handler is sys.stderr
        assert logger.handlers[0].stream is sys.stderr

    @pytest.mark.parametrize("module", modules_contain_logging)
    def test_custom_logger_before_mm_logger_with_higher_level(self, module):
        """Tests that a custom logger created before an MM logger will define
        the level for logging as expected and the MM logger does not overwrite
        the user configuration.

        The logic of the test goes as follows:
        1. Manually setting the level for logging for DEBUG level
        2. Creating an MM logger with level WARNING, that is higher than DEBUG
        3. Checking that the MM logger did not affect the handlers defined or
           the effective level of the logger
        """
        custom_level = logging.DEBUG
        mm_level = logging.WARNING

        logger = logging.getLogger()
        logging.basicConfig(level=custom_level)

        _ = create_logger(module.__name__, level=mm_level)

        assert logging_handler_defined(logger)
        assert logger.getEffectiveLevel() == custom_level
