Development guide
=================

Dependencies
------------

Mr Mustard requires the following to be installed:

* `Python <http://python.org/>`_ >= 3.10, <3.13
* `uv <https://github.com/astral-sh/uv>`_ >= 0.7.0

Installation
------------

For development purposes, it is recommended to install Mr Mustard using ``uv``

.. code-block:: bash

    git clone https://github.com/XanaduAI/MrMustard
    cd MrMustard
    uv sync --all-groups

The ``--all-groups`` flag ensures that all developmental dependencies are included. Note
that ``uv`` will install Mr Mustard into a ``.venv`` virtual environment by default.
Alternatively, ``pip`` is also supported

.. code-block:: bash

    git clone https://github.com/XanaduAI/MrMustard
    cd MrMustard
    pip install --group dev -e .

The ``-e`` flag ensures that edits to the source code will be reflected when
importing Mr Mustard in Python and the ``--group dev`` ensures development dependencies
are also installed (additional flags include ``--group doc`` to build documentation locally
and ``--group interactive`` to support interactive Jupyter notebooks).

Software tests
--------------

The Mr Mustard test suite includes `pytest <https://docs.pytest.org/en/latest/>`_,
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ for coverage reports and
`hypothesis <https://hypothesis.readthedocs.io/en/latest/>`_ for property-based testing.

To ensure that Mr Mustard is working correctly after installation, the test suite
can be run by navigating to the source code folder and running

.. code-block:: bash

    make test

or equivalently

.. code-block:: bash

    uv run pytest

Individual test modules are run by invoking pytest directly from the command line:

.. code-block:: bash

    uv run pytest tests/test_lab/test_states/test_ket.py

The ``--backend`` flag allows specifying the backend used when running the tests (by default ``numpy``).
For example, to use the Jax backend, run the command ``pytest tests/test_lab/test_states/test_ket.py --backend=jax``.

.. note:: **Run options for Mr Mustard tests**

    When running tests, it can be useful to examine a single failing test.
    The following command stops at the first failing test:

    .. code-block:: console

        uv run pytest -x

    For further useful options (e.g. ``-k``, ``-s``, ``--tb=short``, etc.)
    refer to the ``pytest --help`` command line usage description or the
    ``pytest`` online documentation.


Test coverage
^^^^^^^^^^^^^

Test coverage can be checked by running

.. code-block:: bash

    make coverage

The output of the above command will show the coverage percentage of each
file, as well as the line numbers of any lines missing test coverage.

To obtain coverage, the ``pytest-cov`` plugin is needed.

The coverage of a specific file can also be checked by generating a report:

.. code-block:: console

    uv run pytest tests/test_lab/test_states/test_ket.py --cov=mrmustard/location/to/module --cov-report=term-missing

Here the coverage report will be created relative to the module specified by
the path passed to the ``--cov=`` option.

The previously mentioned ``pytest`` options can be combined with the coverage
options. As an example, the ``-k`` option allows you to pass a boolean string
using file names, test class/test function names, and marks. Using ``-k`` in
the following command we can get the report of a specific file while also
filtering out certain tests:

.. code-block:: console

    uv run pytest tests/test_lab/test_states/test_ket.py --cov --cov-report=term-missing -k 'not test_L2_norm'

Passing the ``--cov`` option without any modules specified will generate a
coverage report for all modules of Mr Mustard.

Format and code style
---------------------

Contributions are checked for format alignment and linting in the pipeline.
This process is typically automated via ``pre-commit`

.. code-block:: bash

    pre-commit install

Manually, we can make use of either ``make``

.. code-block:: bash

    make format lint

or by direct calls to ``black`` and ``pylint``

.. code-block:: bash

    black -l 100 mrmustard
    pylint mrmustard

Documentation
-------------

Additional packages are required to build the documentation, as specified in
``pyproject.toml`` under the group ``doc``. These packages can be installed using:

.. code-block:: bash

    uv sync --group doc

from within the top-level directory. To then build the HTML documentation, run

.. code-block:: bash

    make docs

The documentation can be found in the :file:`doc/_build/html/` directory.


Submitting a pull request
-------------------------

Before submitting a pull request, please make sure the following is done:

* **All new features must include a unit test.** If you've fixed a bug or added
  code that should be tested, add a test to the ``tests`` directory.

* **All new functions and code must be clearly commented and documented.**

  Have a look through the source code at some of the existing function docstrings---
  the easiest approach is to simply copy an existing docstring and modify it as appropriate.

  If you do make documentation changes, make sure that the docs build and render correctly by
  running ``make docs``.

* **Ensure that the test suite passes**, by running ``make test``.

* **Make sure the modified code in the pull request conforms to the PEP8 coding standard.**

  Mr Mustard's source code conforms to `PEP8 standards <https://www.python.org/dev/peps/pep-0008/>`_.
  Before submitting the PR, make sure your code is formatted either through the ``pre-commit`` hook or

  .. code-block:: bash

      make format lint

When ready, submit your fork as a `pull request <https://help.github.com/articles/about-pull-requests>`_
to the Mr Mustard repository, filling out the pull request template. This template is added
automatically to the comment box when you create a new issue.

* When describing the pull request, please include as much detail as possible
  regarding the changes made/new features added/performance improvements. If including any
  bug fixes, mention the issue numbers associated with the bugs.

* Once you have submitted the pull request, three things will automatically occur:

  - The **test suite** will automatically run on `GitHub Actions
    <https://github.com/XanaduAI/MrMustard/actions?query=workflow%3ATests>`_
    to ensure that all tests continue to pass.

  - Once the test suite is finished, a **code coverage report** will be generated on
    `Codecov <https://codecov.io/gh/XanaduAI/MrMustard>`_. This will calculate the percentage
    of Mr Mustard covered by the test suite, to ensure that all new code additions
    are adequately tested.

  - Finally, the **code quality** is calculated by
    `Codefactor <https://app.codacy.com/app/XanaduAI/mrmustard/dashboard>`_,
    to ensure all new code additions adhere to our code quality standards.

Based on these reports, we may ask you to make small changes to your branch before
merging the pull request into the master branch. Alternatively, you can also
`grant us permission to make changes to your pull request branch
<https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/>`_.
