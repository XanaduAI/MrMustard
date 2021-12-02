Development guide
=================

Dependencies
------------

Mr. Mustard requires the following libraries be installed:

* `Python <http://python.org/>`_ >= 3.6

as well as the following Python packages:

* `NumPy <http://numpy.org/>`_
* `SciPy <http://scipy.org/>`_
* `The Walrus <https://the-walrus.readthedocs.io>`_ >= 0.17.0
* `Tensorflow <https://www.tensorflow.org/>`_ >= 2.4.0
* `Matplotlib <https://matplotlib.org/>`_
* `Rich <https://pypi.org/project/rich/>`_
* `tqdm <https://tqdm.github.io/>`_


If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version
of Python packaged for scientific computation.

Installation
------------

For development purposes, it is recommended to install the Mr. Mustard source code
using development mode:

.. code-block:: bash

    git clone https://github.com/XanaduAI/MrMustard
    cd MrMustard
    pip install -e .

The ``-e`` flag ensures that edits to the source code will be reflected when
importing StrawberryFields in Python.


PyTorch support
------------------

To use Mr. Mustard with PyTorch using CPU, install it as follows:

.. code-block:: console

    pip install torch

Or, to install PyTorch with GPU and CUDA 10.2 support:

.. code-block:: console

    pip install torch==1.10.0+cu102

for CUDA 11.3 use:

.. code-block:: console

    pip install torch==1.10.0+cu113

Refer to `PyTorch <https://pytorch.org/get-started/locally/>`_ project webpage for more details.

Development environment
-----------------------

Mr. Mustard uses a ``pytest`` suite for testing and ``black`` for formatting. These
dependencies can be installed via ``pip``:

.. code-block:: bash

    pip install -r requirements-dev.txt

Software tests
--------------

The Mr. Mustard test suite includes `pytest <https://docs.pytest.org/en/latest/>`_,
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ for coverage reports and
`hypothesis <https://hypothesis.readthedocs.io/en/latest/>`_ for property-based testing.

To ensure that Mr. Mustard is working correctly after installation, the test suite
can be run by navigating to the source code folder and running

.. code-block:: bash

    make test

Individual test modules are run by invoking pytest directly from the command line:

.. code-block:: bash

    pytest tests/test_fidelity.py

.. note:: **Run options for Mr. Mustard tests**

    When running tests, it can be useful to examine a single failing test.
    The following command stops at the first failing test:

    .. code-block:: console

        pytest -x

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

    pytest tests/test_fidelity.py --cov=mrmustard/location/to/module --cov-report=term-missing

Here the coverage report will be created relative to the module specified by
the path passed to the ``--cov=`` option.

The previously mentioned ``pytest`` options can be combined with the coverage
options. As an example, the ``-k`` option allows you to pass a boolean string
using file names, test class/test function names, and marks. Using ``-k`` in
the following command we can get the report of a specific file while also
filtering out certain tests:

.. code-block:: console

    pytest tests/test_fidelity.py --cov --cov-report=term-missing -k 'not test_fidelity_coherent_state'

Passing the ``--cov`` option without any modules specified will generate a
coverage report for all modules of Mr. Mustard.

Format and code style
---------------------

Contributions are checked for format alignment in the pipeline. With ``black``
installed, changes can be formatted locally using:

.. code-block:: bash

    make format

Contributors without ``make`` installed can run ``black`` directly using:

.. code-block:: bash

    black -l 100 mrmustard

Contributions are checked for format alignment in the pipeline. Changes can be
formatted and linted locally using:

.. code-block:: bash

    make lint

To run both linting and formatting use

.. code-block:: bash

    make format lint

Documentation
-------------

Additional packages are required to build the documentation, as specified in
``doc/requirements.txt``. These packages can be installed using:

.. code-block:: bash

    pip install -r doc/requirements.txt

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

  Mr. Mustard's source code conforms to `PEP8 standards <https://www.python.org/dev/peps/pep-0008/>`_.
  Before submitting the PR, you can autoformat your code changes using the
  `Black <https://github.com/psf/black>`_ Python autoformatter, with max-line length set to 120:

  .. code-block:: bash

      black -l 100 mrmustard/path/to/modified/file.py

  We check all of our code against `Pylint <https://www.pylint.org/>`_ for errors.
  To lint modified files, simply ``pip install pylint``, and then from the source code
  directory, run

  .. code-block:: bash

      pylint mrmustard/path/to/modified/file.py


When ready, submit your fork as a `pull request <https://help.github.com/articles/about-pull-requests>`_
to the Mr. Mustard repository, filling out the pull request template. This template is added
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
    of Mr. Mustard covered by the test suite, to ensure that all new code additions
    are adequately tested.

  - Finally, the **code quality** is calculated by
    `Codefactor <https://app.codacy.com/app/XanaduAI/mrmustard/dashboard>`_,
    to ensure all new code additions adhere to our code quality standards.

Based on these reports, we may ask you to make small changes to your branch before
merging the pull request into the master branch. Alternatively, you can also
`grant us permission to make changes to your pull request branch
<https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/>`_.
