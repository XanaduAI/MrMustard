# Developer Guideline

This is meant as a rough guideline for both new and experienced developers to adhere to the code quality standards that we have within Mr.Mustard.

### Documentation

Docstrings should be formatted as follows:

class MyClass:
    r"""
    Class description.

    .. code-block::
        # Example code

    Args:
        param1: ...
        param2: ...

    Returns:
        ...

    Raises:
        ValueError: If ...
    """

### Typehints

All methods must be typehinted as follows:

def my_method(param1: typehint, ...) -> typehint

### Tests

Ideally tests should be written with 100% coverage in mind. New functionality should be
tested with both unit tests and integration tests. Bug fixes should always include a test for said
bug.

### Pull Requests

A pull request (PR) should fulfill the following criteria before being approved:

- All lines of code should be covered by tests.
- All classes and methods must have proper documentation and type hints (see 'Documentation' and 'Typehints' above).
- All Github workflow checks must be passing.
- All conversations must be resolved.
