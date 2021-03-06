### Before submitting

Please complete the following checklist when submitting a PR:

- [ ] All new features must include a unit test.
      If you've fixed a bug or added code that should be tested, add a test to the
      test directory!

- [ ] All new functions and code must be clearly commented and documented.
      If you do make documentation changes, make sure that the docs build and
      render correctly by running `make docs`.

- [ ] Ensure that the test suite passes, by running `make test`.

- [ ] Ensure that code and tests are properly formatted, by running `make format` or `black -l 100
      <filename>` on any relevant files. You will need to have the Black code format installed:
      `pip install black`.

- [ ] Add a new entry to the `.github/CHANGELOG.md` file, summarizing the
      change, and including a link back to the PR.

- [ ] The Mr Mustard source code conforms to
      [PEP8 standards](https://www.python.org/dev/peps/pep-0008/) except line length.
      We check all of our code against [Pylint](https://www.pylint.org/).
      To lint modified files, simply `pip install pylint`, and then
      run `pylint mrmustard/path/to/file.py`.

When all the above are checked, delete everything above the dashed
line and fill in the pull request template.

------------------------------------------------------------------------------------------------------------

**Context:**

**Description of the Change:**

**Benefits:**

**Possible Drawbacks:**

**Related GitHub Issues:**
