PYTHON3 := $(shell which python3 2>/dev/null)
TESTRUNNER := -m pytest tests -p no:warnings
COVERAGE := --cov=mrmustard --cov-report=html --cov-append

ifdef check
    CHECK := --check
    FIX :=
else
    CHECK :=
    FIX := --fix
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install Mr Mustard"
	@echo "  install-all        to install Mr Mustard with all extras and optional dependencies"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  docs               to build the documentation"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  ruff [check=1]     to run ruff linting and formatting; use with 'check=1' to avoid modifying files"
	@echo "  test               to run the test suite for entire codebase"	
	@echo "  test numpy         to run the test suite with numpy backend"
	@echo "  test jax           to run the test suite with jax backend"
	@echo "  coverage           to generate a coverage report for entire codebase"
	@echo "  clean-coverage     to delete the coverage report"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install Mr Mustard you need to have Python 3 installed"
endif
	uv sync


.PHONY: install-all
install-all:
ifndef PYTHON3
	@echo "To install Mr Mustard you need to have Python 3 installed"
endif
	uv sync --all-extras --group doc

.PHONY: dist
dist:
	uv build

.PHONY : clean
clean:
	rm -rf mrmustard/__pycache__
	rm -rf mrmustard/lab/__pycache__
	rm -rf mrmustard/lab/abstract/__pycache__
	rm -rf mrmustard/math/__pycache__
	rm -rf mrmustard/physics/__pycache__
	rm -rf mrmustard/utils/__pycache__
	rm -rf mrmustard/backends/gaussianbackend/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/test_lab/__pycache__
	rm -rf tests/test_physics/__pycache__
	rm -rf tests/test_utils/__pycache__
	rm -rf dist
	rm -rf build

.PHONY : docs
docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	make -C doc clean
	rm -rf doc/code/api

.PHONY : ruff
ruff:
	ruff check mrmustard/ tests/ $(FIX)
	ruff format mrmustard/ tests/ $(CHECK)

.PHONY: test
test:
	@echo "Testing Mr Mustard..."
	$(PYTHON3) $(TESTRUNNER)

# Backend-specific test targets
test-%:
	@echo "Testing Mr Mustard with $* backend..."
	$(PYTHON3) $(TESTRUNNER) --backend=$*

# Support for "make test <backend>" syntax
ifneq ($(filter numpy jax,$(word 2,$(MAKECMDGOALS))),)
  BACKEND := $(word 2,$(MAKECMDGOALS))
  $(BACKEND):
	@:
  test:
	@echo "Testing Mr Mustard with $(BACKEND) backend..."
	$(PYTHON3) $(TESTRUNNER) --backend=$(BACKEND)
endif

.PHONY : coverage
coverage:
	@echo "Generating coverage report for Mr Mustard..."
	$(PYTHON3) $(TESTRUNNER) $(COVERAGE)

.PHONY : clean-coverage
clean-coverage:
	rm -rf coverage_html_report
