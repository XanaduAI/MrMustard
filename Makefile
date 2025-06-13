PYTHON3 := $(shell which python3 2>/dev/null)
TESTRUNNER := -m pytest tests -p no:warnings
COVERAGE := --cov=mrmustard --cov-report=html --cov-append

ifdef check
    CHECK := --check --diff
    ICHECK := --check
else
    CHECK :=
    ICHECK :=
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install Mr Mustard"
	@echo "  install-all        to install Mr Mustard with all extras and optional dependencies"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  format [check=1]   to run isort and black formatting; use with 'check=1' to check instead of modify"
	@echo "  test               to run the test suite for entire codebase"
	@echo "  test numpy         to run the test suite with numpy backend"
	@echo "  test tensorflow    to run the test suite with tensorflow backend"
	@echo "  test jax           to run the test suite with jax backend"
	@echo "  coverage           to generate a coverage report for entire codebase"

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

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	make -C doc clean
	rm -rf doc/code/api

.PHONY : format
format:
	isort --py 311 --profile black -l 100 -p mrmustard/ tests/ $(ICHECK)
	black -t py310 -t py311 -l 100 mrmustard/ tests/ $(CHECK)

.PHONY : lint
lint:
	pylint mrmustard

.PHONY: test
test:
	@echo "Testing Mr Mustard..."
	$(PYTHON3) $(TESTRUNNER)

# Backend-specific test targets
test-%:
	@echo "Testing Mr Mustard with $* backend..."
	$(PYTHON3) $(TESTRUNNER) --backend=$*

# Support for "make test <backend>" syntax
ifneq ($(filter numpy tensorflow jax,$(word 2,$(MAKECMDGOALS))),)
  BACKEND := $(word 2,$(MAKECMDGOALS))
  $(BACKEND):
	@:
  test:
	@echo "Testing Mr Mustard with $(BACKEND) backend..."
	$(PYTHON3) $(TESTRUNNER) --backend=$(BACKEND)
endif

coverage:
	@echo "Generating coverage report for Mr Mustard..."
	$(PYTHON3) $(TESTRUNNER) $(COVERAGE)

clean-coverage:
	rm -rf coverage_html_report
