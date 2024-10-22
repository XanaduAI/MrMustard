PYTHON3 := $(shell which python3 2>/dev/null)
TESTRUNNER := -m pytest tests -p no:warnings
COVERAGE := --cov=mrmustard --cov-report=html --cov-append

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install Mr Mustard"
	@echo "  install-all        to install Mr Mustard with all extras and optional dependencies"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  format             to run black formatting"
	@echo "  test               to run the test suite for entire codebase"
	@echo "  coverage           to generate a coverage report for entire codebase"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install Mr Mustard you need to have Python 3 installed"
endif
	poetry install


.PHONY: install-all
install-all:
ifndef PYTHON3
	@echo "To install Mr Mustard you need to have Python 3 installed"
endif
	poetry install --all-extras --with dev,doc

.PHONY: dist
dist:
	poetry build

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
	black -l 100 mrmustard/ tests/

.PHONY : lint
lint:
	pylint mrmustard

test:
	@echo "Testing Mr Mustard..."
	$(PYTHON3) $(TESTRUNNER)

coverage:
	@echo "Generating coverage report for Mr Mustard..."
	$(PYTHON3) $(TESTRUNNER) $(COVERAGE)

clean-coverage:
	rm -rf coverage_html_report
