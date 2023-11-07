.DEFAULT: help

help:
	@echo "install"
	@echo "        Install curvlinops and dependencies"
	@echo "uninstall"
	@echo "        Unstall curvlinops"
	@echo "lint"
	@echo "        Run all linting actions"
	@echo "docs"
	@echo "        Build the documentation"
	@echo "install-dev"
	@echo "        Install curvlinops and development tools"
	@echo "install-docs"
	@echo "        Install curvlinops and documentation tools"
	@echo "install-test"
	@echo "        Install curvlinops and testing tools"
	@echo "test-light"
	@echo "        Run pytest on the light part of project and report coverage"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "install-lint"
	@echo "        Install curvlinops and the linter tools"
	@echo "isort"
	@echo "        Run isort (sort imports) on the project"
	@echo "isort-check"
	@echo "        Check if isort (sort imports) would change files"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "conda-env"
	@echo "        Create conda environment 'curvlinops' with dev setup"
	@echo "darglint-check"
	@echo "        Run darglint (docstring check) on the project"
	@echo "pydocstyle-check"
	@echo "        Run pydocstyle (docstring check) on the project"

.PHONY: install

install:
	@pip install -e .

.PHONY: uninstall

uninstall:
	@pip uninstall curvlinops-for-pytorch

.PHONY: docs

docs:
	@cd docs/rtd && make html
	@echo "\nOpen docs/rtd/index.html to see the result."

.PHONY: install-dev

install-dev: install-lint install-test install-docs

.PHONY: install-docs

install-docs:
	@pip install -e .[docs]

.PHONY: install-test

install-test:
	@pip install -e .[test]

.PHONY: test test-light

test:
	@pytest -vx --run-optional-tests=montecarlo --cov=curvlinops --doctest-modules curvlinops test

test-light:
	@pytest -vx --cov=curvlinops --doctest-modules curvlinops test

.PHONY: install-lint

install-lint:
	@pip install -e .[lint]

.PHONY: isort isort-check

isort:
	@isort .

isort-check:
	@isort . --check --diff

.PHONY: black black-check

black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

.PHONY: flake8

flake8:
	@flake8 .

.PHONY: darglint-check

darglint-check:
	@darglint --verbosity 2 curvlinops

.PHONY: pydocstyle-check

pydocstyle-check:
	@pydocstyle --count .

.PHONY: conda-env

conda-env:
	@conda env create --file .conda_env.yml

.PHONY: lint

lint:
	make black-check
	make isort-check
	make flake8
	make darglint-check
	make pydocstyle-check
