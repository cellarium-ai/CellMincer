.PHONY: install lint format FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall cellmincer

lint: FORCE
	ruff check .
	black --check .

docs: FORCE
	cd docs && make html

format: license FORCE
	ruff check --fix .
	black .

typecheck: FORCE
	mypy cellmincer tests

FORCE:
