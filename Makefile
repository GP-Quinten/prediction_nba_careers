.PHONY: style
style:
	@find . -type f -name "*.py" ! -path "./notebooks_scripts/*" ! -path "*/.ipynb_checkpoints/*" ! -path "./mlruns/*" | xargs pylint

.PHONY: black
black:
	poetry run black --check --diff .

.PHONY: test
test: test-unit

.PHONY: test-unit
test-unit:
	poetry run pytest --ignore=.cache --ignore=.venv tests/unit
