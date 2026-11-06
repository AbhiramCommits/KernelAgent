.PHONY: setup lint test smoke experiments report clean format

VENV := .venv
# Use the project virtualenv when `make setup` has created one; otherwise fall
# back to the active interpreter (CI installs into the runner's Python).
PY ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)

setup: $(VENV)/bin/python
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -e ".[dev]"

$(VENV)/bin/python:
	python3.11 -m venv $(VENV)

lint:
	$(PY) -m ruff check kernelagent tests experiments scripts
	$(PY) -m mypy kernelagent

test:
	$(PY) -m pytest -q -m "not gpu and not llm"

smoke:
	$(PY) scripts/smoke.py

experiments:
	$(PY) -m experiments.exp01_strategies
	$(PY) -m experiments.exp02_ablation
	$(PY) -m experiments.exp03_cost_model
	$(PY) -m experiments.exp04_reliability
	$(PY) -c "from experiments.common import assemble_summary; assemble_summary()"

report:
	$(PY) -m kernelagent.cli report

clean:
	rm -rf .kacache build dist .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
