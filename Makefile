.PHONY: setup lint test smoke experiments report clean format

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

setup: .venv/bin/python
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

.venv/bin/python:
	python3.11 -m venv .venv

lint:
	.venv/bin/ruff check kernelagent tests experiments scripts
	.venv/bin/mypy kernelagent

test:
	.venv/bin/pytest -q -m "not gpu and not llm"

smoke:
	.venv/bin/python scripts/smoke.py

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
