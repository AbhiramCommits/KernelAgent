"""CLI smoke tests: doctor and tasks must exit zero."""

from __future__ import annotations

from typer.testing import CliRunner

from kernelagent.cli import app

runner = CliRunner()


def test_doctor_exits_zero() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "capability" in result.output.lower()
    assert "Compiler" in result.output


def test_tasks_exits_zero() -> None:
    result = runner.invoke(app, ["tasks"])
    assert result.exit_code == 0, result.output


def test_version_exits_zero() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
