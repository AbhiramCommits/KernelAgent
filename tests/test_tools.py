"""Tool-layer tests: subprocess safety, schema generation and caching."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

from kernelagent.tools.base import (
    DiskCache,
    content_hash,
    redact,
    run_subprocess,
    tool,
    tool_schemas,
    truncate,
    validate_tool_arguments,
)


def test_run_subprocess_success() -> None:
    result = run_subprocess([sys.executable, "-c", "print('hello')"], timeout_s=10, retries=0)
    assert result.ok
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert result.duration_s >= 0
    assert result.attempts == 1


def test_run_subprocess_timeout_kills_and_retries() -> None:
    result = run_subprocess(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=0.3,
        retries=1,
        backoff_s=0.01,
    )
    assert not result.ok
    assert result.timed_out
    assert result.attempts == 2
    assert result.duration_s < 4.0


def test_run_subprocess_missing_binary() -> None:
    result = run_subprocess(["definitely-not-a-real-binary-ka"], timeout_s=2, retries=0)
    assert not result.ok
    assert "FileNotFoundError" in result.error or "No such" in result.error


def test_output_is_truncated_and_redacted() -> None:
    home = str(Path.home())
    text = (home + "/secret ") * 10000
    assert len(truncate(text, limit=200)) < 400
    assert "truncated" in truncate(text, limit=200)
    assert home not in redact(f"key at {home}/x sk-ant-abcdefghijklmnop")


def test_content_hash_stable() -> None:
    assert content_hash(["a", "b"]) == content_hash(["a", "b"])
    assert content_hash(["a", "b"]) != content_hash(["b", "a"])


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    path = cache.entry("abc123") / "meta.json"
    cache.write_json(path, {"x": 1})
    assert cache.read_json(path) == {"x": 1}
    assert cache.read_json(tmp_path / "missing.json") is None


class _Args(BaseModel):
    task: str
    budget: int = 4


@tool("Run a toy search")
def _toy_tool(args: _Args) -> str:
    return args.task


def test_tool_schema_and_validation() -> None:
    schemas = tool_schemas([_toy_tool])
    assert schemas[0]["name"] == "_toy_tool"
    assert schemas[0]["input_schema"]["properties"]["args"]["properties"]["task"]["type"] == "string"
    validated = validate_tool_arguments(_toy_tool, {"task": "gemm", "budget": 2})
    assert validated["args"].task == "gemm"


def test_tool_validation_rejects_bad_arguments() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        validate_tool_arguments(_toy_tool, {"budget": 2})
