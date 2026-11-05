"""The reliable subprocess boundary every tool shares.

Raw shell output is captured, redacted and truncated here; callers only ever
receive a :class:`ToolResult` or a typed pydantic model. Timeouts kill the whole
process group so a wedged benchmark cannot leak into the next trial.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

from pydantic import BaseModel

from ..config import MAX_TOOL_OUTPUT_CHARS

_SECRET_PATTERNS = (
    re.compile(r"(sk-ant-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(sk-[A-Za-z0-9_\-]{16,})"),
    re.compile(r"(ANTHROPIC_API_KEY\s*=\s*)\S+"),
    re.compile(r"(OPENAI_API_KEY\s*=\s*)\S+"),
)

_TRANSIENT_MARKERS = (
    "resource temporarily unavailable",
    "text file busy",
    "temporarily unavailable",
    "killed",
)


@dataclass
class ToolResult:
    """Normalised result of one subprocess invocation."""

    ok: bool
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    attempts: int = 0
    timed_out: bool = False
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "attempts": self.attempts,
            "timed_out": self.timed_out,
            "error": self.error,
        }


def redact(text: str) -> str:
    """Remove secrets and the user's home directory from tool output."""
    redacted = text
    home = str(Path.home())
    if home and home != "/":
        redacted = redacted.replace(home, "~")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(r"\1<redacted>", redacted)
    return redacted


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Keep the head and tail of long output with an explicit marker."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... [truncated {len(text) - limit} chars] ...\n{text[-tail:]}"


def _is_transient(result: ToolResult) -> bool:
    if result.timed_out:
        return True
    lowered = result.stderr.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS) and result.returncode != 0


def run_subprocess(
    cmd: list[str],
    timeout_s: float = 60.0,
    retries: int = 1,
    backoff_s: float = 0.25,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> ToolResult:
    """Run ``cmd`` with a hard timeout, process-group kill and bounded retries.

    ``retries`` counts *additional* attempts after the first. Only timeouts and
    explicitly transient failures are retried; a compiler error is never retried.
    """
    merged_env = os.environ.copy()
    merged_env.setdefault("LC_ALL", "C")
    if env:
        merged_env.update(env)

    attempts = 0
    last: ToolResult | None = None
    for attempt in range(retries + 1):
        attempts = attempt + 1
        started = time.perf_counter()
        proc: subprocess.Popen[str] | None = None
        timed_out = False
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                stdout, stderr = proc.communicate()
            duration = time.perf_counter() - started
            returncode = proc.returncode if proc.returncode is not None else -1
            result = ToolResult(
                ok=(returncode == 0 and not timed_out),
                returncode=returncode,
                stdout=truncate(redact(stdout or "")),
                stderr=truncate(redact(stderr or "")),
                duration_s=duration,
                attempts=attempts,
                timed_out=timed_out,
                error="timeout" if timed_out else ("" if returncode == 0 else f"exit {returncode}"),
            )
        except OSError as exc:
            duration = time.perf_counter() - started
            result = ToolResult(
                ok=False,
                returncode=-1,
                duration_s=duration,
                attempts=attempts,
                error=f"{type(exc).__name__}: {exc}",
            )
        last = result
        if result.ok or not _is_transient(result) or attempt >= retries:
            return result
        time.sleep(backoff_s * (2**attempt))
    assert last is not None
    return last


def content_hash(parts: list[str | bytes]) -> str:
    """Stable sha256 over file contents, flags and compiler identity."""
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            digest.update(part)
        else:
            digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class DiskCache:
    """A tiny content-addressed cache rooted at ``.kacache``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def entry(self, key: str) -> Path:
        """Directory for one cache key."""
        path = self.root / key[:2] / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def binary(self, key: str, name: str = "kernel") -> Path:
        """Path of a cached binary."""
        return self.entry(key) / name

    def read_json(self, path: Path) -> dict[str, Any] | None:
        """Read a cache metadata file if it exists."""
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Atomically write cache metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        tmp.replace(path)


def find_executable(*names: str) -> str | None:
    """First executable found on PATH."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


# ---------------------------------------------------------------------------
# LLM tool-schema support
# ---------------------------------------------------------------------------


def tool(description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach a JSON-schema tool contract to a function.

    The function must take a single pydantic model argument (or only annotated
    scalar arguments). The generated schema is what the LLM proposer sees.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, parameter in signature.parameters.items():
            annotation = hints.get(name, parameter.annotation)
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                schema = annotation.model_json_schema()
                properties[name] = schema
                required.append(name)
            elif annotation in (int, float, str, bool):
                properties[name] = {"type": {int: "integer", float: "number", str: "string", bool: "boolean"}[annotation]}
                if parameter.default is inspect.Parameter.empty:
                    required.append(name)
        schema = {
            "name": fn.__name__,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
        fn.tool_schema = schema  # type: ignore[attr-defined]
        return fn

    return decorator


def tool_schemas(functions: list[Callable[..., Any]]) -> list[dict[str, Any]]:
    """Collect the schemas of ``@tool``-decorated functions."""
    schemas = []
    for fn in functions:
        schema = getattr(fn, "tool_schema", None)
        if schema is None:
            raise ValueError(f"{fn.__name__} is not a @tool")
        schemas.append(schema)
    return schemas


def validate_tool_arguments(fn: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate LLM-supplied arguments against the function's pydantic model."""
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    validated: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        annotation = hints.get(name, parameter.annotation)
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            model = annotation.model_validate(arguments)
            validated[name] = model
    if not validated:
        raise TypeError(f"{fn.__name__} has no pydantic argument model")
    return validated
