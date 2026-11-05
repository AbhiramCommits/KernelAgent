"""Append-only experiment ledger.

One JSON object per trial is appended to ``results/trials.jsonl``. Each record
carries a run id, the current git SHA and the environment fingerprint, so any
reported number can be traced to the code and machine that measured it.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from .env import probe_environment
from .types import Trial


def git_sha(short: bool = True) -> str:
    """Current git SHA, or ``"unknown"`` outside a repository."""
    cmd = ["git", "rev-parse"] + (["--short", "HEAD"] if short else ["HEAD"])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


class TrialLedger:
    """Writes and reads the JSONL trial ledger."""

    def __init__(self, path: Path | None = None, run_id: str | None = None) -> None:
        self.path = Path(path) if path else config.RESULTS_DIR / "trials.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        env = probe_environment()
        self.environment = {
            "os": env["os"],
            "cpu_model": env["cpu_model"],
            "cores": env["cores"],
            "compiler": env["compiler"],
            "compiler_version": env["compiler_version"],
            "has_nvcc": env["has_nvcc"],
            "has_ncu": env["has_ncu"],
            "has_triton": env["has_triton"],
            "has_torch": env["has_torch"],
            "llm_provider": env["llm_provider"],
        }

    def append(self, trial: Trial) -> dict[str, Any]:
        """Append one trial and return the record that was written."""
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "git_sha": git_sha(),
            "environment": self.environment,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        record.update(trial.model_dump(mode="json"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def append_many(self, trials: list[Trial]) -> None:
        """Append several trials."""
        for trial in trials:
            self.append(trial)

    @staticmethod
    def read(path: Path | None = None) -> list[dict[str, Any]]:
        """Read every record from a ledger file (empty list when absent)."""
        target = Path(path) if path else config.RESULTS_DIR / "trials.jsonl"
        if not target.exists():
            return []
        records: list[dict[str, Any]] = []
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
