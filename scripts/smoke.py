"""Fast CI smoke run: 2 tasks, budget 4, fixed seed, non-empty summary."""

from __future__ import annotations

import json
import sys

from kernelagent import config
from kernelagent.eval.suite import run_suite


def main() -> int:
    out = config.RESULTS_DIR / "smoke" / "summary.json"
    summary = run_suite(
        strategy="mutation",
        budget=4,
        seeds=[0],
        out=out,
        tasks=["sgemm_tiled", "reduce_sum"],
    )
    if not out.exists() or out.stat().st_size == 0:
        print("smoke failed: summary.json is missing or empty")
        return 1
    if summary["problems"] < 4 or summary["trials"] < 4:
        print(f"smoke failed: too little evidence: {summary['problems']} problems, {summary['trials']} trials")
        return 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    print(
        f"smoke ok: {payload['problems']} problems, {payload['trials']} trials, "
        f"geomean={payload['geomean_speedup']:.3f}x -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
