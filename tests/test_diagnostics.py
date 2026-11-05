"""Parser tests against committed real-tool fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelagent.tools.diagnostics import (
    parse_ncu_csv,
    parse_opt_record,
    parse_ptxas_verbose,
    parse_triton_log,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_opt_record_counts_vectorized_and_missed() -> None:
    parsed = parse_opt_record(FIXTURES / "optimization_record.yaml", interest_files={"fix.cpp"})
    assert parsed["source"] == "opt-record"
    assert parsed["vectorized_loops"] == 3
    assert parsed["vector_widths"] == [4, 4, 4]
    assert parsed["missed_count"] >= 1
    missed = parsed["missed_vectorizations"]
    assert any("dependent memory" in item["reason"] for item in missed)
    assert any(item["line"] == 6 for item in missed)
    unroll = parsed["unroll"]
    assert unroll["fully"] + unroll["partially"] + unroll["not"] > 0
    assert parsed["inlined"] >= 1


def test_opt_record_missing_file_is_empty(tmp_path: Path) -> None:
    parsed = parse_opt_record(tmp_path / "nope.yaml")
    assert parsed["records"] == 0
    assert parsed["vectorized_loops"] == 0
    assert parsed["missed_count"] == 0


def test_ptxas_parser() -> None:
    parsed = parse_ptxas_verbose((FIXTURES / "ptxas_verbose.txt").read_text())
    assert parsed["found"] is True
    assert parsed["registers"] == 255  # last kernel in the fixture
    assert parsed["shared_bytes"] == 49152
    assert parsed["spill_stores_bytes"] == 96
    assert parsed["spill_loads_bytes"] == 128
    assert parsed["spills"] == 224


def test_ptxas_parser_empty() -> None:
    parsed = parse_ptxas_verbose("nothing to see")
    assert parsed["found"] is False
    assert parsed["registers"] == 0


def test_ncu_csv_parser() -> None:
    parsed = parse_ncu_csv(FIXTURES / "ncu.csv")
    assert parsed["found"] is True
    assert parsed["rows"] == 6
    counters = parsed["counters"]
    assert counters["occupancy_pct"] == pytest.approx(62.5)
    assert counters["dram_throughput_pct"] == pytest.approx(41.2)
    assert counters["sm_throughput_pct"] == pytest.approx(78.9)
    assert counters["kernel_time_ns"] == pytest.approx(812345.0)
    assert parsed["warp_stalls"]["long_scoreboard"] == pytest.approx(24.5)
    assert parsed["warp_stalls"]["wait"] == pytest.approx(11.0)


def test_ncu_csv_missing(tmp_path: Path) -> None:
    parsed = parse_ncu_csv(tmp_path / "missing.csv")
    assert parsed["found"] is False


def test_triton_log_parser() -> None:
    parsed = parse_triton_log((FIXTURES / "triton.log").read_text())
    assert parsed["source"] == "triton"
    assert parsed["registers"] == 168
    assert parsed["shared_bytes"] == 98304
    assert parsed["num_warps"] == 8
    assert parsed["num_stages"] == 3
