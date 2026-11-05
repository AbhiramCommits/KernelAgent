"""Parsers that turn compiler/profiler output into structured diagnostics.

These are deliberately pure functions over files or strings so they can be
tested against committed fixtures without a GPU or a compiler.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

_LINE_RE = re.compile(r"Line:\s*'?(\d+)")
_WIDTH_RE = re.compile(r"vectorization width:\s*(\d+)")
_INTERLEAVE_RE = re.compile(r"interleaved count:\s*(\d+)")


_BLOCK_RE = re.compile(r"(?m)^--- !(\w+)\s*$")
_PASS_RE = re.compile(r"(?m)^Pass:\s+(\S+)")
_NAME_RE = re.compile(r"(?m)^Name:\s+(\S+)")
_LOC_RE = re.compile(r"File:\s*'?([^'\n,}]*)'?.*?Line:\s*'?(\d+)", re.S)
_ARG_RE = re.compile(r"(?m)^\s*-\s+([A-Za-z0-9_]+):\s*(.*)$")


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    text = text.replace("\\n", " ").replace('\\"', '"')
    return " ".join(text.split())


def _parse_opt_blocks(text: str) -> list[dict[str, Any]]:
    """Fast line parser for clang's optimization-record YAML.

    A real record for one kernel can exceed 3 MB; PyYAML spends seconds on it,
    while this parser only needs the fields the diagnostics use.
    """
    parts = _BLOCK_RE.split(text)
    records: list[dict[str, Any]] = []
    for index in range(1, len(parts) - 1, 2):
        tag = parts[index]
        body = parts[index + 1]
        pass_match = _PASS_RE.search(body)
        name_match = _NAME_RE.search(body)
        if not pass_match:
            continue
        record: dict[str, Any] = {
            "__record__": tag,
            "Pass": pass_match.group(1),
            "Name": name_match.group(1) if name_match else "",
            "Args": [],
        }
        loc_match = _LOC_RE.search(body)
        if loc_match:
            record["DebugLoc"] = {"File": loc_match.group(1), "Line": int(loc_match.group(2))}
        args_start = body.find("\nArgs:")
        if args_start >= 0:
            section = body[args_start:]
            for arg_match in _ARG_RE.finditer(section):
                key = arg_match.group(1)
                value = _unquote(arg_match.group(2))
                record["Args"].append({key: value})
        records.append(record)
    return records


def _arg_number(record: dict[str, Any], key: str) -> int | None:
    args = record.get("Args")
    if not isinstance(args, list):
        return None
    for arg in args:
        if isinstance(arg, dict) and key in arg:
            try:
                return int(float(str(arg[key])))
            except (TypeError, ValueError):
                return None
    return None


def _find_line(record: dict[str, Any]) -> int | None:
    loc = record.get("DebugLoc")
    if isinstance(loc, dict) and loc.get("Line") is not None:
        return int(loc["Line"])
    match = _LINE_RE.search(str(record))
    return int(match.group(1)) if match else None


def _loc_file(record: dict[str, Any]) -> str:
    loc = record.get("DebugLoc")
    if isinstance(loc, dict):
        return str(loc.get("File", ""))
    return ""


def _args_text(args: Any) -> str:
    if not isinstance(args, list):
        return ""
    pieces: list[str] = []
    for arg in args:
        if not isinstance(arg, dict):
            continue
        for key in ("String", "Callee", "Reason", "Pass"):
            value = arg.get(key)
            if value is not None:
                pieces.append(str(value))
    return " ".join(pieces).strip()


def parse_opt_record(
    path: str | Path, interest_files: set[str] | None = None
) -> dict[str, Any]:
    """Parse a clang ``-fsave-optimization-record`` YAML file.

    Returns counts of vectorized loops (with widths), missed vectorizations with
    their real compiler reasons, unroll decisions and inlining notes. When
    ``interest_files`` is given only records from those source basenames are
    counted; this filters out template/STL instantiations that would otherwise
    dominate the diagnostics.
    """
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
    records = _parse_opt_blocks(text) if text.strip() else []

    def interesting(record: dict[str, Any]) -> bool:
        if not interest_files:
            return True
        return _loc_file(record).rsplit("/", 1)[-1] in interest_files

    vectorized = 0
    widths: list[int] = []
    missed: list[dict[str, Any]] = []
    analyses: dict[tuple[str, int | None], list[str]] = {}
    unroll = {"fully": 0, "partially": 0, "not": 0}
    inlined = 0
    not_inlined = 0
    other_remarks = 0

    for record in records:
        if not interesting(record):
            continue
        pass_name = str(record.get("Pass", ""))
        name = str(record.get("Name", ""))
        args_text = _args_text(record.get("Args"))
        tag = str(record.get("__record__", ""))
        line = _find_line(record)
        file_name = _loc_file(record)

        if pass_name == "loop-vectorize":
            if name == "Vectorized":
                vectorized += 1
                factor = _arg_number(record, "VectorizationFactor")
                if factor is None:
                    width_match = _WIDTH_RE.search(args_text)
                    factor = int(width_match.group(1)) if width_match else None
                if factor:
                    widths.append(factor)
            elif tag == "Missed" or name.startswith("Missed"):
                reason = args_text or name or "vectorization missed"
                missed.append(
                    {
                        "reason": reason[:300],
                        "line": line,
                        "file": file_name.rsplit("/", 1)[-1],
                        "name": name,
                    }
                )
            elif tag == "Analysis":
                reason = args_text
                if reason.startswith("loop not vectorized:"):
                    reason = reason.split(":", 1)[1].strip()
                analyses.setdefault((file_name, line), []).append(reason)
            else:
                other_remarks += 1
        elif pass_name == "loop-unroll":
            lowered = name.lower()
            if "not" in lowered:
                unroll["not"] += 1
            elif "partial" in lowered:
                unroll["partially"] += 1
            elif "full" in lowered or "unroll" in lowered:
                unroll["fully"] += 1
        elif pass_name == "inline":
            if name == "Inlined":
                inlined += 1
            elif name in ("NoDefinition", "WillNotBeInlined"):
                not_inlined += 1

    # Attach the specific analysis reason(s) recorded for the same source line.
    for item in missed:
        reasons: list[str] = []
        for (file_name, line), values in analyses.items():
            if line == item["line"] and file_name.rsplit("/", 1)[-1] == item.get("file", ""):
                reasons.extend(values)
        if reasons:
            unique = list(dict.fromkeys(r for r in reasons if r))
            if unique:
                item["reason"] = "; ".join(unique)[:300]

    return {
        "source": "opt-record",
        "records": len(records),
        "vectorized_loops": vectorized,
        "vector_widths": widths,
        "missed_count": len(missed),
        "missed_vectorizations": missed[:24],
        "unroll": unroll,
        "inlined": inlined,
        "not_inlined": not_inlined,
        "other_remarks": other_remarks,
    }


_PTXAS_PATTERNS = {
    "registers": re.compile(r"Used\s+(\d+)\s+registers"),
    "shared_bytes": re.compile(r"(\d+)\s+bytes\s+(?:smem|shared)"),
    "spill_stores_bytes": re.compile(r"(\d+)\s+bytes\s+spill\s+stores"),
    "spill_loads_bytes": re.compile(r"(\d+)\s+bytes\s+spill\s+loads"),
    "stack_frame_bytes": re.compile(r"(\d+)\s+bytes\s+stack\s+frame"),
    "cmem_bytes": re.compile(r"(\d+)\s+bytes\s+cmem"),
}


def parse_ptxas_verbose(stderr: str) -> dict[str, Any]:
    """Parse ``nvcc -Xptxas -v`` / ``ptxas -v`` output."""
    counters: dict[str, int] = {key: 0 for key in _PTXAS_PATTERNS}
    for key, pattern in _PTXAS_PATTERNS.items():
        matches = pattern.findall(stderr)
        if matches:
            counters[key] = int(matches[-1])
    found = any(pattern.search(stderr) for pattern in _PTXAS_PATTERNS.values())
    result: dict[str, Any] = dict(counters)
    result["source"] = "ptxas"
    result["found"] = found
    result["spills"] = counters["spill_stores_bytes"] + counters["spill_loads_bytes"]
    return result


def parse_triton_log(text: str) -> dict[str, Any]:
    """Parse a Triton/ptxas diagnostic log."""
    ptxas = parse_ptxas_verbose(text)
    meta: dict[str, Any] = {}
    for line in text.splitlines():
        for key in ("num_warps", "num_stages", "shared", "name"):
            match = re.search(rf"\b{key}\s*[=:]\s*([A-Za-z0-9_.\-]+)", line)
            if match and key not in meta:
                value = match.group(1)
                try:
                    meta[key] = int(value)
                except ValueError:
                    meta[key] = value
        match = re.search(r"ptxas info\s*:\s*Function properties for\s+(\S+)", line)
        if match:
            meta["function"] = match.group(1)
    ptxas.update(meta)
    ptxas["source"] = "triton"
    return ptxas


_NCU_NORMALIZE = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__time_duration.sum": "kernel_time_ns",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum": "global_load_bytes",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum": "global_store_bytes",
}


def parse_ncu_csv(path: str | Path) -> dict[str, Any]:
    """Parse ``ncu --csv --metrics ...`` output into normalised counters.

    Warp-stall metrics are collected under ``warp_stall_*`` keys.
    """
    target = Path(path)
    if not target.exists():
        return {"source": "ncu", "found": False, "counters": {}, "warp_stalls": {}}
    lines = [line for line in target.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in lines:
        if line.startswith("=="):
            continue
        if "Metric Name" in line and "Metric Value" in line:
            header = next(csv.reader([line]))
            continue
        if header is None:
            continue
        values = next(csv.reader([line]))
        if len(values) == len(header):
            rows.append(dict(zip(header, values, strict=False)))

    counters: dict[str, float] = {}
    warp_stalls: dict[str, float] = {}
    for row in rows:
        metric = row.get("Metric Name", "")
        raw_value = row.get("Metric Value", "")
        unit = row.get("Metric Unit", "")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not metric:
            continue
        counters[metric] = value
        if "warp_issue_stalled" in metric and (metric.endswith(".ratio") or metric.endswith(".pct")):
            reason_match = re.search(r"warp_issue_stalled_(.+?)_per_", metric)
            reason = reason_match.group(1) if reason_match else metric.split(".", 1)[0]
            warp_stalls[reason] = value
        normalized = _NCU_NORMALIZE.get(metric)
        if normalized:
            counters[normalized] = value
        if unit and unit not in ("", "n/a"):
            counters[f"{metric}__unit"] = 0.0

    return {
        "source": "ncu",
        "found": bool(rows),
        "counters": counters,
        "warp_stalls": warp_stalls,
        "rows": len(rows),
    }
