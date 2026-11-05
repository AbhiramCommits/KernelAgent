"""Offline LLM proposer tests using a scripted transport and transcript replay."""

from __future__ import annotations

import json
from pathlib import Path

from kernelagent.agent.llm_agent import (
    LLMResponse,
    ScriptedTransport,
    ToolAgent,
    ToolCall,
    load_transcript,
    save_transcript,
)
from kernelagent.agent.proposer import LLMProposer
from kernelagent.eval.harness import all_tasks
from kernelagent.tasks.registry import variant_from_knobs


def _task():
    return all_tasks()["sgemm_tiled"]


def _default_args() -> dict:
    task = _task()
    return {"knobs": dict(task.default_knobs), "cflags": [], "rationale": "baseline"}


def _responses() -> list[LLMResponse]:
    return [
        LLMResponse(
            text="Let me compile the default first.",
            tool_calls=[ToolCall(id="c1", name="build_variant", arguments=_default_args())],
            input_tokens=100,
            output_tokens=20,
        ),
        LLMResponse(
            text="Submitting the baseline.",
            tool_calls=[ToolCall(id="c2", name="submit_best", arguments=_default_args())],
            input_tokens=80,
            output_tokens=10,
        ),
    ]


def test_tool_agent_runs_offline_and_records_transcript(tmp_path: Path) -> None:
    task = _task()
    transport = ScriptedTransport(_responses())
    agent = ToolAgent(task, task.shapes["small"], transport, max_tool_calls=4, model="scripted")
    variant, transcript = agent.run("no trials yet", [])
    assert variant is not None
    assert variant.key() == task.default_variant().key()
    assert agent.tokens == 210
    assert len(agent.tool_calls) == 2
    assert transcript["model"] == "scripted"
    assert transcript["proposals"]
    path = save_transcript(transcript, tmp_path)
    assert path.exists()
    assert load_transcript(path)["tokens"] == 210


def test_llm_proposer_replays_recording(tmp_path: Path) -> None:
    task = _task()
    agent = ToolAgent(task, task.shapes["small"], ScriptedTransport(_responses()), max_tool_calls=4)
    variant, transcript = agent.run("no trials yet", [])
    path = save_transcript(transcript, tmp_path)
    proposer = LLMProposer(replay=path)
    proposals = proposer.propose(task, task.shapes["small"], [], "no trials yet")
    assert len(proposals) == 1
    assert proposals[0].key() == task.default_variant().key()
    assert proposer.tokens == 210
    assert proposer.propose(task, task.shapes["small"], [], "again") == []


def test_invalid_tool_arguments_are_rejected_then_budget_stops(tmp_path: Path) -> None:
    task = _task()
    bad = LLMResponse(
        tool_calls=[
            ToolCall(id="c1", name="build_variant", arguments={"knobs": {"NOPE": 1}}),
        ],
        input_tokens=5,
        output_tokens=1,
    )
    good = LLMResponse(
        tool_calls=[ToolCall(id="c2", name="submit_best", arguments=_default_args())],
        input_tokens=5,
        output_tokens=1,
    )
    agent = ToolAgent(task, task.shapes["small"], ScriptedTransport([bad, good]), max_tool_calls=3)
    variant, transcript = agent.run("no trials yet", [])
    assert variant is not None
    assert transcript["tool_calls"][0]["result"].get("error")
    assert transcript["tool_calls"][1]["name"] == "submit_best"


def test_tool_call_budget_is_enforced(tmp_path: Path) -> None:
    task = _task()
    responses = [
        LLMResponse(
            tool_calls=[ToolCall(id=f"c{i}", name="build_variant", arguments=_default_args())],
            input_tokens=1,
            output_tokens=1,
        )
        for i in range(10)
    ]
    agent = ToolAgent(task, task.shapes["small"], ScriptedTransport(responses), max_tool_calls=2)
    variant, transcript = agent.run("no trials yet", [])
    assert len(transcript["tool_calls"]) == 2
    assert variant is not None  # falls back to the last successfully built variant


def test_replay_transcript_is_plain_json(tmp_path: Path) -> None:
    payload = {
        "task": "sgemm_tiled",
        "shape": "M=64,N=64,K=64",
        "tokens": 42,
        "proposals": [
            variant_from_knobs(
                _task(), dict(_task().default_knobs), [], "llm"
            ).model_dump(mode="json")
        ],
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    proposer = LLMProposer(replay=path)
    proposals = proposer.propose(_task(), _task().shapes["small"], [], "")
    assert proposals[0].knobs == _task().default_knobs
    assert proposer.tokens == 42
