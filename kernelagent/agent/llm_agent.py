"""LLM tool-calling agent (transport-agnostic).

The proposer exposes exactly five tools to the model: build, run, profile and
check a variant, and submit the best one. Every argument is validated against a
pydantic schema; every episode is written to ``results/transcripts`` and can be
replayed without a network call.
"""

from __future__ import annotations

import contextvars
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from .. import config
from ..tasks.registry import variant_from_knobs
from ..tools.base import tool, tool_schemas
from ..tools.build import CxxBuilder
from ..tools.checker import CorrectnessChecker
from ..tools.profiler import profile_binary
from ..tools.runner import BenchRunner
from ..types import (
    BuildResult,
    CheckResult,
    KernelTask,
    KernelVariant,
    ProfileResult,
    RunResult,
    Shape,
)
from ..workload import flops_for
from .feedback import history_summary


class VariantArgs(BaseModel):
    """Arguments for every kernel tool."""

    knobs: dict[str, int | str] = Field(description="Complete legal knob assignment for the task")
    cflags: list[str] = Field(default_factory=list, description="Optional extra compiler flags")
    rationale: str = Field(default="", description="One sentence explaining the change")


@dataclass
class ToolContext:
    """Per-episode state used by the module-level tool functions."""

    task: KernelTask
    shape: Shape
    builder: CxxBuilder
    runner: BenchRunner
    checker: CorrectnessChecker
    submitted: KernelVariant | None = None
    last_variant: KernelVariant | None = None
    build_cache: dict[str, BuildResult] = field(default_factory=dict)

    def variant(self, args: VariantArgs) -> KernelVariant:
        return variant_from_knobs(self.task, dict(args.knobs), list(args.cflags), "llm")

    def build(self, args: VariantArgs) -> BuildResult:
        variant = self.variant(args)
        self.last_variant = variant
        key = variant.key()
        if key not in self.build_cache:
            self.build_cache[key] = self.builder.build(self.task, variant)
        return self.build_cache[key]


_CURRENT: contextvars.ContextVar[ToolContext | None] = contextvars.ContextVar(
    "kernelagent_tool_context", default=None
)


def _context() -> ToolContext:
    context = _CURRENT.get()
    if context is None:
        raise RuntimeError("tool called outside an LLM episode")
    return context


def _build_summary(build: BuildResult) -> dict[str, Any]:
    diagnostics = build.diagnostics
    return {
        "ok": build.ok,
        "compile_time_s": build.compile_time_s,
        "error": build.error,
        "vectorized_loops": diagnostics.get("vectorized_loops"),
        "missed_vectorizations": diagnostics.get("missed_count"),
        "unroll": diagnostics.get("unroll"),
        "registers": diagnostics.get("registers"),
        "spills": diagnostics.get("spills"),
    }


@tool("Compile one kernel variant and return compiler diagnostics.")
def build_variant(args: VariantArgs) -> dict[str, Any]:
    """Build only."""
    context = _context()
    return _build_summary(context.build(args))


@tool("Build, correctness-check and time one kernel variant.")
def run_variant(args: VariantArgs) -> dict[str, Any]:
    """Build, check, run."""
    context = _context()
    build = context.build(args)
    if not build.ok or build.binary is None:
        return {"ok": False, "error": build.error}
    check = context.checker.check(context.shape, build.binary)
    if not check.passed:
        return {"ok": False, "correct": False, "message": check.message}
    run: RunResult = context.runner.run(
        build.binary, context.shape, reps=3, warmup=1, flops=flops_for(context.task, context.shape)
    )
    return {
        "ok": True,
        "correct": True,
        "ms_median": run.ms_median,
        "ms_mad": run.ms_mad,
        "gflops": run.throughput,
    }


@tool("Build, run and profile one kernel variant, returning performance counters.")
def profile_variant(args: VariantArgs) -> dict[str, Any]:
    """Build, check, run, profile."""
    context = _context()
    build = context.build(args)
    if not build.ok or build.binary is None:
        return {"ok": False, "error": build.error}
    check = context.checker.check(context.shape, build.binary)
    if not check.passed:
        return {"ok": False, "correct": False, "message": check.message}
    run = context.runner.run(
        build.binary, context.shape, reps=3, warmup=1, flops=flops_for(context.task, context.shape)
    )
    profile: ProfileResult = profile_binary(context.task, context.shape, build.binary, run)
    wanted = ("gflops", "gbps", "bandwidth_pct", "arithmetic_intensity", "occupancy_pct")
    counters = {key: profile.counters[key] for key in wanted if key in profile.counters}
    return {"ok": True, "source": profile.source, "ms_median": run.ms_median, "counters": counters}


@tool("Build one variant and compare its output with the NumPy reference.")
def check_variant(args: VariantArgs) -> dict[str, Any]:
    """Build and check only."""
    context = _context()
    build = context.build(args)
    if not build.ok or build.binary is None:
        return {"ok": False, "error": build.error}
    result: CheckResult = context.checker.check(context.shape, build.binary)
    return {
        "ok": True,
        "passed": result.passed,
        "max_abs_err": result.max_abs_err,
        "max_rel_err": result.max_rel_err,
        "message": result.message,
    }


@tool("Submit the best variant found. Ends the episode.")
def submit_best(args: VariantArgs) -> dict[str, Any]:
    """Record the final answer."""
    context = _context()
    context.submitted = context.variant(args)
    return {"submitted": True, "variant": context.submitted.key()}


LLM_TOOLS = [build_variant, run_variant, profile_variant, check_variant, submit_best]
LLM_TOOL_SCHEMAS = tool_schemas(LLM_TOOLS)
_TOOL_BY_NAME = {fn.__name__: fn for fn in LLM_TOOLS}


@dataclass
class ToolCall:
    """One model-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Neutral provider response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Transport(Protocol):
    """Provider adapter boundary; tests inject a scripted transport."""

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


class AnthropicTransport:
    """Anthropic Messages API adapter (lazy import)."""

    def __init__(self, model: str, max_tokens: int = 2048) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system, messages, tools):  # type: ignore[no-untyped-def]
        converted = _to_anthropic_messages(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
                for tool in tools
            ],
            messages=converted,
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", "") == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=calls,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )


class OpenAITransport:
    """OpenAI chat-completions adapter (lazy import)."""

    def __init__(self, model: str, max_tokens: int = 2048) -> None:
        import openai

        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system, messages, tools):  # type: ignore[no-untyped-def]
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, *_to_openai_messages(messages)],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ],
        )
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        usage = response.usage
        return LLMResponse(
            text=message.content or "",
            tool_calls=calls,
            input_tokens=int(usage.prompt_tokens) if usage else 0,
            output_tokens=int(usage.completion_tokens) if usage else 0,
        )


class ScriptedTransport:
    """Offline transport that replays responses (used by tests and fixtures)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, messages, tools):  # type: ignore[no-untyped-def]
        self.calls += 1
        if not self.responses:
            return LLMResponse(text="no more scripted responses")
        return self.responses.pop(0)


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message["content"],
                        }
                    ],
                }
            )
        elif role == "assistant":
            content: list[dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls", []):
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["arguments"],
                    }
                )
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append({"role": "user", "content": message["content"]})
    return converted


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            converted.append(
                {"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]}
            )
        elif role == "assistant":
            converted.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in message.get("tool_calls", [])
                    ]
                    or None,
                }
            )
        else:
            converted.append({"role": "user", "content": message["content"]})
    return converted


def _system_prompt(task: KernelTask) -> str:
    knob_lines = [
        f"- {name}: legal values {spec.values} (current default {task.default_knobs[name]})"
        for name, spec in task.knobs.items()
    ]
    return (
        "You are KernelAgent, an expert performance engineer optimising a compiled kernel.\n"
        f"Task: {task.name}. {task.description}\n"
        "Use the tools to compile, check, time and profile variants. Only submit a variant "
        "that passed the correctness check. Prefer changes justified by the compiler remarks "
        "or profiler counters. Keep each knob assignment complete.\n"
        "Knobs:\n" + "\n".join(knob_lines)
    )


def _user_prompt(task: KernelTask, shape: Shape, feedback: str, history: list[Any]) -> str:
    return (
        f"Shape: {shape.key()}\nFeedback from the latest trial:\n{feedback}\n\n"
        f"Best trials so far:\n{history_summary(history)}\n\n"
        "Propose the next variant and call submit_best when you have one."
    )


class ToolAgent:
    """Runs one tool-calling episode and returns the submitted variant."""

    def __init__(
        self,
        task: KernelTask,
        shape: Shape,
        transport: Transport,
        max_tool_calls: int = 8,
        seed: int = 0,
        model: str = "unknown",
    ) -> None:
        self.task = task
        self.shape = shape
        self.transport = transport
        self.max_tool_calls = max_tool_calls
        self.seed = seed
        self.model = model
        self.tokens = 0
        self.tool_calls: list[dict[str, Any]] = []

    def run(self, feedback: str, history: list[Any]) -> tuple[KernelVariant | None, dict[str, Any]]:
        """Execute the episode; return (variant, transcript payload)."""
        context = ToolContext(
            task=self.task,
            shape=self.shape,
            builder=CxxBuilder(),
            runner=BenchRunner(),
            checker=CorrectnessChecker(self.task),
        )
        token = _CURRENT.set(context)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _user_prompt(self.task, self.shape, feedback, history)}
        ]
        system = _system_prompt(self.task)
        empty_streak = 0
        try:
            while len(self.tool_calls) < self.max_tool_calls and empty_streak < 2:
                response = self.transport.complete(system, messages, LLM_TOOL_SCHEMAS)
                self.tokens += response.tokens
                if not response.tool_calls:
                    empty_streak += 1
                    messages.append({"role": "assistant", "content": response.text or ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use one of the tools, or call submit_best to finish.",
                        }
                    )
                    continue
                empty_streak = 0
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.text,
                        "tool_calls": [
                            {"id": call.id, "name": call.name, "arguments": call.arguments}
                            for call in response.tool_calls
                        ],
                    }
                )
                for call in response.tool_calls:
                    if len(self.tool_calls) >= self.max_tool_calls:
                        break
                    result = self._dispatch(call)
                    self.tool_calls.append(
                        {"name": call.name, "arguments": call.arguments, "result": result}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": json.dumps(result, default=str)[:4000],
                        }
                    )
                    if call.name == "submit_best" and context.submitted is not None:
                        return context.submitted, self._transcript(context)
        finally:
            _CURRENT.reset(token)
        return context.submitted or context.last_variant, self._transcript(context)

    def _dispatch(self, call: ToolCall) -> dict[str, Any]:
        function = _TOOL_BY_NAME.get(call.name)
        if function is None:
            return {"error": f"unknown tool {call.name}"}
        try:
            args = VariantArgs.model_validate(call.arguments)
            return dict(function(args))
        except (ValidationError, ValueError, KeyError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:1000]}

    def _transcript(self, context: ToolContext) -> dict[str, Any]:
        proposals = []
        if context.submitted is not None:
            proposals.append(context.submitted.model_dump(mode="json"))
        elif context.last_variant is not None:
            proposals.append(context.last_variant.model_dump(mode="json"))
        return {
            "task": self.task.name,
            "shape": self.shape.key(),
            "model": self.model,
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "proposals": proposals,
            "recorded_at": datetime.now(UTC).isoformat(),
            "seed": self.seed,
        }


def save_transcript(payload: dict[str, Any], directory: Path | None = None) -> Path:
    """Write an episode transcript and return its path."""
    target_dir = Path(directory) if directory else config.TRANSCRIPTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{payload.get('task', 'task')}_{str(payload.get('shape', 'shape')).replace(',', '-')}"
        f"_{payload.get('seed', 0)}_{uuid.uuid4().hex[:8]}"
    )
    path = target_dir / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def load_transcript(path: str | Path) -> dict[str, Any]:
    """Read a recorded transcript."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a transcript object")
    return payload


def build_transport(provider: str, model: str | None = None) -> tuple[Transport, str]:
    """Construct the configured provider transport."""
    if provider == "anthropic":
        chosen = model or os.environ.get("KA_LLM_MODEL", "claude-sonnet-4-20250514")
        return AnthropicTransport(chosen), chosen
    if provider == "openai":
        chosen = model or os.environ.get("KA_LLM_MODEL", "gpt-4o-mini")
        return OpenAITransport(chosen), chosen
    raise ValueError(f"unknown LLM provider {provider!r}")
