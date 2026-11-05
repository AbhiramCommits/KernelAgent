# Transcripts

Live LLM episodes are written here by `LLMProposer` (one JSON file per episode).
Each transcript records the task, shape, model, token usage, every tool call with
validated arguments and its structured result, and the final submitted variant.
`kernelagent search --strategy llm --replay <file>` replays the recorded
proposal sequence without any network access.

This run had no `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, so no live transcript was
produced. The LLM proposer is exercised entirely offline in
`tests/test_llm_proposer.py` using a scripted transport and transcript replay;
those test transcripts are written to pytest temporary directories, not here.
