# Workspace Substrate Architecture

## Goal

Move RLM from a Python REPL substrate to a generalized workspace substrate while preserving the core recursive inference idea:

1. The root model receives a task and compact workspace instructions.
2. It takes structured actions against a workspace.
3. The runtime executes those actions and returns observations.
4. The model iterates, optionally spawning child RLMs.
5. The run ends with a final answer and a replayable trajectory.

The target is not a terminal coding assistant. It is a task substrate for recursive language model inference across coding, web research, math, and mixed workflows. The first implementation should be Docker-only. Modal and Prime can be deprecated for this fork for now.

## What Makes This Different

The workspace substrate should be different from Codex, Claude Code, opencode, and OpenHands in four ways:

1. Recursion is first-class.
   The core operation is not "run a coding agent"; it is "let an LM recursively spawn smaller RLM problems with their own workspaces."

2. The workspace is an inference substrate, not a user product.
   No initial TUI, IDE, PR flow, git automation, or human approval UX. The public API remains `RLM.completion`.

3. The prompt/tool surface is intentionally small.
   The point is to make the format easier for 7-9B models to follow. A large MCP-style tool universe would work against that goal.

4. Benchmarks drive the architecture.
   The first-class outputs should be final answer, action trajectory, subcall tree, artifacts, usage, and timing. That makes coding, web, and math evaluations easier to compare.

If we keep those constraints, the workspace substrate will not just be another coding agent. It will be a cleaner external-memory and tool substrate for recursive LM inference.
