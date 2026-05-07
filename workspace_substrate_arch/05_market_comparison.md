# Comparison With Existing Coding Agents

Sources checked: OpenAI Codex CLI docs, Anthropic Claude Code docs, opencode docs, OpenHands docs and paper.

## Source Links

- OpenAI Codex CLI: https://developers.openai.com/codex/cli
- Claude Code overview: https://code.claude.com/docs
- opencode docs: https://dev.opencode.ai/docs/
- OpenHands docs: https://docs.openhands.dev/overview/introduction
- OpenHands paper: https://arxiv.org/abs/2407.16741

## Comparisons

### Codex CLI

Codex CLI is a local terminal coding agent. OpenAI describes it as a coding agent that can read, change, and run code on the user's machine in the selected directory. It is optimized for developer workflows around a local repository.

RLM workspace should differ in purpose: it is not primarily a developer-facing coding CLI. It is an inference substrate for recursive decomposition. Coding is one benchmark/task family, not the product center.

Key difference:

- Codex: one agent operating on a repo to help a developer.
- RLM workspace: recursive root/child RLM calls operating over isolated task workspaces, with trajectory data for research and benchmarking.

### Claude Code

Claude Code is an agentic coding tool available through terminal, IDE, desktop, browser, and cloud sessions. It reads codebases, edits files, runs commands, integrates with tools, supports MCP, permissions, hooks, skills, and multiple agents.

RLM workspace should avoid becoming a full developer productivity surface. It should not start with IDE integrations, git workflows, PR creation, broad MCP support, or complex permission UX. Those are valuable product features, but they distract from the substrate question.

Key difference:

- Claude Code: broad interactive coding product with rich integrations and user-facing workflow controls.
- RLM workspace: minimal, benchmarkable action runtime designed to test whether recursive workspaces make smaller models follow the RLM format better.

### opencode

opencode is an open source AI coding agent available as a terminal interface, desktop app, or IDE extension. Its docs emphasize project initialization, `AGENTS.md`, plan/build modes, file references, provider configuration, and terminal-centric development.

RLM workspace can borrow the idea of project instructions and plan/build separation, but should not copy the full terminal-agent product shape.

Key difference:

- opencode: an open terminal-native coding assistant for humans working in a project.
- RLM workspace: a library substrate invoked programmatically by `RLM.completion`, where the "workspace" is the model's external memory and tool surface.

### OpenHands

OpenHands is closest architecturally. It is an open platform for software development agents that write code, use a command line, browse the web, use sandboxed environments, coordinate agents, and run benchmarks such as SWE-Bench and WebArena.

This means the RLM workspace must be especially clear about what it is not trying to duplicate. It should not become a general software-agent platform with many agent classes and UI surfaces. The distinctive piece is recursive language model inference: `rlm_query` is not just another subagent dispatch. It is a child RLM with the same iterative substrate and depth-limited recursion.

Key difference:

- OpenHands: generalist software-agent platform and SDK.
- RLM workspace: recursive inference engine with workspace-backed child calls, designed to study decomposition, context offloading, and small-model format following.
