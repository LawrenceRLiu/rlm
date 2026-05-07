# Migration Plan & Feasibility

## Current Constraint

The current codebase is REPL-first:

- `BaseEnv` exposes `setup`, `load_context`, and `execute_code`.
- The main RLM loop parses only fenced `repl` blocks and sends each block to `environment.execute_code`.
- Results are represented as `REPLResult`, `CodeBlock`, and `RLMIteration`.
- The prompt teaches `context`, `history`, `llm_query`, `rlm_query`, `SHOW_VARS`, and `FINAL_VAR`.
- Persistence means versioned Python variables, not durable workspace state.

The migration should preserve the useful shape of the loop but replace the action substrate underneath it.

## Migration Plan

1. Add new workspace types and parser while leaving REPL code untouched.
2. Add `DockerWorkspaceEnv` with `list_files`, `read_file`, `write_file`, `append_file`, `shell`, and `python`.
3. Add host-backed `llm_query` and `rlm_query` workspace actions.
4. Add host-backed `web_search` and `fetch_url`.
5. Add a workspace system prompt and route `environment="docker_workspace"` through `get_environment`.
6. Add focused tests for parsing, Docker workspace file persistence, shell/python execution, web tool stubs, and recursive child calls.
7. Update logging/visualizer compatibility.
8. Deprecate Modal and Prime in docs for this fork.
9. Only after the workspace path is stable, consider removing direct REPL prompt defaults.

## Feasibility

This is feasible as a staged migration.

The main loop in `rlm/core/rlm.py` is compact enough to generalize from "code blocks" to "actions". The larger cost is not the RLM algorithm; it is changing the assumptions around prompts, parsing, result schemas, persistence, tests, and visualizer output.

The most important design decision is to make workspace mode a new path first, rather than trying to mutate every REPL environment at once.
