# Workspace Substrate Sketch

## Goal

Move RLM from a Python REPL substrate to a generalized workspace substrate while preserving the core recursive inference idea:

1. The root model receives a task and compact workspace instructions.
2. It takes structured actions against a workspace.
3. The runtime executes those actions and returns observations.
4. The model iterates, optionally spawning child RLMs.
5. The run ends with a final answer and a replayable trajectory.

The target is not a terminal coding assistant. It is a task substrate for recursive language model inference across coding, web research, math, and mixed workflows. The first implementation should be Docker-only. Modal and Prime can be deprecated for this fork for now.

## Current Constraint

The current codebase is REPL-first:

- `BaseEnv` exposes `setup`, `load_context`, and `execute_code`.
- The main RLM loop parses only fenced `repl` blocks and sends each block to `environment.execute_code`.
- Results are represented as `REPLResult`, `CodeBlock`, and `RLMIteration`.
- The prompt teaches `context`, `history`, `llm_query`, `rlm_query`, `SHOW_VARS`, and `FINAL_VAR`.
- Persistence means versioned Python variables, not durable workspace state.

The migration should preserve the useful shape of the loop but replace the action substrate underneath it.

## Proposed Architecture

### Core Loop

The RLM loop should become:

1. Build a workspace system prompt.
2. Ask the root LM for the next action or final answer.
3. Parse one or more workspace action blocks.
4. Execute actions through a workspace environment.
5. Append compact observations to the message history.
6. Repeat until final answer or limits.

This keeps the current iterative RLM structure intact, but renames the unit of execution from "code block" to "action".

### New Environment Contract

Introduce a workspace contract alongside the existing REPL contract:

```python
class BaseWorkspaceEnv:
    def setup(self) -> None: ...
    def load_context(self, context_payload: dict | list | str) -> None: ...
    def run_action(self, action: WorkspaceAction) -> WorkspaceObservation: ...
    def snapshot(self) -> WorkspaceSnapshot: ...
    def cleanup(self) -> None: ...
```

The first concrete implementation should be `DockerWorkspaceEnv`.

`LocalREPL` can remain as legacy support during migration, but new work should target the workspace API. Modal and Prime should not be carried forward until the Docker version is proven.

### Workspace State

The workspace should be a mounted Docker directory with durable files:

```text
(Workspace Root)
  _rlm_query_0.txt
  [user_uploaded_files_and_directories]
  /_rlm_notes
    scratch.md
    findings.md
  /_rlm_artifacts
    outputs produced by tools or shell commands
  /_rlm_state
    action_log.jsonl
    workspace_manifest.json
```

The model should not have to remember Python locals. It should inspect files and use _rlm_notes/_rlm_artifacts as durable memory. This would enable us to handle "modalities" such as code, images, pdfs, or other file formats by just dumping them into the workspace root.

**Note** With the `_rlm_` prefix system, I anticipate significantly less chance of "namespace collisions" between user provided files and what RLM needs to use internally. However as a final catch, to prevent unexpected behavior, if there is a clash, then we should immediately throw an error and have the user resolve it.

### Action Format

Use a single structured action block rather than many language-specific tags:

```workspace
{"tool": "read_file", "path": "_rlm_query_0.txt"}
```

The block may contain either one action object or a list of action objects:

```workspace
[
  {"tool": "read_file", "path": "_rlm_query_0.txt"},
  {"tool": "read_file", "path": "_rlm_notes/findings.md"}
]
```

Batching is useful when actions are independent reads or simple information-gathering steps. The runtime should execute a block in order by default, with optional parallel execution later for tools declared as safe and read-only. This gives the model an efficient "do these next few things" interface without requiring the runtime to infer dependencies between actions.

Initial tools:

- `list_files`: list workspace paths. This should act like `ls` (shallow by default) to prevent context window bloat. The runtime must aggressively filter out noise (e.g., `.git`, `node_modules`, `__pycache__`). For the files it does list, it should include metadata on who created and last modified each file (e.g., which child RLM or tool, backed by a background git repo). Because the listing is shallow, this extra info won't cost too many tokens.
- `read_file`: read a bounded file slice.
- `write_file`: create or overwrite a workspace file.
- `append_file`: append notes or results.
- `shell`: run a command inside the Docker workspace.
- `python`: convenience wrapper for running Python inside Docker.
- `web_search`: host-backed search tool returning compact structured results.
- `fetch_url`: host-backed URL fetch with bounded text extraction.
- `llm_query`: plain LM completion through the existing LM handler.
- `rlm_query`: recursive child RLM call with its own workspace.
- `final`: submit final answer.

The model should see only short tool descriptions. Full tool schemas should stay in code/tests, not in the base prompt.

### Turn-to-Turn Interface

**Note** by turn we mean the internal turns between rounds of the RLM root, not the outer turns between user queries.

Each RLM turn should have a simple transcript shape:

1. The model receives the task, compact workspace instructions, current workspace summary, and previous observations.
2. The model returns either a `workspace` action block or a final answer action.
3. The runtime parses the block into `WorkspaceAction` objects.
4. The runtime executes the actions and appends one compact observation message.
5. The next model turn sees the prior assistant action block and the runtime observation.

**Example transcript:**

    ====== Turn 1 ======  
    assistant:
    I need to inspect the context and current notes.

    ```workspace
    [
      {"tool": "list_files", "path": "."},
      {"tool": "read_file", "path": "_rlm_query_0.txt"}
    ]
    ```

    user/runtime:
    Observation:
    - list_files: 12 paths. Important paths: _rlm_query_0.txt, _rlm_notes/scratch.md
    - read_file _rlm_query_0.txt: "Please investigate the error in..."

    ====== Turn 2 ======
    assistant:
    [follow-up assistant message with new `workspace` action block or `final` action]

    user/runtime:
    [follow-up observation message]

    ...

    ====== Turn N ======
    assistant:
    [...]

    user/runtime:
    [follow-up observation message]



The observation should be compact and bounded. Large outputs (Ie above a certain hyperparameter number of lines/words/charachters) should be written to `_rlm_artifacts/` and summarized with a file path and a note on the length (in lines or characters). This avoids filling the root message history while preserving inspectable state in the workspace.

The runtime should reject malformed actions loudly with a parse observation that tells the model exactly what schema failed. This is important for small models. **Note** An interesting alternative would be to use constrained decoding to force the model to output a valid json block.

#### Context Compaction

Like the vanilla RLM implementation, the workspace substrate will inevitably accumulate long message histories over many turns. To mitigate this, we implement a context compaction mechanism:

1. When the token count of the active `message_history` crosses a configured threshold, the runtime pauses and prompts the model to generate a dense summary of the entire trajectory up to that point.
2. The active context window is then "restarted" by replacing the prior history with this new summary.
3. However, unlike the vanilla REPL implementation—which stored the raw history in an ephemeral Python `history` variable—the workspace substrate writes the full, uncompacted action log to a durable file (e.g., `_rlm_state/action_log.jsonl` or `_rlm_notes/trajectory_archive.md`).

This ensures that the model can always use standard file tools (like `read_file`) to recover exact details from earlier in the run if it realizes the summary omitted crucial information, completely avoiding the permanent loss of context.

### Docker Execution

The Docker workspace should be the only execution backend for now.

Host responsibilities:

- Own the RLM loop, LM handler, action parser, and trajectory logger.
- Create a per-completion workspace directory.
- Start a Docker container with that directory mounted as the working directory.
- Dispatch shell/python actions into the container.
- Dispatch web and LM actions from the host, not from ad hoc model-written Python.
- Capture stdout, stderr, exit code, wall time, modified files, and any tool-specific metadata.

Container responsibilities:

- Execute commands in the workspace root.
- Persist files through the mounted volume.
- Avoid knowing about OpenAI/Anthropic/etc. API keys unless explicitly needed.

This is less secure than cloud isolation, but acceptable for the fork's near-term goals. Docker still gives a cleaner boundary than in-process `exec`.

### Result Types

Replace REPL-specific result names in the new path:

```python
@dataclass
class WorkspaceAction:
    tool: str
    args: dict[str, Any]
    raw: str

@dataclass
class WorkspaceObservation:
    tool: str
    stdout: str = ""
    stderr: str = ""
    data: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    execution_time: float | None = None
    rlm_calls: list[RLMChatCompletion] = field(default_factory=list)
    final_answer: str | None = None
```

`RLMIteration` should eventually hold `actions` rather than `code_blocks`. During migration, a compatibility serializer can emit both fields or map actions into code-block-like records for the current visualizer.

### Prompt Shape

The workspace prompt should be much shorter than the current REPL prompt.

It should say:

- Your task instructions are in `_rlm_query_0.txt`.
- Use workspace tools to inspect, compute, search, and write notes.
- Keep durable findings in `_rlm_notes/`.
- Use `llm_query` for simple language subtasks.
- Use `rlm_query` for subtasks that need their own iterative workspace.
- Do not answer until you have inspected enough context.
- End with a `final` action.

Drop `FINAL_VAR` and `SHOW_VARS` from workspace mode. They are Python-local concepts and make smaller models learn an avoidable convention.

### Recursion

`rlm_query` remains the defining feature.

Each recursive child should get:

- Its own Docker workspace initialized from a filtered snapshot of the parent workspace.
- A task instruction file (`_rlm_query_0.txt`) containing the specific child task requested by the parent.
- Its own action loop and trajectory.
- A returned final answer plus metadata.

The default should be copy-on-spawn:

1. Parent calls `rlm_query` with a child task (e.g., `"Analyze the logs in _rlm_artifacts/server_logs.txt"`).
2. Runtime creates a new child workspace directory.
3. Runtime copies the parent workspace into the child workspace.
4. Runtime applies simple excludes for obvious junk: `.git`, `.venv`, `node_modules`, caches, build outputs, and oversized binary files.
5. Runtime writes the child task string to `_rlm_query_0.txt` inside the child's workspace (overwriting the parent's `_rlm_query_0.txt`).
6. Child RLM runs independently over that snapshot. Because it inherited the file system, the parent doesn't need to pass massive data strings in the `rlm_query` call; it just passes file pointers.
7. Child returns final answer, trajectory metadata, and optionally a list of produced artifacts. This will be extracted from the `FINAL_VAR` in the child's assistant message and append to the observation.
8. Parent receives a compact observation containing the child answer and artifact paths.

The child must not mutate the parent workspace directly. Copying the parent workspace gives the child full context without introducing live shared state. If the parent wants to use child artifacts, the runtime can either copy selected child outputs back into `_rlm_artifacts/children/<child_id>/` or expose a follow-up action such as `import_child_artifact`.

This default is intentionally simple. Explicit artifact selection can be added later as an optimization when workspaces become too large or when benchmark isolation needs stricter control.

### Web Search

Web search should be a first-class host tool, not a Python package the model has to import.

Recommended behavior:

- `web_search(query, max_results={n})` returns titles, URLs, snippets, and source metadata. This should be done using the brave API (key will be put in `.env` at the end)
- `fetch_url(url)` returns bounded extracted text plus page metadata.
- Observations should be concise by default.
- Large pages should be saved into `_rlm_artifacts/` and returned as file paths.

This keeps web research compatible with Docker while avoiding API keys and scraping logic inside model-generated code.

### Migration Plan

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

## Source Links

- OpenAI Codex CLI: https://developers.openai.com/codex/cli
- Claude Code overview: https://code.claude.com/docs
- opencode docs: https://dev.opencode.ai/docs/
- OpenHands docs: https://docs.openhands.dev/overview/introduction
- OpenHands paper: https://arxiv.org/abs/2407.16741

## Comparison With Existing Coding Agents

Sources checked: OpenAI Codex CLI docs, Anthropic Claude Code docs, opencode docs, OpenHands docs and paper.

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
