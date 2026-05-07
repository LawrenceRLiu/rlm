# Core Architecture

## Core Loop

The RLM loop should become:

1. Build a workspace system prompt.
2. Ask the root LM for the next action or final answer.
3. Parse one or more workspace action blocks.
4. Execute actions through a workspace environment.
5. Append compact observations to the message history.
6. Repeat until final answer or limits.

This keeps the current iterative RLM structure intact, but renames the unit of execution from "code block" to "action".

## New Environment Contract

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

## Docker Execution

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

## Result Types

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
