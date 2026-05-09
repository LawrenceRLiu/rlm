# Instructions for Claude, Codex, Gemini, and other coding agents.

This guide covers best practices for contributing to the core Recursive Language Models `rlm` library and developing new environments (in `rlm/environments/`) and LM clients (in `rlm/clients/`).

## Current State of the Project:
The RLM substrate has been migrated from the legacy Python REPL to a durable
**Workspace Substrate** backed by a Docker container. The action surface is
XML `<action>` blocks parsed by `rlm/utils/action_parser.py`; "memory" lives
in workspace files (with per-turn git snapshots and a `_rlm_state/provenance.json`
sidecar). Architecture notes are in `workspace_substrate_arch/`. The only
supported environment is `DockerWorkspaceEnv` — all REPL substrates (local,
ipython, modal, prime, daytona, e2b, the old Docker REPL) have been removed.

## Setup

### Repository-specific environment

For this repository, run commands inside the `RLM_substrate` conda environment (this is a python 3.12 environment).

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate RLM_substrate
```

Install Python packages only after activating that environment. Prefer `uv` where appropriate, or use `python -m pip install ...`; do not use raw `pip install ...` because it may target the wrong environment.

We use `uv` for developing `rlm`.
```bash
# Install uv (first time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup blank project if needed
uv init && uv venv --python 3.12
source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# For Modal sandbox support
uv pip install -e ".[modal]"

# For Prime sandbox support
uv pip install -e ".[prime]"
```

## General Guidelines

### Code Style & Typing
- **Formatting**: Strict `ruff` enforcement. All PRs must pass `ruff check --fix .`
- **Typing**: Explicit types preferred
  - **OK**: `cast(...)`, `assert ...` for type narrowing
  - **SOMETIMES OK**: Untyped args for simple cases (e.g., prompt handlers)
  - **NOT OK**: `# type: ignore` without strong justification

### Naming Conventions
- **Methods**: snake_case
- **Classes**: PascalCase (e.g., `LocalREPL`, `PortkeyClient`)
- **Variables**: snake_case
- **Constants**: UPPER_CASE (e.g., `_SAFE_BUILTINS`, `RLM_SYSTEM_PROMPT`)

Do NOT use `_` prefix for private methods unless explicitly requested.

### Error Handling Philosophy
- **Fail fast, fail loud** - No defensive programming or silent fallbacks
- **Minimize branching** - Prefer single code paths; every `if`/`try` needs justification
- **Example**: Missing API key → immediate `ValueError`, not graceful fallback

## Core Repository Development

For PRs to `rlm` core:
```bash
git clone https://github.com/alexzhang13/rlm.git
cd rlm

# Standard development:
uv sync

# Install dev + test dependencies:
uv sync --group dev --group test

# Install pre-commit hooks:
uv run pre-commit install
```

### Dependencies
- Avoid new core dependencies
- Use optional extras for non-essential features (e.g., `modal` extra)
- Exception: tiny deps that simplify widely-used code

### Testing
- `uv run pytest` with discovery under `tests/`
- Write simple, deterministic unit tests
- Update tests when changing functionality
- For isolated environments, mock external services

### Documentation
- Keep concise and actionable
- Update README when behavior changes
- Avoid content duplication

### Scope
- Small, focused diffs
- One change per PR
- Backward compatibility is only desirable if it can be done without introducing excessive maintenance burden
- Delete dead code (don't guard it)

### Checklist

Before a PR:

```bash
# Run style + lint checks:
uv run ruff check --fix .
uv run ruff format .
uv run pre-commit run --all-files

# Run tests:
uv run pytest
```

Ensure docs and tests are updated if necessary, and dead code is deleted. Strive for minimal, surgical diffs.

## Developing LM Clients

LM client implementations live in `rlm/clients/`. All clients must inherit from `BaseLM`.

### Client Pattern

| Base Class | When to Use | Key Methods |
|------------|-------------|-------------|
| `BaseLM` | All LM integrations | `completion`, `acompletion`, `get_usage_summary`, `get_last_usage` |

### Requirements
- Inherit from `BaseLM` in `rlm/clients/base_lm.py`
- Implement all abstract methods: `completion`, `acompletion`, `get_usage_summary`, `get_last_usage`
- Track per-model usage (calls, input/output tokens)
- Handle both string and message list prompts
- Register client in `rlm/clients/__init__.py`

### Example Structure
```python
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

class MyClient(BaseLM):
    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        # Initialize your client
        
    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        # Handle both str and message list formats
        # Track usage with _track_cost()
        # Return response string
        
    def get_usage_summary(self) -> UsageSummary:
        # Return aggregated usage across all calls
```

### Configuration Guidelines
- **Environment variables**: ONLY for API keys (document in README)
- **Hardcode**: Default base URLs, reasonable defaults
- **Arguments**: Essential customization via `__init__()`

## Workspace Substrate

The only supported environment is `DockerWorkspaceEnv` (`rlm/environments/docker_workspace.py`),
which inherits from `BaseWorkspaceEnv` (`rlm/environments/base_workspace.py`).

### Contract

```python
class BaseWorkspaceEnv(ABC):
    def setup(self) -> None: ...
    def load_context(self, context_payload) -> None: ...
    def run_action(self, action: WorkspaceAction) -> WorkspaceObservation: ...
    def snapshot(self, turn: int) -> WorkspaceSnapshot: ...
    def cleanup(self) -> None: ...
```

### Configuration

All workspace knobs live on a composed `WorkspaceConfig` dataclass tree
(`rlm/core/config.py`): `ParseConfig`, `ObservationConfig`, `RecursionConfig`,
`DockerConfig`. Pass it via `RLM(workspace_config=...)`. Environment
variables are reserved for secrets only.

### Action Surface

The model emits XML `<action tool="...">...</action>` elements. The parser
in `rlm/utils/action_parser.py` is a tag-pair scanner (not strict XML) that
handles raw `<` / `&` / nested `<action>` inside bodies. The 10 v0.1 tools
live under `rlm/workspace_tools/`:

`list_directory`, `read_file`, `write_file`, `append_file`, `edit_file`,
`shell`, `python`, `llm_query`, `rlm_query`, `final`.

Read-only tool failures do **not** halt the rest of the turn; mutating tool
failures halt the rest of the batch. Per-call observations above
`observation.max_observation_chars` are spilled to
`_rlm_artifacts/_observations/` and replaced with a summary path.

### Recursion (`rlm_query`)

Implemented in `rlm/workspace_tools/rlm_query.py` with the spawn machinery
in `rlm/core/recursion.py`. Each child gets its own copy-on-spawn workspace
(excludes `.git`, `_rlm_state/snapshots`, `_rlm_artifacts/children`, files
larger than `recursion.copy_on_spawn_max_file_bytes`). At `depth ==
max_depth` the system prompt omits `rlm_query`; if the model emits one
anyway, the runtime returns a loud error observation. Children export
artifacts explicitly via `<artifact path="..."/>` children of `final`; the
runtime copies *only those* into the parent's
`_rlm_artifacts/children/<child_id>/` and includes a path-mapping table in
the parent's observation.

### Container ↔ Host Transport

`DockerWorkspaceEnv` runs the workspace container with the broker
(`rlm/environments/_broker.py`, also embedded in the image at
`docker/workspace_image/rlm_workspace/broker.py`) as PID 1. The host
poller forwards `/pending` requests to `LMHandler` over its TCP socket
and posts responses back to `/respond`. In-container code uses the
`rlm_workspace.client` module (preimported into `python` action bodies)
which exposes `llm_query`, `llm_query_batched`, `rlm_query`,
`rlm_query_batched` (no `model=` argument — the parent's configured
model is used everywhere).

### Logging & Visualization

`RLMLogger.log(WorkspaceIteration)` writes one JSONL line per turn (plus
a `type:"metadata"` header) under `log_dir/`. The Next.js visualizer in
`visualizer/` consumes that file directly: types in
`visualizer/src/lib/types.ts` mirror the Python `to_dict` schemas 1:1.
