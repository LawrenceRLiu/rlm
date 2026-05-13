
---

<h1 align="center" style="font-size:2.8em">
<span>Recursive Language Models (<span style="color:orange">RLM</span>s)</span>
</h1>

<p align="center" style="font-size:1.3em">
  <a href="https://arxiv.org/abs/2512.24601">Full Paper</a> •
  <a href="https://alexzhang13.github.io/blog/2025/rlm/">Blogpost</a> •
  <a href="https://alexzhang13.github.io/rlm/">Documentation</a> •
  <a href="https://github.com/alexzhang13/rlm-minimal">RLM Minimal</a>
</p>

<p align="center">
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/style.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/test.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">
    <img src="media/paper_preview.png" alt="Paper Preview" width="300"/>
  </a>
</p>

## Overview
Recursive Language Models (RLMs) are a task-agnostic inference paradigm for language models (LMs) to handle near-infinite length contexts by enabling the LM to *programmatically* examine, decompose, and recursively call itself over its input. RLMs replace the canonical `llm.completion(prompt, model)` call with a `rlm.completion(prompt, model)` call.

This repository provides an extensible inference engine for RLMs built on the **Workspace Substrate** architecture — a Docker-backed execution environment with durable file-system memory. Unlike REPL-based approaches, the workspace substrate preserves state across actions via mounted directories and git snapshots, enabling reliable multi-turn reasoning and artifact management.

The initial experiments and idea were proposed in a [blogpost](https://alexzhang13.github.io/blog/2025/rlm/) in 2025, with expanded results in an [arXiv preprint](https://arxiv.org/abs/2512.24601).

> [!NOTE]
> This is a working fork of the RLM substrate, maintained by Lawrence Liu (UCLA) with the workspace substrate architecture implementation. The codebase is actively developed and not yet fully tested in production. Open-source contributions are welcome.

## Quick Setup
You can try out RLMs quickly by installing from PyPi:
```bash
pip install rlms
```

> [!IMPORTANT]
> **Docker is required.** The workspace substrate runs code in an isolated Docker container. [Install Docker Desktop](https://docs.docker.com/desktop/setup/install/) before proceeding.

The RLM client uses a workspace environment that runs inside Docker. It provides durable, file-system based memory and supports structured tool calls (file operations, code execution, recursive child RLMs). As an example, we can call RLM completions using Claude 3.5 Sonnet:
```python
from rlm import RLM

rlm = RLM(
    backend="anthropic",
    backend_kwargs={"model_name": "claude-3-5-sonnet-20241022"},
    verbose=True,  # For printing to console with rich, disabled by default.
)

print(rlm.completion("Analyze the first 100 primes and write the results to a file.").response)
```

See `workspace_substrate_arch/` in the repository for detailed architecture documentation.

<details>
<summary><b>Manual Setup</b></summary>

Set up the dependencies with `uv` (or your virtual environment of choice):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12  # change version as needed
uv sync --group dev --group test
```

This project includes a `Makefile` to simplify common tasks.

- `make install`: Install base dependencies.
- `make install-dev`: Install dev and test dependencies.
- `make build-image`: Build the workspace Docker image.
- `make lint`: Run ruff linter.
- `make format`: Run ruff formatter.
- `make test`: Run pytest.
- `make check`: Run lint + format + test.

</details>

## Workspace Substrate
The RLM runtime operates inside a **Docker container with a mounted workspace directory**. This provides:

- **Durable memory**: Files in the workspace survive across iterations
- **Git snapshots**: One commit per turn tracks changes and enables rollback
- **Structured actions**: native OpenAI-compatible/vLLM tool calls for file ops, code execution, and recursive calls; legacy `<action>` XML is retained only for old logs and parser compatibility
- **Isolation**: Code execution is sandboxed; malicious payloads are contained
- **File provenance**: Role-based tagging (`user`, `assistant`, `system`, `child`) tracks which tool created/modified each file
- **Recursive support**: Child RLM instances get a copy-on-spawn snapshot of the parent workspace

### Configuration
The workspace environment is configured via `WorkspaceConfig`, with sub-configs for parsing, observation handling, recursion, and Docker:

```python
from rlm import RLM
from rlm.core.config import DockerConfig, RecursionConfig, WorkspaceConfig

rlm = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": "qwen3-5-9b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
    },
    workspace_config=WorkspaceConfig(
        docker=DockerConfig(
            image="rlm-workspace:0.1.0",
            workspace_root_base="~/.rlm/workspaces",
            exec_timeout_seconds=300,
            cleanup_mode="keep",  # or "tar" / "delete"
        ),
        recursion=RecursionConfig(
            max_concurrent_subcalls=5,
        ),
    ),
    verbose=True,
)
```

### Building the Docker Image
Before running RLM, build the workspace image:
```bash
make build-image
# or with a custom tag:
make build-image IMAGE_TAG=my-workspace:latest
```

The image is based on `python:3.11-slim` with pre-installed scientific/utility libraries (NumPy, Pandas, SciPy, Matplotlib, etc.). See `docker/workspace.Dockerfile` for the full manifest.

### Native vLLM Tool Calls
For Qwen/Qwen3.5 runs, the native scaffold is the default:

```python
from rlm.core.config import WorkspaceConfig

workspace_config = WorkspaceConfig()
```

Serve vLLM with auto tool choice enabled. The single-replica helper now defaults to the `hermes` parser:

```bash
python setup/server_qwen35_single.py --port 8001 --gpus 0 --model Qwen/Qwen3.5-9B
```

The native tool surface uses `run_shell_command`, `run_python_command`, `edit`, `write_file`, `append_file`, `read_file`, `list_directory`, `llm_query`, `rlm_query`, and `final`. `run_python_command` preimports `llm_query`, `llm_query_batched`, `rlm_query`, and `rlm_query_batched` for programmatic document loops and batched calls.

In native mode, OpenAI-compatible backends default to `enable_thinking=False` unless `backend_kwargs` explicitly overrides it. This avoids known Qwen/vLLM cases where thinking text can interfere with tool-call extraction.

Legacy XML is not supported for new public `RLM.completion()` runs. Existing
XML logs remain readable by the visualizer, and the parser compatibility code
is retained for old traces and tests.


### Supported LM Backends
The workspace substrate works with any LM backend supported by the RLM client library. We currently support:
- **API-based**: OpenAI (GPT-4, o1, etc.), Anthropic (Claude), Google (Gemini), Portkey, OpenRouter
- **Local models**: vLLM (via OpenAI-compatible interface), or any model behind an OpenAI API-compatible server

For detailed client implementations and to add support for more backends, see [`rlm/clients/`](https://github.com/alexzhang13/rlm/tree/main/rlm/clients).

## Relevant Reading
* **[Dec '25]** [Recursive Language Models arXiv](https://arxiv.org/abs/2512.24601)
* **[Oct '25]** [Recursive Language Models Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)

If you use this code or repository in your research, please cite:

```bibtex
@misc{zhang2026recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2026},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

## Optional: Trajectory metadata and logging
`RLMChatCompletion` has an optional `metadata` field (default `None`) that holds the full trajectory (run config + all iterations and sub-calls) so you can reconstruct the run. Pass an `RLMLogger` to capture it:

- **In-memory only** (trajectory on `completion.metadata`): `logger=RLMLogger()` (no `log_dir`).
- **Also save to disk** (JSONL for the visualizer): `logger=RLMLogger(log_dir="./logs")`.

## Optional Debugging: Visualizing RLM Trajectories
We provide a simple visualizer to inspect code, sub-LM, and root-LM calls. Use `RLMLogger(log_dir="./logs")` so each completion writes a `.jsonl` file:
```python
from rlm.logger import RLMLogger
from rlm import RLM

logger = RLMLogger(log_dir="./logs")
rlm = RLM(..., logger=logger)
```

To run the visualizer locally, we use Node.js and shadcn/ui:
```
cd visualizer/
npm run dev        # default localhost:3001
```

You'll have the option to select saved `.jsonl` files 
<p align="center">
  <img src="media/visualizer.png" alt="RLM Visualizer Example" width="800"/>
</p>

For terminal inspection, use the packaged text viewer:

```bash
rlm-trace logs/*.jsonl
rlm-trace logs/run.jsonl --turn 3 --prompt --response --actions --observations
rlm-trace logs/run.jsonl --turn 3 --all --children --max-chars 0
```
