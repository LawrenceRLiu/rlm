# Workspace State & Actions

## Workspace State

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

## Action Format

The model interacts with the workspace by outputting structured action blocks. Because embedding multi-line code inside JSON strings is notoriously difficult and breaks whitespace, **we exclusively use XML for all actions.**

XML natively preserves formatting, is extremely robust for LLM generation, and provides a unified interface for both simple reads and complex code writes.

### Action Format (XML)

For simple reads or commands, use empty elements with attributes:

```workspace
<action tool="read_file" path="_rlm_query_0.txt" />
<action tool="web_search" query="rlm agents architecture" />
```

For actions requiring multi-line strings where whitespace matters (like `write_file`, `edit_file`, `python`), put the content inside the element:

```workspace
<action tool="write_file" path="script.py">
def process_data():
    with open("data.txt") as f:
        return f.read()
</action>
```

### Execution Semantics

We explicitly classify tools to manage batching safely:
1. **Read-Only Actions** (`read_file`, `list_files`, `web_search`,etc): Can be safely batched by providing multiple `<action>` blocks sequentially. The runtime executes them and may even parallelize them safely under the hood.
2. **State-Mutating Actions** (`write_file`, `edit_file`, `shell`, `python`, `rlm_query`): Must be executed sequentially. We allow the model to batch them (e.g., scaffolding three files at once in a single generation), but the runtime **MUST halt execution immediately** upon the first error and return the partial result to avoid cascading failures.

### Initial tools:

- `list_files`: list workspace paths (shallow by default like `ls`) to prevent context bloat, filtering out noise (`.git`, `__pycache__`). Includes metadata on who created/modified each file.
- `read_file`: read a bounded file slice.
- `write_file`: create or completely overwrite a workspace file. Use this for new files or small complete rewrites.
- `append_file`: append notes or results. Highly recommended for keeping iterative logs/scratchpads without rewriting the whole file.
- `edit_file`: (Replaces brittle diffs) Search-and-replace block for targeted edits. Avoids line-number drift. Format: `<action tool="edit_file" path="x.py"><search>old</search><replace>new</replace></action>`. **Rule:** The `<search>` block must be a unique substring. If it finds 0 or >1 matches, the tool fails and returns an error unless an `allow_multiple="true"` attribute is provided.
- `shell`: run a command inside the Docker workspace.
- `python`: convenience wrapper for running Python inside Docker. **Note:** The runtime injects `llm_query` and `rlm_query` functions into this environment. If the model needs to programmatically construct complex prompts or map over files before calling an LLM, it should write a Python script and call `rlm_query_batched` (or `llm_query_batched` for simpler tasks that don't require RLM base decomposition) directly from Python. (Python is strongly preferred over Bash here because Bash escaping/quoting for multiline prompts and JSON IO is incredibly fragile).
- `web_search`: host-backed search tool returning compact structured results.
- `fetch_url`: host-backed URL fetch with bounded text extraction.
- `llm_query`: plain LM completion for simple string prompts.
- `rlm_query`: recursive child RLM call for simple string prompts.
- `final`: submit final answer.

The model should see only short tool descriptions. Full tool schemas should stay in code/tests, not in the base prompt.

## Web Search

Web search should be a first-class host tool, not a Python package the model has to import.

Recommended behavior:

- `web_search(query, max_results={n})` returns titles, URLs, snippets, and source metadata. This should be done using the brave API (key will be put in `.env` at the end)
- `fetch_url(url)` returns bounded extracted text plus page metadata.
- Observations should be concise by default.
- Large pages should be saved into `_rlm_artifacts/` and returned as file paths.

This keeps web research compatible with Docker while avoiding API keys and scraping logic inside model-generated code.
