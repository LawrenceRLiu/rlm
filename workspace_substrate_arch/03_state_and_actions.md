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

## Web Search

Web search should be a first-class host tool, not a Python package the model has to import.

Recommended behavior:

- `web_search(query, max_results={n})` returns titles, URLs, snippets, and source metadata. This should be done using the brave API (key will be put in `.env` at the end)
- `fetch_url(url)` returns bounded extracted text plus page metadata.
- Observations should be concise by default.
- Large pages should be saved into `_rlm_artifacts/` and returned as file paths.

This keeps web research compatible with Docker while avoiding API keys and scraping logic inside model-generated code.
