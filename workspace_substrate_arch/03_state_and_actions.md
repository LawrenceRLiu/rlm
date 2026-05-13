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

### Transcript vs Workspace Memory

The chat transcript is not treated as durable memory. The runtime logs raw model responses and action bodies for humans/evals, but the model's next-turn prompt receives only compact replay:

- file-edit action bodies are replaced by receipts;
- older observations are replaced by receipts;
- command output from mutating `python`/`shell` actions is capped;
- a short `<note>` can carry intent, but not large content.

This prevents `write_file` bodies or one-time `read_file` outputs from silently becoming permanent context. If the model needs file contents again, it should explicitly inspect the workspace with `read_file`.

### File Provenance Metadata

To help the model reason about the workspace without needing to memorize its entire action history, the runtime tracks file provenance. When files are listed or read, they are tagged with `created` and `modified` metadata based on the following roles:

- **`user`**: Files that existed at the start of the current RLM's execution. This includes the initial codebase, human-provided files, and files provided as context from a parent RLM. (From the perspective of any RLM, the caller is always the "user").
- **`assistant`**: Files created or directly modified by the current RLM using explicit file-system tools (`write_file`, `edit_file`, `append_file`).
- **`system`**: Files generated indirectly via command execution (`shell` or `python` tools), or internal runtime state files (e.g., `_rlm_query_0.txt`).
- **`child`**: Artifacts explicitly returned and imported into the workspace from a recursive `rlm_query` call.

**Tracking Mechanism:** The runtime implements this by taking a fast directory snapshot (recording file paths, sizes, and `mtime` modification timestamps) *before* and *after* each action is executed. 
- If a file's `mtime` changes during the execution of an explicit file tool (e.g., `write_file`), the runtime tags it as `assistant`.
- If a file's `mtime` changes during the execution of a `shell` or `python` command, the runtime assumes the tool generated/modified it and tags it as `system`.

## Action Format

The model interacts with the workspace by outputting structured action blocks. Because embedding multi-line code inside JSON strings is notoriously difficult and breaks whitespace, **we exclusively use XML for all actions.**

XML natively preserves formatting, is extremely robust for LLM generation, and provides a unified interface for both simple reads and complex code writes.

**Parsing Strategy (Important):** Standard XML parsers will crash if the assistant writes code containing unescaped characters like `<` or `&` (e.g., `if a < b:`). Therefore, the host-side parser must NOT be a strict XML parser. It must be a **tolerant, regex-based extractor** that simply finds the `<action ...>` opening tag and the `</action>` closing tag, and extracts the raw string between them. This prevents parsing failures when handling raw Python, Bash, or HTML files.

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

### Optional Turn Notes

The model may include one short note before its actions:

```workspace
<note>Next inspect _rlm_notes/proof.md and fix the beta derivation.</note>
<action tool="read_file" path="_rlm_notes/proof.md" start_line="1" end_line="120" />
```

The note is not a general scratchpad. It is a bounded intent anchor for the next turns: current plan, open questions, and file paths to revisit. The runtime omits notes that are too long, span too many lines, contain code fences, or contain action XML. Durable findings still belong in workspace files.

### Execution Semantics

We explicitly classify tools to manage batching safely:
1. **Read-Only Actions** (`read_file`, `list_directory`, `web_search`,etc): Can be safely batched by providing multiple `<action>` blocks sequentially. The runtime executes them and may even parallelize them safely under the hood. **If a read-only action fails (e.g., file not found), the runtime does NOT halt.** It simply records the error for that specific tool call and continues executing the remaining read-only actions in the batch.
2. **State-Mutating Actions** (`write_file`, `edit_file`, `shell`, `python`, `rlm_query`): Must be executed sequentially. We allow the model to batch them (e.g., scaffolding three files at once in a single generation), but the runtime **MUST halt execution immediately** upon the first error and return the partial result to avoid cascading failures.

## Tool Interface & Definitions

The workspace provides a set of initial tools to the model. While the model should only see short tool descriptions in its base prompt (with full schemas remaining in code/tests), the actual implementation and capability of these tools are defined below. 

### 1. `list_directory`
- **Description**: Lists workspace paths to prevent context bloat. Filters out noise like `.git` and `__pycache__`.
- **Arguments**:
  - `path` (optional): The directory to list. Defaults to the workspace root.
- **Behavior**: Returns a compact, shallow file-tree representation of the directory contents, including inline metadata about who created and last modified each item. Will include all the folders in this directory, but not their contents.
- **Example Usage**:
  ```workspace
  <action tool="list_directory" path="_rlm_artifacts" />
  ```  

### 2. `read_file`
- **Description**: Reads a bounded slice of a file to avoid context window explosion.
- **Arguments**:
  - `path` (required): The path to the file.
  - `start_line` (optional): The 1-indexed line to start reading from. Defaults to 1.
  - `end_line` (optional): The 1-indexed line to stop reading at. Defaults to bounded max (e.g., 500 lines).
- **Behavior**: Returns the contents of the specified file slice along with a header containing the total line count and inline metadata (created by, last modified by).
- **Example Usage**:
  ```workspace
  <action tool="read_file" path="src/main.py" start_line="10" end_line="50" />
  ```

### 3. `write_file`
- **Description**: Creates a new file or completely overwrites an existing workspace file. 
- **Arguments**:
  - `path` (required): The path to the file.
  - `content` (implicit text content): The text to write to the file.
- **Behavior**: Writes the provided content directly to the file, replacing anything that was there. Use this for new files or small complete rewrites. The body is stored in the full JSONL trajectory but is not replayed verbatim to the model on later turns; later prompt history contains only a receipt. Use `read_file` to inspect what was written.
- **Example Usage**:
  ```workspace
  <action tool="write_file" path="scripts/run.sh">
  #!/bin/bash
  echo "Hello World"
  </action>
  ```

### 4. `append_file`
- **Description**: Appends text to an existing file. Highly recommended for keeping iterative logs or scratchpads without rewriting the whole file.
- **Arguments**:
  - `path` (required): The path to the file.
  - `content` (implicit text content): The text to append to the file.
- **Behavior**: Adds the content to the end of the file. Creates the file if it does not exist. As with `write_file`, the appended body is logged for traceability but later prompt replay contains only a receipt.
- **Example Usage**:
  ```workspace
  <action tool="append_file" path="_rlm_notes/scratch.md">
  ## New Finding
  The parser fails on edge case X.
  </action>
  ```

### 5. `edit_file`
- **Description**: Search-and-replace block for targeted edits. Replaces brittle unified diffs and avoids line-number drift.
- **Arguments**:
  - `path` (required): The path to the file.
  - `allow_multiple` (optional): Boolean, defaults to `false`. If `true`, replaces all occurrences.
  - `<search>` (child element): The exact substring to find.
  - `<replace>` (child element): The new substring to replace it with.
- **Behavior**: **Rule:** The `<search>` block must be a unique substring in the file. If it finds 0 or >1 matches, the tool fails and returns an error unless `allow_multiple="true"` is provided. Failed edit observations include the reason (for example, search text not found or multiple matches). Successful edit bodies are logged but later prompt replay contains only a receipt.
- **Example Usage**:
  ```workspace
  <action tool="edit_file" path="src/processor.py" allow_multiple="false">
  <search>
  def process_data(data):
      if not data:
          return None
      return data.lower()
  </search>
  <replace>
  def process_data(data):
      if not data:
          return None
      
      # Now safely strip and lower the data
      cleaned = data.strip().lower()
      return cleaned
  </replace>
  </action>
  ```

### 6. `shell`
- **Description**: Runs a bash command inside the Docker workspace for inspection, tests, build steps, and diagnostics.
- **Arguments**:
  - `command` (implicit text content): The bash command to run.
- **Behavior**: Executes the command, capturing `stdout`, `stderr`, and the exit code. `shell` is intentionally described as a scratch/validation tool, not the preferred path for ordinary durable edits. The command source may be replayed in later prompts (capped), while stdout from commands that changed files is capped more aggressively.
- **Example Usage**:
  ```workspace
  <action tool="shell">pytest tests/test_core.py</action>
  ```

### 7. `python`
- **Description**: Convenience wrapper for running scratch Python inside Docker for computation, parsing, validation, tests, and diagnostics.
- **Arguments**:
  - `code` (implicit text content): The python script to execute.
- **Behavior**: Runs the Python code. `python` should not be used as a substitute for simple durable file edits or for printing large generated artifacts. Use file tools for durable edits and `read_file` for inspection. The script source may be replayed in later prompts (capped), while stdout from scripts that changed files is capped more aggressively.
  - **HTTP Broker Requirement:** Because the Python script runs inside an isolated Docker container without API keys, this requires an HTTP Broker running inside the container (polled by the host) to proxy the LLM requests securely.
  - **Batching & Concurrency:** When a script calls `rlm_query_batched` with many prompts, the runtime enforces a top-level hyperparameter (`MAX_CONCURRENT_RLMS`, e.g., N=5). It will spin up N child workspaces at a time, wait for them to finish, and then spin up the next N, accepting the wall-time slowdown to prevent host resource exhaustion.
- **Example Usage**:
  ```workspace
  <action tool="python">
  import json
  data = json.loads(open('data.json').read())
  print(f"Processed {len(data)} items")
  </action>
  ```

#### 7.1. Python Injected Functions

The runtime injects the following helper functions into the Python namespace so the model can programmatically construct prompts or map over files:

- `def llm_query(prompt: str) -> str:` Plain string-in, string-out LLM completion.
- `def rlm_query(prompt: str) -> str:` Spawns a child workspace. Exported artifacts are saved to `_rlm_artifacts/children/child_{turn}_{idx}/`. Returns the exact structured observation string (containing the child's text answer and the artifact path mapping table) so the Python script can read the answer and locate the files.
- `def llm_query_batched(prompts: list[str]) -> list[str]:` Executes `llm_query` over a list of strings. Returns a list of string responses in the same order.
- `def rlm_query_batched(prompts: list[str]) -> list[str]:` Executes `rlm_query` over a list of strings. Returns a list of structured observation strings in the same order.

### 8. `llm_query`
- **Description**: Plain LM completion for simple string prompts.
- **Arguments**:
  - `prompt` (implicit text content): The prompt to send to the LLM.
- **Behavior**: Returns a direct completion without REPL/action loop iteration.
- **Example Usage**:
  ```workspace
  <action tool="llm_query">Summarize the above error logs.</action>
  ```

### 9. `rlm_query`
- **Description**: Recursive child RLM call for tasks requiring multi-step reasoning or file system interaction.
- **Arguments**:
  - `prompt` (implicit text content): The prompt to send to the child RLM.
- **Behavior**: Spawns a child RLM that has its own action loop to complete the task before returning the final answer to the parent.
- **Example Usage**:
  ```workspace
  <action tool="rlm_query">Investigate the root cause of the memory leak in src/engine.py</action>
  ```

### 10. `final`
- **Description**: Submit final answer to conclude the task. Can optionally attach file paths to return as artifacts to the caller (either a parent RLM or the human user).
- **Arguments**:
  - `answer` (child element or implicit text content): The final response.
  - `<artifact>` (optional child elements): Explicit file paths to be passed back.
- **Behavior**: Halts the RLM loop and returns the answer. If this is a child RLM, it copies any specified artifacts back to the parent workspace (under `_rlm_artifacts/children/<child_id>/`). If this is the root RLM, the specified artifacts are surfaced directly to the human user alongside the final text answer.
- **Example Usage**:
  ```workspace
  <action tool="final">
      <answer>The bug was caused by an off-by-one error. I have fixed it in src/utils.py.</answer>
      <artifact path="src/utils.py" />
  </action>
  ```

---

## Deferred / Disabled Tools

*Note: The following tools are currently disabled. We are focusing on getting the core workspace tools working first, as the engineering and implementation details for web integration need further examination.* Forgoe implementing these for now.

### 11. `web_search` (Disabled)
- **Description**: Host-backed search tool returning compact structured results. Web search should be a first-class host tool, not a Python package the model has to import. This keeps web research compatible with Docker while avoiding API keys and scraping logic inside model-generated code.
- **Arguments**:
  - `query` (required): The search query (can be provided as an attribute or child element).
  - `max_results` (optional): Int, number of results to return.
- **Behavior**: Returns titles, URLs, snippets, and source metadata. This should be done using the Brave API (key will be put in `.env` on the host side). Observations should be concise by default.
- **Example Usage**:
  ```workspace
  <action tool="web_search" query="RLM agents architecture" max_results="5" />
  ```

### 12. `fetch_url` (Disabled)
- **Description**: Host-backed URL fetch with bounded text extraction.
- **Arguments**:
  - `url` (required): The URL to fetch.
- **Behavior**: Returns bounded extracted text plus page metadata. To handle large pages gracefully, content should be automatically saved into `_rlm_artifacts/` and returned as file paths to prevent context window bloat.
- **Example Usage**:
  ```workspace
  <action tool="fetch_url" url="https://example.com/docs" />
  ```
