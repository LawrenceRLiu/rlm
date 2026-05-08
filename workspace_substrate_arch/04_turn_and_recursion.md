# Turn-to-Turn Interface & Recursion
## Turn-to-Turn Interface

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
    <action tool="list_directory" path="." />
    <action tool="read_file" path="_rlm_query_0.txt" />
    ```

    user/runtime:
    Observation:
    - list_directory [path="."]:
      [Directory: .]
      ├── _rlm_query_0.txt (file | created: system | modified: system)
      ├── _rlm_notes/ (dir | created: system | modified: assistant)
      ├── src/ (dir | created: user | modified: user)
      ... (9 other items)
    - read_file [path="_rlm_query_0.txt"]:
      [File: _rlm_query_0.txt | Lines: 1-1/1 | Created: system | Modified: system]
      "Please investigate the error in..."

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

This transcript gets fed into the next model call, and so on. **IMPORTANT**: We need the model to output something above and below the workspace block. This should naturally be a summary of what the model plans to approach the problem etc, ie it would be able to provided anchoring for the next model call.

The observation should be compact and bounded. Large outputs (Ie above a certain hyperparameter number of lines/words/charachters) should be written to `_rlm_artifacts/` and summarized with a file path and a note on the length (in lines or characters). This avoids filling the root message history while preserving inspectable state in the workspace. This truncation should operate on a per observation/tool call basis. Ie if the first tool call in the action block returns a large output, that and only that should be truncated, the rest of the tool calls (if they are not as long), should be included fully.

The runtime should reject malformed actions loudly with a parse observation that tells the model exactly what schema failed. This is important for small models. **Note** An interesting alternative would be to use constrained decoding to force the model to output a valid json block.

### Context Compaction

Like the vanilla RLM implementation, the workspace substrate will inevitably accumulate long message histories over many turns. To mitigate this, we implement a context compaction mechanism:

1. When the token count of the active `message_history` crosses a configured threshold, the runtime pauses and prompts the model to generate a dense summary of the entire trajectory up to that point.
2. The active context window is then "restarted" by replacing the prior history with this new summary.
3. However, unlike the vanilla REPL implementation—which stored the raw history in an ephemeral Python `history` variable—the workspace substrate writes the full, uncompacted action log to a durable file (e.g., `_rlm_state/action_log.jsonl` or `_rlm_notes/trajectory_archive.md`).

This ensures that the model can always use standard file tools (like `read_file`) to recover exact details from earlier in the run if it realizes the summary omitted crucial information, completely avoiding the permanent loss of context.

**Note:** The implementation of this feature can be delayed for a bit. For now we can focus on everything else first.

## Prompt Shape

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

## Recursion

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
7. Child returns its final answer and optionally explicitly defines a list of produced artifacts using the `<action tool="final">` block.
8. Runtime extracts the requested artifacts and copies them from the child workspace into the parent workspace under `_rlm_artifacts/children/<child_id>/`.
9. Parent receives a compact observation containing the child answer and the parent-relative paths to the copied artifacts.

The child must not mutate the parent workspace directly. Copying the parent workspace gives the child full context without introducing live shared state. The explicit selection of artifacts via the `final` tool guarantees that massive intermediate scratch files or logs are ignored by default.

### Example: Returning Child Artifacts

When a child has finished its task (e.g., compiling a dataset), it issues a `final` action with explicit artifact paths:

    ```workspace
    <action tool="final">
        <answer>I have downloaded, cleaned, and compiled the dataset.</answer>
        <artifact path="_rlm_artifacts/cleaned_data.csv" />
        <artifact path="src/data_loader.py" />
    </action>
    ```

The runtime intercepts this action, halts the child, and copies those exact paths from the child's isolated container to the parent's directory structure under `_rlm_artifacts/children/<child_id>/`.

The parent RLM then receives the following compact observation, avoiding context window bloat:

    user/runtime:
    Observation: Child RLM completed.
    Answer: I have downloaded, cleaned, and compiled the dataset.
    Artifacts imported:
    - _rlm_artifacts/children/child_42/cleaned_data.csv
    - _rlm_artifacts/children/child_42/data_loader.py

This mechanism allows pointers to be passed instead of massive text blobs, enabling smooth multi-modal and structured data workflows across recursion depths.

### Maximum Depth Handling

The legacy Python REPL substrate handled maximum recursion depth by silently falling back from `rlm_query` to `llm_query`. In the Workspace Substrate, we adhere to the "fail fast, fail loud" philosophy to prevent unexpected behavior:

1. **System Prompt Omission (Prevention):** When initializing a workspace at `max_depth`, the `rlm_query` tool is dynamically omitted from the system prompt's tool list. The prompt is modified to explicitly state: *"Note: You are at maximum recursion depth. You cannot spawn further child RLMs. You must solve the task directly or use `llm_query`."*
2. **Runtime Rejection (Correction):** If the model hallucinates a `<action tool="rlm_query">` block, the runtime does **not** silently route it to `llm_query`. Instead, it immediately halts the tool execution and returns a loud observation error: `Error: Maximum recursion depth reached. The 'rlm_query' tool is unavailable.`

This prevents the model from assuming a child is actively doing file-system work (which a plain `llm_query` cannot do) and forces it to adapt its strategy.
