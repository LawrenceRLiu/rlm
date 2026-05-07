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

### Context Compaction

Like the vanilla RLM implementation, the workspace substrate will inevitably accumulate long message histories over many turns. To mitigate this, we implement a context compaction mechanism:

1. When the token count of the active `message_history` crosses a configured threshold, the runtime pauses and prompts the model to generate a dense summary of the entire trajectory up to that point.
2. The active context window is then "restarted" by replacing the prior history with this new summary.
3. However, unlike the vanilla REPL implementation—which stored the raw history in an ephemeral Python `history` variable—the workspace substrate writes the full, uncompacted action log to a durable file (e.g., `_rlm_state/action_log.jsonl` or `_rlm_notes/trajectory_archive.md`).

This ensures that the model can always use standard file tools (like `read_file`) to recover exact details from earlier in the run if it realizes the summary omitted crucial information, completely avoiding the permanent loss of context.

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
7. Child returns final answer, trajectory metadata, and optionally a list of produced artifacts. This will be extracted from the `FINAL_VAR` in the child's assistant message and append to the observation.
8. Parent receives a compact observation containing the child answer and artifact paths.

The child must not mutate the parent workspace directly. Copying the parent workspace gives the child full context without introducing live shared state. If the parent wants to use child artifacts, the runtime can either copy selected child outputs back into `_rlm_artifacts/children/<child_id>/` or expose a follow-up action such as `import_child_artifact`.

This default is intentionally simple. Explicit artifact selection can be added later as an optimization when workspaces become too large or when benchmark isolation needs stricter control.
