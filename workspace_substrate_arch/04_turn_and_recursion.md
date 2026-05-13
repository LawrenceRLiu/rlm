# Turn-to-Turn Interface & Recursion
## Turn-to-Turn Interface

**Note** by turn we mean the internal turns between rounds of the RLM root, not the outer turns between user queries.

Each RLM turn should have a simple transcript shape:

1. The model receives the task, compact workspace instructions, recent observations, older observation receipts, action receipts, snapshots, and any valid short turn notes.
2. The model returns optional `<note>...</note>` intent text plus one or more action blocks, or a final answer action.
3. The runtime parses the block into `WorkspaceAction` objects.
4. The runtime executes the actions and records a full-fidelity iteration in the JSONL trajectory.
5. The next model turn is rebuilt from compact model-facing replay, not from the raw assistant response.

**Example transcript:**

    ====== Turn 1 ======  
    assistant:
    <note>Inspect the root task and workspace layout first.</note>

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
    [compact replay includes the turn 1 note, action receipts, and observations]
    <note>Use the task details to choose the next file/tool to inspect.</note>
    [new `workspace` action block or `final` action]

    user/runtime:
    [follow-up observation message]

    ...

    ====== Turn N ======
    assistant:
    [...]

    user/runtime:
    [follow-up observation message]

The raw transcript is logged, but it is not fed back verbatim forever. The model-facing replay is intentionally compact. Prior file-edit bodies are hidden behind receipts, recent observations are visible, older observations age into receipts, and snapshots summarize changed files. If the model needs exact file contents again, it must call `read_file`.

For short planning continuity, the model may include one bounded `<note>...</note>` before its actions. A valid note is replayed as:

```workspace
<turn_note turn="3">
Read _rlm_notes/proof.md next and verify the beta derivation.
</turn_note>
```

The note is for intent only. It should be short and should contain current plan, open questions, or file paths to revisit. It must not contain file contents, code blocks, proofs, large outputs, generated artifacts, or action XML. Overlong/content-like notes are replaced by an omitted-note receipt rather than truncated, so dumping content into a note is not rewarded.

The observation should be compact and bounded. Large outputs (Ie above a certain hyperparameter number of lines/words/charachters) should be written to `_rlm_artifacts/` and summarized with a file path and a note on the length (in lines or characters). This avoids filling the root message history while preserving inspectable state in the workspace. This truncation should operate on a per observation/tool call basis. Ie if the first tool call in the action block returns a large output, that and only that should be truncated, the rest of the tool calls (if they are not as long), should be included fully.

Prompt-history shaping currently has explicit knobs under `PromptHistoryConfig`:

- `full_observation_turns`: how many most-recent completed turns keep full observations in replay;
- `max_command_body_replay_chars`: cap for replaying `python`/`shell` source;
- `max_mutating_command_body_replay_chars`: smaller source cap when the command changed files;
- `max_mutating_command_stdout_replay_chars`: cap for stdout from mutating command actions;
- `max_turn_note_chars` / `max_turn_note_lines`: bounds for optional turn notes.

The runtime should reject malformed actions loudly with a parse observation that tells the model exactly what schema failed. This is important for small models. **Note** An interesting alternative would be to use constrained decoding to force the model to output a valid json block.

### Context Compaction

Like the vanilla RLM implementation, the workspace substrate can still accumulate long message histories over many turns. The current implementation already performs lightweight prompt-history shaping by rebuilding model-facing replay from receipts, aged observations, snapshots, and bounded notes. A heavier summary-based compaction mechanism can still be added later:

1. When the token count of the active `message_history` crosses a configured threshold, the runtime pauses and prompts the model to generate a dense summary of the entire trajectory up to that point.
2. The active context window is then "restarted" by replacing the prior replay receipts with this new summary.
3. However, unlike the vanilla REPL implementation—which stored the raw history in an ephemeral Python `history` variable—the workspace substrate writes the full, uncompacted action log to a durable file (e.g., `_rlm_state/action_log.jsonl` or `_rlm_notes/trajectory_archive.md`).

This ensures that the model can always use standard file tools (like `read_file`) to recover exact details from earlier in the run if it realizes the summary omitted crucial information, completely avoiding the permanent loss of context.

**Note:** The current receipt/aged-observation replay is already implemented. The heavier summary-generation variant described above remains future work.

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

**Note on Performance (Future Work):** While a naive `cp` with excludes is the default for MVP, copying massive directories (like datasets) is slow and I/O intensive. Future iterations of the workspace substrate should leverage **Docker OverlayFS**. By providing a read-only bind mount of the parent workspace (Lower layer) and an empty directory for the child (Upper layer), we can achieve instant, zero-copy child spawning with perfect isolation (Copy-on-Write).

### Example: Returning Child Artifacts

When a child has finished its task (e.g., compiling a dataset), it issues a `final` action with explicit artifact paths:

    ```workspace
    <action tool="final">
        <answer>I have downloaded, cleaned, and compiled the dataset. You can find it at _rlm_artifacts/cleaned_data.csv.</answer>
        <artifact path="_rlm_artifacts/cleaned_data.csv" />
        <artifact path="src/data_loader.py" />
    </action>
    ```

The runtime intercepts this action, halts the child, and copies those exact paths from the child's isolated container to the parent's directory structure under `_rlm_artifacts/children/<child_id>/` (where `<child_id>` follows the format `child_{turn_count}_{idx}`, we will construct this <child_id> on the fly by just seeing for the latest turn, how many children have already been recorded, and doing the last one + 1)

**Handling Path Invalidation:** Because the child writes its `answer` from the perspective of its own workspace, it will frequently reference the original paths (e.g., `_rlm_artifacts/cleaned_data.csv`). If the parent tries to read that exact path, it will read its own stale version of the file (or the file won't exist). To solve this without using brittle regex string replacements on the LLM's output, the runtime injects a **Path Mapping Table** into the parent's observation. This will be constructed on the fly from the artifact paths returned by the child and their new locations in the parent workspace. 

The parent RLM receives the following compact observation that wraps around the child's answer and the path mappings:
```
user/runtime:
    Observation: Child RLM completed.
    
    Answer: <child answer, example: I have downloaded, cleaned, and compiled the dataset. You can find it at _rlm_artifacts/cleaned_data.csv.> 
    
    [Runtime Note: The child's exported files have been safely isolated. Translate any paths mentioned in the answer above using this mapping:]
    Artifact Mapping:
    - _rlm_artifacts/cleaned_data.csv -> _rlm_artifacts/children/child_3_0/cleaned_data.csv
    - src/data_loader.py -> _rlm_artifacts/children/child_3_0/data_loader.py
```

This mechanism allows pointers to be passed instead of massive text blobs, keeping the parent's workspace safe from destructive overwrites while providing the LLM with explicitly clear path resolution logic.

### Maximum Depth Handling

The legacy Python REPL substrate handled maximum recursion depth by silently falling back from `rlm_query` to `llm_query`. In the Workspace Substrate, we adhere to the "fail fast, fail loud" philosophy to prevent unexpected behavior:

1. **System Prompt Omission (Prevention):** When initializing a workspace at `max_depth`, the `rlm_query` tool is dynamically omitted from the system prompt's tool list. The prompt is modified to explicitly state: *"Note: You are at maximum recursion depth. You cannot spawn further child RLMs. You must solve the task directly or use `llm_query`."*
2. **Runtime Rejection (Correction):** If the model hallucinates a `<action tool="rlm_query">` block, the runtime does **not** silently route it to `llm_query`. Instead, it immediately halts the tool execution and returns a loud observation error: `Error: Maximum recursion depth reached. The 'rlm_query' tool is unavailable.`

This prevents the model from assuming a child is actively doing file-system work (which a plain `llm_query` cannot do) and forces it to adapt its strategy.
