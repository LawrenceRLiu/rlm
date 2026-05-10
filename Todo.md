# ToDOs for this fork for the RLMs

## Analysis
Previously we ran a short analysis of RLMs on OOLONG for the more modern models, ie Qwen3.6 and Gemma 4. We found that while Gemma 4 largely can follow the RLM format, the Qwen3.6 models (both 27B dense and 35B-A3B MoE could not). At the same time, the Gemma 4 model was able to follow the RLM format. 


## Very General
- [ ] Make the RLM format more easy to follow for a LLM, thus hopefully enabling Qwen3.6/Qwen3.5 models, hopefully at a size of around 9B natively, this would open us up for the next step of the project
- [ ] Enable RLMs to be used in a more general way, ie ready support of coding, math, web research, and other such tasks.

## How to establish that:

- [ ] Change the substrate from a python based REPL to a generalized workspace with prebuilt tools for web search
    - [x] Conduct a high level sketch of how this would work, store it in `./workspace_substrate_arch/` (originally `workspace_sketch.md`) also analyze the RLM prompt/systems and see if we can simplify the setup with these new changes. Hopefully this would enable us to also explore things at 7-9B parameter scales, which would be very exciting.
    - [ ] Implement the workspace, and test it on a simple RLM task, such as a simple web search task, or a simple coding task, and see if the model can follow
    - [ ] Benchmark 
        - [ ] For coding: SWE-Bench, Terminal Bench, etc 
        - [ ] For web search: WebArena, ToDO: find the deep research benchmarks etc. 
        - [ ] For math: AIME 2025 (search disabled), and maybe some other ones 

# Helper Functions/things that would be nice
- [ ] Integration with sglang (if the system prompt is long enough that radix attention would make a runtime difference)
- [ ] Integration with openrouter so we can eval a wider variety of models and also easily switch between them.

# Bumps from first end-to-end run (2026-05-09, Gemma 4 31B)
Surfaced by the four smoke tasks documented in `workspace_substrate_arch/dev/2026-05-09_first_run_traces_gemma-4-31B-it.md`. Setup details in `workspace_substrate_arch/SETUP.md`.

- [ ] **Add `pytest` (and probably `pytest-asyncio`, `pip` upgrades) to the workspace Docker image.** Right now the model has to `pip install pytest` mid-run when a task says "run the tests" — costs a turn, doesn't persist across runs since each run gets a fresh container. Edit `docker/workspace.Dockerfile`'s pip-install list. (3b turn 2 — `_setup_runs/logs/rlm_2026-05-09_17-12-59_48e29be2.jsonl`.)
- [ ] **Run the workspace container as the host user, not root.** Files the model writes land `root:root`, so cleaning up kept workspaces (`cleanup_mode="keep"`) requires `sudo`. Pass `--user $(id -u):$(id -g)` in `rlm/environments/docker_workspace.py:222-256`; may also need to chown `/workspace` inside the image or fix mount ownership.
- [ ] **Populate `observations[].rlm_calls` with child trajectories.** When the parent calls `rlm_query`, the parent's JSONL observation has the artifact list but `rlm_calls` is `[]` — the child's per-turn actions/observations don't appear anywhere in the log. The visualizer already has a "Sub-LM Calls" section in `ActionCard` waiting for this data, so once the producer fills it, the UI will render it for free. Alternative: write each child to its own JSONL under `log_dir/children/`. (3c turn 2 — `_setup_runs/logs/rlm_2026-05-09_17-14-57_5d23f2bb.jsonl`.)
- [ ] **Visualizer can't drill into child trajectories.** Direct downstream of the above — even if you load the parent's JSONL into the visualizer, you can't navigate into what each `rlm_query` child did because the data isn't there. Track separately so we don't forget the UI side once the producer is fixed (no UI work likely needed, but worth verifying).
- [ ] **Build end-to-end tests that *force* each substrate guard rule to fire.** Our smoke runs were "happy-path" enough that several documented rules were never exercised. Each one needs a deliberate test:
    - `mutating-tool failure halts the rest of the batch` — the rule wasn't actually hit in 3b because the model only emitted one mutating action per turn. Test: a turn that emits `<action tool="shell">false</action>` then `<action tool="write_file" path="x">x</action>` and assert the write didn't run.
    - Observation spill (`_rlm_artifacts/_observations/`) when output > `observation.max_observation_chars` — never tripped in our 4 runs. Test: a turn that emits `<action tool="shell">cat _rlm_query_0.txt</action>` against a >16 KB seed file and assert the spill file appears with the original bytes.
    - `rlm_query` at `depth >= max_depth` returning a loud error observation — exercised in unit tests but not end-to-end with a real LM.
    - Parse-retry inner loop hitting `parse.max_action_parse_retries` — Gemma 4 had 0 parse retries across all 4 runs, so the retry-exhaustion path is untested with a real LM.
    - Stop conditions other than `final` — `max_iterations`, `max_budget`, `max_timeout`, `max_tokens`, `max_errors`. None of these tripped in our runs. Each deserves a deliberate test that drives the loop into the limit.