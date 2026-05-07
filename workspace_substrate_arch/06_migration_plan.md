# Migration Plan & Feasibility

## Current Constraint

The current codebase is REPL-first:

- `BaseEnv` exposes `setup`, `load_context`, and `execute_code`.
- The main RLM loop parses only fenced `repl` blocks and sends each block to `environment.execute_code`.
- Results are represented as `REPLResult`, `CodeBlock`, and `RLMIteration`.
- The prompt teaches `context`, `history`, `llm_query`, `rlm_query`, `SHOW_VARS`, and `FINAL_VAR`.
- Persistence means versioned Python variables, not durable workspace state.

The migration should preserve the useful shape of the loop but replace the action substrate underneath it. **Note** Because this is a fork and because we are moving quick and dirty, do not worry about making this backwards compatible. This is a separate repo / project entirely. We can break things!


## Things we plan to deprecate for now:
- Modal and Prime Intellect sandboxes, only docker will be supported for now. Also disable bare running without a docker, because we are doing workspace level tasks, i don't want to clean up messy stuff and potentiall breaking stuff. 

## Things we want to keep and not break:
- Visualization (this will be CRITICAL, if we can take snapshots at every turn that would be great!)
- base RLM loop logic, 

