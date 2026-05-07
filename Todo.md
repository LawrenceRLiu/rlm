# ToDOs for this fork for the RLMs

## Analysis
Previously we ran a short analysis of RLMs on OOLONG for the more modern models, ie Qwen3.6 and Gemma 4. We found that while Gemma 4 largely can follow the RLM format, the Qwen3.6 models (both 27B dense and 35B-A3B MoE could not). At the same time, the Gemma 4 model was able to follow the RLM format. 


## Very General
- [ ] Make the RLM format more easy to follow for a LLM, thus hopefully enabling Qwen3.6/Qwen3.5 models, hopefully at a size of around 9B natively, this would open us up for the next step of the project
- [ ] Enable RLMs to be used in a more general way, ie ready support of coding, math, web research, and other such tasks.

## How to establish that:

- [ ] Change the substrate from a python based REPL to a generalized workspace with prebuilt tools for web search
    - [ ] Conduct a high level sketch of how this would work, store it in `./workspace_sketch.md` also analyze the RLM prompt/systems and see if we can simplify the setup with these new changes. Hopefully this would enable us to also explore things at 7-9B parameter scales, which would be very exciting.
    - [ ] Implement the workspace, and test it on a simple RLM task, such as a simple web search task, or a simple coding task, and see if the model can follow
    - [ ] Benchmark 
        - [ ] For coding: SWE-Bench, Terminal Bench, etc 
        - [ ] For web search: WebArena, ToDO: find the deep research benchmarks etc. 
        - [ ] For math: AIME 2025 (search disabled), and maybe some other ones 

# Helper Functions/things that would be nice
- [ ] Integration with sglang (if the system prompt is long enough that radix attention would make a runtime difference)
- [ ] Integration with openrouter so we can eval a wider variety of models and also easily switch between them.