from rlm.core.rlm import RLM
from rlm.environments.local_repl import LocalREPL
from rlm.clients.openai import OpenAIClient

if __name__ == "__main__":
    import os
    env = LocalREPL()
    client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", "dummy"), model_name="gpt-4o")
    rlm = RLM(environment=env, lm_client=client)
    print("Successfully instantiated RLM")
