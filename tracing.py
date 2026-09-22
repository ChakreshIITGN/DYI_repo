"""
All Langfuse code lives here. One class, four methods.

    A TRACE      = one run of your program.
    A GENERATION = one LLM call inside it.
    Nesting them is what draws the tree in the dashboard.
"""

import config


class Tracer:
    def __init__(self):
        from langfuse import Langfuse
        self.client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            base_url=config.LANGFUSE_HOST,
        )

    def trace(self, name):
        """The parent span. Returns Langfuse's own object, so use it with `with`."""
        return self.client.start_as_current_observation(as_type="span", name=name)

    def record(self, name, model, prompt, output,
               tokens_in=0, tokens_out=0, metadata=None):
        """One finished LLM call, nested under whatever trace is open."""
        with self.client.start_as_current_observation(
                as_type="generation", name=name, model=model) as gen:
            gen.update(
                input=prompt,
                output=output,
                usage_details={"input": tokens_in, "output": tokens_out},
                metadata=metadata or {},
            )

    def flush(self):
        """Langfuse sends in a background thread. Without this, a short script
        exits before anything is sent and the dashboard stays empty."""
        self.client.flush()


def make_tracer():
    return Tracer()