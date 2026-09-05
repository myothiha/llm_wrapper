"""Google Gemini wrapper.

Uses the current `google-genai` SDK (`from google import genai`), not the
retired `google-generativeai` package:

    pip install google-genai

The API key is read from GEMINI_API_KEY (or GOOGLE_API_KEY) unless one is
passed. Get one at https://aistudio.google.com/apikey.

Exposes the same callable the rest of this repo expects —
``model(prompt, max_new_tokens=…) -> str`` — so it drops into
`BiomedicalKnowledgeGeneratorWithLLM` and the evaluation scripts unchanged.
Images may be passed as well, since Gemini is natively multimodal.
"""

import os

DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"

# Gemini 3 models think before answering, and thinking tokens are billed
# against max_output_tokens. A budget sized for the answer alone can therefore
# come back empty, so the wrapper raises rather than returning "" silently.
THINKING_MODEL_PREFIXES = ("gemini-3", "gemini-2.5")


class GeminiAIWrapper:
    """Call a Gemini model and get plain text back.

    Args:
        model_name: e.g. ``"gemini-3.1-pro-preview"`` (default) or
            ``"gemini-2.5-pro"`` for the current stable Pro tier.
        temperature: Sampling temperature.
        api_key: Overrides ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
        system_instruction: Optional system prompt applied to every call.
        thinking_budget: Thinking tokens to allow. ``None`` lets the model
            decide, ``0`` disables thinking where the model permits it.
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        temperature=0.2,
        api_key=None,
        system_instruction=None,
        thinking_budget=None,
    ):
        try:
            from google import genai
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "GeminiAIWrapper requires the 'google-genai' package: pip install google-genai"
            ) from error

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key. Set GEMINI_API_KEY in the root project's .env "
                "(or pass api_key=). Create one at https://aistudio.google.com/apikey"
            )

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction
        self.thinking_budget = thinking_budget

    @property
    def is_thinking_model(self):
        return self.model_name.startswith(THINKING_MODEL_PREFIXES)

    def _build_config(self, max_output_tokens, temperature):
        from google.genai import types

        config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if self.system_instruction:
            config["system_instruction"] = self.system_instruction
        if self.thinking_budget is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**config)

    @staticmethod
    def _build_contents(prompt, images):
        """Gemini takes a flat list of parts; PIL images are accepted directly."""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        return [prompt] + list(images) if images else prompt

    def __call__(self, prompt, images=None, **kwargs):
        max_output_tokens = kwargs.get("max_new_tokens", 2048)
        temperature = kwargs.get("temperature", self.temperature)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=self._build_contents(prompt, images),
            config=self._build_config(max_output_tokens, temperature),
        )

        text = getattr(response, "text", None)
        if text:
            return text

        raise RuntimeError(self._explain_empty(response, max_output_tokens))

    def _explain_empty(self, response, max_output_tokens):
        """Turn an empty completion into something actionable."""
        reason = None
        candidates = getattr(response, "candidates", None)
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)

        detail = f"{self.model_name} returned no text (finish_reason={reason})."
        if str(reason).endswith("MAX_TOKENS") and self.is_thinking_model:
            detail += (
                f" This is a thinking model and thinking tokens count against"
                f" max_output_tokens ({max_output_tokens}), so the budget was spent"
                f" before the answer began. Raise max_new_tokens, or construct the"
                f" wrapper with thinking_budget=0 to disable thinking."
            )
        elif str(reason).endswith("SAFETY"):
            detail += " The prompt or response was blocked by a safety filter."
        return detail

    def get_model_name(self):
        return self.model_name


if __name__ == "__main__":
    wrapper = GeminiAIWrapper()
    print(wrapper.get_model_name())
    print(wrapper("Name the four chambers of the heart in one sentence.", max_new_tokens=4096))
