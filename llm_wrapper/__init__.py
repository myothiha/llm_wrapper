"""Wrapper exports.

Each wrapper pulls in its own backend SDK, and no single environment can
satisfy all of them at once (the Gemma wrapper needs transformers>=4.50 for
Gemma3ForConditionalGeneration, while spacy-transformers pins transformers
<4.50). Importing a wrapper whose backend is absent therefore leaves the name
bound to None instead of breaking the whole package for every other caller.
"""

try:
    from .claude_ai_wrapper import ClaudeAIWrapper
except ImportError:
    ClaudeAIWrapper = None

try:
    from .gemma_lvlm_wrapper import LVLMWrapper as GemmaLVLMWrapper
except ImportError:
    GemmaLVLMWrapper = None

try:
    from .gpt_llm_wrapper import GPTLLMWrapper
except ImportError:
    GPTLLMWrapper = None

try:
    from .mistral_ai_wrapper import MistralAIWrapper
except ImportError:
    MistralAIWrapper = None

__all__ = [
    "ClaudeAIWrapper",
    "GemmaLVLMWrapper",
    "GPTLLMWrapper",
    "MistralAIWrapper",
]