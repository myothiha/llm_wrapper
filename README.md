# llm_loader

A lightweight library for loading and interacting with LLMs through various backends.

## Backends

| Backend | README |
|---|---|
| Ollama | [ollama/README.md](ollama/README.md) |

## Quick Start

Add the required variables to the **root project's `.env`** file. See each backend's README for the full list of variables needed.

## Wrappers

| Wrapper | Backend | Install | Credentials |
|---|---|---|---|
| `ClaudeAIWrapper` | Anthropic API | `pip install anthropic` | `ANTHROPIC_API_KEY` |
| `GeminiAIWrapper` | Google Gemini API | `pip install google-genai` | `GEMINI_API_KEY` |
| `GPTLLMWrapper` | local HF weights | `transformers` | — |
| `GemmaLVLMWrapper` | local Gemma 3 weights | `transformers>=4.50` | — |
| `MistralAIWrapper` | Mistral API | `pip install mistralai` | `MISTRAL_API_KEY` |
| `OllamaWrapper` | Ollama server | — | `ollama_host` |

Every wrapper is the same callable: `model(prompt, max_new_tokens=…) -> str`.
The multimodal ones also accept `images=[pil_image, …]`.

### Gemini

```python
from llm_wrapper import GeminiAIWrapper

llm = GeminiAIWrapper()                                  # gemini-3.1-pro-preview
llm = GeminiAIWrapper(model_name="gemini-2.5-pro")       # current stable Pro
print(llm("Define pleural effusion.", max_new_tokens=4096))
```

Gemini 3 and 2.5 models **think before answering, and thinking tokens count
against `max_output_tokens`**. A budget sized for the answer alone can come back
empty; the wrapper raises with that explanation rather than returning `""`.
Either raise `max_new_tokens` or pass `thinking_budget=0` to turn thinking off.
