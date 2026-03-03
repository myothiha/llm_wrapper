# Ollama Container Setup and Wrapper Class

- [Files Structure](#files-structure)
- [Setup Ollama Container](#setup-ollama-container)
- [Configuration (.env)](#configuration-env)
- [Manage Models](#manage-models)
- [OllamaWrapper Class Usage](#ollamawrapper-class-usage)

## Files Structure
 - `ollama/ollama_wrapper.py` — wrapper class used to interact with Ollama models.
 - `docker-compose.yaml` — orchestrates the Ollama container with GPU support.
 - `docker-compose-cpu.yaml` — CPU-only variant of the docker-compose setup.
 - `.env.example` — template listing all required and optional environment variables.

## Setup Ollama Container
 - Create a new folder called `ollama-api` and copy `docker-compose.yaml` to that folder.
 - Run `cd ollama-api`.
 - Run `docker-compose up` in the folder.
 - Once running, the Ollama service will be available at [http://localhost:11434](http://localhost:11434).
 - Use the wrapper class in `ollama_wrapper.py` to send requests to the running Ollama container.

## Configuration (.env)

The wrapper reads configuration from the **root project's `.env`** file (loaded automatically via `python-dotenv`). Add the following variables to that file:

| Variable | Required | Default | Description |
|---|---|---|---|
| `ollama_host` | ✅ Yes | — | URL of the Ollama API server, e.g. `http://localhost:11434` |
| `GPU_TEMP_THRESHOLD` | No | `80` | GPU temperature (°C) at which inference is paused for cooling |
| `GPU_COOLDOWN_TIME` | No | `60` | Seconds to wait during a cooldown before retrying |

> **Note:** `GPU_TEMP_THRESHOLD` and `GPU_COOLDOWN_TIME` are tunable per-deployment. The defaults (80 °C / 60 s) are safe for most setups, but you can lower the threshold or extend the cooldown time for more aggressive thermal protection.

## Manage Models
 - To list available models inside the Ollama container: `docker exec -it ollama ollama list`
 - To pull a new model (example: Llama 3 8B): `docker exec -it ollama ollama pull llama3:8b`

## OllamaWrapper Class Usage

### For text-only LLM
We can use a text-only LLM for single-turn or multi-turn conversations depending on the use case. By default, it is single-turn.

**Single-turn example:**
```python
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper

llm = OllamaWrapper(model="gpt-oss:20b")
response = llm("Hello, world!")
print(response)
```

**Multi-turn example:**
```python
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper

llm = OllamaWrapper(model="gpt-oss:20b", multi_turn=True)
llm.set_system_prompt("You are a helpful assistant.")  # only for multi-turn mode

response = llm("What is the capital of France?")
print(response)

response = llm("What is its population?")
print(response)

# Print the conversation history
print(llm.history)
```

### For LLM with image input
Right now image input is single-turn only.

**Single-turn multimodal example:**
```python
from libs.llm_loader.ollama.ollama_wrapper import OllamaWrapper

llm = OllamaWrapper(model="llava:34b")
response = llm(images=["example.jpg"], prompt="Describe the provided image.")
print(response)
```