"""
Argus — Autonomous Research Agent
Exports the shared `app` Agent instance used across all modules.

Authentication is handled automatically via environment variables:
  NEBIUS_API_KEY  — used by AgentField/LiteLLM for all app.ai() calls
  NEBIUS_MODEL    — optional LiteLLM/Nebius model override
"""
import logging
import os
from urllib.parse import urlparse

from agentfield import Agent, AIConfig
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_NEBIUS_MODEL = "nebius/openai/gpt-oss-120b"
NEBIUS_MODEL_PLACEHOLDER = "your_valid_nebius_model_id_here"
DEPRECATED_NEBIUS_MODELS = {
    "openai/gpt-oss-20b",
    "nebius/openai/gpt-oss-20b",
}


def get_nebius_model() -> str:
    """Return the LiteLLM-compatible Nebius model name."""
    model = os.getenv("NEBIUS_MODEL", "").strip()
    if not model or model == NEBIUS_MODEL_PLACEHOLDER:
        model = DEFAULT_NEBIUS_MODEL
    elif not model.startswith("nebius/"):
        model = f"nebius/{model}"
    return model


def _configured_agentfield_base_url() -> str:
    """Return an explicitly configured AgentField server URL, if present."""
    base_url = (
        os.getenv("AGENTFIELD_BASE_URL")
        or os.getenv("AGENTFIELD_SERVER")
        or os.getenv("AGENTFIELD_SERVER_URL")
        or ""
    ).strip().rstrip("/")

    if base_url:
        return base_url

    ws_url = os.getenv("AGENTFIELD_MEMORY_WS_URL", "").strip()
    if ws_url:
        parsed = urlparse(ws_url)
        if parsed.scheme in {"ws", "wss"} and parsed.netloc:
            http_scheme = "https" if parsed.scheme == "wss" else "http"
            return f"{http_scheme}://{parsed.netloc}"

    return ""


class ArgusAgent(Agent):
    """AgentField Agent with local-only fallbacks for Argus."""

    _memory_warning_logged = False
    _model_warning_logged = False

    def _register_memory_event_listeners(self):
        if not _configured_agentfield_base_url():
            if not ArgusAgent._memory_warning_logged:
                logger.warning(
                    "AgentField memory server not configured; running without persistent memory/events."
                )
                ArgusAgent._memory_warning_logged = True
            return

        return super()._register_memory_event_listeners()

    async def ai(self, *args, **kwargs):
        requested_model = kwargs.get("model")
        if requested_model in DEPRECATED_NEBIUS_MODELS:
            kwargs["model"] = NEBIUS_MODEL
            if not ArgusAgent._model_warning_logged:
                logger.warning(
                    "Deprecated Nebius model %s requested; using NEBIUS_MODEL=%s instead.",
                    requested_model,
                    NEBIUS_MODEL,
                )
                ArgusAgent._model_warning_logged = True

        return await super().ai(*args, **kwargs)


NEBIUS_MODEL = get_nebius_model()
AGENTFIELD_BASE_URL = _configured_agentfield_base_url()

app = ArgusAgent(
    node_id="argus-research-agent",
    ai_config=AIConfig(model=NEBIUS_MODEL),
    agentfield_server=AGENTFIELD_BASE_URL,
    auto_register=bool(AGENTFIELD_BASE_URL),
)
