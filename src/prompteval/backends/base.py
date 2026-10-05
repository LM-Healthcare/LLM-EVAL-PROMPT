"""Backend interface shared by all model providers."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path


class RetryableError(Exception):
    """Transient failure (rate limit, overload, timeout, 5xx): retry with backoff."""


class FatalError(Exception):
    """Configuration or request error (e.g. HTTP 400): stop this model's run."""


@dataclass
class Generation:
    text: str
    thinking: str = ""
    finish_reason: str = ""
    model_returned: str = ""
    usage: dict = field(default_factory=dict)
    request_params: dict = field(default_factory=dict)


class Backend:
    """A model endpoint. `cfg` is the model entry of the experiment config."""

    supports_seed = False

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg.get("model", "")
        self.params = dict(cfg.get("params") or {})
        self.extra_body = dict(cfg.get("extra_body") or {})
        self.max_tokens = int(cfg.get("max_tokens", 2048))

    def generate(self, system: str, user: str, image_path: str | None = None,
                 seed: int | None = None) -> Generation:
        raise NotImplementedError

    @staticmethod
    def encode_image(path: str) -> tuple[str, str]:
        mime = mimetypes.guess_type(path)[0] or "image/png"
        data = base64.b64encode(Path(path).read_bytes()).decode()
        return mime, data

    def describe(self) -> dict:
        """Static request settings, logged with every response."""
        return {
            "backend": self.cfg.get("backend"),
            "model": self.model,
            "max_tokens": self.max_tokens,
            "params": self.params,
            "extra_body": self.extra_body,
        }
