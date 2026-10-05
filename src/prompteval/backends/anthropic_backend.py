"""Anthropic Messages API backend.

Notes for the Claude 5.5 generation (checked October 2026, see METHODOLOGY.md):
- non-default temperature / top_p / top_k return HTTP 400, so leave `params` empty;
- Opus 5.5 always thinks; Sonnet 5.5 turns off up-front thinking with
  extra_body: {thinking: {type: between_tools}} at effort high or below.
Everything in `params` is passed as a top-level argument, everything in
`extra_body` is sent verbatim in the request body, so new API fields need no
code change. No prompt caching is requested.
"""

from __future__ import annotations

import os

from .base import Backend, FatalError, Generation, RetryableError


class AnthropicBackend(Backend):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise FatalError("pip install anthropic") from e
        self._anthropic = anthropic
        key = os.environ.get(cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not key:
            raise FatalError(f"Missing API key in ${cfg.get('api_key_env', 'ANTHROPIC_API_KEY')}")
        self.client = anthropic.Anthropic(
            api_key=key, max_retries=0, timeout=float(cfg.get("timeout", 600))
        )

    def generate(self, system, user, image_path=None, seed=None) -> Generation:
        a = self._anthropic
        content: list[dict] = []
        if image_path:
            mime, data = self.encode_image(image_path)
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": data}})
        content.append({"type": "text", "text": user})
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": content}], **self.params)
        if system:
            kwargs["system"] = system
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            resp = self.client.messages.create(**kwargs)
        except (a.RateLimitError, a.APIConnectionError, a.APITimeoutError, a.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code in (408, 409, 429) or e.status_code >= 500:
                raise RetryableError(str(e)) from e
            raise FatalError(f"HTTP {e.status_code}: {e.message}") from e

        texts, thoughts = [], []
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                texts.append(block.text)
            elif btype == "thinking":
                thoughts.append(getattr(block, "thinking", "") or "")
        usage = {}
        if resp.usage is not None:
            usage = {k: v for k, v in resp.usage.model_dump().items()
                     if isinstance(v, (int, float)) or isinstance(v, dict)}
        return Generation(
            text="\n".join(texts).strip(),
            thinking="\n".join(thoughts).strip(),
            finish_reason=str(resp.stop_reason or ""),
            model_returned=str(resp.model or ""),
            usage=usage,
        )
