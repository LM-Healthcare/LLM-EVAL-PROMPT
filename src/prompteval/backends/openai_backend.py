"""OpenAI and OpenAI-compatible (vLLM, SGLang, Ollama, OpenRouter...) chat backend.

For Qwen3.5 served by vLLM, thinking is switched off with
extra_body: {chat_template_kwargs: {enable_thinking: false}} and the recommended
non-thinking sampling (temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5)
goes in `params` / `extra_body` (top_k is not an OpenAI field, so it goes in
extra_body). A per-run seed is sent when `seed_per_run` is true.
"""

from __future__ import annotations

import os

from .base import Backend, FatalError, Generation, RetryableError


class OpenAIBackend(Backend):
    supports_seed = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise FatalError("pip install openai") from e
        self._openai = openai
        key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env)
        if not key:
            if cfg.get("backend") == "openai_compatible":
                key = "EMPTY"  # local servers usually ignore the key
            else:
                raise FatalError(f"Missing API key in ${key_env}")
        self.client = openai.OpenAI(
            api_key=key, base_url=cfg.get("base_url"), max_retries=0,
            timeout=float(cfg.get("timeout", 600)),
        )
        self.token_param = cfg.get("max_tokens_param", "max_tokens")  # or max_completion_tokens

    def generate(self, system, user, image_path=None, seed=None) -> Generation:
        o = self._openai
        if image_path:
            mime, data = self.encode_image(image_path)
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text", "text": user},
            ]
        else:
            content = user
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        kwargs = dict(model=self.model, messages=messages, **self.params)
        kwargs[self.token_param] = self.max_tokens
        if seed is not None:
            kwargs["seed"] = int(seed)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except (o.RateLimitError, o.APIConnectionError, o.APITimeoutError, o.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        except o.APIStatusError as e:
            if e.status_code in (408, 409, 429) or e.status_code >= 500:
                raise RetryableError(str(e)) from e
            raise FatalError(f"HTTP {e.status_code}: {e.message}") from e

        choice = resp.choices[0]
        msg = choice.message
        thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        usage = resp.usage.model_dump() if resp.usage is not None else {}
        return Generation(
            text=(msg.content or "").strip(),
            thinking=str(thinking).strip(),
            finish_reason=str(choice.finish_reason or ""),
            model_returned=str(resp.model or ""),
            usage={k: v for k, v in usage.items() if v is not None},
        )
