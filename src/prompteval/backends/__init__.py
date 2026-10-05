from .base import Backend, FatalError, Generation, RetryableError


def make_backend(cfg: dict) -> Backend:
    kind = cfg.get("backend")
    if kind == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(cfg)
    if kind in ("openai", "openai_compatible"):
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(cfg)
    if kind == "mock":
        from .mock_backend import MockBackend
        return MockBackend(cfg)
    raise ValueError(f"Unknown backend: {kind}")


__all__ = ["Backend", "FatalError", "Generation", "RetryableError", "make_backend"]
