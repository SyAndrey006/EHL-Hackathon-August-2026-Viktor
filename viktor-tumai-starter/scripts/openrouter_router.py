#!/usr/bin/env python3
"""Run the stateful session router through OpenRouter.

The routing policy lives in :mod:`session_router`; this file only supplies an
OpenRouter-compatible backend and a small CLI.  Prices deliberately come from
``cost_model.py`` so live cost estimates use the same schedule as the offline
evaluation scripts.  OpenRouter model ids can be supplied through environment
variables because the challenge's model names are aliases, not necessarily the
provider ids exposed by OpenRouter.

Examples::

    python scripts/openrouter_router.py --dry-run "Summarize this project"
    set OPENROUTER_API_KEY=...
    set OPENROUTER_LUNA_MODEL=openai/gpt-4o-mini
    python scripts/openrouter_router.py "Write a short project summary"
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

try:  # Support both ``python scripts/openrouter_router.py`` and package imports.
    from .cost_model import load_pricing, price_of
    from .session_router import ALL_MODELS, CONTEXT_LIMITS, SessionRouter
except ImportError:  # pragma: no cover - exercised by the direct-script CLI.
    from cost_model import load_pricing, price_of
    from session_router import ALL_MODELS, CONTEXT_LIMITS, SessionRouter


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ALIAS_ENV = {
    "gpt-5.6-luna": "OPENROUTER_LUNA_MODEL",
    "gpt-5.6-terra": "OPENROUTER_TERRA_MODEL",
    "gpt-5.6-sol": "OPENROUTER_SOL_MODEL",
}


def _json_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


class OpenRouterBackend:
    """Minimal dependency-free adapter for OpenRouter's chat-completions API."""

    def __init__(self, *, api_key: str | None = None, model_aliases: Mapping[str, str] | None = None,
                 timeout: float = 90.0, opener: Any = urllib.request.urlopen):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY before using the live adapter.")
        self.model_aliases = dict(model_aliases or {})
        self.timeout = timeout
        self.opener = opener
        self.pricing = load_pricing()

    def provider_model(self, model: str) -> str:
        env_name = ALIAS_ENV.get(model)
        return self.model_aliases.get(model) or (os.environ.get(env_name) if env_name else None) or model

    def call(self, model: str, task: Any) -> dict[str, Any]:
        provider_model = self.provider_model(model)
        body = json.dumps({
            "model": provider_model,
            "messages": [{"role": "user", "content": _json_text(task)}],
        }).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_NAME", "Viktor session router"),
        }
        request = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers, method="POST")
        try:
            with self.opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        if payload.get("error"):
            raise RuntimeError(f"OpenRouter error: {payload['error']}")
        choices = payload.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            text = content if isinstance(content, str) else _json_text(content)
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        input_price, _, output_price = price_of(model, self.pricing)
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        return {
            "model": model,
            "provider_model": provider_model,
            "text": text,
            "response": payload,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "confidence": 1.0,
        }


def build_openrouter_router(*, api_key: str | None = None,
                            models: Sequence[str] = ALL_MODELS,
                            model_aliases: Mapping[str, str] | None = None,
                            timeout: float = 90.0) -> tuple[SessionRouter, OpenRouterBackend]:
    """Build a live router whose aliases retain the repository's price keys."""
    backend = OpenRouterBackend(api_key=api_key, model_aliases=model_aliases, timeout=timeout)
    limits = {model: CONTEXT_LIMITS.get(model, 128_000) for model in models}
    router = SessionRouter(models=models, model_call=backend.call, context_limits=limits)
    return router, backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="*", help="message to route")
    parser.add_argument("--dry-run", action="store_true", help="route without making an API call")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    message = " ".join(args.message).strip() or "Explain the routing policy."

    if args.dry_run:
        router = SessionRouter()
    else:
        router, _ = build_openrouter_router(timeout=args.timeout)
    result = router.route(message)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
