#!/usr/bin/env python3
"""Layered, stateful model router.

The module is provider-neutral: applications inject ``model_call`` and may inject
real tokenizers, embeddings, judges, and test runners.  The default adapters are
offline and deterministic, which makes the router usable in local evaluations.

Usage (demo): ``python scripts/session_router.py``
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence


ALL_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
# OpenAI documents a 1.05M context window and 128K max output for this family.
CONTEXT_LIMITS = {m: 1_050_000 for m in ALL_MODELS}
CHEAPEST_TIER = "gpt-5.6-luna"
MAX_DEPTH = 2
TTL_SECONDS = 30 * 60
TTL_TURNS = 6
LOW_THRESHOLD = 0.35
PROMOTE_MARGIN = 0.15

EDIT_VERBS = r"edit|update|revise|change|fix|modify|rewrite|continue|extend|remove|delete"
PRONOUNS = r"it|this|that|these|those|the (file|document|image|chart|result|one)"
PIVOT_PHRASES = r"instead|actually|new task|different question|switch gears|forget that"
SINGLE_SHOT = r"^(what is|who is|when is|define|translate|calculate|summarize|yes|no)\b"
TASK_TYPES = ("code", "analysis", "writing", "research", "general")
ROUTING_TABLE = {
    "code": {"short": "gpt-5.6-terra", "medium": "gpt-5.6-sol", "long": "gpt-5.6-sol"},
    "analysis": {"short": "gpt-5.6-terra", "medium": "gpt-5.6-sol", "long": "gpt-5.6-sol"},
    "writing": {"short": "gpt-5.6-luna", "medium": "gpt-5.6-terra", "long": "gpt-5.6-terra"},
    "research": {"short": "gpt-5.6-terra", "medium": "gpt-5.6-sol", "long": "gpt-5.6-sol"},
    "general": {"short": CHEAPEST_TIER, "medium": "gpt-5.6-terra", "long": "gpt-5.6-sol"},
}


def now_ts() -> float:
    return time.time()


def estimate_tokens(text: Any, model: str | None = None) -> int:
    """Fallback tokenizer.  Replace via ``tokenizer_for`` for production."""
    return max(1, len(json.dumps(text, ensure_ascii=False) if not isinstance(text, str) else text) // 4)


OPENAI_PRICES_PER_MILLION = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (4.00, 20.00),
}


class OpenAIBackend:
    """OpenAI Responses API adapter used by ``SessionRouter``.

    The SDK import is lazy so the router remains importable for offline tests.
    Install the SDK and set ``OPENAI_API_KEY`` before using this adapter.
    """

    def __init__(self, client: Any = None, *, api_key: str | None = None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc
            client = OpenAI(api_key=api_key)
        self.client = client

    @staticmethod
    def _text(task: Any) -> str:
        if isinstance(task, str):
            return task
        return json.dumps(task, ensure_ascii=False, default=str)

    def call(self, model: str, task: Any) -> dict[str, Any]:
        response = self.client.responses.create(model=model, input=self._text(task))
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        input_price, output_price = OPENAI_PRICES_PER_MILLION[model]
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        return {"model": model, "text": getattr(response, "output_text", ""),
                "response": response, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "cost": cost, "confidence": 1.0}

    def judge(self, judge_model: str, result: Any, task: Any) -> bool:
        prompt = ("You are a ground-truth judge. Return only PASS or FAIL.\n"
                  f"Acceptance criteria/task: {self._text(task)}\n"
                  f"Candidate result: {self._text(result)}")
        judged = self.call(judge_model, prompt)
        return judged["text"].strip().upper().startswith("PASS")


def build_openai_router(*, client: Any = None, api_key: str | None = None,
                        models: Sequence[str] = ALL_MODELS) -> tuple[SessionRouter, OpenAIBackend]:
    """Build a live router using OpenAI's three frontier tiers."""
    backend = OpenAIBackend(client, api_key=api_key)
    router = SessionRouter(models=models, model_call=backend.call,
                           judge_call=lambda judge, result, task: backend.judge(judge, result, task),
                           context_limits={m: CONTEXT_LIMITS[m] for m in models})
    return router, backend


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", text.lower()))


def _embed(text: str, dimensions: int = 32) -> list[float]:
    values = [0.0] * dimensions
    for word in _words(text):
        digest = hashlib.sha256(word.encode()).digest()
        for i in range(0, 8, 2):
            values[int.from_bytes(digest[i:i + 2], "big") % dimensions] += 1.0
    norm = math.sqrt(sum(x * x for x in values)) or 1.0
    return [x / norm for x in values]


def _similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Artifact:
    id: str
    type: str
    created_turn: int
    expires_at: float
    summary: str


@dataclass
class SessionState:
    sticky_model: str | None = None
    artifact_registry: list[Artifact] = field(default_factory=list)
    cumulative_tokens_by_model: dict[str, int] = field(default_factory=dict)
    last_message_ts: float = field(default_factory=now_ts)
    multi_turn_expected: str = "unknown"
    classification_history: list[str] = field(default_factory=list)
    escalation_depth: int = 0
    task_budget_ceiling: float = 1.0
    cumulative_task_cost: float = 0.0
    topic_centroid_embedding: list[float] | None = None
    turn: int = 0


@dataclass
class RouteResult:
    route: str
    model: str | None = None
    resolved_reference: str | None = None
    subtasks: list[str] = field(default_factory=list)
    clarification: str | None = None
    results: Any = None
    logs: list[dict[str, Any]] = field(default_factory=list)


class SessionRouter:
    def __init__(self, *, models: Sequence[str] = ALL_MODELS,
                 model_call: Callable[[str, Any], Any] | None = None,
                 tokenizer_for: Callable[[str], Callable[[Any], int]] | None = None,
                 context_limits: Mapping[str, int] | None = None,
                 classification_call: Callable[[str], Mapping[str, Any]] | None = None,
                 judge_call: Callable[[str, Any, Any], bool | None] | None = None,
                 test_runner: Callable[[Any], bool] | None = None):
        self.models = tuple(models)
        self.model_call = model_call or (lambda model, task: {"model": model, "task": task, "cost": 0.0, "confidence": 1.0})
        self.tokenizer_for = tokenizer_for or (lambda model: lambda value: estimate_tokens(value, model))
        self.context_limits = dict(context_limits or {m: CONTEXT_LIMITS.get(m, 128_000) for m in self.models})
        self.classification_call = classification_call
        self.judge_call, self.test_runner = judge_call, test_runner
        self.logs: list[dict[str, Any]] = []

    def _log(self, session: SessionState, **fields: Any) -> None:
        self.logs.append({"conversation_id": fields.pop("conversation_id", None), "turn_id": session.turn,
                          "resolution_layer": fields.pop("resolution_layer", None), **fields})

    def _prune(self, s: SessionState, ts: float) -> None:
        s.artifact_registry[:] = [a for a in s.artifact_registry if a.expires_at > ts and s.turn - a.created_turn <= TTL_TURNS]

    def _classify(self, message: str) -> str:
        if self.classification_call:
            try:
                answer = self.classification_call(message)
                if answer.get("confidence", 1.0) >= 0.5 and answer.get("task_type") in TASK_TYPES:
                    return str(answer["task_type"])
            except Exception:
                pass  # Layer 1 remains available when the optional service is down.
        terms = {"code": "code python api bug function script repository", "analysis": "analyze compare metric data why", "writing": "write rewrite email story document", "research": "research sources investigate latest", "general": "help question explain"}
        vector = _embed(message)
        return max(terms, key=lambda t: _similarity(vector, _embed(terms[t])))

    def _length_bucket(self, message: str) -> str:
        n = estimate_tokens(message)
        return "short" if n < 150 else "medium" if n < 1500 else "long"

    def _compound(self, message: str) -> tuple[str | None, list[str], dict[str, Any] | None]:
        clauses = [x.strip() for x in re.split(r"\s*(?:;|\n|\band then\b|\bthen\b)\s*", message, flags=re.I) if x.strip()]
        imperatives = len(re.findall(r"\b(?:and|then|also|first|second|next)\b", message, re.I)) > 0 or len(clauses) > 1
        if not imperatives: return None, [], None
        parts = [_embed(c) for c in clauses]
        sim = min((_similarity(parts[i], parts[j]) for i in range(len(parts)) for j in range(i)), default=1.0)
        overlap = len(set.intersection(*[_words(c) for c in clauses])) if clauses else 0
        structure = "parallel" if sim < LOW_THRESHOLD and overlap == 0 else "sequential"
        return structure, clauses, {"clause_similarity": round(sim, 4), "entity_overlap": overlap}

    def route(self, message: str, session: SessionState | None = None, *, conversation_id: str | None = None,
              timestamp: float | None = None) -> RouteResult:
        s = session or SessionState(); ts = timestamp or now_ts(); s.turn += 1; self._prune(s, ts)
        full_text = message + " " + " ".join(a.summary for a in s.artifact_registry)
        s.cumulative_tokens_by_model = {m: self.tokenizer_for(m)(full_text) for m in self.models}
        candidates = [m for m in self.models if self.context_limits.get(m, 0) > s.cumulative_tokens_by_model[m]]
        forced = None
        if s.sticky_model not in candidates: forced = "context_window_exceeded" if s.sticky_model else None
        if not candidates: return RouteResult("compression_required", clarification="Session context exceeded every model; summarize/compress and retry.")
        warm = ts - s.last_message_ts < 300
        s.last_message_ts = ts
        resolved, bind_conf = None, None
        if re.search(f"(?:{EDIT_VERBS}).*(?:{PRONOUNS})|(?:{PRONOUNS}).*(?:{EDIT_VERBS})", message, re.I):
            matches = s.artifact_registry[-2:]
            if len(matches) == 1: resolved, bind_conf = matches[0].id, "high"
            elif matches: resolved, bind_conf = matches[-1].id, "low"
        task_type = self._classify(message); s.classification_history.append(task_type)
        bucket = self._length_bucket(message)
        structure, subtasks, basis = self._compound(message)
        if re.search(PIVOT_PHRASES, message, re.I): s.topic_centroid_embedding = _embed(message)
        if re.search(SINGLE_SHOT, message, re.I): s.multi_turn_expected = "unlikely"
        default = ROUTING_TABLE[task_type][bucket]
        if default not in candidates:
            default = candidates[0]
        if forced: self._log(s, conversation_id=conversation_id, forced_escalation_reason=forced)
        if structure == "parallel":
            s.task_budget_ceiling = 1.0 * len(subtasks); s.cumulative_task_cost = 0.0
            results = [self.execute_with_cascade(default, x, s) for x in subtasks]
            self._log(s, conversation_id=conversation_id, resolution_layer="L1", task_type=task_type,
                      length_bucket=bucket, compound_structure=structure, multi_turn_expected=s.multi_turn_expected,
                      model_used=default, judge_model_used=None, cache_hit=warm, escalation_depth=s.escalation_depth,
                      cost=s.cumulative_task_cost, downstream_outcome="partial" if any(isinstance(x, dict) and x.get("status") == "partial" for x in results) else "complete",
                      forced_escalation_reason=forced, fallback_triggered=None, compound_decision_basis=basis,
                      bind_confidence=bind_conf)
            return RouteResult("parallel", default, resolved, subtasks, results=results, logs=self.logs[-len(subtasks):])
        if structure == "sequential":
            chain = max((ROUTING_TABLE[self._classify(x)][self._length_bucket(x)] for x in subtasks), key=lambda m: self.models.index(m) if m in self.models else 0)
            s.sticky_model = chain
            results = self.execute_as_chain(chain, subtasks, s)
            self._log(s, conversation_id=conversation_id, resolution_layer="L1", task_type=task_type,
                      length_bucket=bucket, compound_structure=structure, multi_turn_expected=s.multi_turn_expected,
                      model_used=chain, judge_model_used=None, cache_hit=warm, escalation_depth=s.escalation_depth,
                      cost=s.cumulative_task_cost, downstream_outcome="complete", forced_escalation_reason=forced,
                      fallback_triggered=None, compound_decision_basis=basis, bind_confidence=bind_conf)
            return RouteResult("sequential", chain, resolved, subtasks, results=results, logs=self.logs[-1:])
        model = s.sticky_model if warm and s.sticky_model in candidates else default
        s.sticky_model = model
        s.task_budget_ceiling = max(s.task_budget_ceiling, 1.0); s.cumulative_task_cost = 0.0; s.escalation_depth = 0
        result = self.execute_with_cascade(model, {"message": message, "task_type": task_type, "reference": resolved}, s)
        self._log(s, conversation_id=conversation_id, resolution_layer="L0" if resolved else "L1",
                  task_type=task_type, length_bucket=bucket, compound_structure=None,
                  multi_turn_expected=s.multi_turn_expected, model_used=model, judge_model_used=None,
                  cache_hit=warm, escalation_depth=s.escalation_depth,
                  cost=s.cumulative_task_cost, downstream_outcome="complete", forced_escalation_reason=forced,
                  fallback_triggered=None, compound_decision_basis=None, bind_confidence=bind_conf)
        return RouteResult("standalone", model, resolved, results=result, logs=self.logs[-1:])

    def execute_as_chain(self, model: str, subtasks: Sequence[str], s: SessionState) -> list[Any]:
        return [self.execute_with_cascade(model, task, s) for task in subtasks]

    def execute_with_cascade(self, model: str, task: Any, s: SessionState) -> Any:
        result = self.model_call(model, task); s.cumulative_task_cost += float(result.get("cost", 0.0)) if isinstance(result, dict) else 0.0
        while not self.passes_ground_truth_check(result, task, model) and s.escalation_depth < MAX_DEPTH:
            if s.cumulative_task_cost > s.task_budget_ceiling: return {"status": "partial", "result": result, "note": "Budget ceiling reached."}
            s.escalation_depth += 1; idx = self.models.index(model) if model in self.models else 0; model = self.models[min(idx + 1, len(self.models) - 1)]
            result = self.model_call(model, task); s.cumulative_task_cost += float(result.get("cost", 0.0)) if isinstance(result, dict) else 0.0
        return result

    def passes_ground_truth_check(self, result: Any, task: Any, produced_model: str) -> bool:
        if isinstance(task, dict) and task.get("has_tests_or_exit_code") and self.test_runner: return bool(self.test_runner(result))
        if self.judge_call:
            judge = next((m for m in self.models if m != produced_model), None)
            if judge: return self.judge_call(judge, result, task)
        return True  # unverified is deliverable when no judge is configured


class ReviewAgent:
    """Offline tuning helper; it never mutates a live ``SessionRouter``."""

    def daily_report(self, logs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        rows = list(logs); total = max(1, len(rows))
        by_layer = {layer: sum(r.get("resolution_layer") == layer for r in rows) / total
                    for layer in ("L0", "L1", "L2", "L3")}
        cells: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            cells.setdefault(f"{row.get('task_type')}:{row.get('length_bucket')}", []).append(row)
        escalation_cells = {key: sum(bool(r.get("escalation_depth")) for r in value) / len(value)
                            for key, value in cells.items()}
        return {"layer_distribution": by_layer, "escalation_rate_by_cell": escalation_cells,
                "cells_over_30_percent": [k for k, v in escalation_cells.items() if v > 0.30]}

    def stage(self, current: Mapping[str, Any], proposed: Mapping[str, Any], *, risk: str) -> dict[str, Any]:
        if risk not in {"low", "medium", "high"}: raise ValueError("risk must be low, medium, or high")
        return {"status": "auto_apply" if risk == "low" else "approval_required", "risk": risk,
                "before": dict(current), "after": dict(proposed), "created_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    if "--live" in sys.argv:
        router, _ = build_openai_router()
        message = " ".join(x for x in sys.argv[1:] if x != "--live") or "Explain the routing policy."
    else:
        router = SessionRouter()
        message = "Write a short summary of this project"
    result = router.route(message)
    print(json.dumps(result.__dict__, indent=2, default=str))
