"""
core/llm.py — Ollama client wrapper for all agent roles.

Design rules:
  * Every LLM output that drives a decision must be JSON, schema-validated.
  * Every call site provides a deterministic fallback — the POC never dies
    because a model rambled.
  * Engine is swappable (llama3.1:8b default, deepseek-r1:7b toggle for
    Planner/Critic). DeepSeek <think> blocks are stripped from JSON but
    captured for the ledger.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

def set_reasoner(model: str) -> None:
    """Swap the reasoning model at runtime (Fast=llama3.2:3b / Quality=llama3.1:8b)."""
    MODELS["reasoner"] = model


MODELS = {
    "reasoner": "llama3.1:8b",       # intent, planner, critic, rca, recommend
    "reasoner_alt": "deepseek-r1:7b",  # optional toggle — thinking traces
    "narrator": "gemma2:2b",          # report prose, personas
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class LLMResult:
    ok: bool
    content: str = ""
    parsed: Optional[dict] = None
    thinking: str = ""          # DeepSeek <think> trace, if any → ledger
    error: str = ""
    used_fallback: bool = False


def _post(model: str, messages: list[dict], temperature: float = 0.1,
          timeout: int = 75) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction: strip think blocks, fences, find outermost {}."""
    text = THINK_RE.sub("", text)
    fence = JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def call_json(role: str, system: str, user: str,
              required_keys: tuple[str, ...] = (),
              fallback: Optional[Callable[[], dict]] = None,
              retries: int = 1, temperature: float = 0.1,
              model_override: Optional[str] = None) -> LLMResult:
    """Call an agent role expecting JSON with required_keys. Falls back on failure."""
    model = model_override or MODELS[role]
    last_err = ""
    for attempt in range(retries + 1):
        try:
            raw = _post(model, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], temperature=temperature)
        except Exception as e:  # noqa: BLE001 — network/model failure
            last_err = f"ollama call failed: {e}"
            continue
        thinking = "\n".join(THINK_RE.findall(raw)).strip()
        parsed = extract_json(raw)
        if parsed is not None and all(k in parsed for k in required_keys):
            return LLMResult(ok=True, content=raw, parsed=parsed, thinking=thinking)
        last_err = f"missing keys or unparseable JSON (attempt {attempt + 1})"
        # tighten the instruction on retry
        user = user + "\n\nRespond with ONLY a valid JSON object. No prose."
    if fallback is not None:
        return LLMResult(ok=True, parsed=fallback(), used_fallback=True,
                         error=last_err)
    return LLMResult(ok=False, error=last_err)


def call_text(role: str, system: str, user: str,
              fallback_text: str = "", temperature: float = 0.4) -> LLMResult:
    """Free-text call (narration). Falls back to template text on failure."""
    model = MODELS[role]
    try:
        raw = _post(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=temperature)
        return LLMResult(ok=True, content=THINK_RE.sub("", raw).strip())
    except Exception as e:  # noqa: BLE001
        return LLMResult(ok=bool(fallback_text), content=fallback_text,
                         used_fallback=True, error=str(e))


def warm_up(roles: tuple[str, ...] = ("reasoner", "narrator")) -> dict[str, float]:
    """Preload models into Ollama memory with a 1-token ping per role.
    Returns seconds per role; raises nothing (best-effort)."""
    import time
    timings: dict[str, float] = {}
    for role in roles:
        model = MODELS.get(role)
        if not model or model in {m for r, m in MODELS.items() if r in timings
                                  and MODELS[r] == model}:
            pass
        t0 = time.time()
        try:
            _post(model, [{"role": "user", "content": "ok"}], timeout=150)
            timings[role] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            timings[role] = -1.0
    return timings
