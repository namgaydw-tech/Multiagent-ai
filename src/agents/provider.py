"""LLM provider abstraction for the multi-agent engine (Phase 3, section 3.1).

Backends are selected by environment variables only — no API key, vendor URL or model
name is hard-coded:

========  ==========================================================
``LLM_BACKEND``     ``deterministic`` (default) | ``openai_compatible`` |
                    ``ollama`` | ``huggingface`` | ``local`` | ``cloud``
``LLM_BASE_URL``    endpoint base URL for HTTP backends
``LLM_API_KEY``     bearer token for HTTP backends (never hard-coded here)
``LLM_MODEL``       model name reported in logs / audit trails
========  ==========================================================

Behaviour contract:

* timeouts, retry with exponential backoff, JSON extraction/repair, strict
  Pydantic schema validation of every payload;
* per-call logging of prompt hash, token counts, latency, model version and
  retry/error metadata (consumed by ``orchestration/persistence.py``);
* when no LLM is configured (``deterministic`` backend — the default on this
  machine) or an LLM repeatedly fails schema validation, the caller may supply
  a ``deterministic`` builder that derives the payload **only from values
  actually present in the input record**. The result is flagged with
  ``fallback_reason`` so audit trails never pretend an LLM produced it.

Research prototype — not a medical device.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import yaml
from pydantic import BaseModel, ValidationError

from src.preprocessing.regensburg import ROOT

CONFIG_PATH = ROOT / "config/agents.yaml"

# Backends that speak the OpenAI /chat/completions protocol when given a base URL.
_OPENAI_PROTOCOL = {"openai_compatible", "huggingface", "local", "cloud"}
_KNOWN_BACKENDS = _OPENAI_PROTOCOL | {"ollama", "deterministic"}

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RETRIES = 3
BACKOFF_BASE_S = 0.5  # sleep = BACKOFF_BASE_S * 2**attempt (tests may set 0)


class ProviderError(RuntimeError):
    """Raised when a backend fails and no deterministic builder is available."""


@dataclass
class ProviderCall:
    """One logged provider invocation (persisted into stage audit files)."""

    stage: str
    backend: str
    model_version: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int
    fallback_reason: str | None = None
    error: str | None = None
    timestamp: str = ""
    schema_name: str = ""


@dataclass
class ProviderResult:
    """Schema-validated payload plus full provenance for the audit trail."""

    payload: BaseModel
    call: ProviderCall
    raw_text: str = ""


def _load_agent_config() -> dict[str, Any]:
    """Read ``config/agents.yaml`` (authoritative provider settings)."""
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def prompt_hash(system: str, user: str) -> str:
    """Stable sha256 of the exact prompt pair (audit reproducibility)."""
    return hashlib.sha256(f"{system}\n<<<>>>\n{user}".encode("utf-8")).hexdigest()


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output.

    Handles fenced ```json blocks and leading prose. Raises ``ValueError`` if no
    parseable object exists — callers retry or fall back; nothing is invented.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```", 2)[1]
        if candidate.startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(candidate[start : i + 1])
                if isinstance(obj, dict):
                    return obj
                raise ValueError("top-level JSON value is not an object")
    raise ValueError("unbalanced JSON object in model output")


class LLMProvider:
    """Configurable multi-backend provider with retries and strict validation."""

    def __init__(self, backend: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, model: str | None = None,
                 timeout_s: float | None = None, retries: int | None = None,
                 temperature: float | None = None, max_tokens: int | None = None,
                 backoff_base_s: float = BACKOFF_BASE_S,
                 config: dict[str, Any] | None = None) -> None:
        """Resolve settings: explicit args > environment > config/agents.yaml."""
        cfg = (config if config is not None else _load_agent_config()).get("llm_provider", {}) or {}

        def resolve(explicit: Any, env_key: str, cfg_key: str, default: Any) -> Any:
            if explicit is not None:
                return explicit
            env_val = os.environ.get(env_key, "").strip()
            if env_val:
                return env_val
            val = cfg.get(cfg_key, default)
            if isinstance(val, str) and val.strip().startswith("${") and val.strip().endswith("}"):
                return default  # unsubstituted ${VAR} placeholder in config/agents.yaml
            return val

        self.backend = str(resolve(backend, "LLM_BACKEND", "backend", "deterministic") or "deterministic").lower()
        if self.backend in ("", "none", "mock", "rule_based"):
            self.backend = "deterministic"
        if self.backend not in _KNOWN_BACKENDS:
            raise ProviderError(
                f"unknown LLM_BACKEND {self.backend!r}; expected one of {sorted(_KNOWN_BACKENDS)}")
        self.base_url = resolve(base_url, "LLM_BASE_URL", "base_url", None)
        self.api_key = resolve(api_key, "LLM_API_KEY", "api_key", None)
        self.model = resolve(model, "LLM_MODEL", "model", None)
        self.timeout_s = float(resolve(timeout_s, "", "timeout_s", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)
        self.retries = int(resolve(retries, "", "retries", DEFAULT_RETRIES) or DEFAULT_RETRIES)
        self.temperature = float(temperature if temperature is not None else cfg.get("temperature", 0.0))
        self.max_tokens = int(max_tokens if max_tokens is not None else cfg.get("max_tokens", 2000))
        self.backoff_base_s = float(backoff_base_s)
        self.log: list[ProviderCall] = []

    # ------------------------------------------------------------------ helpers
    @property
    def model_version(self) -> str:
        """Model identifier recorded in every audit entry."""
        if self.backend == "deterministic":
            return "deterministic-rules-v1 (no LLM inference)"
        return str(self.model or "unspecified-model")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """~4 chars/token estimate when the backend reports no usage."""
        return max(1, len(text) // 4)

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST JSON with timeout; raise ``ProviderError`` on HTTP/URL failures."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"connection to {url} failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"non-JSON response from {url}") from exc

    def _chat(self, system: str, user: str) -> tuple[str, int, int]:
        """One backend round-trip → (text, input_tokens, output_tokens)."""
        if self.backend == "ollama":
            base = (self.base_url or "http://127.0.0.1:11434").rstrip("/")
            data = self._post_json(f"{base}/api/chat", {
                "model": self.model or "llama3", "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            })
            text = (data.get("message") or {}).get("content", "")
            return text, self._estimate_tokens(system + user), self._estimate_tokens(text)
        if self.backend in _OPENAI_PROTOCOL:
            if not self.base_url:
                raise ProviderError(f"backend {self.backend!r} requires LLM_BASE_URL")
            base = self.base_url.rstrip("/")
            url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
            data = self._post_json(url, {
                "model": self.model or "unknown-model",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            })
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            usage = data.get("usage") or {}
            return (text,
                    int(usage.get("prompt_tokens") or self._estimate_tokens(system + user)),
                    int(usage.get("completion_tokens") or self._estimate_tokens(text)))
        raise ProviderError(f"backend {self.backend!r} has no chat implementation")

    # ------------------------------------------------------------------ main API
    def complete_json(self, *, stage: str, system: str, user: str,
                      schema: type[BaseModel],
                      deterministic: Callable[[], dict[str, Any]] | None = None) -> ProviderResult:
        """Generate and strictly validate one stage payload.

        For the ``deterministic`` default backend the payload comes from
        ``deterministic()`` (which must derive values only from the record).
        For HTTP backends: up to ``retries`` attempts with exponential backoff,
        re-feeding the validation error each time; if every attempt fails and a
        deterministic builder exists, it is used and the result carries a
        ``fallback_reason`` — otherwise ``ProviderError`` is raised (stage fails
        loudly; fields are never silently invented).
        """
        if self.backend == "deterministic":
            if deterministic is None:
                raise ProviderError(f"stage {stage!r}: deterministic backend requires a builder")
            start = time.perf_counter()
            payload = schema.model_validate(deterministic())
            call = ProviderCall(
                stage=stage, backend="deterministic", model_version=self.model_version,
                prompt_hash=prompt_hash(system, user),
                input_tokens=0, output_tokens=0,
                latency_s=round(time.perf_counter() - start, 4), attempts=1,
                schema_name=schema.__name__,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self.log.append(call)
            return ProviderResult(payload=payload, call=call)

        start = time.perf_counter()
        last_error: str | None = None
        in_tok = out_tok = 0
        attempts = 0
        raw_text = ""
        for attempt in range(self.retries):
            attempts = attempt + 1
            try:
                text, in_tok, out_tok = self._chat(system, user)
                raw_text = text
                payload = schema.model_validate(extract_json(text))
                call = ProviderCall(
                    stage=stage, backend=self.backend, model_version=self.model_version,
                    prompt_hash=prompt_hash(system, user), input_tokens=in_tok,
                    output_tokens=out_tok, latency_s=round(time.perf_counter() - start, 4),
                    attempts=attempts, schema_name=schema.__name__,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                self.log.append(call)
                return ProviderResult(payload=payload, call=call)
            except (ProviderError, ValueError, ValidationError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries - 1 and self.backoff_base_s > 0:
                    time.sleep(self.backoff_base_s * (2 ** attempt))
                # re-prompt once with the validator feedback appended
                user = user + f"\n\nYour previous output failed validation: {last_error}. Return ONLY the JSON object."
        # all attempts failed
        if deterministic is not None:
            payload = schema.model_validate(deterministic())
            call = ProviderCall(
                stage=stage, backend=f"{self.backend}+deterministic_fallback",
                model_version=self.model_version, prompt_hash=prompt_hash(system, user),
                input_tokens=in_tok, output_tokens=out_tok,
                latency_s=round(time.perf_counter() - start, 4), attempts=attempts,
                fallback_reason=f"llm_failed_after_{attempts}_attempts: {last_error}",
                error=last_error, schema_name=schema.__name__,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self.log.append(call)
            return ProviderResult(payload=payload, call=call)
        err = ProviderError(f"stage {stage!r}: {attempts} attempts failed: {last_error}")
        self.log.append(ProviderCall(
            stage=stage, backend=self.backend, model_version=self.model_version,
            prompt_hash=prompt_hash(system, user), input_tokens=in_tok, output_tokens=out_tok,
            latency_s=round(time.perf_counter() - start, 4), attempts=attempts,
            error=last_error, schema_name=schema.__name__,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ))
        raise err

    # ------------------------------------------------------------------ audit helpers
    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        """All logged calls for a stage as plain dicts (JSON-serializable)."""
        from dataclasses import asdict
        return [asdict(c) for c in self.log if c.stage == stage]

    def total_tokens(self) -> int:
        """Sum of logged input+output tokens across all calls."""
        return sum(c.input_tokens + c.output_tokens for c in self.log)
