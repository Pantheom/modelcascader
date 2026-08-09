"""
debug_logger.py
Structured JSON debug logger — verbose, per-request event tracing.

Purpose
-------
This is distinct from telemetry.py, which records one compact analytics row
per routing decision.  This logger emits one JSON line per *event* within a
request (G1 invoked, G2 invoked, dispatch started, …), giving a full trace
for a single request that can be grepped by request_id.

Output
------
Logs are written to logs/debug.jsonl (rotating).
Each line is a self-contained JSON object:

    {
        "ts":         "2026-01-15T10:23:45.123456+00:00",
        "request_id": "a3f1c2d4-...",
        "event":      "REQUEST_RECEIVED",
        ... event-specific fields ...
    }

Verbosity / redaction
---------------------
Controlled by the DEBUG_LOG_LEVEL environment variable (default: INFO).

  INFO  (default) — structural events only; prompt/response text is OMITTED.
  DEBUG           — includes truncated previews (first 200 chars) of prompt
                    and response for debugging.  Full text is NEVER logged at
                    any level; API keys are never referenced.

This redaction behaviour is intentional for privacy and security in shared
environments.  See README for details.

Usage
-----
    from modelcascader.debug_logger import DebugLogger

    dl = DebugLogger()                  # reads DEBUG_LOG_LEVEL from env
    dl.request_received(request_id, prompt_len=len(prompt), prompt=prompt)
    dl.g1_result(request_id, score=0.312, threshold=0.42, decision="tier_1", failed=False)
    dl.g2_result(request_id, score=0.501, threshold=0.48, decision="tier_3", failed=False)
    dl.dispatch_started(request_id, provider="groq", model="llama-3.3-70b-versatile", tier="tier_1")
    dl.dispatch_result(request_id, latency_ms=230.4, success=True, response=response_text)
    dl.response_sent(request_id, total_latency_ms=310.7, final_tier="tier_1")
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOG_FILE = "logs/debug.jsonl"
_PREVIEW_MAX_CHARS = 200  # max chars included in previews at DEBUG level


# ---------------------------------------------------------------------------
# DebugLogger
# ---------------------------------------------------------------------------

class DebugLogger:
    """
    Emits structured JSON debug events to logs/debug.jsonl.

    Instantiate once at service startup and share the instance.
    Each public method corresponds to one pipeline event.
    """

    def __init__(self, log_file: str = _DEFAULT_LOG_FILE) -> None:
        # ── Resolve log level from env ────────────────────────────────────
        level_name = os.environ.get("DEBUG_LOG_LEVEL", "INFO").upper()
        self._level = getattr(logging, level_name, logging.INFO)
        self._is_debug = self._level <= logging.DEBUG

        # ── Set up rotating JSONL file handler ────────────────────────────
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("modelcascader.debug")
        logger.propagate = False  # do not echo to the console / root logger
        logger.setLevel(logging.DEBUG)  # always write; we gate at emit time

        if not logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

        self._logger = logger

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event: str, request_id: str, **fields) -> None:
        """Serialize and write one JSON log line."""
        record = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "request_id": request_id,
            "event": event,
            **fields,
        }
        self._logger.debug(json.dumps(record, separators=(",", ":")))

    @staticmethod
    def _preview(text: str | None) -> str | None:
        """Return a truncated preview string (DEBUG only)."""
        if text is None:
            return None
        if len(text) <= _PREVIEW_MAX_CHARS:
            return text
        return text[:_PREVIEW_MAX_CHARS] + f"…[+{len(text) - _PREVIEW_MAX_CHARS} chars]"

    # ------------------------------------------------------------------
    # Public event emitters
    # ------------------------------------------------------------------

    def request_received(
        self,
        request_id: str,
        prompt_len: int,
        prompt: str,
    ) -> None:
        """
        Log that a request was received.

        At INFO+: logs prompt_len only.
        At DEBUG:  also includes a 200-char preview of the prompt.
        """
        fields: dict = {"prompt_len": prompt_len}
        if self._is_debug:
            fields["prompt_preview"] = self._preview(prompt)
        self._emit("REQUEST_RECEIVED", request_id, **fields)

    def g1_result(
        self,
        request_id: str,
        score: float | None,
        threshold: float,
        decision: str,
        failed: bool,
    ) -> None:
        """Log the G1 gatekeeper invocation and outcome."""
        self._emit(
            "G1_RESULT",
            request_id,
            score=round(score, 6) if score is not None else None,
            threshold=threshold,
            decision=decision,
            failed=failed,
        )

    def g2_result(
        self,
        request_id: str,
        score: float | None,
        threshold: float,
        decision: str,
        failed: bool,
    ) -> None:
        """Log the G2 gatekeeper invocation and outcome."""
        self._emit(
            "G2_RESULT",
            request_id,
            score=round(score, 6) if score is not None else None,
            threshold=threshold,
            decision=decision,
            failed=failed,
        )

    def dispatch_started(
        self,
        request_id: str,
        provider: str,
        model: str,
        tier: str,
    ) -> None:
        """Log that provider dispatch is about to start."""
        self._emit(
            "DISPATCH_STARTED",
            request_id,
            provider=provider,
            model=model,
            tier=tier,
        )

    def dispatch_result(
        self,
        request_id: str,
        latency_ms: float,
        success: bool,
        response: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        """
        Log the outcome of a provider dispatch call.

        At INFO+: logs latency and success/failure flag only.
        At DEBUG:  also includes a 200-char preview of the response text.
        """
        fields: dict = {
            "latency_ms": round(latency_ms, 1),
            "success": success,
        }
        if error_msg is not None:
            fields["error_msg"] = error_msg
        if self._is_debug and response is not None:
            fields["response_preview"] = self._preview(response)
        self._emit("DISPATCH_RESULT", request_id, **fields)

    def fail_safe_triggered(
        self,
        request_id: str,
        reason: str,
        tier_escalated_to: str,
    ) -> None:
        """Log a fail-safe escalation event."""
        self._emit(
            "FAIL_SAFE_TRIGGERED",
            request_id,
            reason=reason,
            tier_escalated_to=tier_escalated_to,
        )

    def response_sent(
        self,
        request_id: str,
        total_latency_ms: float,
        final_tier: str,
        http_status: int = 200,
    ) -> None:
        """Log that the final response was assembled and dispatched."""
        self._emit(
            "RESPONSE_SENT",
            request_id,
            total_latency_ms=round(total_latency_ms, 1),
            final_tier=final_tier,
            http_status=http_status,
        )

    def error_sent(
        self,
        request_id: str,
        http_status: int,
        error_code: str,
        stage: str,
        message: str,
    ) -> None:
        """Log that an error response was sent."""
        self._emit(
            "ERROR_SENT",
            request_id,
            http_status=http_status,
            error_code=error_code,
            stage=stage,
            message=message,
        )
