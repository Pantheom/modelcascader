"""
server.py
Three-Tier Cascade Router — local server.

Endpoints
---------
  POST /v1/route-and-generate   Hardened REST API (versioned, full schema validation,
                                  structured error responses, debug logging).
  GET  /v1/health               Service health + provider key status + loaded thresholds.

  POST /route-and-generate      Legacy alias — same handler, same v1 response shape.
                                  Kept so the test frontend (frontend/index.html) continues
                                  to work without any frontend changes.
  GET  /                        Serves frontend/index.html.
  GET  /frontend/*              Serves static files from the frontend/ directory.

Usage
-----
    # From the project root, with the venv active:
    python server.py

    # With API keys (required for generation; routing works without):
    set GROQ_API_KEY=gsk-...        # Windows — Tier 1 (Llama 3.3 70B)
    set GEMINI_API_KEY=...          # Windows — Tier 2 & 3 (Gemini Flash)
    export GROQ_API_KEY=gsk-...     # macOS/Linux

    # Optional: control debug log verbosity (default: INFO)
    set DEBUG_LOG_LEVEL=DEBUG       # Windows
    export DEBUG_LOG_LEVEL=DEBUG    # macOS/Linux

    # Optional: change the port (default 8765)
    python server.py --port 9000

Then open http://localhost:8765 in your browser.

Architecture
------------
Uses only Python stdlib (http.server + json + threading).
Single-threaded (one request at a time), which is fine for a single-user debug tool.

The routing path calls the existing CascadeRouter.route() and providers.generate()
without duplicating any logic.  This file only wraps the pipeline in a strict
API contract and adds structured debug logging.

Debug logging
-------------
Structured JSON events are written to logs/debug.jsonl (one line per pipeline event).
Verbosity is controlled by the DEBUG_LOG_LEVEL env var:
  INFO  (default) — structural events only; prompt/response text omitted.
  DEBUG           — includes 200-char truncated previews of prompt and response.
Full prompt/response text is never logged at any level; API keys are never logged.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pydantic import ValidationError

# ── Bootstrap: ensure project root is on sys.path ────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Set placeholder OPENAI_API_KEY before RouteLLM import ────────────────────
# RouteLLM's similarity_weighted router instantiates OpenAI() at module import
# time. Setting a placeholder satisfies the constructor without making real calls.
# If a real key is already set, this no-op guard leaves it untouched.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "placeholder-for-routellm-import"

from modelcascader import CascadeRouter, load_config, build_router_pool
from modelcascader.telemetry import TelemetryLogger
from modelcascader.providers import get_client, generate
from modelcascader.api_schemas import (
    RouteRequest,
    RouteResponse,
    RoutingInfo,
    TimingInfo,
    ErrorDetail,
    ErrorResponse,
)
from modelcascader.debug_logger import DebugLogger

# ── Logging (human-readable console logger for startup/infra messages) ────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ── Global singletons (initialised once at startup) ───────────────────────────
_config = None
_router = None
_debug_log: DebugLogger | None = None
_frontend_dir = ROOT / "frontend"


def _init_router():
    global _config, _router, _debug_log
    logger.info("Loading config...")
    _config = load_config(ROOT / "config" / "cascade_config.yaml")
    pool = build_router_pool(_config)
    tel = TelemetryLogger(_config.telemetry)
    _router = CascadeRouter(_config, pool, tel)
    _debug_log = DebugLogger()
    level_name = os.environ.get("DEBUG_LOG_LEVEL", "INFO").upper()
    logger.info(
        "Cascade router ready. Debug log level: %s → logs/debug.jsonl", level_name
    )


# ── Provider key detection (for /v1/health) ───────────────────────────────────

_PROVIDER_KEY_ENV: dict[str, list[str]] = {
    "openai":    ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "groq":      ["GROQ_API_KEY"],
    "google":    ["GOOGLE_API_KEY", "GEMINI_API_KEY"],  # either satisfies google-genai
}

_PLACEHOLDER = "placeholder-for-routellm-import"


def _provider_has_key(provider: str) -> bool:
    """Return True iff the provider has a non-placeholder API key configured."""
    for env_var in _PROVIDER_KEY_ENV.get(provider, []):
        val = os.environ.get(env_var, "")
        if val and val != _PLACEHOLDER:
            return True
    return False


# ── Error code detection helpers ──────────────────────────────────────────────

_AUTH_SIGNALS = (
    "api_key", "authentication", "401", "x-api-key",
    "invalid api key", "api_key_missing", "unauthenticated",
)


def _is_auth_error(exc: Exception) -> bool:
    return any(s in str(exc).lower() for s in _AUTH_SIGNALS)


def _friendly_missing_key_message(exc: Exception, provider: str, model: str) -> str:
    key_names = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq":      "GROQ_API_KEY",
        "google":    "GEMINI_API_KEY (note: GOOGLE_API_KEY takes precedence if also set)",
    }
    key_hint = key_names.get(provider, f"the API key for '{provider}'")
    return (
        f"Authentication failed for provider '{provider}' (model: {model}). "
        f"Set {key_hint} in your environment."
    )


# ── Internal exception used to signal API errors from helpers ─────────────────

class _ApiError(Exception):
    """Carries a structured error to be serialised as ErrorResponse.

    routing_context is an optional dict matching the RoutingInfo schema.
    When present, the HTTP handler attaches it to the error response body
    so the frontend can display G1/G2 scores even when generation fails.
    """

    def __init__(
        self,
        code: str,
        message: str,
        stage: str,
        http_status: int,
        routing_context: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.http_status = http_status
        self.routing_context: dict | None = routing_context


# ── Core v1 handler logic ─────────────────────────────────────────────────────

def _handle_v1_route_and_generate(request_id: str, req: RouteRequest) -> RouteResponse:
    """
    Run the cascade router, then call the winning tier's LLM.

    All structured debug log events are emitted here.
    Raises _ApiError on any recoverable failure so the HTTP handler can
    build a consistent error response without catching raw exceptions.
    """
    prompt = req.prompt
    t_start = time.perf_counter()

    # ── Log: request received ─────────────────────────────────────────────
    _debug_log.request_received(request_id, prompt_len=len(prompt), prompt=prompt)

    # ── Step 1: routing ───────────────────────────────────────────────────
    try:
        routing_result = _router.route(prompt)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Routing failed:\n%s", request_id[:8], tb)
        raise _ApiError(
            code="INTERNAL_ERROR",
            message=f"Routing pipeline raised an unexpected error: {exc}",
            stage="routing",
            http_status=500,
        ) from exc

    tier = routing_result.tier
    tier_cfg = _config.tiers.get(tier)
    g1_threshold = _config.gatekeeper_1.threshold
    g2_threshold = _config.gatekeeper_2.threshold

    # ── Log: G1 result ────────────────────────────────────────────────────
    g1_fired = "gatekeeper_1" in routing_result.gatekeepers_fired
    g1_failed = g1_fired and routing_result.g1_score is None
    if g1_fired:
        # Decision: if g1_score is None → failed; if score < threshold → tier_1;
        # otherwise → escalated to G2
        if g1_failed:
            g1_decision = "FAIL_SAFE_ESCALATE"
        elif routing_result.g1_score < g1_threshold:  # type: ignore[operator]
            g1_decision = "tier_1"
        else:
            g1_decision = "escalate_to_g2"
        _debug_log.g1_result(
            request_id,
            score=routing_result.g1_score,
            threshold=g1_threshold,
            decision=g1_decision,
            failed=g1_failed,
        )

    # ── Log: G2 result (if invoked) ───────────────────────────────────────
    g2_fired = "gatekeeper_2" in routing_result.gatekeepers_fired
    if g2_fired:
        g2_failed = routing_result.g2_score is None
        if g2_failed:
            g2_decision = "FAIL_SAFE_ESCALATE"
        elif routing_result.g2_score < g2_threshold:  # type: ignore[operator]
            g2_decision = "tier_2"
        else:
            g2_decision = "tier_3"
        _debug_log.g2_result(
            request_id,
            score=routing_result.g2_score,
            threshold=g2_threshold,
            decision=g2_decision,
            failed=g2_failed,
        )

    # ── Log: fail-safe (if triggered) ────────────────────────────────────
    if routing_result.fail_safe_triggered:
        _debug_log.fail_safe_triggered(
            request_id,
            reason="Gatekeeper errored or timed out — escalated by fail-safe policy.",
            tier_escalated_to=tier,
        )

    logger.info(
        "[%s] Routed to %s (%s) via %s  g1=%.4f  g2=%s",
        request_id[:8],
        tier,
        tier_cfg.model,
        routing_result.gatekeepers_fired,
        routing_result.g1_score or 0.0,
        f"{routing_result.g2_score:.4f}" if routing_result.g2_score is not None else "N/A",
    )

    # ── Step 2: generation ────────────────────────────────────────────────
    _debug_log.dispatch_started(
        request_id, provider=tier_cfg.provider, model=tier_cfg.model, tier=tier
    )
    gen_start = time.perf_counter()
    response_text: str | None = None

    try:
        client = get_client(tier_cfg)
        messages = [{"role": "user", "content": prompt}]
        response_text = generate(client, tier_cfg, messages)
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000
        _debug_log.dispatch_result(
            request_id, latency_ms=gen_latency_ms, success=True, response=response_text
        )
        logger.info("[%s] Generation complete (%.1f ms)", request_id[:8], gen_latency_ms)
    except Exception as exc:
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000
        err_str = str(exc)
        _debug_log.dispatch_result(
            request_id, latency_ms=gen_latency_ms, success=False, error_msg=err_str
        )

        # Build routing context so the frontend can show G1/G2 scores
        # even though generation failed.  Mirrors the RoutingInfo schema.
        routing_ctx = {
            "gatekeepers_fired": routing_result.gatekeepers_fired,
            "g1_score": routing_result.g1_score,
            "g1_threshold": g1_threshold,
            "g2_score": routing_result.g2_score if g2_fired else None,
            "g2_threshold": g2_threshold if g2_fired else None,
        }
        # Also include tier/model so the UI badge can still render
        routing_ctx["final_tier"] = tier
        routing_ctx["model_used"] = tier_cfg.model
        routing_ctx["provider_used"] = tier_cfg.provider
        routing_ctx["tier_label"] = tier_cfg.label
        routing_ctx["routing_latency_ms"] = round(routing_result.routing_latency_ms, 1)
        routing_ctx["fail_safe_triggered"] = routing_result.fail_safe_triggered

        if _is_auth_error(exc):
            msg = _friendly_missing_key_message(exc, tier_cfg.provider, tier_cfg.model)
            raise _ApiError(
                code="MISSING_API_KEY",
                message=msg,
                stage="generation",
                http_status=424,
                routing_context=routing_ctx,
            ) from exc

        raise _ApiError(
            code="PROVIDER_ERROR",
            message=f"Generation failed ({tier_cfg.provider}/{tier_cfg.model}): {err_str}",
            stage="generation",
            http_status=502,
            routing_context=routing_ctx,
        ) from exc

    # ── Step 3: assemble response ─────────────────────────────────────────
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    response = RouteResponse(
        request_id=request_id,
        final_tier=tier,  # type: ignore[arg-type]
        model_used=tier_cfg.model,
        provider_used=tier_cfg.provider,
        response_text=response_text,
        routing=RoutingInfo(
            gatekeepers_fired=routing_result.gatekeepers_fired,
            g1_score=routing_result.g1_score,
            g1_threshold=g1_threshold,
            g2_score=routing_result.g2_score if g2_fired else None,
            g2_threshold=g2_threshold if g2_fired else None,
        ),
        timing=TimingInfo(
            routing_latency_ms=round(routing_result.routing_latency_ms, 1),
            generation_latency_ms=round(gen_latency_ms, 1),
            total_latency_ms=round(total_latency_ms, 1),
        ),
        fail_safe_triggered=routing_result.fail_safe_triggered,
    )
    _debug_log.response_sent(
        request_id, total_latency_ms=total_latency_ms, final_tier=tier, http_status=200
    )
    return response


# ── Health endpoint ───────────────────────────────────────────────────────────

def _handle_health() -> dict:
    """
    Return service health information.

    Includes per-provider key-configured booleans (never the key values)
    and the current thresholds loaded from config.
    """
    # Determine which providers are actually used by the configured tiers
    configured_providers = {
        name: _config.tiers.get(name).provider
        for name in ("tier_1", "tier_2", "tier_3")
    }
    provider_set = set(configured_providers.values())

    return {
        "status": "ok",
        "providers": {
            p: _provider_has_key(p) for p in sorted(provider_set)
        },
        "thresholds": {
            "gatekeeper_1": _config.gatekeeper_1.threshold,
            "gatekeeper_2": _config.gatekeeper_2.threshold,
        },
        "tiers": {
            tier_name: {
                "provider": _config.tiers.get(tier_name).provider,
                "model": _config.tiers.get(tier_name).model,
                "label": _config.tiers.get(tier_name).label,
            }
            for tier_name in ("tier_1", "tier_2", "tier_3")
        },
    }


# ── Request handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    """HTTP handler — routes the v1 API endpoints plus legacy frontend paths."""

    def log_message(self, format, *args):  # suppress default access log noise
        logger.debug("HTTP %s %s", self.path, args)

    # ── GET ───────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/v1/health":
            self._send_json(200, _handle_health())
        elif self.path in ("/", "/index.html"):
            self._serve_file(_frontend_dir / "index.html", "text/html; charset=utf-8")
        elif self.path.startswith("/frontend/"):
            rel = self.path.lstrip("/")
            self._serve_file(ROOT / rel, "application/octet-stream")
        else:
            self._send_json(404, {"error": f"Not found: {self.path}"})

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self._send_json(404, {"error": f"File not found: {path}"})
            return
        data = path.read_bytes()
        if path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── POST ──────────────────────────────────────────────────────────────

    def do_POST(self):
        if self.path in ("/v1/route-and-generate", "/route-and-generate"):
            self._handle_route_post()
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    def _handle_route_post(self):
        """Shared handler for both /v1/route-and-generate and /route-and-generate."""
        request_id = str(uuid.uuid4())

        # ── Parse JSON body ───────────────────────────────────────────────
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_error_response(
                request_id=request_id,
                code="VALIDATION_ERROR",
                message=f"Request body is not valid JSON: {exc}",
                stage="validation",
                http_status=422,
            )
            return

        # ── Validate with Pydantic ────────────────────────────────────────
        try:
            req = RouteRequest.model_validate(payload)
        except ValidationError as exc:
            # Extract the first error message for a clear, specific response
            errors = exc.errors()
            first = errors[0] if errors else {}
            field = ".".join(str(p) for p in first.get("loc", ["prompt"]))
            msg = first.get("msg", str(exc))
            self._send_error_response(
                request_id=request_id,
                code="VALIDATION_ERROR",
                message=f"Validation failed on field '{field}': {msg}",
                stage="validation",
                http_status=422,
            )
            return

        # ── Run the pipeline ──────────────────────────────────────────────
        try:
            result = _handle_v1_route_and_generate(request_id, req)
            self._send_json(200, result.model_dump())
        except _ApiError as api_err:
            _debug_log.error_sent(
                request_id,
                http_status=api_err.http_status,
                error_code=api_err.code,
                stage=api_err.stage,
                message=api_err.message,
            )
            # If routing succeeded but generation failed, attach the routing
            # context so the frontend can still render G1/G2 scores and the
            # tier badge even when there is no response_text.
            body: dict = {
                "request_id": request_id,
                "error": {
                    "code": api_err.code,
                    "message": api_err.message,
                    "stage": api_err.stage,
                },
            }
            if api_err.routing_context is not None:
                body["routing"] = api_err.routing_context
            self._send_json(api_err.http_status, body)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("[%s] Unhandled exception:\n%s", request_id[:8], tb)
            _debug_log.error_sent(
                request_id,
                http_status=500,
                error_code="INTERNAL_ERROR",
                stage="generation",
                message=str(exc),
            )
            self._send_error_response(
                request_id=request_id,
                code="INTERNAL_ERROR",
                message=f"An unexpected internal error occurred: {exc}",
                stage="generation",
                http_status=500,
            )

    # ── Response helpers ──────────────────────────────────────────────────

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_response(
        self,
        request_id: str,
        code: str,
        message: str,
        stage: str,
        http_status: int,
    ):
        error_body = ErrorResponse(
            request_id=request_id,
            error=ErrorDetail(code=code, message=message, stage=stage),  # type: ignore[arg-type]
        )
        self._send_json(http_status, error_body.model_dump())

    def do_OPTIONS(self):
        """Pre-flight CORS for browsers that send OPTIONS first."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cascade Router local test server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help=(
            "Host to bind to (default: 0.0.0.0 — listens on all interfaces). "
            "Use 127.0.0.1 to restrict to localhost only."
        ),
    )
    args = parser.parse_args()

    _init_router()

    server = HTTPServer((args.host, args.port), Handler)
    logger.info("Server ready at http://%s:%d", args.host, args.port)
    logger.info("REST API: POST http://%s:%d/v1/route-and-generate", args.host, args.port)
    logger.info("Health:   GET  http://%s:%d/v1/health", args.host, args.port)
    logger.info("Test UI:  open the above base URL in your browser.")
    logger.info("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
