"""
api_schemas.py
Pydantic v2 models for the POST /v1/route-and-generate REST API contract.

These models define the canonical request/response/error shapes.  They are the
single source of truth — server.py uses them for both validation (input) and
serialisation (output).  Do NOT hardcode JSON key names anywhere else.

Request
-------
POST /v1/route-and-generate
Content-Type: application/json

{
    "prompt": "<non-empty string>"
}

Success response (200)
---------------------
{
    "request_id": "<uuid4>",
    "final_tier": "tier_1" | "tier_2" | "tier_3",
    "model_used":  "<model id string>",
    "provider_used": "<provider string>",
    "response_text": "<generated text>",
    "routing": {
        "gatekeepers_fired": ["gatekeeper_1"] | ["gatekeeper_1", "gatekeeper_2"],
        "g1_score":     <float>,
        "g1_threshold": <float>,
        "g2_score":     <float | null>,
        "g2_threshold": <float | null>
    },
    "timing": {
        "routing_latency_ms":    <float>,
        "generation_latency_ms": <float>,
        "total_latency_ms":      <float>
    },
    "fail_safe_triggered": <bool>
}

Error response (4xx / 5xx)
--------------------------
{
    "request_id": "<uuid4>",
    "error": {
        "code":    "<ERROR_CODE>",
        "message": "<human-readable, specific>",
        "stage":   "validation" | "routing" | "generation"
    }
}

Error codes
-----------
VALIDATION_ERROR   — malformed or missing prompt (HTTP 422)
MISSING_API_KEY    — provider key absent for the routed tier (HTTP 424)
PROVIDER_ERROR     — provider API returned an error (HTTP 502)
TIMEOUT            — routing or generation timed out (HTTP 502)
INTERNAL_ERROR     — unexpected exception (HTTP 500)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    """Validated request body for POST /v1/route-and-generate."""

    prompt: str = Field(..., description="The user query to route and generate a response for.")

    @field_validator("prompt")
    @classmethod
    def prompt_must_be_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("'prompt' must be a non-empty string (whitespace-only is not accepted).")
        return stripped  # return the stripped version so routing never sees leading/trailing whitespace


# ---------------------------------------------------------------------------
# Response — success
# ---------------------------------------------------------------------------

class RoutingInfo(BaseModel):
    """Gatekeeper scores and thresholds for the routing decision."""

    gatekeepers_fired: list[str] = Field(
        ...,
        description="Which gatekeepers were invoked. G2 never appears when G1 short-circuits to Tier 1.",
    )
    g1_score: float | None = Field(..., description="Win-probability score from Gatekeeper 1.")
    g1_threshold: float = Field(..., description="Configured threshold for Gatekeeper 1.")
    g2_score: float | None = Field(None, description="Win-probability score from Gatekeeper 2. null if G2 was not invoked.")
    g2_threshold: float | None = Field(None, description="Configured threshold for Gatekeeper 2. null if G2 was not invoked.")


class TimingInfo(BaseModel):
    """Wall-clock timing broken down by pipeline stage."""

    routing_latency_ms: float = Field(..., description="Time spent in G1+G2 routing (ms).")
    generation_latency_ms: float = Field(..., description="Time spent in provider LLM call (ms).")
    total_latency_ms: float = Field(..., description="Total end-to-end latency (ms).")


class RouteResponse(BaseModel):
    """Success response body for POST /v1/route-and-generate."""

    request_id: str = Field(..., description="Server-generated UUID4 for this request.")
    final_tier: Literal["tier_1", "tier_2", "tier_3"] = Field(..., description="Tier chosen by the router.")
    model_used: str = Field(..., description="Provider-specific model identifier used for generation.")
    provider_used: str = Field(..., description="Provider name (openai, anthropic, groq, google).")
    response_text: str | None = Field(
        None,
        description=(
            "Generated text from the LLM. null only when generation failed but "
            "routing succeeded (the error field is also populated in that case)."
        ),
    )
    routing: RoutingInfo
    timing: TimingInfo
    fail_safe_triggered: bool = Field(
        ..., description="True if a gatekeeper errored and the fail-safe escalation fired."
    )


# ---------------------------------------------------------------------------
# Response — error
# ---------------------------------------------------------------------------

_ErrorCode = Literal[
    "VALIDATION_ERROR",
    "MISSING_API_KEY",
    "PROVIDER_ERROR",
    "TIMEOUT",
    "INTERNAL_ERROR",
]

_ErrorStage = Literal["validation", "routing", "generation"]


class ErrorDetail(BaseModel):
    """Machine-readable error descriptor."""

    code: _ErrorCode = Field(..., description="Stable error code for programmatic handling.")
    message: str = Field(..., description="Human-readable description; names the missing key or failing provider.")
    stage: _ErrorStage = Field(..., description="Pipeline stage where the failure occurred.")


class ErrorResponse(BaseModel):
    """Consistent error envelope returned on all 4xx/5xx responses."""

    request_id: str = Field(..., description="Server-generated UUID4 echoed back for correlation.")
    error: ErrorDetail
