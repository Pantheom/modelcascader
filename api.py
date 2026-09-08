"""
api.py
Routing-only FastAPI service for the Model Cascader.

Endpoint
--------
POST /route
{
    "prompt": "<non-empty string>"
}

Response (200)
--------------
{
    "prompt": "<original prompt>",
    "tier":  1 | 2 | 3,
    "model": "<model id string>",
    "score": <float>          # score of the last gatekeeper that fired
}

Error (422 / 500)
-----------------
{
    "detail": "<message>"
}

Run locally
-----------
    uvicorn api:app --reload --port 8000

Run on EC2
-----------
    nohup uvicorn api:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# RouteLLM similarity_weighted router instantiates OpenAI() at import time.
# Setting a placeholder satisfies the constructor if key is not yet set in env.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "placeholder-for-routellm-import"

from modelcascader import (
    CascadeRouter,
    TelemetryLogger,
    build_router_pool,
    load_config,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App state - initialised once at startup
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and warm up the CascadeRouter exactly once at startup."""
    config_path = os.environ.get("CONFIG_PATH", "config/cascade_config.yaml")
    logger.info("Loading config from: %s", config_path)

    config = load_config(config_path)
    pool = build_router_pool(config)
    telemetry = TelemetryLogger(config.telemetry)
    router = CascadeRouter(config, pool, telemetry)

    _state["router"] = router
    _state["config"] = config

    logger.info("CascadeRouter ready. Serving requests.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Model Cascader - Routing API",
    description="Accepts a prompt and returns the tier, model, and routing score.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API key security
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(key: str = Security(_API_KEY_HEADER)) -> None:
    """Dependency: reject requests that do not supply the correct API key."""
    expected = os.environ.get("API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: API_KEY env var is not set.",
        )
    if key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Supply it in the X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    prompt: str = Field(..., description="The user query to route.")

    @field_validator("prompt")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("prompt must be a non-empty, non-whitespace string.")
        return stripped


class RouteResponse(BaseModel):
    prompt: str
    tier: int = Field(..., description="Tier number: 1, 2, or 3.")
    model: str = Field(..., description="Model identifier chosen for this tier.")
    score: float = Field(
        ...,
        description=(
            "Score of the last gatekeeper that fired. "
            "G1 score when tier=1; G2 score when tier=2 or tier=3."
        ),
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

_TIER_MAP = {"tier_1": 1, "tier_2": 2, "tier_3": 3}


@app.post("/route", response_model=RouteResponse, dependencies=[Depends(_verify_api_key)])
def route_prompt(body: RouteRequest) -> RouteResponse:
    """
    Run the cascade router on the incoming prompt.

    Returns the tier number, model assigned to that tier, and the
    deciding gatekeeper score (for debug purposes).
    """
    router: CascadeRouter = _state["router"]
    config = _state["config"]

    try:
        result = router.route(body.prompt)
    except Exception as exc:
        logger.exception("Routing failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Routing error: {exc}") from exc

    tier_int = _TIER_MAP[result.tier]
    tier_cfg = config.tiers.get(result.tier)

    # Score of the last gatekeeper that actually made the call.
    # tier=1 -> only G1 ran -> return g1_score
    # tier=2 or tier=3 -> G2 ran and made the final call -> return g2_score
    if tier_int == 1:
        deciding_score: float = result.g1_score if result.g1_score is not None else 0.0
    else:
        deciding_score = result.g2_score if result.g2_score is not None else 0.0

    return RouteResponse(
        prompt=body.prompt,
        tier=tier_int,
        model=tier_cfg.model,
        score=round(deciding_score, 6),
    )


# ---------------------------------------------------------------------------
# Health check - useful for EC2 / load balancer probes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}

