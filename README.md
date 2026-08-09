# Model Cascade Router

A Python service that routes incoming queries to one of **three LLM tiers** (Small → Medium → Large) using two chained [RouteLLM](https://github.com/lm-sys/RouteLLM) `Controller` instances as binary gatekeepers. Routing decisions are made by RouteLLM's pretrained lightweight routers (`mf` or `bert`) — no heuristics, no local LLM for routing.

---

## Architecture

```
Query ──► Gatekeeper 1 (G1)  [asks: "Tier 1 enough, or escalate?"]
           │  score < threshold_1  →  Tier 1 — Small     ✅ STOP (G2 never called)
           └  score ≥ threshold_1  →  Gatekeeper 2 (G2)  [asks: "Tier 2 enough, or Tier 3?"]
                                         │  score < threshold_2  →  Tier 2 — Medium  ✅ STOP
                                         └  score ≥ threshold_2  →  Tier 3 — Large   ✅ STOP
```

Two independent `Controller` instances are used because RouteLLM only supports binary (weak vs. strong) decisions. The cascade is implemented as explicit application code in [`modelcascader/cascade.py`](modelcascader/cascade.py), not inside RouteLLM.

---

## Quick Start

```bash
# 1. Create a virtualenv with Python 3.12 or earlier (3.13+ lacks pre-built wheels for some deps)
virtualenv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set API keys — only needed for LLM generation, NOT for routing with bert
export GROQ_API_KEY=gsk-...        # Tier 1 — Llama 3.3 70B Versatile
export GEMINI_API_KEY=...          # Tier 2 & 3 — Gemini 2.5 Flash / 3.5 Flash
# export OPENAI_API_KEY=sk-...     # only if using OpenAI tiers
# export ANTHROPIC_API_KEY=...     # only if using Anthropic tiers
# Note: if GOOGLE_API_KEY is also set, google-genai uses it instead of GEMINI_API_KEY.
# Unset GOOGLE_API_KEY or set it to the same value to avoid silent auth failures.

# 4. Run the evaluation harness (routing only — no generation calls)
python eval/evaluate.py

# 5. Use the router in your own code
python - <<'EOF'
from modelcascader import CascadeRouter, load_config, build_router_pool
from modelcascader.telemetry import TelemetryLogger

config = load_config("config/cascade_config.yaml")
pool   = build_router_pool(config)
tel    = TelemetryLogger(config.telemetry)
router = CascadeRouter(config, pool, tel)

result = router.route("Explain the Riemann hypothesis in simple terms.")
print(result.tier)                # e.g. "tier_2"
print(result.g1_score)            # e.g. 0.183
print(result.routing_latency_ms)
EOF
```

---

## Configuration

All knobs live in [`config/cascade_config.yaml`](config/cascade_config.yaml). Zero application code changes are needed to:

- **Change a threshold** — edit `gatekeeper_1.threshold` or `gatekeeper_2.threshold`.
- **Change a tier's model** — edit `tiers.tier_N.model`.
- **Change a tier's provider** — edit `tiers.tier_N.provider` (`"openai"`, `"anthropic"`, `"groq"`, or `"google"`).
- **Change the router type** — edit `gatekeeper_N.router` (`"mf"` or `"bert"`).

---

## Gatekeeper Model Pairs — What They Mean and Why They Must Differ

Each gatekeeper must be configured with the **model pair that matches the actual routing decision it makes**:

| Gatekeeper | Question it answers | `weak_model` | `strong_model` |
|-----------|---------------------|-------------|----------------|
| G1 | "Is Tier 1 (Small) enough?" | Tier 1 model | Tier 2 model |
| G2 | "Is Tier 2 (Medium) enough?" | Tier 2 model | Tier 3 model |

**Never copy-paste the same model pair into both gatekeepers.** If both G1 and G2 share the same pair, they ask the same question and will always produce identical raw scores for every query. Only the threshold would differ — G2 would never actually evaluate whether a query needs Tier 3.

### Router-specific behaviour

The two routers handle model pairs differently:

| Router | Uses `weak_model`/`strong_model`? | API key required? | Score is pair-independent? |
|--------|----------------------------------|------------------|-----------------------------|
| `bert` | ❌ Ignored entirely | No | ✅ Yes — scores identical regardless of pair |
| `mf`   | ✅ Used for model-specific embeddings | Yes (OpenAI) | ❌ No — scores change with pair |

**Implication**: with `router: bert`, even with correctly distinct model pairs, G1 and G2 will still return the same raw score for the same query — because the BERT classifier is purely query-complexity-based and does not consult the model pair at all. The thresholds still differ, so routing decisions can differ, but the underlying *scores* will be numerically identical.

With `router: mf`, distinct model pairs produce genuinely distinct scores because the matrix factorization model looks up model-specific embedding vectors. This is the recommended choice for production where you want fully independent G1 and G2 signals.

**For local dev / offline eval**: `bert` is fine. Just understand that G1 and G2 scores will match numerically — only the threshold comparison differs.  
**For production**: use `mf` with a real `OPENAI_API_KEY` to get truly independent scores per gatekeeper.

---

## Understanding `threshold_1` and `threshold_2`

These are not arbitrary floats you invent. They are the **output of calibration** and represent different routing decisions:

| Threshold | Decision it encodes | Calibration question |
|-----------|--------------------|-----------------------|
| `threshold_1` | "Is this query too complex for Tier 1?" | "What fraction of **all** queries should leave Tier 1?" |
| `threshold_2` | "Of queries that escaped Tier 1, does this one need Tier 3?" | "Of **already-escalated** queries, what fraction should reach Tier 3?" |

A query's **win-probability score** ∈ [0, 1] is produced by the router for each gatekeeper:
- `score < threshold` → route to the **weaker** (cheaper) tier.
- `score ≥ threshold` → escalate to the **stronger** (more capable) tier.

---

## Calibrating Thresholds

`routellm.calibrate_threshold` is a **command-line tool**, run offline once per gatekeeper. It takes a target escalation percentage (`--strong-model-pct`) and returns the threshold float to paste into the YAML.

**You calibrate by percentage, not by picking a float.**

> **Warning — calibration order matters:**
> 1. **Calibrate G1 first** on your full query sample.
> 2. **Calibrate G2 second** on only the subset that G1 actually escalates.
>
> Never calibrate both gatekeepers on the same full dataset. G2 should only ever see the traffic G1 passes up, and calibrating it on the full set will skew its threshold toward queries it will never receive in production.

> **Note on sample size:** The 30-query sample bundled in `sample_queries.txt` is a smoke test only. For reliable threshold calibration, use at least **100+ queries per category** (simple / medium / complex) drawn from your actual query distribution.

### Current model trio (Llama 3.3 70B / Gemini 2.5 Flash / Gemini 3.5 Flash)

Thresholds were calibrated specifically for this trio (2026-07-23). The old GPT-based thresholds (0.44878 / 0.54198) no longer apply and must not be reused.

| Gatekeeper | Calibration target | `--strong-model-pct` | Threshold |
|------------|-------------------|---------------------|----------|
| G1 | 45% of all traffic escalates past Llama 70B | `0.45` | `0.42013` |
| G2 | 25% of escalated traffic reaches Gemini 3.5 Flash | `0.25` | `0.48456` |

**Capability rationale:**
- *Llama 3.3 70B Versatile* — strong open-source model. Handles factual Q&A, basic coding, short explanations, most medium tasks well. Struggles with deep multi-domain reasoning and comprehensive design docs. ~55% of mixed traffic stays at Tier 1.
- *Gemini 2.5 Flash → 3.5 Flash* — adjacent flash-class generations with a narrower gap than GPT-4o-mini→GPT-4o. 2.5 Flash handles most escalated traffic; only the hardest ~25% need 3.5 Flash.

**Eval results on 30-query sample:**

| Tier | Model | Count | % |
|------|-------|------:|--:|
| Tier 1 — Small | `llama-3.3-70b-versatile` (Groq) | 16 | 53.3% |
| Tier 2 — Medium | `gemini-2.5-flash` (Google) | 6 | 20.0% |
| Tier 3 — Large | `gemini-3.5-flash` (Google) | 8 | 26.7% |

Fail-safes triggered: **0 / 30** ✅

### Gatekeeper 1 calibration (run on full traffic sample)

```bash
# Target: 45% of all queries escalate past Tier 1 (Llama 3.3 70B handles ~55%)
python -m routellm.calibrate_threshold \
  --routers bert \
  --strong-model-pct 0.45
# Output: threshold = 0.42013   ← paste into gatekeeper_1.threshold
```

### Gatekeeper 2 calibration (run on G1's escalated subset only)

```bash
# Target: 25% of escalated queries go to Tier 3 (Gemini 3.5 Flash)
python -m routellm.calibrate_threshold \
  --routers bert \
  --strong-model-pct 0.25
# Output: threshold = 0.48456   ← paste into gatekeeper_2.threshold
```

---

## Evaluating Tier Distribution

Before deploying, run the eval harness to verify your thresholds produce a sensible split:

```bash
python eval/evaluate.py --queries sample_queries.txt --out results/eval.jsonl
```

No LLM generation is performed; only the RouteLLM Controllers are called. The output shows:
- % of queries resolved at each tier
- Mean G1 / G2 scores per tier (to spot if thresholds need shifting)
- Fail-safe count (should be 0 in a healthy environment)

With `router: bert`, expect `g1_score` and `g2_score` to be numerically identical for the same query (both read from the same local BERT classifier). What differs is which threshold each is compared against.

To run against your own queries, replace `sample_queries.txt` with any file containing one query per line.

---

## Fail-Safe Behaviour

If a Controller call **errors or times out** (configurable via `routing.timeout_seconds`):

1. The failure is logged at `WARNING` level with the query ID.
2. Routing **escalates** — goes to the next tier up, never silently to the cheapest tier.
3. `fail_safe_triggered: true` is recorded in the telemetry JSONL.

This ensures quality is protected under degraded conditions at the cost of slightly higher spend.

---

## Telemetry

Every routing decision is appended to `logs/routing_telemetry.jsonl` (configurable). Each record is a single JSON object:

```json
{
  "ts": "2024-01-15T10:23:45.123456+00:00",
  "query_id": "a3f1c2d4-...",
  "prompt_preview": "Explain the Riemann hypothesis...",
  "gatekeepers_fired": ["gatekeeper_1"],
  "g1_score": 0.047,
  "g2_score": null,
  "final_tier": "tier_1",
  "tier_label": "Small",
  "routing_latency_ms": 3.8,
  "fail_safe_triggered": false
}
```

Use these logs to analyse score distributions over real traffic and recalibrate thresholds accordingly.



## Adding a New Provider

1. Add the provider literal to `TierConfig.provider` in [`config_loader.py`](modelcascader/config_loader.py).
2. Add a `get_client()` branch and a `_generate_*()` helper in [`providers.py`](modelcascader/providers.py).
3. Update the tier's `provider:` field in `cascade_config.yaml`.

No changes to routing logic (`cascade.py`) are required.

---

## When Swapping Models

Recalibrate from scratch whenever any tier's model changes. **Do not carry forward old thresholds** — the capability gap between tiers changes with every model swap.

1. **Update config + provider dispatch** — set new model IDs in `cascade_config.yaml`, add a provider branch in `providers.py` if the new model uses a different API.
2. **Decide fresh escalation percentages** — based on each new model's real capability level, not inherited from the previous trio. Ask: "What fraction of my actual query mix can this tier handle without escalation?"
3. **Calibrate G1 first** on your full query sample at the new percentage.
4. **Calibrate G2 second** on G1's escalated subset only, at its own fresh percentage.
5. **Verify G2_threshold > G1_threshold** (required when both gatekeepers use `bert` — they share a score axis).
6. **Re-run `eval/evaluate.py`** — confirm all three tiers are populated and the split looks appropriate.
7. **Test generation end-to-end** via the test frontend before trusting the new config in production.

---

## Extending to N Tiers

The current design is intentionally 3-tier. To add a 4th tier, add a `gatekeeper_3` config block, a `tier_4` block under `tiers`, and one more gate in `cascade.py`'s `route()` method following the same pattern. The short-circuit and fail-safe logic generalises cleanly.

---

## Local Test UI

A minimal browser-based test tool lets you type a prompt and see the full routing decision plus the actual generated response.

### Starting the server

```bash
# Activate the venv first
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Set API keys for the tiers you want to test generation on.
# Routing (G1/G2 scoring) works without keys — only generation needs them.
set GROQ_API_KEY=gsk-...        # Tier 1 — Llama 3.3 70B Versatile (Windows)
set GEMINI_API_KEY=...          # Tier 2 & 3 — Gemini 2.5/3.5 Flash (Windows)
# export GROQ_API_KEY=gsk-...   # macOS / Linux
# export GEMINI_API_KEY=...     # macOS / Linux

# ⚠ If GOOGLE_API_KEY is also set in your environment (e.g. from another tool),
#   google-genai will use it silently instead of GEMINI_API_KEY.
#   Unset it or set it to the same value to avoid a confusing auth failure.

# Start the server (default port 8765)
python server.py

# Optional: use a different port
python server.py --port 9000
```

Then open **http://localhost:8765** in your browser.

### What API keys are required

| Tier | Provider | Model | Key needed |
|------|----------|-------|------------|
| Tier 1 — Small | `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Tier 2 — Medium | `google` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| Tier 3 — Large | `google` | `gemini-3.5-flash` | `GEMINI_API_KEY` |

If the required key is missing for the tier a query is routed to, the UI shows a specific error message identifying which key is missing rather than failing silently.

The routing step (G1 + G2 scoring) runs entirely locally with the `bert` router and does **not** need any API key.

### What the UI shows

- **Tier badge** — colour-coded (green = Tier 1, amber = Tier 2, red = Tier 3) with the actual model name
- **Gatekeeper grid** — G1 and G2 scores vs their thresholds, with a plain-text explanation of the decision (e.g. `0.312 < 0.449 → Tier 1 (stop)` or `0.551 ≥ 0.542 → Tier 3`)
- **G2 "not invoked"** — shown when G1 short-circuited to Tier 1 (G2 was never called)
- **Routing latency** and **generation latency** as separate numbers
- **Generated response** from the winning tier's model
- **Specific error messages** if a key is missing or the backend is unreachable

---

## REST API

The server exposes a versioned REST API at `/v1/`. The legacy `/route-and-generate` path remains as a compatibility alias for the test frontend and returns the same response shape.

### `POST /v1/route-and-generate`

Run a prompt through the cascade router and return the generated response with full routing diagnostics.

#### Request

```
Content-Type: application/json

{
  "prompt": "<non-empty string>"
}
```

- `request_id` is **not accepted** from the client — it is always generated server-side (UUID4) and echoed back in every response.
- Whitespace-only prompts are rejected with 422 (same as a missing `prompt` field).

#### Success response — `200 OK`

```json
{
  "request_id": "a3f1c2d4-e5b6-7890-abcd-ef1234567890",
  "final_tier": "tier_1",
  "model_used": "llama-3.3-70b-versatile",
  "provider_used": "groq",
  "response_text": "2 + 2 equals 4.",
  "routing": {
    "gatekeepers_fired": ["gatekeeper_1"],
    "g1_score": 0.083,
    "g1_threshold": 0.42013,
    "g2_score": null,
    "g2_threshold": null
  },
  "timing": {
    "routing_latency_ms": 4.2,
    "generation_latency_ms": 312.7,
    "total_latency_ms": 316.9
  },
  "fail_safe_triggered": false
}
```

`g2_score` and `g2_threshold` are `null` when G1 short-circuited to Tier 1 (G2 was never invoked).

#### Error response — all `4xx` / `5xx`

Every error — regardless of status code — returns the same envelope:

```json
{
  "request_id": "a3f1c2d4-...",
  "error": {
    "code": "MISSING_API_KEY",
    "message": "Authentication failed for provider 'groq' (model: llama-3.3-70b-versatile). Set GROQ_API_KEY in your environment.",
    "stage": "generation"
  }
}
```

| `code` | HTTP status | `stage` | Meaning |
|---|---|---|---|
| `VALIDATION_ERROR` | 422 | `validation` | Missing or empty `prompt`, or malformed JSON body |
| `MISSING_API_KEY` | 424 | `generation` | The API key for the routed tier's provider is absent or invalid |
| `PROVIDER_ERROR` | 502 | `generation` | Provider API returned an error (not auth-related) |
| `TIMEOUT` | 502 | `routing` | A gatekeeper call timed out (fail-safe fires separately) |
| `INTERNAL_ERROR` | 500 | `routing` or `generation` | Unexpected exception — check server logs |

---

### `GET /v1/health`

Quick status check — confirms the service is running, which providers have valid keys configured, and what thresholds are currently loaded.

```json
{
  "status": "ok",
  "providers": {
    "google": true,
    "groq": false
  },
  "thresholds": {
    "gatekeeper_1": 0.42013,
    "gatekeeper_2": 0.48456
  },
  "tiers": {
    "tier_1": { "provider": "groq", "model": "llama-3.3-70b-versatile", "label": "Small" },
    "tier_2": { "provider": "google", "model": "gemini-2.5-flash", "label": "Medium" },
    "tier_3": { "provider": "google", "model": "gemini-3.5-flash", "label": "Large" }
  }
}
```

`providers` values are **booleans only** — the key values themselves are never returned.

---

### `curl` Examples

**Success case:**

```bash
curl -s -X POST http://localhost:8765/v1/route-and-generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}' \
  | python -m json.tool
```

**Missing prompt → 422:**

```bash
curl -s -X POST http://localhost:8765/v1/route-and-generate \
  -H "Content-Type: application/json" \
  -d '{}' \
  | python -m json.tool
# → {"request_id": "...", "error": {"code": "VALIDATION_ERROR", ...}}
```

**Whitespace-only prompt → 422:**

```bash
curl -s -X POST http://localhost:8765/v1/route-and-generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "   "}' \
  | python -m json.tool
```

**Health check:**

```bash
curl -s http://localhost:8765/v1/health | python -m json.tool
```

**Filter debug log by request_id** (replace with the UUID from a real response):

```bash
grep "a3f1c2d4" logs/debug.jsonl | python -m json.tool
```

---

## Debug Logging

Structured JSON debug logs are written to **`logs/debug.jsonl`** (rotating, max 10 MB, 5 backups).

Each line is a self-contained JSON object representing one pipeline event:

```json
{"ts":"2026-08-06T18:00:00.123456+00:00","request_id":"a3f1c2d4-...","event":"REQUEST_RECEIVED","prompt_len":27}
{"ts":"2026-08-06T18:00:00.128456+00:00","request_id":"a3f1c2d4-...","event":"G1_RESULT","score":0.083,"threshold":0.42013,"decision":"tier_1","failed":false}
{"ts":"2026-08-06T18:00:00.129012+00:00","request_id":"a3f1c2d4-...","event":"DISPATCH_STARTED","provider":"groq","model":"llama-3.3-70b-versatile","tier":"tier_1"}
{"ts":"2026-08-06T18:00:00.441234+00:00","request_id":"a3f1c2d4-...","event":"DISPATCH_RESULT","latency_ms":312.2,"success":true}
{"ts":"2026-08-06T18:00:00.441890+00:00","request_id":"a3f1c2d4-...","event":"RESPONSE_SENT","total_latency_ms":316.9,"final_tier":"tier_1","http_status":200}
```

Every line shares the same `request_id`, so a complete per-request trace can be extracted with a single `grep`.

### Events logged per request

| Event | When |
|---|---|
| `REQUEST_RECEIVED` | Body parsed, prompt validated |
| `G1_RESULT` | After G1 returns (or errors) |
| `G2_RESULT` | After G2 returns (if invoked) |
| `FAIL_SAFE_TRIGGERED` | If any gatekeeper errored and fail-safe fired |
| `DISPATCH_STARTED` | Just before provider API call |
| `DISPATCH_RESULT` | After provider returns (success or error) |
| `RESPONSE_SENT` | Final 200 assembled and written |
| `ERROR_SENT` | Any 4xx/5xx sent to the caller |

### `DEBUG_LOG_LEVEL` — controlling verbosity

Set this environment variable before starting the server:

```bash
# Windows
set DEBUG_LOG_LEVEL=DEBUG    # include truncated prompt/response previews
set DEBUG_LOG_LEVEL=INFO     # structural events only (default)
set DEBUG_LOG_LEVEL=WARNING  # errors and warnings only

# macOS / Linux
export DEBUG_LOG_LEVEL=DEBUG
```

### Redaction policy

> **Privacy / security note:** The debug log is designed to be safe to store in shared environments.
>
> - At `INFO` level (default): prompt and response text are **omitted entirely** from all log lines.
> - At `DEBUG` level: a **truncated preview** (first 200 characters) of the prompt and response is included. Full text is never written at any level.
> - API keys are **never** referenced in any log line at any level.
>
> If you need to share debug logs for troubleshooting, `INFO` level is safe to share without scrubbing. `DEBUG`-level logs should be treated as potentially containing user data.

### Telemetry vs debug logs

| | `logs/routing_telemetry.jsonl` | `logs/debug.jsonl` |
|---|---|---|
| **Purpose** | Analytics — one compact row per query for threshold tuning | Debugging — full per-request event trace |
| **Granularity** | One line per request | Several lines per request (one per event) |
| **Prompt text** | First 80 chars (always) | Omitted at INFO; 200-char preview at DEBUG |
| **Use case** | Offline score distribution analysis, calibration | Tracing individual request failures |

---

## File Structure

```
modelcascader/
├── config/
│   └── cascade_config.yaml     # All routing knobs — thresholds, tiers, telemetry
├── modelcascader/
│   ├── __init__.py
│   ├── api_schemas.py          # Pydantic v2 REST API schema (request / response / error)
│   ├── config_loader.py        # Pydantic v2 schema + YAML loader
│   ├── debug_logger.py         # Structured JSON debug event logger
│   ├── router_pool.py          # RouteLLM Controller init + score extraction
│   ├── cascade.py              # Two-gate escalation logic (the orchestration core)
│   ├── telemetry.py            # JSONL rotating log + console logger (analytics)
│   └── providers.py            # Provider dispatch (OpenAI, Anthropic, Groq, Google Gemini)
├── eval/
│   └── evaluate.py             # Tier distribution report over sample queries
├── logs/
│   ├── routing_telemetry.jsonl # Per-query analytics log
│   └── debug.jsonl             # Per-event debug trace log
├── sample_queries.txt          # Example queries spanning all three tiers
├── server.py                   # HTTP server — REST API + test UI
├── README.md
├── requirements.txt
└── pyproject.toml
```
