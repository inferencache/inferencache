"""Dashboard control REST API mounted at /api."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..state import (
    get_analytics,
    get_cache_dir,
    get_engine,
    is_batch_running,
    query_index_db,
    register_sse_client,
    unregister_sse_client,
    write_index_db,
)
from . import analyze as _analyze
from . import db as _db
from .models import ApplyThresholdRequest, BatchConfig, FlagRequest, RunConfig, ThresholdUpdate
from .runner import (
    expand_matrix,
    list_presets,
    list_suite_names,
    load_preset,
    run_batch,
    run_suite,
    save_uploaded_suite,
)

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    return {"status": "ok", "cache_dir": str(get_cache_dir())}


@router.get("/suites")
async def get_suites():
    return {"suites": list_suite_names()}


@router.post("/upload-suite")
async def upload_suite(file: UploadFile = File(...)):
    content = await file.read()
    name = (file.filename or "custom").rsplit(".", 1)[0]
    suffix = "." + (file.filename or "custom.json").rsplit(".", 1)[-1].lower()
    if suffix not in (".json", ".csv"):
        raise HTTPException(400, "Only .json and .csv files are accepted")
    dest = save_uploaded_suite(name, suffix, content)
    return {"saved": name, "path": str(dest)}


@router.post("/run-suite")
async def start_run(config: RunConfig, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_suite, config, run_id)
    return {"run_id": run_id}


@router.get("/events")
async def sse_events():
    client_q = register_sse_client()

    async def generator() -> AsyncIterator[str]:
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(client_q.get(), timeout=30)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            unregister_sse_client(client_q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stats")
async def get_stats(model: str = "gpt-4o-mini"):
    engine = get_engine(model, default_endpoint="dashboard/run-suite")
    stats = engine.cache_store.stats(top_n=10)
    return {
        "total_entries": stats.total_entries,
        "total_hits": stats.total_hits,
        "exact_hits": stats.exact_hits,
        "semantic_hits": stats.semantic_hits,
        "hit_rate": stats.hit_rate,
        "top_entries": stats.top_entries,
    }


@router.post("/clear")
async def clear_cache(model: str | None = None):
    from ..state import all_engines

    for key, engine in list(all_engines().items()):
        engine_model = key.split(":")[0]
        if model is None or engine_model == model:
            deleted = engine.cache_store.clear(model=model)
            return {"deleted": deleted, "model": model or "all"}
    return {"deleted": 0}


@router.post("/set-threshold")
async def set_threshold(update: ThresholdUpdate):
    engine = get_engine(update.model, default_endpoint="dashboard/run-suite")
    engine.set_threshold(update.threshold)
    return {"threshold": update.threshold, "model": update.model}


@router.get("/analytics/hit-rate")
@router.get("/cache-stats/hit-rate")
async def analytics_hit_rate(
    model: str = "gpt-4o-mini",
    window_hours: int = 24,
    bucket_minutes: int = 30,
):
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None, get_analytics().hit_rate_over_time, model, window_hours, bucket_minutes
    )
    return {"data": rows, "model": model, "window_hours": window_hours}


@router.get("/analytics/cost-saved")
@router.get("/cache-stats/cost-saved")
async def analytics_cost_saved(model: str = "gpt-4o-mini", window_hours: int = 24):
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None, get_analytics().cost_saved_cumulative, model, window_hours
    )
    return {"data": rows, "model": model, "window_hours": window_hours}


@router.get("/analytics/endpoints")
@router.get("/cache-stats/endpoints")
async def analytics_endpoints(
    model: str = "gpt-4o-mini",
    window_hours: int = 24,
    limit: int = 20,
):
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None, get_analytics().endpoint_breakdown, model, window_hours, limit
    )
    return {"data": rows, "model": model, "window_hours": window_hours}


@router.get("/analytics/similarity-dist")
@router.get("/cache-stats/similarity-dist")
async def analytics_similarity_dist(
    model: str = "gpt-4o-mini",
    window_hours: int = 24,
    buckets: int = 20,
):
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None, get_analytics().similarity_distribution, model, window_hours, buckets
    )
    return {"data": rows, "model": model, "window_hours": window_hours}


@router.get("/analytics/generative-hit-rate")
@router.get("/cache-stats/generative-hit-rate")
async def analytics_generative_hit_rate(
    model: str = "gpt-4o-mini",
    window_hours: int = 24,
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, get_analytics().generative_hit_rate, model, window_hours
    )
    return {"data": result, "model": model, "window_hours": window_hours}


@router.get("/analytics/stale-miss-rate")
@router.get("/cache-stats/stale-miss-rate")
async def analytics_stale_miss_rate(
    model: str = "gpt-4o-mini",
    window_hours: int = 24,
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, get_analytics().stale_miss_rate, model, window_hours
    )
    return {"data": result, "model": model, "window_hours": window_hours}


@router.get("/analytics/tier-breakdown")
@router.get("/cache-stats/tier-breakdown")
async def analytics_tier_breakdown(model: str = "gpt-4o-mini", window_hours: int = 24):
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None, get_analytics().tier_breakdown, model, window_hours
    )
    return {"data": rows, "model": model, "window_hours": window_hours}


@router.get("/alerts/check")
async def alerts_check(model: str = "gpt-4o-mini", budget_usd_per_hour: float = 1.0):
    loop = asyncio.get_event_loop()
    state = await loop.run_in_executor(
        None, lambda: get_analytics().alert_check(model, budget_usd_per_hour)
    )
    return state


@router.post("/calls/{call_id}/flag")
async def flag_call(call_id: int, body: FlagRequest):
    rows = query_index_db("SELECT hit_type FROM calls WHERE id = ?", (call_id,))
    if not rows:
        raise HTTPException(404, f"Call {call_id} not found")
    if rows[0]["hit_type"] != "semantic":
        raise HTTPException(400, "Only semantic hits can be flagged as false positives")
    write_index_db(
        "UPDATE calls SET false_positive = ? WHERE id = ?",
        (1 if body.flagged else 0, call_id),
    )
    return {"call_id": call_id, "flagged": body.flagged}


@router.get("/tuning/false-positives")
async def tuning_false_positives(model: str = "gpt-4o-mini", limit: int = 50):
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None, get_analytics().false_positive_queue, model, limit
    )
    return {"data": rows, "model": model}


@router.get("/calls")
async def list_calls(
    endpoint: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    conditions = []
    params: list = []
    if endpoint is not None:
        conditions.append("endpoint = ?")
        params.append(endpoint)
    if session_id is not None:
        conditions.append("session_id = ?")
        params.append(session_id)
    if model is not None:
        conditions.append("model = ?")
        params.append(model)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params += [limit, offset]
    rows = query_index_db(
        f"SELECT * FROM calls {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        tuple(params),
    )
    return {"data": rows, "limit": limit, "offset": offset}


@router.get("/runs")
async def list_runs(limit: int = 50, offset: int = 0, batch_id: str | None = None):
    loop = asyncio.get_event_loop()
    runs = await loop.run_in_executor(None, _db.list_runs, limit, offset, batch_id)
    total = await loop.run_in_executor(None, _db.count_runs)
    return {"runs": runs, "total": total}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    loop = asyncio.get_event_loop()
    run = await loop.run_in_executor(None, _db.get_run, run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    loop = asyncio.get_event_loop()
    deleted = await loop.run_in_executor(None, _db.delete_run, run_id)
    if not deleted:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return {"deleted": run_id}


@router.get("/experiment-presets")
async def get_presets():
    return {"presets": list_presets()}


@router.get("/experiment-presets/{preset_id}")
async def get_preset(preset_id: str):
    try:
        return load_preset(preset_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/run-batch")
async def start_batch(batch: BatchConfig, background_tasks: BackgroundTasks):
    if is_batch_running():
        raise HTTPException(409, "A batch is already running")
    if not batch.batch_id:
        batch.batch_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_batch, batch)
    cells = expand_matrix(batch.base, batch.matrix)
    return {"batch_id": batch.batch_id, "total_cells": len(cells)}


@router.get("/batch-status")
async def batch_status():
    return {"running": is_batch_running()}


@router.get("/entries")
async def get_entries(model: str = "gpt-4o-mini", limit: int = 500):
    """Return cache entries for the map view."""
    engine = get_engine(model, default_endpoint="dashboard/run-suite")
    entries = engine.cache_store.list_entries(limit=limit)
    return {"entries": entries}


@router.get("/analyze")
async def analyze_runs(batch_id: str | None = None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _analyze.analyze, batch_id)


@router.get("/recommendations")
async def get_recommendations():
    return _db.load_tuning()


@router.post("/apply-threshold")
async def apply_threshold(req: ApplyThresholdRequest):
    tuning = _db.load_tuning()
    recs = tuning.get("recommendations", [])
    if not recs:
        raise HTTPException(404, "No recommendations available — run /analyze first")

    applied = None
    for rec in recs:
        if req.suite_name and rec.get("suite") != req.suite_name:
            continue
        if req.model and rec.get("model") != req.model:
            continue
        if rec.get("cache_mode", "cold") != req.cache_mode:
            continue
        model = rec["model"]
        threshold = rec["optimal_threshold"]
        engine = get_engine(model, default_endpoint="dashboard/run-suite")
        engine.set_threshold(threshold)
        applied = {"model": model, "suite": rec["suite"], "threshold": threshold}
        break

    if not applied:
        rec = recs[0]
        engine = get_engine(rec["model"], default_endpoint="dashboard/run-suite")
        engine.set_threshold(rec["optimal_threshold"])
        applied = {
            "model": rec["model"],
            "suite": rec["suite"],
            "threshold": rec["optimal_threshold"],
        }

    return {"applied": applied}
