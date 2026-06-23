"""
Test suite runner — executes prompt suites and batch experiments.

Extracted from the dashboard backend; shared by the proxy control API.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from itertools import product
from pathlib import Path
from typing import Any

import httpx

from ..state import broadcast_sse, get_engine, set_batch_running
from . import db as _db
from .models import BatchConfig, RunConfig

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUILTIN_SUITES_DIR = _DATA_DIR / "prompt_suites"
PRESETS_DIR = _DATA_DIR / "experiment_presets"


def _upload_suites_dir() -> Path:
    from ..state import get_cache_dir

    d = get_cache_dir() / "suites"
    d.mkdir(parents=True, exist_ok=True)
    return d


_INPUT_COST_PER_TOKEN: dict[str, float] = {
    "openai": 0.000001,
    "anthropic": 0.000003,
}

_PREFIX_DISCOUNT: dict[str, float] = {
    "openai": 0.50,
    "anthropic": 0.90,
}

MODEL_COSTS: dict[str, float] = {
    "gpt-5.5": 0.00003,
    "gpt-5.4": 0.000015,
    "gpt-5.4-mini": 0.0000045,
    "gpt-5.3-chat-latest": 0.000015,
    "gpt-5.2": 0.000014,
    "gpt-5.1": 0.000012,
    "gpt-5.1-mini": 0.000003,
    "gpt-5": 0.00001,
    "gpt-5-mini": 0.000003,
    "gpt-4.1": 0.000008,
    "gpt-4.1-mini": 0.0000016,
    "gpt-4.1-nano": 0.0000004,
    "o4-mini": 0.0000044,
    "o3": 0.00004,
    "o3-mini": 0.0000044,
    "o1": 0.00006,
    "o1-mini": 0.000012,
    "gpt-4o": 0.000005,
    "gpt-4o-mini": 0.0000006,
    "gpt-4-turbo": 0.00001,
    "gpt-4": 0.00003,
    "gpt-3.5-turbo": 0.0000005,
    "claude-opus-4-8": 0.000025,
    "claude-sonnet-4-6": 0.000015,
    "claude-haiku-4-5-20251001": 0.000004,
    "claude-3-5-sonnet-20241022": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
    "claude-sonnet-4-20250514": 0.000015,
}


def estimate_cost(model: str, tokens: int) -> float:
    return round(tokens * MODEL_COSTS.get(model, 0.000003), 8)


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_cached_input_tokens(provider: str, usage: dict) -> int:
    if provider == "openai":
        details = usage.get("prompt_tokens_details") or {}
        return int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    if provider == "anthropic":
        return int(usage.get("cache_read_input_tokens") or 0)
    return 0


def _compute_tier_savings(
    provider: str,
    model: str,
    cached_input_tokens: int,
    tokens_input: int,
    tokens_output: int,
    cost_usd: float,
) -> dict[str, float | int]:
    input_rate = _INPUT_COST_PER_TOKEN.get(provider, 0.000001)
    discount = _PREFIX_DISCOUNT.get(provider, 0.50)
    tier2_cost_saved = round(cached_input_tokens * input_rate * discount, 8)
    tier3_hit = int(tokens_input > 0 and cached_input_tokens >= tokens_input)
    output_cost = tokens_output * MODEL_COSTS.get(model, 0.000003)
    tier3_cost_saved = round(output_cost, 8) if tier3_hit else 0.0
    return {
        "tier2_cached_input_tokens": cached_input_tokens,
        "tier2_cost_saved": tier2_cost_saved,
        "tier3_hit": tier3_hit,
        "tier3_cost_saved": tier3_cost_saved,
    }


def _uses_max_completion_tokens(model: str) -> bool:
    m = model.lower()
    if re.match(r"^o\d", m):
        return True
    if m.startswith("gpt-4o") or m.startswith("chatgpt-4o"):
        return True
    if re.match(r"^gpt-[4-9]\.", m):
        return True
    if re.match(r"^gpt-[5-9]", m):
        return True
    if m.startswith("codex-"):
        return True
    return False


def _is_reasoning_model(model: str) -> bool:
    return bool(re.match(r"^o\d", model.lower()))


def _is_gpt5_family(model: str) -> bool:
    return bool(re.match(r"^gpt-[5-9]", model.lower()))


def _uses_internal_reasoning(model: str) -> bool:
    return _is_reasoning_model(model) or _is_gpt5_family(model)


def _openai_output_token_limits(model: str) -> list[int]:
    if _uses_internal_reasoning(model):
        return [8192, 16384]
    return [1024]


def _openai_token_limit_kwargs(model: str, limit: int) -> dict[str, int]:
    if _uses_max_completion_tokens(model):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


def _openai_chat_body(model: str, prompt: str, limit: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **_openai_token_limit_kwargs(model, limit),
    }
    if _uses_internal_reasoning(model):
        body["reasoning_effort"] = "low"
    return body


def _reasoning_tokens_used(data: dict) -> int:
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or 0)


def _extract_openai_text(data: dict) -> str:
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if message.get("refusal"):
        raise ValueError(f"OpenAI refused: {message['refusal']}")
    if choice.get("finish_reason") == "length":
        reasoning = _reasoning_tokens_used(data)
        detail = f" ({reasoning} hidden reasoning tokens)" if reasoning else ""
        raise ValueError(
            f"OpenAI hit the output token cap before producing text{detail}. "
            "Retrying with a larger budget."
        )
    raise ValueError("OpenAI returned empty content")


async def _openai_completion_once(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    api_key: str,
    limit: int,
) -> tuple[str, int, int, int]:
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=_openai_chat_body(model, prompt, limit),
    )
    if not resp.is_success:
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise ValueError(f"OpenAI {resp.status_code} — {msg}")
    data = resp.json()
    text = _extract_openai_text(data)
    usage = data.get("usage", {})
    tokens_input = usage.get("prompt_tokens", count_tokens(prompt))
    tokens_output = usage.get("completion_tokens", count_tokens(text))
    cached_input = _extract_cached_input_tokens("openai", usage)
    return text, tokens_input, tokens_output, cached_input


async def call_openai(prompt: str, model: str, api_key: str) -> tuple[str, int, int, int]:
    limits = _openai_output_token_limits(model)
    timeout = 180.0 if _uses_internal_reasoning(model) else 60.0
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for limit in limits:
            try:
                return await _openai_completion_once(client, prompt, model, api_key, limit)
            except ValueError as exc:
                msg = str(exc)
                if "Retrying with a larger budget" in msg or "empty content" in msg:
                    last_exc = exc
                    continue
                raise
    raise last_exc or ValueError("OpenAI request failed")


async def call_anthropic(prompt: str, model: str, api_key: str) -> tuple[str, int, int, int]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if not resp.is_success:
            try:
                body = resp.json()
                msg = body.get("error", {}).get("message", resp.text)
            except Exception:
                msg = resp.text
            raise ValueError(f"Anthropic {resp.status_code} — {msg}")
        data = resp.json()
        text = data["content"][0].get("text", "")
        if not text.strip():
            raise ValueError("Anthropic returned empty content")
        usage = data.get("usage", {})
        tokens_input = usage.get("input_tokens", count_tokens(prompt))
        tokens_output = usage.get("output_tokens", count_tokens(text))
        cached_input = _extract_cached_input_tokens("anthropic", usage)
        return text, tokens_input, tokens_output, cached_input


def _suite_search_dirs() -> list[Path]:
    return [_upload_suites_dir(), BUILTIN_SUITES_DIR]


def load_suite(suite_name: str) -> list[str]:
    for suites_dir in _suite_search_dirs():
        json_path = suites_dir / f"{suite_name}.json"
        if json_path.exists():
            data = json.loads(json_path.read_text())
            if isinstance(data, list):
                return [str(p) for p in data]
            if isinstance(data, dict) and "prompts" in data:
                return [str(p) for p in data["prompts"]]
        csv_path = suites_dir / f"{suite_name}.csv"
        if csv_path.exists():
            reader = csv.DictReader(io.StringIO(csv_path.read_text()))
            rows = list(reader)
            if rows and "prompt" in rows[0]:
                col = "prompt"
            elif rows:
                col = list(rows[0].keys())[0]
            else:
                col = "prompt"
            return [r[col] for r in rows if r.get(col)]
    raise ValueError(f"Suite '{suite_name}' not found. Available: {list_suite_names()}")


def list_suite_names() -> list[str]:
    names: set[str] = set()
    for suites_dir in _suite_search_dirs():
        if not suites_dir.exists():
            continue
        for p in suites_dir.iterdir():
            if p.suffix in (".json", ".csv"):
                names.add(p.stem)
    return sorted(names)


def load_suite_groups(suite_name: str) -> dict[str, str]:
    for suites_dir in _suite_search_dirs():
        json_path = suites_dir / f"{suite_name}.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        mapping: dict[str, str] = {}
        for group in data.get("groups", []):
            gid = group.get("id", "")
            for prompt in group.get("prompts", []):
                mapping[str(prompt)] = gid
        return mapping
    return {}


def prompt_hash(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


def expand_matrix(base: dict, matrix: dict) -> list[dict]:
    keys = list(matrix.keys())
    values = [matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys]
    cells = []
    for combo in product(*values):
        cell = {**base}
        for k, v in zip(keys, combo):
            cell[k] = v
        cells.append(cell)
    return cells


def list_presets() -> list[dict]:
    presets = []
    if not PRESETS_DIR.exists():
        return presets
    for p in sorted(PRESETS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        matrix = data.get("matrix", {})
        cell_count = 1
        for v in matrix.values():
            cell_count *= len(v) if isinstance(v, list) else 1
        presets.append({
            "id": p.stem,
            "batch_id": data.get("batch_id", p.stem),
            "description": data.get("description", ""),
            "cell_count": cell_count,
        })
    return presets


def load_preset(preset_id: str) -> dict:
    path = PRESETS_DIR / f"{preset_id}.json"
    if not path.exists():
        raise ValueError(f"Preset '{preset_id}' not found")
    return json.loads(path.read_text())


def save_uploaded_suite(name: str, suffix: str, content: bytes) -> Path:
    dest = _upload_suites_dir() / f"{name}{suffix}"
    dest.write_bytes(content)
    return dest


EmitFn = Callable[[dict], Awaitable[None]]


async def run_suite(config: RunConfig, run_id: str) -> None:
    async def emit(event: dict) -> None:
        await broadcast_sse(json.dumps({"run_id": run_id, **event}))

    try:
        await _run_suite(config, run_id, emit)
    except Exception as exc:
        await emit({"event_type": "error", "message": str(exc)})


async def run_batch(batch: BatchConfig) -> None:
    set_batch_running(True)

    async def emit(event: dict) -> None:
        await broadcast_sse(json.dumps(event))

    try:
        cells = expand_matrix(batch.base, batch.matrix)
        total_cells = len(cells)
        await emit({
            "event_type": "batch_start",
            "batch_id": batch.batch_id,
            "total_cells": total_cells,
            "description": batch.description,
        })

        completed = 0
        skipped = 0

        for cell_idx, cell in enumerate(cells):
            suite = cell.get("suite_name", "general_qa")
            model = cell.get("model", batch.base.get("model", "gpt-4o-mini"))
            threshold = float(cell.get("threshold", batch.base.get("threshold", 0.85)))
            cache_mode = cell.get("cache_mode", batch.base.get("cache_mode", "warm"))

            if batch.skip_existing:
                loop = asyncio.get_event_loop()
                existing = await loop.run_in_executor(
                    None,
                    _db.find_batch_cell,
                    batch.batch_id,
                    suite,
                    model,
                    threshold,
                    cache_mode,
                )
                if existing:
                    skipped += 1
                    await emit({
                        "event_type": "batch_cell_skip",
                        "batch_id": batch.batch_id,
                        "cell_index": cell_idx,
                        "total_cells": total_cells,
                        "run_id": existing,
                        "suite_name": suite,
                        "model": model,
                        "threshold": threshold,
                        "cache_mode": cache_mode,
                    })
                    continue

            run_id = str(uuid.uuid4())[:8]
            config = RunConfig(
                suite_name=suite,
                model=model,
                provider=cell.get("provider", batch.base.get("provider", "openai")),
                threshold=threshold,
                repeat_factor=int(
                    cell.get("repeat_factor", batch.base.get("repeat_factor", 2))
                ),
                delay_between_ms=int(
                    cell.get("delay_between_ms", batch.base.get("delay_between_ms", 200))
                ),
                openai_api_key=batch.openai_api_key,
                anthropic_api_key=batch.anthropic_api_key,
                batch_id=batch.batch_id,
                cache_mode=cache_mode,
            )

            await emit({
                "event_type": "batch_cell_start",
                "batch_id": batch.batch_id,
                "cell_index": cell_idx,
                "total_cells": total_cells,
                "run_id": run_id,
                "suite_name": suite,
                "model": model,
                "threshold": threshold,
                "cache_mode": cache_mode,
            })

            async def cell_emit(event: dict) -> None:
                await broadcast_sse(json.dumps({"run_id": run_id, **event}))

            try:
                await _run_suite(config, run_id, cell_emit)
                completed += 1
            except Exception as exc:
                await emit({
                    "event_type": "batch_cell_error",
                    "batch_id": batch.batch_id,
                    "cell_index": cell_idx,
                    "run_id": run_id,
                    "message": str(exc),
                })

            await emit({
                "event_type": "batch_cell_complete",
                "batch_id": batch.batch_id,
                "cell_index": cell_idx,
                "total_cells": total_cells,
                "run_id": run_id,
                "completed": completed,
                "skipped": skipped,
            })

        await emit({
            "event_type": "batch_complete",
            "batch_id": batch.batch_id,
            "total_cells": total_cells,
            "completed": completed,
            "skipped": skipped,
        })
    finally:
        set_batch_running(False)


async def _run_suite(config: RunConfig, run_id: str, emit: EmitFn) -> None:
    try:
        prompts = load_suite(config.suite_name)
    except ValueError as e:
        await emit({"event_type": "error", "message": str(e)})
        return

    group_map = load_suite_groups(config.suite_name)
    call_sequence: list[str] = []
    for _ in range(config.repeat_factor):
        call_sequence.extend(prompts)

    total = len(call_sequence)
    engine = get_engine(
        config.model,
        config.threshold,
        config.provider,
        default_endpoint="dashboard/run-suite",
    )
    engine.set_threshold(config.threshold)

    if config.cache_mode == "cold":
        engine.cache_store.clear(model=config.model)

    await emit({
        "event_type": "start",
        "total_prompts": total,
        "suite": config.suite_name,
        "model": config.model,
        "threshold": config.threshold,
        "repeat_factor": config.repeat_factor,
        "batch_id": config.batch_id,
        "cache_mode": config.cache_mode,
    })

    summary = {
        "total_calls": 0,
        "cache_hits": 0,
        "exact_hits": 0,
        "semantic_hits": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "total_time_ms": 0.0,
        "api_errors": 0,
    }

    call_records: list[dict] = []
    error_records: list[dict] = []

    for idx, prompt in enumerate(call_sequence):
        await emit({
            "event_type": "call_start",
            "prompt_index": idx,
            "total_prompts": total,
            "prompt_preview": prompt[:80],
            "model": config.model,
        })

        t0 = time.perf_counter()
        result = engine.lookup(prompt, endpoint="dashboard/run-suite", session_id=run_id)
        lookup_ms = result.latency_ms

        tokens = 0
        tokens_input: int | None = None
        tokens_output: int | None = None
        cost = 0.0
        response_text = ""
        tier2_cached_tokens = 0
        tier3_hit = False

        if result.hit:
            response_text = result.response or ""
            tokens = count_tokens(response_text)
            summary["cache_hits"] += 1
            if result.hit_type == "exact":
                summary["exact_hits"] += 1
            else:
                summary["semantic_hits"] += 1
        else:
            try:
                cached_input = 0
                if config.provider == "openai":
                    response_text, tokens_input, tokens_output, cached_input = await call_openai(
                        prompt, config.model, config.openai_api_key
                    )
                elif config.provider == "anthropic":
                    response_text, tokens_input, tokens_output, cached_input = await call_anthropic(
                        prompt, config.model, config.anthropic_api_key
                    )
                else:
                    raise ValueError(f"Unknown provider: {config.provider}")

                tokens = (tokens_input or 0) + (tokens_output or 0)
                cost = estimate_cost(config.model, tokens)
                tier_savings = _compute_tier_savings(
                    config.provider,
                    config.model,
                    cached_input,
                    tokens_input or 0,
                    tokens_output or 0,
                    cost,
                )
                tier2_cached_tokens = int(tier_savings["tier2_cached_input_tokens"])
                tier3_hit = bool(tier_savings["tier3_hit"])
                engine.store(
                    prompt,
                    response_text,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cost_usd=cost,
                    endpoint="dashboard/run-suite",
                    session_id=run_id,
                    tier2_cached_input_tokens=tier2_cached_tokens,
                    tier3_hit=int(tier3_hit),
                    tier2_cost_saved=float(tier_savings["tier2_cost_saved"]),
                    tier3_cost_saved=float(tier_savings["tier3_cost_saved"]),
                )
            except Exception as exc:
                summary["api_errors"] += 1
                err_event = {
                    "event_type": "error",
                    "prompt_index": idx,
                    "total_prompts": total,
                    "prompt_preview": prompt[:80],
                    "model": config.model,
                    "message": str(exc),
                }
                error_records.append(err_event)
                await emit(err_event)
                continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary["total_calls"] += 1
        summary["total_tokens"] += tokens
        summary["total_cost_usd"] += cost
        summary["total_time_ms"] += elapsed_ms

        prompts_per_round = len(prompts)
        round_idx = idx // prompts_per_round if prompts_per_round else 0

        call_event = {
            "event_type": "call",
            "prompt_index": idx,
            "total_prompts": total,
            "prompt_preview": prompt[:80],
            "hit": result.hit,
            "hit_type": result.hit_type,
            "similarity": result.similarity,
            "best_similarity": result.best_similarity,
            "latency_ms": round(elapsed_ms, 1),
            "lookup_ms": round(lookup_ms, 1),
            "tokens_used": tokens,
            "cost_usd": cost,
            "model": config.model,
            "response_preview": response_text[:120],
            "prompt_hash": prompt_hash(prompt, config.model),
            "group_id": group_map.get(prompt, ""),
            "round_idx": round_idx,
            "call_id": result.call_id,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "endpoint": "dashboard/run-suite",
            "session_id": run_id,
            "matched_prompt": (
                result.entry.prompt
                if result.hit_type == "semantic" and result.entry
                else None
            ),
            "tier1_hit": result.hit or result.tier1_hit,
            "tier2_cached_tokens": tier2_cached_tokens,
            "tier3_hit": tier3_hit,
        }
        call_records.append(call_event)
        await emit(call_event)

        if config.delay_between_ms > 0:
            await asyncio.sleep(config.delay_between_ms / 1000)

    hit_rate = summary["cache_hits"] / summary["total_calls"] if summary["total_calls"] else 0.0
    cpt = MODEL_COSTS.get(config.model, 0.000003)
    tokens_saved = 0
    cost_saved = 0.0
    for rec in call_records:
        if rec.get("hit") and rec.get("hit_type") != "miss":
            tok = rec.get("tokens_used") or 200
            tokens_saved += tok
            cost_saved += tok * cpt

    final_summary = {
        **summary,
        "hit_rate": round(hit_rate, 4),
        "tokens_saved": tokens_saved,
        "cost_saved": round(cost_saved, 8),
        "prompt_index": total,
        "total_prompts": total,
        "error_messages": [e["message"] for e in error_records[:8]],
    }
    await emit({"event_type": "summary", **final_summary})

    config_dict = config.model_dump()
    config_dict.pop("openai_api_key", None)
    config_dict.pop("anthropic_api_key", None)
    if summary["api_errors"] > 0 and summary["total_calls"] == 0:
        config_dict["status"] = "error"
    elif summary["api_errors"] > 0:
        config_dict["status"] = "partial"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _db.save_run,
        run_id,
        config_dict,
        final_summary,
        call_records,
        error_records,
    )
