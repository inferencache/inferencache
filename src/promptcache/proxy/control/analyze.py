"""Analyze saved runs and produce threshold recommendations."""

from __future__ import annotations

from typing import Any

from . import db as _db

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _counterfactual_hit_rate(calls: list[dict], threshold: float) -> float:
    if not calls:
        return 0.0
    exact = sum(1 for c in calls if c.get("hit_type") == "exact")
    semantic = sum(
        1
        for c in calls
        if c.get("hit_type") == "semantic"
        or (
            c.get("hit_type") == "miss"
            and (c.get("best_similarity") or c.get("similarity") or 0) >= threshold
        )
    )
    return (exact + semantic) / len(calls)


def _confidence(sample_runs: int) -> str:
    if sample_runs >= 5:
        return "high"
    if sample_runs >= 2:
        return "medium"
    return "low"


def analyze(batch_id: str | None = None) -> dict[str, Any]:
    runs = _db.get_all_runs_for_analysis(batch_id)
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        key = (run["suite_name"], run["model"], run.get("cache_mode", "warm"))
        groups.setdefault(key, []).append(run)

    recommendations: list[dict] = []
    near_misses: list[dict] = []
    suite_rankings: list[dict] = []

    for (suite, model, cache_mode), group_runs in groups.items():
        all_calls: list[dict] = []
        for run in group_runs:
            all_calls.extend(run.get("calls", []))

        if not all_calls:
            continue

        best_t = 0.85
        best_rate = 0.0
        curve = []
        for t in THRESHOLDS:
            rate = _counterfactual_hit_rate(all_calls, t)
            curve.append({"threshold": t, "hit_rate": round(rate, 4)})
            if rate >= best_rate:
                best_rate = rate
                best_t = t

        recommendations.append({
            "suite": suite,
            "model": model,
            "cache_mode": cache_mode,
            "optimal_threshold": best_t,
            "expected_hit_rate": round(best_rate, 4),
            "confidence": _confidence(len(group_runs)),
            "sample_runs": len(group_runs),
            "curve": curve,
        })

        current_threshold = group_runs[-1].get("threshold", 0.85) if group_runs else 0.85
        for call in all_calls:
            if call.get("hit_type") != "miss":
                continue
            bs = call.get("best_similarity") or 0
            if bs > 0 and bs >= current_threshold - 0.10:
                near_misses.append({
                    "suite": suite,
                    "model": model,
                    "prompt_preview": call.get("prompt_preview", ""),
                    "best_similarity": round(bs, 4),
                    "threshold": current_threshold,
                    "group_id": call.get("group_id", ""),
                })
        near_misses.sort(key=lambda x: x["best_similarity"], reverse=True)

        avg_hit = sum(r.get("hit_rate", 0) for r in group_runs) / len(group_runs)
        suite_rankings.append({
            "suite": suite,
            "model": model,
            "cache_mode": cache_mode,
            "avg_hit_rate": round(avg_hit, 4),
            "runs": len(group_runs),
        })

    suite_rankings.sort(key=lambda x: x["avg_hit_rate"], reverse=True)
    near_misses = near_misses[:20]

    result = {
        "recommendations": recommendations,
        "near_misses": near_misses,
        "suite_rankings": suite_rankings,
        "total_runs_analyzed": len(runs),
    }

    if recommendations:
        _db.save_tuning(recommendations)

    return result
