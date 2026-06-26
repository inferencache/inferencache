"""
ttl.py — Temporal validity metadata for cache entries.

Classifies prompts at write time into one of four TTL classes and
provides default policies for expiry computation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "TTLClass",
    "TTLPolicy",
    "DEFAULT_POLICIES",
    "TTLClassifier",
]


class TTLClass(str, Enum):
    PERMANENT = "permanent"
    SESSION = "session"
    TIME_WINDOWED = "time_windowed"
    EPHEMERAL = "ephemeral"


@dataclass
class TTLPolicy:
    ttl_class: TTLClass
    max_age_secs: float | None  # None = no time limit (PERMANENT/SESSION)
    session_bound: bool  # True = invalidated when session_id changes


DEFAULT_POLICIES: dict[TTLClass, TTLPolicy] = {
    TTLClass.PERMANENT: TTLPolicy(TTLClass.PERMANENT, None, False),
    TTLClass.SESSION: TTLPolicy(TTLClass.SESSION, None, True),
    TTLClass.TIME_WINDOWED: TTLPolicy(TTLClass.TIME_WINDOWED, 86400.0, False),  # 24h
    TTLClass.EPHEMERAL: TTLPolicy(TTLClass.EPHEMERAL, 300.0, False),  # 5m
}


class TTLClassifier:
    """
    Lightweight rule-based classifier. No LLM call, no embedding.
    Runs in microseconds.
    """

    _RULES: list[tuple[TTLClass, list[str]]] = [
        # EPHEMERAL: real-time queries
        (
            TTLClass.EPHEMERAL,
            [
                r"\b(open|current|active|right now|at the moment)\b.*\b(prs?|pull requests?|issues?|bugs?|errors?|builds?|deploy)\b",
                r"\b(prs?|pull requests?)\b.*\b(open|current|active|right now|at the moment)\b",
                r"\b(build|pipeline|ci)\b.*(status|passing|failing|running)",
                r"\bwhat('s| is) (running|active|live|deployed)\b",
                r"\b(latest|recent) (error|exception|log|trace)\b",
            ],
        ),
        # SESSION: file/repo/codebase-specific context
        (
            TTLClass.SESSION,
            [
                r"\b(this|the) (file|repo|codebase|project|module|function|class)\b",
                r"\b(summarize|explain|describe).*(file|code|implementation|function)",
                r"\bwhat (does this|is this|is in this)\b",
                r"\b(failing|passing) (test|spec|suite)\b",
                r"\b(my|our) (code|implementation|approach)\b",
            ],
        ),
        # TIME_WINDOWED: current-state queries without real-time urgency
        (
            TTLClass.TIME_WINDOWED,
            [
                r"\b(latest|current|recent|new|updated)\b.*(model|library|version|release|benchmark|paper)",
                r"\bbest practices?\b",
                r"\b(what|which).*(recommend|use|prefer)\b",
                r"\b(news|update|change).*(about|in|for)\b",
                r"\b(how|what).*(in \d{4}|today|this (year|month|week))\b",
            ],
        ),
        # PERMANENT: facts, math, concepts, syntax
        (
            TTLClass.PERMANENT,
            [
                r"\b(what is|explain|define|how does)\b.*(algorithm|concept|formula|theorem|syntax)",
                r"\b(regex|pattern) for\b",
                r"\bhow to\b.*(configure|install|set up)\b",
                r"\b(git|bash|sql|python|javascript)\b.*(command|syntax|flag|option)",
            ],
        ),
    ]

    def classify(self, prompt: str) -> TTLClass:
        lowered = prompt.lower()
        for ttl_class, patterns in self._RULES:
            for pattern in patterns:
                if re.search(pattern, lowered):
                    return ttl_class
        return TTLClass.PERMANENT
