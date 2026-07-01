#!/usr/bin/env python3
"""
cache_tester.py — Interactive REPL for exercising the inferencache proxy.

Wraps the OpenAI chat.completions API but points at your local proxy
instead of api.openai.com, so every message you send goes through the
real intercept → lookup → forward/write-back path. Each response is
annotated with the cache verdict (HIT/MISS, hit type, similarity,
latency) so you can see exactly what the cache is doing in real time.

This intentionally bypasses Claude Code / Cursor and their OAuth
complexity — it's a direct, scriptable way to validate the proxy
mechanics with a plain API key.

Usage:
    export OPENAI_API_KEY=sk-...
    python cache_tester.py

    # Or pass the key directly:
    python cache_tester.py --api-key sk-...

    # Point at a different proxy host/port:
    python cache_tester.py --proxy-url http://localhost:8080

    # Run the built-in scripted test sequence instead of the REPL:
    python cache_tester.py --script

Commands inside the REPL:
    /help            Show available commands
    /script          Run the built-in test sequence
    /model <name>    Switch model (default: gpt-4o-mini)
    /history         Show this session's call history
    /clear           Clear the screen
    /quit or /exit   Exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────
# ANSI color helpers — no external deps, works in any real terminal
# ──────────────────────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    AMBER = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_COLOR = _supports_color()


def paint(text: str, *codes: str) -> str:
    if not _COLOR:
        return text
    return "".join(codes) + text + C.RESET


# ──────────────────────────────────────────────────────────────────────────
# Call result + history
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    prompt: str
    response_text: str
    hit_type: str          # 'exact' | 'semantic' | 'miss'
    similarity: float
    latency_ms: float
    model: str


@dataclass
class Session:
    proxy_url: str
    api_key: str
    model: str = "gpt-4o-mini"
    history: list[CallRecord] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# The actual proxy call — this is the part that exercises inferencache
# ──────────────────────────────────────────────────────────────────────────

class ProxyCallError(Exception):
    pass


def call_proxy(session: Session, prompt: str, *, timeout: float = 60.0) -> CallRecord:
    """
    Send a chat.completions request to the proxy and return a CallRecord
    with the cache verdict attached.

    Reads X-Cache / X-Cache-Similarity / X-Cache-Latency-Ms response
    headers — these are set by inferencache's proxy on every request,
    hit or miss. A 'miss' response won't carry a similarity score above
    0, but inferencache may still report best_similarity via headers in
    future versions; this client reads what's present and defaults
    sanely otherwise.
    """
    url = session.proxy_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "model": session.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session.api_key}",
        },
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            wall_ms = (time.perf_counter() - t0) * 1000
            raw = resp.read()
            headers = resp.headers
            payload = json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise ProxyCallError(f"HTTP {e.code} from proxy: {detail}") from e
    except urllib.error.URLError as e:
        raise ProxyCallError(
            f"Could not reach proxy at {url} — is `inferencache serve` running? ({e.reason})"
        ) from e

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProxyCallError(f"Unexpected response shape from proxy: {payload}")

    hit_type = headers.get("X-Cache", "miss")
    try:
        similarity = float(headers.get("X-Cache-Similarity", "0") or "0")
    except ValueError:
        similarity = 0.0
    try:
        # Prefer the proxy's own reported latency if present; fall back to
        # our own wall-clock measurement (covers the miss/forward path,
        # which the proxy may not annotate with a header).
        latency_ms = float(headers.get("X-Cache-Latency-Ms", "0") or "0")
        if latency_ms <= 0:
            latency_ms = wall_ms
    except ValueError:
        latency_ms = wall_ms

    record = CallRecord(
        prompt=prompt,
        response_text=text,
        hit_type=hit_type,
        similarity=similarity,
        latency_ms=latency_ms,
        model=session.model,
    )
    session.history.append(record)
    return record


# ──────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────

def _verdict_badge(record: CallRecord) -> str:
    if record.hit_type == "exact":
        return paint(" HIT · exact ", C.BOLD, C.GREEN)
    if record.hit_type == "semantic":
        return paint(f" HIT · semantic ({record.similarity:.2f}) ", C.BOLD, C.AMBER)
    return paint(" MISS ", C.BOLD, C.RED)


def print_result(record: CallRecord) -> None:
    badge = _verdict_badge(record)
    meta = paint(f"{record.latency_ms:.1f}ms", C.GRAY)
    print(f"\n{badge}  {meta}")
    print(record.response_text)
    print()


def print_banner(session: Session) -> None:
    line = "─" * 60
    print(paint(line, C.DIM))
    print(paint("  inferencache — cache tester", C.BOLD, C.CYAN))
    print(paint(f"  proxy:  {session.proxy_url}", C.GRAY))
    print(paint(f"  model:  {session.model}", C.GRAY))
    print(paint(line, C.DIM))
    print("Type a prompt and press enter. /help for commands.\n")


def print_help() -> None:
    print(
        """
Commands:
  /help            Show this message
  /script          Run the built-in test sequence (exact, semantic, miss)
  /model <name>    Switch model (current default: gpt-4o-mini)
  /history         Show this session's call history
  /clear           Clear the screen
  /quit, /exit     Exit
"""
    )


def print_history(session: Session) -> None:
    if not session.history:
        print(paint("No calls yet this session.", C.GRAY))
        return
    print()
    for i, r in enumerate(session.history, 1):
        badge = _verdict_badge(r)
        prompt_preview = r.prompt if len(r.prompt) <= 60 else r.prompt[:57] + "..."
        print(f"{i:>2}. {badge}  {paint(prompt_preview, C.GRAY)}")
    print()


# ──────────────────────────────────────────────────────────────────────────
# Built-in scripted test sequence
#
# Designed to exercise all three cache paths in a predictable order so
# you can confirm the proxy is behaving correctly without having to
# improvise prompts on the spot.
# ──────────────────────────────────────────────────────────────────────────

SCRIPT_STEPS: list[tuple[str, str]] = [
    (
        "Baseline miss",
        "What is the capital of France?",
    ),
    (
        "Exact repeat — should HIT exact",
        "What is the capital of France?",
    ),
    (
        "Semantic paraphrase — should HIT semantic",
        "What's France's capital city?",
    ),
    (
        "Different topic — should MISS",
        "Explain how a binary search tree works.",
    ),
    (
        "Exact repeat of step 4 — should HIT exact",
        "Explain how a binary search tree works.",
    ),
    (
        "Loose paraphrase of step 4 — semantic, may or may not hit depending on threshold",
        "Can you walk me through binary search trees?",
    ),
]


def run_script(session: Session) -> None:
    print(paint("\nRunning scripted test sequence...\n", C.BOLD, C.CYAN))
    expectations_met = 0
    for i, (label, prompt) in enumerate(SCRIPT_STEPS, 1):
        print(paint(f"[{i}/{len(SCRIPT_STEPS)}] {label}", C.BOLD))
        print(paint(f'    prompt: "{prompt}"', C.GRAY))
        try:
            record = call_proxy(session, prompt)
        except ProxyCallError as e:
            print(paint(f"    ERROR: {e}", C.RED))
            print()
            continue
        badge = _verdict_badge(record)
        print(f"    → {badge}  {paint(f'{record.latency_ms:.1f}ms', C.GRAY)}")
        preview = record.response_text[:80].replace("\n", " ")
        print(paint(f"    response: {preview}{'...' if len(record.response_text) > 80 else ''}", C.GRAY))
        print()
        # brief pause so the dashboard Live tab visibly updates row-by-row,
        # and so write-back has a moment to complete before the next call
        time.sleep(0.3)

    print(paint("Script complete.", C.BOLD, C.CYAN))
    print(
        paint(
            "Check the dashboard Live tab — you should see 6 rows: "
            "1 miss, 1 exact, 1 semantic, 1 miss, 1 exact, and 1 that depends on your threshold.",
            C.GRAY,
        )
    )
    print()


# ──────────────────────────────────────────────────────────────────────────
# REPL
# ──────────────────────────────────────────────────────────────────────────

def repl(session: Session) -> None:
    print_banner(session)
    while True:
        try:
            line = input(paint("> ", C.CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print_help()
            continue
        if line == "/script":
            run_script(session)
            continue
        if line == "/history":
            print_history(session)
            continue
        if line == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner(session)
            continue
        if line.startswith("/model "):
            session.model = line.split(" ", 1)[1].strip()
            print(paint(f"Model set to: {session.model}", C.GRAY))
            continue
        if line.startswith("/"):
            print(paint(f"Unknown command: {line}. Try /help.", C.RED))
            continue

        try:
            record = call_proxy(session, line)
        except ProxyCallError as e:
            print(paint(f"\nError: {e}\n", C.RED))
            continue
        print_result(record)


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive terminal client for testing the inferencache proxy."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to $OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--proxy-url",
        default=os.environ.get("INFERENCACHE_PROXY_URL", "http://localhost:8080"),
        help="Base URL of the running inferencache proxy (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use for requests (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--script",
        action="store_true",
        help="Run the built-in test sequence once and exit, instead of entering the REPL.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            paint(
                "No API key found. Set $OPENAI_API_KEY or pass --api-key.",
                C.RED,
            )
        )
        sys.exit(1)

    session = Session(proxy_url=args.proxy_url, api_key=args.api_key, model=args.model)

    if args.script:
        print_banner(session)
        run_script(session)
        return

    repl(session)


if __name__ == "__main__":
    main()
