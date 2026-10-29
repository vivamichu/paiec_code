"""Small Claude client using only the standard library (like the competition runtime).

ask(prompt) returns the reply text and records tokens and cost.
Every reply is saved in data/cache/claude_replies.jsonl, so asking the same
prompt again is free.

The key is read from ANTHROPIC_API_KEY. It is never printed.
"""

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
REPLY_CACHE = KIT / "data" / "cache" / "claude_replies.jsonl"

# dollars per million tokens (input, output); Haiku price is for prompts up to 100K
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}

MODEL = "claude-haiku-5-5"
PRICE_IN = 0.10 / 1000000    # dollars per input token
PRICE_OUT = 0.50 / 1000000   # dollars per output token (thinking tokens count here too)


def parse_last_line(reply):
    """Read the probability from the reply's LAST line; None if that line is not just a number.

    Sonnet 5.5 explains first and puts the number on the last line, so my_model5's
    first-number rule often picks a stray number (a year, a model size).
    Accepted last lines: 0.3  **0.3**  30%  <answer>0.3</answer>  Answer: 0.3  Probability: 0.3
    A reply that ends in reasoning (or is cut off) gives None: no guess.
    """
    if reply is None:
        return None
    lines = []
    for line in reply.strip().splitlines():
        if line.strip() != "":
            lines.append(line.strip())
    if len(lines) == 0:
        return None
    line = lines[-1]
    line = re.sub(r"<[^>]*>", "", line)   # drop tags like <answer>
    line = line.replace("*", "")          # drop bold
    line = re.sub(r"^(answer|probability)\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
    line = line.strip().rstrip(".").rstrip("%").strip()
    if re.fullmatch(r"\d*\.?\d+", line) is None:
        return None
    number = float(line)
    # 65 or 65% means 0.65
    if number > 1:
        number = number / 100
    if number < 0 or number > 1:
        return None
    return number


def set_model(name):
    """Switch every later ask() and make_key() to this model and its prices."""
    global MODEL, PRICE_IN, PRICE_OUT
    MODEL = name
    PRICE_IN = PRICES[name][0] / 1000000
    PRICE_OUT = PRICES[name][1] / 1000000

# running totals for this process
totals = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "dollars": 0.0}
_lock = threading.Lock()   # several threads may call ask() at once


def load_cache():
    cache = {}
    if REPLY_CACHE.exists():
        for line in REPLY_CACHE.read_text().splitlines():
            row = json.loads(line)
            cache[row["key"]] = row
    return cache


_cache = load_cache()


def make_key(prompt, system, max_tokens, effort, thinking):
    text = json.dumps([MODEL, prompt, system, max_tokens, effort, thinking])
    return hashlib.sha1(text.encode()).hexdigest()


def ask(prompt, system=None, max_tokens=2000, effort="low", thinking=None):
    """Send one prompt. Returns the reply text.

    thinking="disabled" turns the model's hidden thinking off (shorter, cheaper).
    Thinking tokens count against max_tokens: a too-small limit gives an empty reply.
    """
    key = make_key(prompt, system, max_tokens, effort, thinking)
    if key in _cache:
        totals["cached"] = totals["cached"] + 1
        return _cache[key]["text"]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key == "":
        raise RuntimeError("ANTHROPIC_API_KEY is not set in this shell")

    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    if thinking == "disabled":
        body["thinking"] = {"type": "disabled"}

    result = None
    for attempt in range(4):
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode()[:300]
            if error.code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(2 ** attempt * 2)   # wait and try again
                continue
            raise RuntimeError("Claude API error " + str(error.code) + ": " + detail)

    text = ""
    for block in result["content"]:
        if block["type"] == "text":
            text = text + block["text"]
    usage = result["usage"]
    dollars = usage["input_tokens"] * PRICE_IN + usage["output_tokens"] * PRICE_OUT

    row = {"key": key, "text": text, "stop_reason": result.get("stop_reason"),
           "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
    with _lock:
        totals["calls"] = totals["calls"] + 1
        totals["input_tokens"] = totals["input_tokens"] + usage["input_tokens"]
        totals["output_tokens"] = totals["output_tokens"] + usage["output_tokens"]
        totals["dollars"] = totals["dollars"] + dollars
        _cache[key] = row
        REPLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(REPLY_CACHE, "a") as file:
            file.write(json.dumps(row) + "\n")
    return text
