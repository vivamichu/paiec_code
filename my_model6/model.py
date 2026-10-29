"""S4c: S4b with a stronger prior (4 instead of 2) when Claude's guess is used. S4b: S3 plus Claude's guess as the starting point while few labels are known
(BLE's "ask the LLM" idea, one call per item, no agent loop).

Claude part: while this pair has at most LLM_MAX_LABELS own labels,
             p0 starts from Claude's guess pulled toward 0.5:
             p0 = 0.5 + LAM * (claude - 0.5)
             Public test (200 items, 2026-10-22): raw guess is overconfident,
             LAM about 0.5 gave Brier 0.236 vs 0.25 at budget 0.
S3 part:     p0 also moves toward the siblings' rate on this benchmark (W).
S1 part:     p = (PAIR_STRENGTH * p0 + s) / (PAIR_STRENGTH + n)
S2 part:     nudge p by how siblings did on THIS item vs their usual rate (G).

No key, API error, timeout or unreadable reply: Claude part is skipped (= S3).
Standard library only. The key is read from submission_config.json
({"api_keys": {"ANTHROPIC_API_KEY": "..."}}) or the ANTHROPIC_API_KEY variable.
"""

import json
import os
import re
import time
import urllib.request
from pathlib import Path

PRIOR_MEAN = 0.5    # base rate for an unseen subject-benchmark pair
PAIR_STRENGTH = 2   # pseudo-count of the prior
LLM_PAIR_STRENGTH = 4  # pseudo-count when the prior is Claude's guess (tuned on seed 0; S4b used 2)
G = 0.4             # how strongly a sibling's result on the same item moves p
W = 0.8             # how far the siblings' benchmark rate moves p0
LAM = 0.75          # how far Claude's guess moves p0 (tuned on seed 0; 0 = S3)
LLM_MAX_LABELS = 31 # use Claude while the pair has at most this many labels (tuned: all budgets)
LLM_SHIFT = 0.0     # added to Claude's pulled guess: corrects a too low / too high average (to tune)

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 2000   # thinking counts against this; too small gives empty replies
TIMEOUT = 20        # seconds per API request
MAX_FAILURES = 5    # after this many failed questions in a row, stop asking (API down or key empty)
MAX_ITEM_CHARS = 4000


def read_key():
    try:
        config = json.loads((Path(__file__).parent / "submission_config.json").read_text())
        key = config["api_keys"]["ANTHROPIC_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


API_KEY = read_key()
_replies = {}   # prompt -> reply text, within one worker
_failures = [0]  # failed questions in a row, within one worker


# ---- Claude (prompt written by the user, dev/my_prompt.py) ----

def shorten(text, max_chars):
    """Cut long text. Item texts can be 100,000+ characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[cut]"


def build_prompt(subject, item):
    """Return the text we send to Claude for this (subject, item)."""
    # cut the long item text and drop the anonymous benchmark id
    item = dict(item)
    item["item_content"] = shorten(item["item_content"], MAX_ITEM_CHARS)
    item.pop("benchmark_id", None)

    prompt = """
    You are a LLM expert. You know the capabilities of many LLMs, and you can predict how well they will do on a task. 
    You are given a subject (a LLM) and an item (a task). Your job is to predict the probability that the subject will answer the item correctly.
    Subject information:""" + str(subject) + """
    Item information:""" + str(item) + """

    Please provide a single number between 0 and 1, representing the probability that the subject will answer the item correctly. 
    Avoid giving a probability of exactly 0 or 1, as we want to avoid overconfidence. Make sure to consider the subject's capabilities, the task's difficulty, and any other relevant information when making your prediction.
    Your answer should be a single number, with no additional text or explanation.
    """
    return prompt.strip()


def parse_probability(reply):
    """Turn Claude's reply into a float between 0 and 1. Return None if you cannot."""
    if reply is None:
        return None
    # first number in the text, like 0.7, .7 or 65
    match = re.search(r"\d*\.?\d+", reply)
    if match is None:
        return None
    number = float(match.group())
    # 65 or 65% means 0.65
    if number > 1:
        number = number / 100
    if number < 0 or number > 1:
        return None
    return number


def ask_claude(prompt):
    """One API request. Returns the reply text, or None on any problem.
    Local tests replace this function with a cache lookup (dev/evaluator.py)."""
    if API_KEY == "" or _failures[0] >= MAX_FAILURES:
        return None
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": prompt}],
    }
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2 * attempt)   # rate limit or network hiccup: wait, then retry
        try:
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode(),
                headers={
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                result = json.loads(response.read())
            text = ""
            for block in result["content"]:
                if block["type"] == "text":
                    text = text + block["text"]
            _failures[0] = 0
            return text
        except Exception:
            continue
    _failures[0] = _failures[0] + 1
    return None


def claude_guess(subject, item):
    """Claude's probability for this item, or None."""
    prompt = build_prompt(subject, item)
    if prompt not in _replies:
        _replies[prompt] = ask_claude(prompt)
    return parse_probability(_replies[prompt])


# ---- helpers (plumbing) ----

def item_key(item):
    """Two items are the same item if benchmark, text and features match.
    interactors are left out: they can differ between subjects."""
    return (item.get("benchmark_id", ""), item.get("item_content", ""), item.get("item_features", ""))


def pair_counts(labeled, subject, benchmark_id):
    """Successes and number of labels of one subject on one benchmark."""
    successes = 0
    count = 0
    for (obs_subject, obs_item), label in labeled:
        if obs_subject == subject and obs_item.get("benchmark_id", "") == benchmark_id:
            successes = successes + int(label)
            count = count + 1
    return successes, count


def sibling_counts(labeled, subject, benchmark_id):
    """Successes and number of labels of all OTHER subjects on this benchmark."""
    successes = 0
    count = 0
    for (obs_subject, obs_item), label in labeled:
        if obs_subject != subject and obs_item.get("benchmark_id", "") == benchmark_id:
            successes = successes + int(label)
            count = count + 1
    return successes, count


def sibling_labels(labeled, subject, item):
    """Labels on the same item from OTHER subjects: list of (other_subject, label)."""
    target = item_key(item)
    found = []
    for (obs_subject, obs_item), label in labeled:
        if obs_subject != subject and item_key(obs_item) == target:
            found.append((obs_subject, int(label)))
    return found


def smoothed(successes, count, prior_mean=PRIOR_MEAN, strength=PAIR_STRENGTH):
    return (strength * prior_mean + successes) / (strength + count)


# ---- the model ----

def predict(input, labeled=None):
    subject, item = input
    if labeled is None:
        labeled = []
    benchmark_id = item.get("benchmark_id", "")
    s, n = pair_counts(labeled, subject, benchmark_id)

    # Claude: starting point while few labels are known
    p0 = PRIOR_MEAN
    strength = PAIR_STRENGTH
    if LAM > 0 and n <= LLM_MAX_LABELS:
        guess = claude_guess(subject, item)
        if guess is not None:
            p0 = PRIOR_MEAN + LAM * (guess - PRIOR_MEAN) + LLM_SHIFT
            strength = LLM_PAIR_STRENGTH   # Claude's guess is worth more than a flat 0.5

    # S3: move toward the siblings' rate on this benchmark
    s_sib, n_sib = sibling_counts(labeled, subject, benchmark_id)
    sibling_rate = smoothed(s_sib, n_sib)   # 0.5 when there are no sibling labels
    p0 = p0 + W * (sibling_rate - PRIOR_MEAN)
    if p0 < 0.02:
        p0 = 0.02
    if p0 > 0.98:
        p0 = 0.98

    # S1: own labels on this benchmark
    p = smoothed(s, n, p0, strength)

    # S2: how did siblings do on THIS item, compared with their usual rate?
    differences = []
    for other_subject, label in sibling_labels(labeled, subject, item):
        s_other, n_other = pair_counts(labeled, other_subject, benchmark_id)
        # leave this label out of the sibling's usual rate
        usual = smoothed(s_other - label, n_other - 1)
        differences.append(label - usual)

    if len(differences) > 0:
        p = p + G * sum(differences) / len(differences)

    # never fully certain
    if p < 0.02:
        p = 0.02
    if p > 0.98:
        p = 0.98
    return float(p)
