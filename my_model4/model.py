"""S4: S3 plus the subject's public track record (BLE's find_model / get_scores idea).

S1 part:   p = (PAIR_STRENGTH * p0 + s) / (PAIR_STRENGTH + n)
           s, n = successes and labels of THIS subject on THIS benchmark.
S3 part:   p0 moves from 0.5 toward the siblings' rate on this benchmark (W).
S4 part:   p0 also moves by the subject's usual gap on public benchmarks (V):
           gap = its rate minus the benchmark average, averaged over the public
           benchmarks where its normalized_name appears (track_record.json),
           shrunk toward 0 with GAP_STRENGTH extra benchmarks at gap 0.
S2 part:   nudge p by how siblings did on THIS item vs their usual rate (G).

Unknown name and no sibling labels: identical to S2 (and to S1 without item siblings).
Standard library only; stateless.
"""

import json
from pathlib import Path

PRIOR_MEAN = 0.5    # base rate for an unseen subject-benchmark pair
PAIR_STRENGTH = 2   # pseudo-count of the prior
G = 0.4             # how strongly a sibling's result on the same item moves p
W = 0.8             # how far the siblings' benchmark rate moves p0
V = 1.0             # how far the public track record moves p0 (to tune)
GAP_STRENGTH = 1    # shrink the track record: extra benchmarks with gap 0

# name -> {public benchmark: gap}
TRACK_RECORD = {}
try:
    TRACK_RECORD = json.loads((Path(__file__).parent / "track_record.json").read_text())
except Exception:
    TRACK_RECORD = {}

# local testing only: anonymous benchmark id -> public benchmark to leave out.
# Empty on the platform (hidden benchmarks are never in the public table).
EXCLUDE = {}


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


def smoothed(successes, count, prior_mean=PRIOR_MEAN):
    return (PAIR_STRENGTH * prior_mean + successes) / (PAIR_STRENGTH + count)


def track_record_gap(subject, benchmark_id):
    """The subject's usual gap on public benchmarks, shrunk toward 0."""
    name = subject.get("normalized_name", "")
    if name not in TRACK_RECORD:
        return 0.0
    skip = EXCLUDE.get(benchmark_id, "")
    total = 0.0
    count = 0
    for bench, gap in TRACK_RECORD[name].items():
        if bench != skip:
            total = total + gap
            count = count + 1
    return total / (count + GAP_STRENGTH)


# ---- the model ----

def predict(input, labeled=None):
    subject, item = input
    if labeled is None:
        labeled = []
    benchmark_id = item.get("benchmark_id", "")

    # S3: start from the siblings' rate on this benchmark, pulled toward 0.5
    s_sib, n_sib = sibling_counts(labeled, subject, benchmark_id)
    sibling_rate = smoothed(s_sib, n_sib)   # 0.5 when there are no sibling labels
    p0 = PRIOR_MEAN + W * (sibling_rate - PRIOR_MEAN)

    # S4: plus the subject's usual gap on public benchmarks
    p0 = p0 + V * track_record_gap(subject, benchmark_id)
    if p0 < 0.02:
        p0 = 0.02
    if p0 > 0.98:
        p0 = 0.98

    # S1: own labels on this benchmark
    s, n = pair_counts(labeled, subject, benchmark_id)
    p = smoothed(s, n, p0)

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
