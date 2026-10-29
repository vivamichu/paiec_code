"""S3: S2 plus a benchmark-level prior from other subjects ("siblings").

S1 part:   p = (PAIR_STRENGTH * p0 + s) / (PAIR_STRENGTH + n)
           s, n = successes and labels of THIS subject on THIS benchmark.
S3 part:   p0 is 0.5 moved toward the siblings' rate on this benchmark:
           p0 = 0.5 + W * (sibling rate - 0.5)
           Public data: one other subject's rate correlates 0.43 with this
           subject's rate on the same benchmark (REPORT_NOTES.md, 2026-10-28).
S2 part:   nudge p by how siblings did on THIS item vs their usual rate.

No sibling labels on the benchmark: identical to S2 (and to S1 without item siblings).
Standard library only; stateless.
"""

PRIOR_MEAN = 0.5    # base rate for an unseen subject-benchmark pair
PAIR_STRENGTH = 2   # pseudo-count of the prior
G = 0.4             # how strongly a sibling's result on the same item moves p
W = 0.8             # how far the siblings' benchmark rate moves p0 (tuned on seed 0, pool 5)


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
