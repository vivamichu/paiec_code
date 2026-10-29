"""S2: smoothed pair mean (S1) plus same-item evidence from other subjects ("siblings").

S1 part:   p = (PAIR_STRENGTH * PRIOR_MEAN + s) / (PAIR_STRENGTH + n)
           s, n = successes and labels of THIS subject on THIS benchmark.

Sibling part (to write): when another subject on the same benchmark has a
revealed label on the SAME item, nudge p by how that subject did on the item
compared with its own usual rate. Public data: correlation 0.42 between the two
subjects' deviations on the same item (REPORT_NOTES.md, 2026-10-28).

Standard library only; stateless.
"""

PRIOR_MEAN = 0.5    # base rate for an unseen subject-benchmark pair
PAIR_STRENGTH = 2   # pseudo-count of the prior
G = 0.4             # how strongly a sibling's result on the same item moves p (to tune)


# ---- helpers (plumbing) ----

def item_key(item):
    """Two items are the same item if benchmark, text and features match.
    interactors are left out: they can differ between subjects."""
    return (item.get("benchmark_id", ""), item.get("item_content", ""), item.get("item_features", ""))


def subject_key(subject):
    """A subject is the whole subject dict (name plus all settings)."""
    return tuple(sorted(subject.items()))


def pair_counts(labeled, subject, benchmark_id):
    """Successes and number of labels of one subject on one benchmark."""
    successes = 0
    count = 0
    for (obs_subject, obs_item), label in labeled:
        if obs_subject == subject and obs_item.get("benchmark_id", "") == benchmark_id:
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


def smoothed(successes, count):
    return (PAIR_STRENGTH * PRIOR_MEAN + successes) / (PAIR_STRENGTH + count)


# ---- the model ----

def predict(input, labeled=None):
    subject, item = input
    if labeled is None:
        labeled = []
    benchmark_id = item.get("benchmark_id", "")

    s, n = pair_counts(labeled, subject, benchmark_id)
    p = smoothed(s, n)

    # sibling rule: how did other subjects do on THIS item, compared with their usual rate?
    differences = []
    for other_subject, label in sibling_labels(labeled, subject, item):
        s_other, n_other = pair_counts(labeled, other_subject, benchmark_id)
        # leave this label out of the sibling's usual rate,
        # otherwise the difference is pulled toward zero (strongly when it has few labels)
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
