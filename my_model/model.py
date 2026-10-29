"""Beta-smoothed success rate for a subject–benchmark pair.

Prediction = posterior mean of the pair's success rate under a Beta prior:

    p = (PAIR_STRENGTH * PRIOR_MEAN + successes) / (PAIR_STRENGTH + n)

where successes / n count the acquired labels for the same subject (all
visible attributes) on the same anonymous benchmark_id. With no matching
labels (budget 0) this returns PRIOR_MEAN. Unlike the unsmoothed empirical
mean, one label never pushes the prediction to 0 or 1.

Constants were tuned leave-one-benchmark-out on the public binary benchmarks
(dev/tune_smoothed.py). Pooling other subjects' labels on the same benchmark
was tried and did not help. Standard library only; stateless.
"""

from __future__ import annotations

PRIOR_MEAN = 0.5     # base rate for an unseen subject–benchmark pair
PAIR_STRENGTH = 2   # pseudo-count; 31 labels outweigh the prior ~6:1


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    bench = item.get("benchmark_id", "")

    successes = n = 0
    for (obs_subject, obs_item), label in labeled or []:
        if obs_subject == subject and obs_item.get("benchmark_id", "") == bench:
            successes += int(label)
            n += 1

    return float((PAIR_STRENGTH * PRIOR_MEAN + successes) / (PAIR_STRENGTH + n))
