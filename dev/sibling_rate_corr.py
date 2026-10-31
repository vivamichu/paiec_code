"""How much a benchmark's difficulty is shared between subjects (public benchmarks only).

For every subject-benchmark pair with at least 80 items: correlate its success rate with
(a) the mean rate of all other subjects on the same benchmark, and
(b) the rate of ONE random other subject (what a platform round usually gives).

Report: Step 3 ("correlates 0.43 with one other subject, 0.65 with the average of all others").
Run: .venv/bin/python dev/sibling_rate_corr.py
"""

import random
import sys

import numpy as np

sys.path.insert(0, "dev")
from rates_table import load_rates

rates = load_rates()
rates = rates[rates.n_items >= 80]
counts = rates.groupby("bench").subject_id.count()
rates = rates[rates.bench.isin(counts[counts >= 2].index)]
print("pairs", len(rates), "benchmarks", rates.bench.nunique())

# (a) leave-one-out mean of the other subjects on the same benchmark
rate_and_others = []
for bench, group in rates.groupby("bench"):
    total = group.rate.sum()
    n = len(group)
    for rate in group.rate:
        rate_and_others.append((rate, (total - rate) / (n - 1)))
rate_and_others = np.array(rate_and_others)
print("corr(rate, mean of all other subjects): %.3f"
      % np.corrcoef(rate_and_others[:, 0], rate_and_others[:, 1])[0, 1])

# (b) one random other subject
rng = random.Random(0)
rate_and_one = []
for bench, group in rates.groupby("bench"):
    bench_rates = group.rate.tolist()
    for i in range(len(bench_rates)):
        others = []
        for j in range(len(bench_rates)):
            if j != i:
                others.append(j)
        rate_and_one.append((bench_rates[i], bench_rates[rng.choice(others)]))
rate_and_one = np.array(rate_and_one)
print("corr(rate, one random other subject): %.3f"
      % np.corrcoef(rate_and_one[:, 0], rate_and_one[:, 1])[0, 1])
