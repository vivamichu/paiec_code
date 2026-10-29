"""Feasibility check: does a name's gap on OTHER benchmarks predict its gap here?

gap = subject success rate - benchmark average rate (from rates_table.py).
For every (benchmark B, name N):
  predicted gap = average gap of N on all benchmarks except B
  actual gap    = gap of N on B
B is never used to build its own prediction (leave-one-benchmark-out).

Run:  python dev/strength_check.py
"""

import sys

import numpy as np

sys.path.insert(0, "dev")
from rates_table import load_rates

MIN_ITEMS = 30        # ignore subjects with fewer items
MIN_SUBJECTS = 3      # ignore benchmarks with fewer subjects (gap means little)

rates = load_rates()
print("rows:", len(rates))

# keep named subjects with enough items
rates = rates[rates.normalized_name.notna()]
rates = rates[rates.n_items >= MIN_ITEMS]

# keep benchmarks with enough subjects
subjects_per_bench = rates.groupby("bench").size()
good_benchmarks = subjects_per_bench[subjects_per_bench >= MIN_SUBJECTS].index
rates = rates[rates.bench.isin(good_benchmarks)]
print("rows kept:", len(rates), " benchmarks:", rates.bench.nunique(), " names:", rates.normalized_name.nunique())

# one gap per (name, benchmark): average over setting variants of the same name
gaps = rates.groupby(["normalized_name", "bench"]).gap.mean().reset_index()

# gap_of[name] = {benchmark: gap}
gap_of = {}
for i in range(len(gaps)):
    name = gaps.normalized_name.iloc[i]
    bench = gaps.bench.iloc[i]
    gap = gaps.gap.iloc[i]
    if name not in gap_of:
        gap_of[name] = {}
    gap_of[name][bench] = gap

# leave one benchmark out
n_other_list = []
sum_other_list = []
actual_list = []
for name in gap_of:
    for held_out in gap_of[name]:
        other_gaps = []
        for bench in gap_of[name]:
            if bench != held_out:
                other_gaps.append(gap_of[name][bench])
        if len(other_gaps) == 0:
            continue
        n_other_list.append(len(other_gaps))
        sum_other_list.append(sum(other_gaps))
        actual_list.append(gap_of[name][held_out])

n_other = np.array(n_other_list)
sum_other = np.array(sum_other_list)
actual = np.array(actual_list)
predicted = sum_other / n_other
print("pairs (name, held-out benchmark) with data elsewhere:", len(actual))


def show(label, mask, shrink_k=0.0):
    # shrink: pretend there are k extra benchmarks with gap 0
    pred = sum_other[mask] / (n_other[mask] + shrink_k)
    act = actual[mask]
    if len(act) < 5:
        return
    corr = np.corrcoef(pred, act)[0, 1]
    error_zero = np.mean(act ** 2)            # always predict gap 0
    error_pred = np.mean((act - pred) ** 2)   # predict the average gap
    print(label, " n =", len(act), " corr =", round(corr, 3),
          " mean sq error: zero =", round(error_zero, 4), " predicted =", round(error_pred, 4))


print()
print("by number of OTHER benchmarks the name appears in (no shrinking):")
show("all          ", n_other >= 1)
show("1 other      ", n_other == 1)
show("2 others     ", n_other == 2)
show("3 to 5 others", (n_other >= 3) & (n_other <= 5))
show("6 to 10      ", (n_other >= 6) & (n_other <= 10))
show("11 or more   ", n_other >= 11)

print()
print("all pairs, with shrinking k (k extra imaginary benchmarks at gap 0):")
for k in [0, 1, 2, 4, 8]:
    show("k = " + str(k), n_other >= 1, shrink_k=k)

# how much does the gap itself vary? (context for the errors above)
print()
print("std of actual gap:", round(np.std(actual), 3))
