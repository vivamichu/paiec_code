"""How much does subject B's result on an item tell about subject A on the SAME item, beyond A's pair rate?

Report: the same-question correlation behind Step 2 (0.42 over 31,020 question pairs in 76 benchmarks).
Run:  .venv/bin/python dev/same_item_corr.py
"""
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "dev")
from evaluator import load_groups

groups = load_groups()
by_bench = defaultdict(list)
for g in groups:
    by_bench[g["bench"]].append(g)

rng = random.Random(0)
xs, ys, per_bench = [], [], []
for bench, gs in by_bench.items():
    if len(gs) < 2:
        continue
    bx, by = [], []
    for trial in range(10):
        a, b = rng.sample(gs, 2)
        common = sorted(set(a["items"]) & set(b["items"]))
        if len(common) < 20:
            continue
        rate_a = np.mean([np.mean(a["items"][i][2]) for i in a["items"]])
        rate_b = np.mean([np.mean(b["items"][i][2]) for i in b["items"]])
        for i in rng.sample(common, min(50, len(common))):
            ya = np.mean(a["items"][i][2])
            yb = rng.choice(b["items"][i][2])   # one revealed run of B
            bx.append(yb - rate_b)
            by.append(ya - rate_a)
    if len(bx) > 30:
        per_bench.append((bench, np.corrcoef(bx, by)[0, 1], len(bx)))
        xs += bx
        ys += by

print("benchmarks with >= 2 subjects:", len(per_bench), " item pairs:", len(xs))
print("pooled corr(B residual, A residual): %.3f" % np.corrcoef(xs, ys)[0, 1])
c = np.array([p[1] for p in per_bench])
print("per-benchmark corr: median %.3f  25%% %.3f  75%% %.3f" % (np.median(c), np.percentile(c, 25), np.percentile(c, 75)))
