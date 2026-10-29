"""Build my_model4/track_record.json: each model name's gap on every public benchmark.

gap = subject success rate - benchmark average rate (from rates_table.py),
averaged over setting variants of the same name. Same filters as strength_check.py.
Only benchmarks with release == "public".

Run:  .venv/bin/python dev/build_track_record.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "dev")
import evaluator
from rates_table import load_rates

KIT = Path(__file__).resolve().parents[1]
OUT = KIT / "my_model4" / "track_record.json"
MIN_ITEMS = 30        # ignore subjects with fewer items
MIN_SUBJECTS = 3      # ignore benchmarks with fewer subjects (gap means little)

rates = load_rates()
rates = rates[rates.normalized_name.notna()]
rates = rates[rates.n_items >= MIN_ITEMS]

public = []
for bench in rates.bench.unique():
    if evaluator.is_public(bench):
        public.append(bench)
rates = rates[rates.bench.isin(public)]

subjects_per_bench = rates.groupby("bench").size()
good_benchmarks = subjects_per_bench[subjects_per_bench >= MIN_SUBJECTS].index
rates = rates[rates.bench.isin(good_benchmarks)]

# one gap per (name, benchmark)
gaps = rates.groupby(["normalized_name", "bench"]).gap.mean().reset_index()

table = {}
for i in range(len(gaps)):
    name = str(gaps.normalized_name.iloc[i])
    bench = str(gaps.bench.iloc[i])
    if name not in table:
        table[name] = {}
    table[name][bench] = round(float(gaps.gap.iloc[i]), 4)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(table, sort_keys=True))
print("names:", len(table), " benchmarks:", gaps.bench.nunique(), " entries:", len(gaps))
print("saved", OUT, " size KB: %.0f" % (OUT.stat().st_size / 1000))
