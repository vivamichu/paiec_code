#!/usr/bin/env python3
"""Grid-search my_submission/model.py constants with leave-one-benchmark-out.

For each held-out benchmark, pick the config with the best mean ALC on the
other benchmarks' groups, then report that config's ALC on the held-out one.
All configs see identical draws/splits (same seed).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_alc import KIT, evaluate, load_groups, load_predict  # noqa: E402

GRID = {
    "PRIOR_MEAN": [0.25, 0.3, 0.35, 0.4, 0.5],
    "PAIR_STRENGTH": [1.0, 2.0, 3.0, 5.0, 8.0],
}


def main(draws=20, seed=0):
    groups = load_groups()
    mod = load_predict(KIT / "my_submission")
    rows = []
    for values in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, values))
        for k, v in cfg.items():
            setattr(mod, k, v)
        df = evaluate(mod.predict, groups, draws=draws, seed=seed)
        per = df.groupby("bench").alc.agg(["sum", "count"])
        rows.append({**cfg, **{f"{b}_sum": per.loc[b, "sum"] for b in per.index},
                     **{f"{b}_n": per.loc[b, "count"] for b in per.index}, "alc": df.alc.mean()})
    res = pd.DataFrame(rows)
    benches = sorted(c[:-4] for c in res.columns if c.endswith("_sum"))
    for b in benches:
        res[b] = res[f"{b}_sum"] / res[f"{b}_n"]
    print("Top configs (in-sample, all benchmarks):")
    print(res.sort_values("alc")[list(GRID) + benches + ["alc"]].head(10).round(4).to_string(index=False))
    print("\nLeave-one-benchmark-out:")
    held_sum = held_n = 0.0
    for b in benches:
        others = [o for o in benches if o != b]
        train = sum(res[f"{o}_sum"] for o in others) / sum(res[f"{o}_n"] for o in others)
        best = res.loc[train.idxmin()]
        held_sum += best[f"{b}_sum"]
        held_n += best[f"{b}_n"]
        print(f"  held-out {b}: chosen {dict((k, best[k]) for k in GRID)} -> held-out ALC {best[b]:.4f}")
    print(f"  LOBO overall ALC (group-weighted): {held_sum / held_n:.4f}")


if __name__ == "__main__":
    main()
