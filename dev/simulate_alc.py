#!/usr/bin/env python3
"""Local Brier-ALC simulator for the streaming evaluation protocol.

Replays the competition protocol on the public binary benchmarks in
measurement-db/ and scores any submission directory's predict():

  - eligible groups: (subject, benchmark) with 80..1000 distinct items
    (the 1000 cap is our reading of "groups that exceed the remaining
    budget are skipped" under the 1,000-item draw limit)
  - draw: whole groups under a 1,000-item limit, preferring benchmarks and
    subjects less represented in the draw, random tie-break
  - per group: 50/50 item split (odd count: extra item to acquisition),
    acquisition items in random order, default random acquisition
  - budgets 0,1,3,7,15,31 with nested label prefixes; all groups finish a
    budget before the next one; `labeled` holds evidence across all groups
  - each acquired label is one random recorded run of the item
  - Brier over every response row of the evaluation half
  - ALC weights .1,.2,.2,.2,.2,.1; final score = unweighted mean over groups

Usage:
  python dev/simulate_alc.py SUBMISSION_DIR [--draws 20] [--seed 0]
  python dev/simulate_alc.py SUBMISSION_DIR --isolated   # every group alone
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

KIT = Path(__file__).resolve().parents[1]
DB = KIT / "measurement-db"
BENCHMARKS = ["multi_swebench", "real_webagents", "researchcodebench"]
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
DRAW_LIMIT = 1000
SUBJECT_FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra"]


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and v != v) else str(v)


def alias(kind: str, raw: str) -> str:
    return f"{kind}_{int(hashlib.sha1(raw.encode()).hexdigest(), 16) % 900000 + 100000}"


def load_groups(benchmarks=BENCHMARKS):
    """Return list of groups: dict(bench, subject_id, subject, items{item_id: (item_dict, [runs])})."""
    groups = []
    for name in benchmarks:
        d = DB / name
        subjects = pd.read_parquet(d / "subjects.parquet").set_index("subject_id")
        items = pd.read_parquet(d / "items.parquet").set_index("item_id")
        resp = pd.read_parquet(d / "response.parquet", columns=["subject_id", "item_id", "response", "interactors"])
        bench_alias = alias("benchmark", name)
        item_cache = {}
        for sid, sub in resp.groupby("subject_id"):
            runs = sub.groupby("item_id").response.apply(lambda x: [int(v) for v in x]).to_dict()
            if not 80 <= len(runs) <= DRAW_LIMIT:
                continue
            row = subjects.loc[sid]
            subject = {f: _s(row.get(f)) for f in SUBJECT_FIELDS}
            inter = sub.groupby("item_id").interactors.first().to_dict()
            gitems = {}
            for iid, r in runs.items():
                if iid not in item_cache:
                    it = items.loc[iid]
                    item_cache[iid] = {"item_content": _s(it.get("content")),
                                       "item_features": _s(it.get("item_features")),
                                       "interactors": "", "benchmark_id": bench_alias}
                item = dict(item_cache[iid], interactors=_s(inter.get(iid)))
                gitems[iid] = (item, r)
            groups.append({"bench": name, "subject_id": sid, "subject": subject, "items": gitems})
    return groups


def draw_groups(groups, rng):
    chosen, used = [], 0
    bcount, scount = defaultdict(int), defaultdict(int)
    remaining = list(groups)
    while True:
        fits = [g for g in remaining if used + len(g["items"]) <= DRAW_LIMIT]
        if not fits:
            return chosen
        rng.shuffle(fits)
        g = min(fits, key=lambda g: (bcount[g["bench"]], scount[g["subject_id"]]))
        chosen.append(g)
        remaining.remove(g)
        used += len(g["items"])
        bcount[g["bench"]] += 1
        scount[g["subject_id"]] += 1


def run_draw(predict, draw, rng):
    """Score one draw. Returns {group_key: [brier at each budget]}."""
    plans = []
    for g in draw:
        ids = sorted(g["items"])
        rng.shuffle(ids)
        n_acq = (len(ids) + 1) // 2
        acq, ev = ids[:n_acq], ids[n_acq:]
        # one random recorded run per acquired item
        acq_labels = [(g["items"][i][0], rng.choice(g["items"][i][1])) for i in acq]
        plans.append((g, acq_labels, ev))
    out = {}
    for b in BUDGETS:
        labeled = []
        for g, acq_labels, _ in plans:
            for item, y in acq_labels[:b]:
                labeled.append([[g["subject"], item], y])
        for g, _, ev in plans:
            se, n = 0.0, 0
            for iid in ev:
                item, runs = g["items"][iid]
                p = float(predict([dict(g["subject"]), dict(item)], [[[dict(s), dict(i)], y] for (s, i), y in labeled]))
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"prediction out of range: {p}")
                for y in runs:
                    se += (p - y) ** 2
                    n += 1
            out.setdefault((g["bench"], g["subject_id"]), []).append(se / n)
    return out


def load_predict(submission_dir: Path):
    spec = importlib.util.spec_from_file_location(f"sub_{abs(hash(submission_dir))}", submission_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir))
    spec.loader.exec_module(mod)
    sys.path.remove(str(submission_dir))
    return mod


def evaluate(predict, groups, draws=20, seed=0, isolated=False):
    """Return DataFrame rows: draw, bench, subject_id, brier@budget..., alc."""
    rng = random.Random(seed)
    rows = []
    if isolated:
        draw_sets = [[g] for g in groups]
    else:
        draw_sets = [draw_groups(groups, rng) for _ in range(draws)]
    for k, draw in enumerate(draw_sets):
        for (bench, sid), curve in run_draw(predict, draw, rng).items():
            alc = sum(w * c for w, c in zip(WEIGHTS, curve))
            rows.append({"draw": k, "bench": bench, "subject_id": sid,
                         **{f"b{b}": c for b, c in zip(BUDGETS, curve)}, "alc": alc})
    return pd.DataFrame(rows)


def report(df, label=""):
    cols = [f"b{b}" for b in BUDGETS] + ["alc"]
    per_bench = df.groupby("bench")[cols].mean()
    overall = df[cols].mean()
    print(f"\n{label}  groups scored={len(df)}")
    print(per_bench.round(4).to_string())
    print("overall (mean over groups):", " ".join(f"{c}={overall[c]:.4f}" for c in cols))
    return overall["alc"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission_dir", type=Path)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--isolated", action="store_true", help="score each group alone (no cross-group evidence)")
    args = ap.parse_args()
    groups = load_groups()
    mod = load_predict(args.submission_dir.resolve())
    df = evaluate(mod.predict, groups, args.draws, args.seed, args.isolated)
    report(df, f"{args.submission_dir} ({'isolated' if args.isolated else f'{args.draws} draws'})")


if __name__ == "__main__":
    main()
