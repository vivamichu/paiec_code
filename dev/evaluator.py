"""Local Brier-ALC evaluator over the whole measurement-db.

Copies the competition protocol and scores a submission folder's predict().

How one draw works:
  - group = one subject on one benchmark, with 80 to 1000 distinct items
  - groups are drawn until 1000 items are used (least-used benchmarks first)
  - each group's items are split 50/50: acquisition half and evaluation half
  - budgets 0, 1, 3, 7, 15, 31: reveal that many acquisition labels
  - Brier is computed on the evaluation half
  - ALC = weighted mean of the six Brier scores (weights .1 .2 .2 .2 .2 .1)

Building the groups is slow, so they are saved in data/cache/groups.pkl.
Delete that file (or use --rebuild) to build them again.

Usage:
  python dev/evaluator.py my_model --draws 20 --seed 0
  python dev/evaluator.py my_model --benchmarks financebench real_webagents
"""

import argparse
import hashlib
import importlib.util
import pickle
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

KIT = Path(__file__).resolve().parents[1]
DB = KIT / "data" / "measurement-db"
CACHE = KIT / "data" / "cache" / "groups.pkl"

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
MIN_ITEMS = 80
DRAW_LIMIT = 1000
MAX_SUBJECTS = 25  # subjects kept per benchmark, to keep it fast

SUBJECT_FIELDS = [
    "normalized_name", "provider", "release_date", "access_date",
    "harness", "reasoning_effort", "harness_version", "subject_features_extra",
]


def to_text(value):
    # missing values become an empty string, like in the competition
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def make_alias(name):
    # stable fake benchmark id, like benchmark_123456
    number = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 900000 + 100000
    return "benchmark_" + str(number)


def load_benchmark(name):
    """Return the groups of one benchmark."""
    folder = DB / name / "formatted_tables"

    # ids are stored as categories because some files have millions of rows
    resp = pq.read_table(
        folder / "responses.parquet",
        columns=["subject_id", "item_id", "response", "interactors"],
        read_dictionary=["subject_id", "item_id", "interactors"],
    ).to_pandas()
    resp = resp.dropna(subset=["response"])
    if len(resp) == 0:
        return []

    # keep subjects with 80 to 1000 distinct items
    items_per_subject = resp.groupby("subject_id", observed=True).item_id.nunique()
    eligible = []
    for subject_id in items_per_subject.index:
        count = items_per_subject[subject_id]
        if count >= MIN_ITEMS and count <= DRAW_LIMIT:
            eligible.append(str(subject_id))
    eligible.sort()
    random.Random(0).shuffle(eligible)
    eligible = eligible[:MAX_SUBJECTS]
    if len(eligible) == 0:
        return []

    subjects = pd.read_parquet(folder / "subjects.parquet")
    subjects = subjects.drop_duplicates("subject_id").set_index("subject_id")
    items = pd.read_parquet(folder / "items.parquet")
    items = items.drop_duplicates("item_id").set_index("item_id")
    alias = make_alias(name)

    resp = resp[resp.subject_id.astype(str).isin(eligible)]

    groups = []
    shared_items = {}  # one copy of each item, shared by all subjects
    for subject_id, subject_rows in resp.groupby("subject_id", observed=True):
        subject_id = str(subject_id)
        if subject_id not in subjects.index:
            continue

        row = subjects.loc[subject_id]
        subject = {}
        for field in SUBJECT_FIELDS:
            subject[field] = to_text(row.get(field))

        group_items = {}
        for item_id, item_rows in subject_rows.groupby("item_id", observed=True):
            item_id = str(item_id)
            if item_id not in items.index:
                continue
            if item_id not in shared_items:
                item_row = items.loc[item_id]
                shared_items[item_id] = {
                    "item_content": to_text(item_row.get("content")),
                    "item_features": to_text(item_row.get("item_features")),
                    "benchmark_id": alias,
                }
            interactors = to_text(item_rows.interactors.iloc[0])
            runs = []
            for value in item_rows.response:
                runs.append(int(value))
            group_items[item_id] = (shared_items[item_id], interactors, runs)

        if len(group_items) >= MIN_ITEMS:
            groups.append({
                "bench": name,
                "subject_id": subject_id,
                "subject": subject,
                "items": group_items,
            })
    return groups


RELEASE = {}  # benchmark folder -> release status


def is_public(name):
    if name not in RELEASE:
        table = pd.read_parquet(DB / name / "formatted_tables" / "benchmarks.parquet")
        RELEASE[name] = str(table["release"].iloc[0])
    return RELEASE[name] == "public"


def load_groups(benchmarks=None, rebuild=False):
    """All groups, optionally only for some benchmark folders."""
    if CACHE.exists() and not rebuild:
        groups = pickle.loads(CACHE.read_bytes())
    else:
        groups = []
        for folder in sorted(DB.iterdir()):
            if not (folder / "formatted_tables" / "responses.parquet").exists():
                continue
            print("loading", folder.name, "...", flush=True)
            try:
                groups = groups + load_benchmark(folder.name)
            except Exception as error:
                print("  skipped", folder.name, ":", error)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(groups))

    # dataset card: use only benchmarks with release == "public"
    public = []
    dropped = 0
    for group in groups:
        if is_public(group["bench"]):
            public.append(group)
        else:
            dropped = dropped + 1
    groups = public
    if dropped > 0:
        print("dropped", dropped, "groups from withheld benchmarks")

    if benchmarks:
        wanted = []
        for group in groups:
            if group["bench"] in benchmarks:
                wanted.append(group)
        groups = wanted
    return groups


def full_item(group, item_id):
    """The item dict exactly as predict() receives it."""
    entry = group["items"][item_id]
    item = dict(entry[0])
    item["interactors"] = entry[1]
    return item


def draw_groups(groups, rng):
    """Pick whole groups until the 1000-item limit is used up."""
    chosen = []
    used = 0
    bench_count = {}
    subject_count = {}
    remaining = list(groups)

    while True:
        # groups that still fit
        fits = []
        for group in remaining:
            if used + len(group["items"]) <= DRAW_LIMIT:
                fits.append(group)
        if len(fits) == 0:
            return chosen

        # take the least-used benchmark and subject (random order breaks ties)
        rng.shuffle(fits)
        best = None
        best_score = None
        for group in fits:
            score = (bench_count.get(group["bench"], 0),
                     subject_count.get(group["subject_id"], 0))
            if best is None or score < best_score:
                best = group
                best_score = score

        chosen.append(best)
        remaining.remove(best)
        used = used + len(best["items"])
        bench_count[best["bench"]] = bench_count.get(best["bench"], 0) + 1
        subject_count[best["subject_id"]] = subject_count.get(best["subject_id"], 0) + 1


def run_draw(predict, draw, rng):
    """Score one draw. Returns {(bench, subject_id): [Brier at each budget]}."""
    # split every group and fix the order in which labels are revealed
    plans = []
    for group in draw:
        item_ids = sorted(group["items"])
        rng.shuffle(item_ids)
        n_acquisition = (len(item_ids) + 1) // 2
        acquisition_ids = item_ids[:n_acquisition]
        evaluation_ids = item_ids[n_acquisition:]

        labels = []
        for item_id in acquisition_ids:
            runs = group["items"][item_id][2]
            labels.append((full_item(group, item_id), rng.choice(runs)))
        plans.append((group, labels, evaluation_ids))

    scores = {}
    for budget in BUDGETS:
        # evidence from all groups, first `budget` labels of each
        labeled = []
        for group, labels, evaluation_ids in plans:
            for item, label in labels[:budget]:
                labeled.append([[group["subject"], item], label])

        for group, labels, evaluation_ids in plans:
            total_error = 0.0
            count = 0
            for item_id in evaluation_ids:
                item = full_item(group, item_id)
                runs = group["items"][item_id][2]
                p = float(predict([group["subject"], item], labeled))
                if p < 0.0 or p > 1.0:
                    raise ValueError("prediction out of range: " + str(p))
                for label in runs:
                    total_error = total_error + (p - label) ** 2
                    count = count + 1
            key = (group["bench"], group["subject_id"])
            if key not in scores:
                scores[key] = []
            scores[key].append(total_error / count)
    return scores


def draw_small_pool(groups, rng, n_benchmarks, max_group_items):
    """Platform-like draw: only a few benchmarks and small groups.

    The Codabench logs showed about 7 benchmarks and 8-9 pairs per evaluation,
    so one benchmark often gives two subjects ("siblings"). The normal draw
    rarely does that because it spreads over all benchmarks.
    """
    small = []
    for group in groups:
        if len(group["items"]) <= max_group_items:
            small.append(group)
    benches = set()
    for group in small:
        benches.add(group["bench"])
    benches = sorted(benches)
    chosen = rng.sample(benches, min(n_benchmarks, len(benches)))
    pool = []
    for group in small:
        if group["bench"] in chosen:
            pool.append(group)
    return draw_groups(pool, rng)


def evaluate(predict, groups, draws=20, seed=0, pool=0, max_group_items=DRAW_LIMIT):
    """One row per (draw, group): Brier at each budget and the ALC.

    pool > 0 uses the platform-like small-pool draw with that many benchmarks.
    """
    rng = random.Random(seed)
    rows = []
    for draw_number in range(draws):
        if pool > 0:
            draw = draw_small_pool(groups, rng, pool, max_group_items)
        else:
            draw = draw_groups(groups, rng)
        scores = run_draw(predict, draw, rng)
        for key in scores:
            curve = scores[key]
            alc = 0.0
            for weight, brier in zip(WEIGHTS, curve):
                alc = alc + weight * brier
            row = {"draw": draw_number, "bench": key[0], "subject_id": key[1]}
            for budget, brier in zip(BUDGETS, curve):
                row["b" + str(budget)] = brier
            row["alc"] = alc
            rows.append(row)
    return pd.DataFrame(rows)


def report(df, label=""):
    columns = ["b0", "b1", "b3", "b7", "b15", "b31", "alc"]
    print()
    print(label, "group-scores =", len(df), " benchmarks =", df.bench.nunique())
    print(df.groupby("bench")[columns].mean().round(4).to_string())

    overall = df[columns].mean()
    line = "overall:"
    for column in columns:
        line = line + " " + column + "=" + format(overall[column], ".4f")
    print(line)

    per_draw = df.groupby("draw").alc.mean()
    print("ALC per draw: mean =", round(per_draw.mean(), 4),
          " std =", round(per_draw.std(), 4), "(draw-to-draw noise)")
    return overall["alc"]


def load_predict(submission_dir):
    """Import model.py from a submission folder and return its predict."""
    spec = importlib.util.spec_from_file_location("submission_model", submission_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir))
    spec.loader.exec_module(module)
    sys.path.remove(str(submission_dir))

    # models with public lookup tables: leave each group's own benchmark out
    if hasattr(module, "EXCLUDE"):
        for folder in DB.iterdir():
            module.EXCLUDE[make_alias(folder.name)] = folder.name

    # models that ask Claude: read replies from the local cache, never the API
    # (fill the cache first with dev/prefetch_llm.py)
    if hasattr(module, "ask_claude"):
        module.ask_claude = cached_claude_reply
    return module.predict


def cached_claude_reply(prompt):
    """Cached reply for this prompt (same settings as my_model5), or None."""
    import claude_client
    key = claude_client.make_key(prompt, None, 2000, "low", None)
    if key in claude_client._cache:
        return claude_client._cache[key]["text"]
    MISSING_REPLIES.append(prompt)
    return None


MISSING_REPLIES = []   # prompts asked locally that had no cached reply


def main():
    parser = argparse.ArgumentParser(description="Local Brier-ALC evaluator")
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmarks", nargs="*", help="only these benchmark folders")
    parser.add_argument("--rebuild", action="store_true", help="build the groups cache again")
    args = parser.parse_args()

    groups = load_groups(args.benchmarks, args.rebuild)
    names = set()
    for group in groups:
        names.add(group["bench"])
    print(len(groups), "groups from", len(names), "benchmarks")

    predict = load_predict(args.submission_dir.resolve())
    df = evaluate(predict, groups, args.draws, args.seed)
    report(df, str(args.submission_dir))


if __name__ == "__main__":
    main()
