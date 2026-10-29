"""Score several submission folders on the SAME draws and save the results for the report.

Every model sees identical draws (same seed), so differences between models are paired:
we report the mean difference per draw and its standard error.

Run:
  .venv/bin/python dev/final_eval.py --draws 100 --seed 0
Output:
  results/final_eval_seed<seed>.csv   one row per (model, draw, group): Brier at each budget and ALC
  printed summary table
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "dev")
import claude_client
import evaluator

KIT = Path(__file__).resolve().parents[1]

# name shown in the report -> submission folder
MODELS = {
    "constant 0.5": KIT / "sample_code_submission",
    "empirical mean (organisers)": KIT / "paiec_baseline" / "empirical_mean",
    "S0 smoothed (p0 0.33, k 5)": KIT / "dev" / "baselines" / "s0_smoothed_033_k5",
    "S1 smoothed (p0 0.5, k 2)": KIT / "my_model",
    "S2 S1 + same-item siblings": KIT / "my_model2",
    "S3 S2 + benchmark siblings": KIT / "my_model3",
    "S4 S3 + public track record": KIT / "my_model4",
    "S4b S3 + Claude prior": KIT / "my_model5",
    "S4c S4b retuned": KIT / "my_model6",
    "S4b Haiku, last line": KIT / "my_model5",
    "S4b Sonnet 5.5, last line": KIT / "my_model5",
}
# models that read another Claude model's cached replies (default: Haiku 5.5)
LLM_MODEL = {
    "S4b Sonnet 5.5, last line": "claude-sonnet-5-5",
}
# models that read only the LAST line of each reply (Sonnet explains first, then gives the number)
# and treat a refused reply as no guess
LAST_LINE = ["S4b Haiku, last line", "S4b Sonnet 5.5, last line"]


def reply_without_refusals(prompt):
    """Cached reply, or None if missing or refused (a cut-off refusal may hold a stray number)."""
    key = claude_client.make_key(prompt, None, 2000, "low", None)
    if key not in claude_client._cache:
        evaluator.MISSING_REPLIES.append(prompt)
        return None
    row = claude_client._cache[key]
    if row["stop_reason"] == "refusal":
        return None
    return row["text"]

REFERENCE = "S1 smoothed (p0 0.5, k 2)"

parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pool", type=int, default=0, help="platform-like draw: this many benchmarks per draw")
parser.add_argument("--max-group-items", type=int, default=1000, help="with --pool: only groups this small")
parser.add_argument("--models", nargs="*", help="only these model names (default: all)")
parser.add_argument("--reference", default=REFERENCE, help="model the differences are taken against")
parser.add_argument("--tag", default="", help="added to the output file name")
args = parser.parse_args()
REFERENCE = args.reference
if args.models:
    keep = {}
    for name in MODELS:
        if name in args.models:
            keep[name] = MODELS[name]
    MODELS = keep
# the organizers' baselines are not in our repository: skip a model whose folder is missing
for name in list(MODELS):
    if not MODELS[name].exists():
        print("skipping", name, "- folder not found:", MODELS[name])
        del MODELS[name]

groups = evaluator.load_groups()
print(len(groups), "groups loaded")

tables = []
for name in MODELS:
    print("scoring", name, "...", flush=True)
    claude_client.set_model(LLM_MODEL.get(name, "claude-haiku-5-5"))
    predict = evaluator.load_predict(MODELS[name])
    if name in LAST_LINE:
        predict.__globals__["parse_probability"] = claude_client.parse_last_line
        predict.__globals__["ask_claude"] = reply_without_refusals
    df = evaluator.evaluate(predict, groups, args.draws, args.seed, args.pool, args.max_group_items)
    df.insert(0, "model", name)
    tables.append(df)
results = pd.concat(tables, ignore_index=True)

tag = "seed%d" % args.seed
if args.pool > 0:
    tag = tag + "_pool%d_max%d" % (args.pool, args.max_group_items)
if args.tag:
    tag = tag + "_" + args.tag
out =KIT / "results" / ("final_eval_%s.csv" % tag)
out.parent.mkdir(exist_ok=True)
results.to_csv(out, index=False)
print("saved", out)

# summary: mean Brier per budget and ALC over all group-scores
columns = ["b0", "b1", "b3", "b7", "b15", "b31", "alc"]
summary = results.groupby("model")[columns].mean()
print()
print(summary.round(4).to_string())

# paired difference to the reference model, per draw
per_draw = results.groupby(["model", "draw"]).alc.mean().unstack(0)
print()
print("ALC difference to", REFERENCE, "(mean over draws, standard error):")
for name in MODELS:
    if name == REFERENCE or REFERENCE not in MODELS:
        continue
    diff = per_draw[name] - per_draw[REFERENCE]
    print("  %-30s %+.4f  (se %.4f)" % (name, diff.mean(), diff.std() / np.sqrt(len(diff))))
print()
one = results[results.model == list(MODELS)[0]]
print("groups scored per draw: mean", round(one.groupby("draw").size().mean(), 1),
      " distinct benchmarks:", results.bench.nunique())
# share of pairs that have a sibling (another subject of the same benchmark in the same draw)
with_sibling = 0
for draw_number, draw_rows in one.groupby("draw"):
    counts = draw_rows.bench.value_counts()
    for bench in draw_rows.bench:
        if counts[bench] > 1:
            with_sibling = with_sibling + 1
print("share of pairs with a sibling in their draw: %.2f" % (with_sibling / len(one)))

if len(evaluator.MISSING_REPLIES) > 0:
    print()
    print("WARNING:", len(evaluator.MISSING_REPLIES), "Claude replies were not cached (treated as no guess).",
          "Run dev/prefetch_llm.py with the same draw settings (and --model) first.")
