"""Tune one constant of a model (default: G of my_model2) on platform-like draws.
Same draws for every value. Value 0 is the reference.

Run:  .venv/bin/python dev/tune_sibling.py --draws 100 --seed 0 --pool 5
      .venv/bin/python dev/tune_sibling.py --folder my_model3 --constant W
Only tune on seed 0; confirm the chosen G with dev/final_eval.py on seed 1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "dev")
import evaluator

KIT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pool", type=int, default=5)
parser.add_argument("--max-group-items", type=int, default=250)
parser.add_argument("--values", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6, 0.8])
parser.add_argument("--folder", default="my_model2")
parser.add_argument("--constant", default="G")
parser.add_argument("--set", nargs="*", default=[], help="fix other constants, like LLM_MAX_LABELS=3")
parser.add_argument("--reference", type=float, default=0.0, help="value the others are compared with")
args = parser.parse_args()

groups = evaluator.load_groups()
predict = evaluator.load_predict(KIT / args.folder)
for setting in args.set:
    name, value = setting.split("=")
    predict.__globals__[name] = float(value)
    print("fixed", name, "=", value)

per_draw = {}
for g in args.values:
    predict.__globals__[args.constant] = g   # 0 switches the rule off
    df = evaluator.evaluate(predict, groups, args.draws, args.seed, args.pool, args.max_group_items)
    per_draw[g] = df.groupby("draw").alc.mean()
    means = df[["b0", "b1", "b3", "b7", "b15", "b31", "alc"]].mean()
    line = "%s %g:" % (args.constant, g)
    for column in ["b0", "b1", "b3", "b7", "b15", "b31", "alc"]:
        line = line + " %s=%.4f" % (column, means[column])
    print(line, flush=True)

print()
print("ALC difference to %s=%s, paired over draws:" % (args.constant, args.reference))
for g in args.values:
    if g == args.reference:
        continue
    diff = per_draw[g] - per_draw[args.reference]
    print("  %s %g: %+.4f (se %.4f)" % (args.constant, g, diff.mean(), diff.std() / np.sqrt(len(diff))))

if len(evaluator.MISSING_REPLIES) > 0:
    print()
    print("WARNING:", len(evaluator.MISSING_REPLIES), "Claude replies were not cached (treated as no guess).",
          "Run dev/prefetch_llm.py with the same draw settings first.")
