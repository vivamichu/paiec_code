"""Feasibility check: do item_features tokens shared with LABELED items carry signal?

For each group: random 50/50 split, first B acquisition items are "labeled".
For each evaluation item, find labeled items of the same pair that share at least
one item_features token (key=value). Compare their mean outcome with the pair mean.

  coverage = share of evaluation items with at least one such labeled neighbour
  signal   = corr(neighbour mean - pair mean, outcome - pair mean) over covered items

Analysis only, not a model. Run:
  .venv/bin/python dev/feature_signal_check.py
"""

import random
import sys

import numpy as np

sys.path.insert(0, "dev")
from evaluator import load_groups


def tokens(item):
    # item_features is "key=value;key=value"; each part is one token
    parts = item[0]["item_features"].split(";")
    result = set()
    for part in parts:
        part = part.strip()
        if part != "":
            result.add(part)
    return result


groups = load_groups()
rng = random.Random(0)
sample = rng.sample(groups, 400)

for budget in [3, 7, 15, 31]:
    covered = 0
    total = 0
    x = []   # neighbour mean - pair mean
    y = []   # outcome - pair mean
    for group in sample:
        ids = sorted(group["items"])
        rng.shuffle(ids)
        n_acq = (len(ids) + 1) // 2
        labeled = ids[:n_acq][:budget]
        evaluation = ids[n_acq:]

        labels = {}
        for item_id in labeled:
            runs = group["items"][item_id][2]
            labels[item_id] = rng.choice(runs)
        pair_mean = sum(labels.values()) / len(labels)
        labeled_tokens = {}
        for item_id in labeled:
            labeled_tokens[item_id] = tokens(group["items"][item_id])

        # tokens every labeled item has (like split=test) say nothing: ignore them
        common = None
        for item_id in labeled:
            if common is None:
                common = set(labeled_tokens[item_id])
            else:
                common = common & labeled_tokens[item_id]

        for item_id in evaluation:
            total = total + 1
            target = tokens(group["items"][item_id]) - common
            neighbours = []
            for other in labeled:
                if len(target & labeled_tokens[other]) > 0:
                    neighbours.append(labels[other])
            # keep only informative neighbours: not every labeled item (then it is just the pair mean)
            if len(neighbours) == 0 or len(neighbours) == len(labeled):
                continue
            covered = covered + 1
            runs = group["items"][item_id][2]
            x.append(sum(neighbours) / len(neighbours) - pair_mean)
            y.append(sum(runs) / len(runs) - pair_mean)

    x = np.array(x)
    y = np.array(y)
    corr = np.corrcoef(x, y)[0, 1] if len(x) > 2 else float("nan")
    print("budget", budget, " coverage %.2f" % (covered / total), " n", len(x), " corr %.3f" % corr)
