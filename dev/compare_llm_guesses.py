"""Compare the raw guesses of several LLMs on the same items (cache only, free).

For every item that my_model5 scores in the draws, read each model's cached reply
(last line, refusal = no guess) and compare it with the item's real pass rate.

Report: cross-company comparison (Haiku 5.5, Sonnet 5.5, GPT-5 nano).
Run: .venv/bin/python dev/compare_llm_guesses.py --draws 30 --seed 1 --pool 5 --max-group-items 150
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "dev")
import claude_client
import evaluator

KIT = Path(__file__).resolve().parents[1]
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "gpt-5-nano"]

parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=30)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--pool", type=int, default=5)
parser.add_argument("--max-group-items", type=int, default=150)
args = parser.parse_args()

groups = evaluator.load_groups()
build_prompt = evaluator.load_predict(KIT / "my_model5").__globals__["build_prompt"]

# step 1: the prompts my_model5 is asked in these draws
asked = set()


def record(input, labeled=None):
    asked.add(build_prompt(input[0], input[1]))
    return 0.5


evaluator.evaluate(record, groups, args.draws, args.seed, args.pool, args.max_group_items)

# step 2: the real runs (0/1 labels) behind each prompt
runs_of = {}
for group in groups:
    for item_id in group["items"]:
        prompt = build_prompt(group["subject"], evaluator.full_item(group, item_id))
        if prompt in asked:
            runs_of[prompt] = group["items"][item_id][2]
prompts = sorted(runs_of)
print("items:", len(prompts))

# step 3: each model's guess (None = missing, refused or unreadable)
guesses = {}
for model in MODELS:
    claude_client.set_model(model)
    guesses[model] = []
    for prompt in prompts:
        key = claude_client.make_key(prompt, None, 2000, "low", None)
        guess = None
        if key in claude_client._cache:
            row = claude_client._cache[key]
            if row["stop_reason"] != "refusal":
                guess = claude_client.parse_last_line(row["text"])
        guesses[model].append(guess)

truth = np.array([np.mean(runs_of[p]) for p in prompts])


def brier(guess_list, use):
    """Mean squared error, each used item weighted once (some items have millions of runs)."""
    total = 0.0
    count = 0
    for i in range(len(prompts)):
        if use[i]:
            runs = np.array(runs_of[prompts[i]])
            total = total + np.mean((guess_list[i] - runs) ** 2)
            count = count + 1
    return total / count


# step 4: one line per model, on the items that model answered
print()
print("real pass rate: mean %.3f" % truth.mean())
print("%-20s %8s %10s %10s %12s %12s" % ("model", "answered", "mean guess", "corr", "Brier raw", "Brier 0.5"))
for model in MODELS:
    use = [g is not None for g in guesses[model]]
    g = np.array([x for x in guesses[model] if x is not None])
    t = truth[np.array(use)]
    print("%-20s %8d %10.3f %10.3f %12.4f %12.4f" % (
        model, sum(use), g.mean(), np.corrcoef(g, t)[0, 1],
        brier(guesses[model], use), brier([0.5] * len(prompts), use)))

# step 5: do the models agree with each other? (items all of them answered)
use_all = []
for i in range(len(prompts)):
    ok = True
    for model in MODELS:
        if guesses[model][i] is None:
            ok = False
    use_all.append(ok)
print()
print("correlation between guesses, on the", sum(use_all), "items every model answered:")
for a in range(len(MODELS)):
    for b in range(a + 1, len(MODELS)):
        x = [guesses[MODELS[a]][i] for i in range(len(prompts)) if use_all[i]]
        y = [guesses[MODELS[b]][i] for i in range(len(prompts)) if use_all[i]]
        print("  %-18s %-18s %.3f" % (MODELS[a], MODELS[b], np.corrcoef(x, y)[0, 1]))
