"""Analyse the cached Claude guesses from llm_prior_test.py (no API calls, free).

Two questions:
  1. Is the shrink factor stable? (fit on one half of the samples, test on the other)
  2. Does Claude know anything about the ITEM beyond the subject's overall rate on
     that benchmark (the "pair rate")? Labels will teach us the pair rate anyway,
     so only the item-level part can help at budgets 3 to 31.

Run with the same --n and --seed as llm_prior_test.py:
  .venv/bin/python dev/llm_prior_analysis.py --n 200
"""

import argparse
import random
import sys

import numpy as np

sys.path.insert(0, "dev")
import claude_client
import my_prompt
from evaluator import full_item, load_groups

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-thinking", action="store_true")
args = parser.parse_args()
thinking = "disabled" if args.no_thinking else None

# same sampling as llm_prior_test.py
groups = load_groups()
rng = random.Random(args.seed)
guesses = []
outcomes = []
pair_rates = []
for i in range(args.n):
    group = rng.choice(groups)
    item_id = rng.choice(sorted(group["items"]))
    runs = group["items"][item_id][2]

    # pair rate WITHOUT this item (leave it out, so it does not contain the answer)
    total = 0
    count = 0
    for other_id in group["items"]:
        if other_id == item_id:
            continue
        other_runs = group["items"][other_id][2]
        total = total + sum(other_runs) / len(other_runs)
        count = count + 1

    prompt = my_prompt.build_prompt(group["subject"], full_item(group, item_id))
    before = claude_client.totals["calls"]
    reply = claude_client.ask(prompt, thinking=thinking)
    if claude_client.totals["calls"] > before:
        print("warning: this reply was not cached, a paid call was made")
    p = my_prompt.parse_probability(reply)
    if p is None:
        p = 0.5

    guesses.append(p)
    outcomes.append(sum(runs) / len(runs))
    pair_rates.append(total / count)

guesses = np.array(guesses)
outcomes = np.array(outcomes)
pair_rates = np.array(pair_rates)


def brier(p, y):
    # y is the share of correct runs; this is the mean Brier over runs
    return np.mean(p ** 2 - 2 * p * y + y)


def best_lam(p, y):
    d = p - 0.5
    return np.sum(d * (y - 0.5)) / np.sum(d ** 2)


# 1. shrink factor with 2-fold cross-validation
half_a = np.arange(len(guesses)) % 2 == 0
half_b = ~half_a
lam_a = best_lam(guesses[half_a], outcomes[half_a])
lam_b = best_lam(guesses[half_b], outcomes[half_b])
test_a = brier(0.5 + lam_b * (guesses[half_a] - 0.5), outcomes[half_a])
test_b = brier(0.5 + lam_a * (guesses[half_b] - 0.5), outcomes[half_b])
print("samples:", len(guesses))
print()
print("1. shrink toward 0.5, fitted on one half, tested on the other")
print("   lam on half A:", round(lam_a, 3), "  lam on half B:", round(lam_b, 3))
print("   Brier, cross-validated:", round((test_a + test_b) / 2, 4), "  (always 0.5 = 0.25)")

# 2. item-level signal beyond the pair rate
residual = outcomes - pair_rates
print()
print("2. what does Claude know?")
print("   corr(guess, outcome):            ", round(np.corrcoef(guesses, outcomes)[0, 1], 3))
print("   corr(guess, pair rate):          ", round(np.corrcoef(guesses, pair_rates)[0, 1], 3),
      " <- knowledge about subject/benchmark")
print("   corr(guess, outcome - pair rate):", round(np.corrcoef(guesses, residual)[0, 1], 3),
      " <- knowledge about the item itself")
print("   Brier, pair rate alone (oracle, as if labels were unlimited):", round(brier(pair_rates, outcomes), 4))

# can Claude improve on the oracle pair rate? p = pair_rate + g * (guess - 0.5), g fitted cross-validated
def best_g(p, y, r):
    d = p - 0.5
    return np.sum(d * (y - r)) / np.sum(d ** 2)


g_a = best_g(guesses[half_a], outcomes[half_a], pair_rates[half_a])
g_b = best_g(guesses[half_b], outcomes[half_b], pair_rates[half_b])
mix_a = np.clip(pair_rates[half_a] + g_b * (guesses[half_a] - 0.5), 0.01, 0.99)
mix_b = np.clip(pair_rates[half_b] + g_a * (guesses[half_b] - 0.5), 0.01, 0.99)
print("   Brier, pair rate + Claude nudge (cross-validated):",
      round((brier(mix_a, outcomes[half_a]) + brier(mix_b, outcomes[half_b])) / 2, 4),
      "  nudge g:", round(g_a, 3), "/", round(g_b, 3))
