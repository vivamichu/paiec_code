"""Test: does Claude's item-level guess beat a flat 0.5 at budget 0?

Samples (subject, item) pairs from the evaluation half of random groups,
asks Claude through my_prompt.py, and compares Brier scores.
Replies are cached, so rerunning the same samples is free.

Run:
  set -a; source .env; set +a
  .venv/bin/python dev/llm_prior_test.py --n 200 --max-dollars 0.5
  .venv/bin/python dev/llm_prior_test.py --n 3 --dry-run      # only show prompts, no cost
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
parser.add_argument("--max-dollars", type=float, default=0.5)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--no-thinking", action="store_true", help="turn the model's hidden thinking off")
args = parser.parse_args()
thinking = "disabled" if args.no_thinking else None

# 1. sample pairs: pick a random group, then a random item of that group
groups = load_groups()
rng = random.Random(args.seed)
samples = []
for i in range(args.n):
    group = rng.choice(groups)
    item_id = rng.choice(sorted(group["items"]))
    runs = group["items"][item_id][2]
    samples.append({
        "bench": group["bench"],
        "subject": group["subject"],
        "item": full_item(group, item_id),
        "outcome": sum(runs) / len(runs),   # share of correct runs
    })
print("sampled", len(samples), "pairs from", len(set([s["bench"] for s in samples])), "benchmarks")

# 2. dry run: show the first prompts and stop
if args.dry_run:
    for sample in samples[:3]:
        prompt = my_prompt.build_prompt(sample["subject"], sample["item"])
        print("-" * 60)
        print(prompt)
        print("(characters:", len(prompt), ")")
    raise SystemExit(0)

# 3. ask Claude
probabilities = []
failed = 0
for number, sample in enumerate(samples):
    if claude_client.totals["dollars"] > args.max_dollars:
        print("stopping: spent more than", args.max_dollars, "dollars")
        break
    prompt = my_prompt.build_prompt(sample["subject"], sample["item"])
    reply = claude_client.ask(prompt, thinking=thinking)
    p = my_prompt.parse_probability(reply)
    if p is None or p < 0 or p > 1:
        failed = failed + 1
        p = 0.5   # an unreadable reply counts as a flat 0.5, like a real submission would
    probabilities.append(p)
    if (number + 1) % 25 == 0:
        print("done", number + 1, " spent $", round(claude_client.totals["dollars"], 4))

# 4. score
outcomes = []
guesses = []
benches = []
for sample, p in zip(samples, probabilities):
    if p is not None:
        outcomes.append(sample["outcome"])
        guesses.append(p)
        benches.append(sample["bench"])
outcomes = np.array(outcomes)
guesses = np.array(guesses)

# Brier over runs: with outcome = share correct, expected squared error is
# p^2 - 2*p*share + share  (since y*y = y for 0/1 runs)
brier_claude = np.mean(guesses ** 2 - 2 * guesses * outcomes + outcomes)
brier_half = np.mean(0.25 + 0 * outcomes)
mean_outcome = np.mean(outcomes)
brier_mean = np.mean(mean_outcome ** 2 - 2 * mean_outcome * outcomes + outcomes)

print()
print("pairs scored:", len(guesses), " unreadable replies:", failed)
print("Brier, always 0.5:                  ", round(brier_half, 4))
print("Brier, Claude:                      ", round(brier_claude, 4))
print("Brier, best constant (the average): ", round(brier_mean, 4), "(unfair: uses the answers)")
print("correlation guess vs outcome:       ", round(np.corrcoef(guesses, outcomes)[0, 1], 3))
print("mean guess:", round(np.mean(guesses), 3), " mean outcome:", round(mean_outcome, 3))
# shrink Claude's guess toward 0.5: p2 = 0.5 + lam * (p - 0.5)
# best lam from least squares (one number fitted on this sample, so slightly optimistic)
distance = guesses - 0.5
lam = np.sum(distance * (outcomes - 0.5)) / np.sum(distance ** 2)
shrunk = 0.5 + lam * distance
brier_shrunk = np.mean(shrunk ** 2 - 2 * shrunk * outcomes + outcomes)
print("best shrink lam:", round(lam, 3), " (1 = trust Claude fully, 0 = ignore it)")
print("Brier after shrinking:              ", round(brier_shrunk, 4))
for lam_test in [0.25, 0.5, 0.75]:
    test = 0.5 + lam_test * distance
    print("   lam", lam_test, "->", round(np.mean(test ** 2 - 2 * test * outcomes + outcomes), 4))

# calibration table: mean outcome for each range of guesses
print()
print("guess range    n   mean guess   mean outcome")
for low, high in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
    mask = (guesses >= low) & (guesses < high)
    if mask.sum() > 0:
        print(" ", low, "-", min(high, 1.0), "  ", mask.sum(), "  ", round(guesses[mask].mean(), 2), "      ", round(outcomes[mask].mean(), 2))

print()
t = claude_client.totals
print("calls:", t["calls"], " from cache:", t["cached"])
print("tokens in:", t["input_tokens"], " out:", t["output_tokens"], " dollars: $", round(t["dollars"], 4))
if t["calls"] > 0:
    print("per call: in", round(t["input_tokens"] / t["calls"]), " out", round(t["output_tokens"] / t["calls"]),
          " $", round(t["dollars"] / t["calls"], 5))
