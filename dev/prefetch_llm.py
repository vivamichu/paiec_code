"""Fill the Claude reply cache for every item that my_model5 will score in a set of draws.

Local tests of my_model5 never call the API (dev/evaluator.py reads the cache),
so run this first with the same --draws/--seed/--pool/--max-group-items.

Free:   .venv/bin/python dev/prefetch_llm.py --draws 20 --seed 0 --pool 5 --dry-run
Costs:  set -a; source .env; set +a
        .venv/bin/python dev/prefetch_llm.py --draws 20 --seed 0 --pool 5 --max-dollars 1.0
Stop any time with Ctrl-C: replies already paid for stay in the cache.
--model asks a different Claude model (same prompt); --limit N asks only N random prompts (a pilot).
"""

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "dev")
import claude_client
import evaluator

KIT = Path(__file__).resolve().parents[1]
DOLLARS_PER_CALL = 0.00021   # measured on 200 calls, 2026-10-22

parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pool", type=int, default=5)
parser.add_argument("--max-group-items", type=int, default=250)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--max-dollars", type=float, default=1.0)
parser.add_argument("--dry-run", action="store_true", help="only count prompts and estimate cost")
parser.add_argument("--model", default="claude-haiku-5-5", help="Claude model to ask")
parser.add_argument("--limit", type=int, default=0, help="ask only this many random prompts (pilot)")
args = parser.parse_args()
claude_client.set_model(args.model)

# the measured cost per call is for Haiku; other models cost more per token
# (and may think longer, so treat their estimate as a lower bound)
haiku_price = claude_client.PRICES["claude-haiku-5-5"][0]
dollars_per_call = DOLLARS_PER_CALL * claude_client.PRICES[args.model][0] / haiku_price

groups = evaluator.load_groups()
model_code = evaluator.load_predict(KIT / "my_model5").__globals__
build_prompt = model_code["build_prompt"]
parse_probability = model_code["parse_probability"]

# replay the draws with a dummy model that only records what it is asked
prompts = []
seen = set()


def record(input, labeled=None):
    prompt = build_prompt(input[0], input[1])
    if prompt not in seen:
        seen.add(prompt)
        prompts.append(prompt)
    return 0.5


evaluator.evaluate(record, groups, args.draws, args.seed, args.pool, args.max_group_items)

missing = []
for prompt in prompts:
    key = claude_client.make_key(prompt, None, 2000, "low", None)
    if key not in claude_client._cache:
        missing.append(prompt)
print("model:", args.model)
print("items in these draws:", len(prompts), " already cached:", len(prompts) - len(missing),
      " to ask:", len(missing), " estimated cost: $%.2f" % (len(missing) * dollars_per_call))
if args.limit > 0:
    random.Random(0).shuffle(missing)
    missing = missing[:args.limit]
    print("pilot: asking only", len(missing), "random prompts")
if args.dry_run or len(missing) == 0:
    sys.exit(0)

done = {"asked": 0, "failed": 0, "skipped": 0}


def fetch(prompt):
    if claude_client.totals["dollars"] >= args.max_dollars:
        done["skipped"] = done["skipped"] + 1
        return
    try:
        claude_client.ask(prompt, max_tokens=2000, effort="low")
    except Exception as error:
        done["failed"] = done["failed"] + 1
        if done["failed"] <= 3:
            print("  failed:", str(error)[:150], flush=True)
    done["asked"] = done["asked"] + 1
    if done["asked"] % 200 == 0:
        print("  asked", done["asked"], "of", len(missing),
              " spent $%.3f" % claude_client.totals["dollars"], flush=True)


with ThreadPoolExecutor(max_workers=args.threads) as pool:
    list(pool.map(fetch, missing))

t = claude_client.totals
print("new calls:", t["calls"], " failed:", done["failed"], " skipped (dollar cap):", done["skipped"])
print("tokens in/out: %d / %d" % (t["input_tokens"], t["output_tokens"]),
      " spent $%.3f" % t["dollars"])
if t["calls"] > 0:
    print("cost per call: $%.5f" % (t["dollars"] / t["calls"]))

# why replies ended: "refusal" or "max_tokens" give no usable guess (the model falls back to Step 3)
# and can the model's parser read them? (a sentence before the number would be misread)
stops = {}
guesses = []
unreadable = 0
unreadable_first = 0
for prompt in missing:
    key = claude_client.make_key(prompt, None, 2000, "low", None)
    if key in claude_client._cache:
        row = claude_client._cache[key]
        reason = str(row["stop_reason"])
        stops[reason] = stops.get(reason, 0) + 1
        if parse_probability(row["text"]) is None:
            unreadable_first = unreadable_first + 1
        guess = claude_client.parse_last_line(row["text"])
        if reason == "refusal":
            guess = None
        if guess is None:
            unreadable = unreadable + 1
        else:
            guesses.append(guess)
        if args.limit > 0 and len(guesses) + unreadable <= 5:
            print("  reply:", repr(row["text"][:80]), " read as:", guess)
print("stop reasons:", stops)
if len(guesses) > 0:
    print("unreadable replies: first number", unreadable_first, " last line", unreadable,
          " guesses (last line): mean %.2f  min %.2f  max %.2f"
          % (sum(guesses) / len(guesses), min(guesses), max(guesses)))
