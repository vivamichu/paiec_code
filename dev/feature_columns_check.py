"""Do any item columns tell us which questions are harder, INSIDE a benchmark?

Looks at what predict() can see about an item:
  (a) item_features keys that sound like difficulty (difficulty, level, tier, hard, grade, ...)
  (b) any numeric item_features value (bigger number = harder?)
  (c) non-English text (share of non-ASCII characters)
  (d) interactors (do they vary inside a benchmark?)

For each, the within-benchmark Spearman correlation with the item's success rate
(averaged over subjects). Negative = "more of this, fewer correct answers".

Run:  .venv/bin/python dev/feature_columns_check.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

DB = Path(__file__).resolve().parents[1] / "data" / "measurement-db"
MIN_SUBJECTS = 3
DIFFICULTY_WORDS = ["difficult", "level", "tier", "hard", "complex", "grade"]
SKIP_WORDS = ["id", "index", "idx", "seed", "hash", "commit", "date", "year", "time", "version", "url"]
WORD_SCALE = {"easy": 1, "simple": 1, "low": 1, "basic": 1, "beginner": 1,
              "medium": 2, "moderate": 2, "intermediate": 2, "mid": 2,
              "hard": 3, "difficult": 3, "high": 3, "advanced": 3, "expert": 3, "challenging": 3}


def parse_features(text):
    # "key=value;key=value" -> dict
    result = {}
    for part in str(text or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def to_number(value):
    value = value.lower().strip()
    if value in WORD_SCALE:
        return float(WORD_SCALE[value])
    match = re.fullmatch(r"-?\d+(\.\d+)?k?", value)
    if match is None:
        return None
    if value.endswith("k"):
        return float(value[:-1]) * 1000
    return float(value)


difficulty_rows = []
numeric_rows = []
ascii_rows = []
interactor_rows = []

for folder in sorted(DB.iterdir()):
    t = folder / "formatted_tables"
    if not (t / "responses.parquet").exists():
        continue
    resp = pq.read_table(t / "responses.parquet", columns=["subject_id", "item_id", "response", "interactors"],
                         read_dictionary=["subject_id", "item_id", "interactors"]).to_pandas()
    resp = resp.dropna(subset=["response"])
    if len(resp) == 0 or resp.subject_id.nunique() < MIN_SUBJECTS:
        continue
    interactor_rows.append((folder.name, resp.interactors.astype(str).nunique()))

    by_item = resp.groupby("item_id", observed=True).agg(rate=("response", "mean"),
                                                         n_subj=("subject_id", "nunique"))
    by_item = by_item[by_item.n_subj >= MIN_SUBJECTS].reset_index()
    if len(by_item) < 30:
        continue
    by_item["item_id"] = by_item.item_id.astype(str)
    items = pd.read_parquet(t / "items.parquet", columns=["item_id", "content", "item_features"])
    items = items.drop_duplicates("item_id")
    items["item_id"] = items.item_id.astype(str)
    data = by_item.merge(items, on="item_id", how="inner")
    if len(data) < 30:
        continue

    # (c) non-ASCII share of the item text
    shares = []
    for text in data.content.astype(str):
        text = text[:3000]
        if len(text) == 0:
            shares.append(0.0)
        else:
            non_ascii = 0
            for ch in text:
                if ord(ch) > 127:
                    non_ascii = non_ascii + 1
            shares.append(non_ascii / len(text))
    shares = np.array(shares)
    if shares.std() > 0.01:
        ascii_rows.append((folder.name, len(data), spearmanr(shares, data.rate).correlation))

    # (a) and (b) feature keys
    parsed = []
    for text in data.item_features:
        parsed.append(parse_features(text))
    keys = set()
    for features in parsed:
        for key in features:
            keys.add(key)
    for key in sorted(keys):
        lower = key.lower()
        values = []
        rates = []
        for features, rate in zip(parsed, data.rate):
            if key in features:
                number = to_number(features[key])
                if number is not None:
                    values.append(number)
                    rates.append(rate)
        if len(values) < 30 or len(set(values)) < 2:
            continue
        corr = spearmanr(values, rates).correlation
        row = (folder.name, key, len(values), len(set(values)), corr)
        is_difficulty = False
        for word in DIFFICULTY_WORDS:
            if word in lower:
                is_difficulty = True
        is_skipped = False
        for word in SKIP_WORDS:
            if word == lower or lower.endswith("_" + word) or lower.startswith(word + "_"):
                is_skipped = True
        if is_difficulty:
            difficulty_rows.append(row)
        elif not is_skipped:
            numeric_rows.append(row)

print("(a) difficulty-like keys (within-benchmark Spearman with success; negative = harder)")
if len(difficulty_rows) == 0:
    print("   none found")
for bench, key, n, distinct, corr in difficulty_rows:
    print("   %-28s %-28s n=%5d values=%3d corr=%+.2f" % (bench, key[:28], n, distinct, corr))

print()
corrs = np.array([row[4] for row in numeric_rows])
print("(b) other numeric keys: %d (benchmark, key) combinations in %d benchmarks"
      % (len(numeric_rows), len(set([row[0] for row in numeric_rows]))))
if len(corrs) > 0:
    print("   median corr %+.2f, share negative %.2f, share |corr| > 0.2: %.2f"
          % (np.median(corrs), (corrs < 0).mean(), (np.abs(corrs) > 0.2).mean()))
    strongest = sorted(numeric_rows, key=lambda row: -abs(row[4]))[:12]
    for bench, key, n, distinct, corr in strongest:
        print("   %-28s %-28s n=%5d values=%3d corr=%+.2f" % (bench, key[:28], n, distinct, corr))

print()
c = np.array([row[2] for row in ascii_rows])
print("(c) non-English text share: %d benchmarks with varying share" % len(ascii_rows))
if len(c) > 0:
    print("   median corr %+.2f, share negative %.2f" % (np.median(c), (c < 0).mean()))
    for bench, n, corr in sorted(ascii_rows, key=lambda row: row[2])[:6]:
        print("   %-28s n=%5d corr=%+.2f" % (bench, n, corr))

print()
varying = [row for row in interactor_rows if row[1] > 1]
print("(d) interactors: %d of %d benchmarks have more than one value" % (len(varying), len(interactor_rows)))
for bench, count in varying[:8]:
    print("   %-28s distinct values=%d" % (bench, count))
