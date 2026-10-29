"""Feasibility check: can a model that sees ONLY the item predict how hard it is,
on benchmarks it has never seen?

Target: an item's success rate averaged over all subjects (model-agnostic difficulty).
Features: TF-IDF of item_features + item text, plus text length.
Model: ridge regression. Split: 5 folds by BENCHMARK (test benchmarks never seen in training).

Two numbers matter:
  overall corr      - includes "this benchmark is easy/hard" (labels learn that fast anyway)
  within-benchmark  - "this item is harder than the others of its benchmark":
                      the part that could help at budgets 3 to 31

Run:  .venv/bin/python dev/item_difficulty_check.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

DB = Path(__file__).resolve().parents[1] / "data" / "measurement-db"
MAX_ITEMS = 300        # items per benchmark (big benchmarks would dominate otherwise)
MIN_SUBJECTS = 3       # need several subjects for a stable item difficulty
MAX_CHARS = 2000       # text kept per item

rows = []
for folder in sorted(DB.iterdir()):
    t = folder / "formatted_tables"
    if not (t / "responses.parquet").exists():
        continue
    resp = pq.read_table(t / "responses.parquet", columns=["subject_id", "item_id", "response"],
                         read_dictionary=["subject_id", "item_id"]).to_pandas().dropna(subset=["response"])
    if len(resp) == 0 or resp.subject_id.nunique() < MIN_SUBJECTS:
        continue
    by_item = resp.groupby("item_id", observed=True).agg(rate=("response", "mean"),
                                                         n_subj=("subject_id", "nunique"))
    by_item = by_item[by_item.n_subj >= MIN_SUBJECTS].reset_index()
    if len(by_item) < 20:
        continue
    by_item["item_id"] = by_item.item_id.astype(str)
    if len(by_item) > MAX_ITEMS:
        by_item = by_item.sample(MAX_ITEMS, random_state=0)
    items = pd.read_parquet(t / "items.parquet", columns=["item_id", "content", "item_features"])
    items = items.drop_duplicates("item_id")
    items["item_id"] = items.item_id.astype(str)
    merged = by_item.merge(items, on="item_id", how="inner")
    for i in range(len(merged)):
        content = str(merged.content.iloc[i] or "")
        features = str(merged.item_features.iloc[i] or "")
        rows.append({
            "bench": folder.name,
            "rate": merged.rate.iloc[i],
            "text": features + " || " + content[:MAX_CHARS],
            "log_len": np.log1p(len(content)),
        })

data = pd.DataFrame(rows)
print("items:", len(data), " benchmarks:", data.bench.nunique())

y = data.rate.values
groups = data.bench.values
pred_text = np.zeros(len(data))
pred_len = np.zeros(len(data))

for train, test in GroupKFold(n_splits=5).split(data, y, groups):
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    x_train = vectorizer.fit_transform(data.text.values[train])
    x_test = vectorizer.transform(data.text.values[test])
    length_train = csr_matrix(data.log_len.values[train].reshape(-1, 1))
    length_test = csr_matrix(data.log_len.values[test].reshape(-1, 1))

    model = Ridge(alpha=3.0)
    model.fit(hstack([x_train, length_train]).tocsr(), y[train])
    pred_text[test] = model.predict(hstack([x_test, length_test]).tocsr())

    # baseline: text length only
    small = Ridge(alpha=1.0)
    small.fit(data.log_len.values[train].reshape(-1, 1), y[train])
    pred_len[test] = small.predict(data.log_len.values[test].reshape(-1, 1))


def within(values, labels):
    # remove each benchmark's own mean
    out = values.copy()
    for bench in np.unique(labels):
        mask = labels == bench
        out[mask] = values[mask] - values[mask].mean()
    return out


y_within = within(y, groups)
for name, pred in [("TF-IDF + length", pred_text), ("length only", pred_len)]:
    overall = np.corrcoef(pred, y)[0, 1]
    within_corr = np.corrcoef(within(pred, groups), y_within)[0, 1]
    per_bench = []
    for bench in np.unique(groups):
        mask = groups == bench
        if mask.sum() >= 30 and np.std(pred[mask]) > 0 and np.std(y[mask]) > 0:
            per_bench.append(np.corrcoef(pred[mask], y[mask])[0, 1])
    per_bench = np.array(per_bench)
    print()
    print(name)
    print("  overall corr (held-out benchmarks):  %.3f" % overall)
    print("  within-benchmark corr:               %.3f" % within_corr)
    print("  per-benchmark corr: median %.3f, share > 0.2: %.2f, share < 0: %.2f (n=%d)"
          % (np.median(per_bench), (per_bench > 0.2).mean(), (per_bench < 0).mean(), len(per_bench)))
