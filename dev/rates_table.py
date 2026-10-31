"""Success rate of every subject on every PUBLIC benchmark (withheld ones are dropped).

One row per subject per benchmark. Columns:
  bench, subject_id, n_items, rate, normalized_name, provider
Saved in data/cache/rates.parquet (delete it to build again).

Use it for offline analysis. A model built from it must be tested
leave-one-benchmark-out, otherwise it sees the answers.

    from rates_table import load_rates
    rates = load_rates()
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluator

KIT = Path(__file__).resolve().parents[1]
DB = KIT / "data" / "measurement-db"
CACHE = KIT / "data" / "cache" / "rates.parquet"


def rates_for_benchmark(bench):
    """Rates of all subjects of one benchmark."""
    folder = DB / bench / "formatted_tables"

    resp = pq.read_table(
        folder / "responses.parquet",
        columns=["subject_id", "item_id", "response"],
        read_dictionary=["subject_id", "item_id"],
    ).to_pandas()
    resp = resp.dropna(subset=["response"])
    if len(resp) == 0:
        return None

    by_subject = resp.groupby("subject_id", observed=True)
    table = pd.DataFrame({
        "n_items": by_subject.item_id.nunique(),
        "rate": by_subject.response.mean(),
        "mean_rate": resp.response.mean(),
        "std_rate": by_subject.response.std(),
        "min_rate": by_subject.response.min(),
        "max_rate": by_subject.response.max(),
        "gap": by_subject.response.mean() - resp.response.mean(),
    })
    table = table.reset_index()
    table["subject_id"] = table.subject_id.astype(str)

    # add the name and provider of each subject
    subjects = pd.read_parquet(folder / "subjects.parquet",
                               columns=["subject_id", "normalized_name", "provider"])
    subjects = subjects.drop_duplicates("subject_id")
    table = table.merge(subjects, on="subject_id", how="left")

    table.insert(0, "bench", bench)
    return table


def load_rates(rebuild=False):
    if CACHE.exists() and not rebuild:
        rates = pd.read_parquet(CACHE)
        # an old cache may still hold withheld benchmarks
        public = []
        for bench in rates.bench:
            public.append(evaluator.is_public(bench))
        return rates[public].reset_index(drop=True)

    tables = []
    for folder in sorted(DB.iterdir()):
        if not (folder / "formatted_tables" / "responses.parquet").exists():
            continue
        if not evaluator.is_public(folder.name):
            continue
        try:
            table = rates_for_benchmark(folder.name)
        except Exception as error:
            print("skipped", folder.name, ":", error)
            continue
        if table is not None:
            tables.append(table)

    rates = pd.concat(tables, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    rates.to_parquet(CACHE)
    return rates


if __name__ == "__main__":
    rates = load_rates(rebuild=True)
    print(rates.shape, rates.bench.nunique(), "benchmarks", rates.normalized_name.nunique(), "names")
    print(rates.head().to_string())
