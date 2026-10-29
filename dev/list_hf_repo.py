"""List what the HF dataset repos contain (no downloads) and compare with what we have locally.

Lists the top level, then each benchmark folder's top level and its formatted_tables/
(not the huge raw/ folders). Prints progress per folder.

Run:  .venv/bin/python dev/list_hf_repo.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, RepoFile, RepoFolder

KIT = Path(__file__).resolve().parents[1]
REPO = "aims-foundations/measurement-db"
api = HfApi()
inventory = json.load(open(KIT / "data" / "benchmark_inventory.json"))
pinned = inventory["revision"]

main_info = api.dataset_info(REPO, revision="main")
print("pinned revision:", pinned[:12], " main revision:", main_info.sha[:12],
      " same" if main_info.sha.startswith(pinned[:12]) else " DIFFERENT")


def list_dir(path, revision):
    files = []
    folders = []
    for entry in api.list_repo_tree(REPO, repo_type="dataset", revision=revision, path_in_repo=path,
                                    recursive=False):
        if isinstance(entry, RepoFile):
            files.append((entry.path.rsplit("/", 1)[-1], entry.size))
        elif isinstance(entry, RepoFolder):
            folders.append(entry.path.rsplit("/", 1)[-1])
    return files, folders


top_files, top_folders = list_dir("", pinned)
print("\ntop-level files:")
for name, size in top_files:
    print("   %-45s %9.2f MB" % (name, size / 1e6))
print("benchmark folders:", len(top_folders))

eligible = set()
for row in inventory["benchmarks"]:
    if row["eligible_by_type"]:
        eligible.add(row["folder"])

root_kinds = Counter()
table_kinds = Counter()
table_bytes = defaultdict(int)
sub_folders = Counter()
for number, folder in enumerate(sorted(top_folders)):
    if folder.startswith("."):
        continue
    files, folders = list_dir(folder, pinned)
    for name, size in files:
        root_kinds[name] += 1
    for name in folders:
        sub_folders[name] += 1
    if "formatted_tables" in folders:
        t_files, _ = list_dir(folder + "/formatted_tables", pinned)
        for name, size in t_files:
            key = name + (" (binary benchmark)" if folder in eligible else "")
            table_kinds[key] += 1
            table_bytes[key] += size
    if number % 25 == 0:
        print("  ...", number, "folders listed", flush=True)

print("\nsub-folders inside benchmark folders:")
for name, count in sub_folders.most_common():
    print("   %-30s %4d" % (name, count))
print("\nfiles at a benchmark folder's top level:")
for name, count in root_kinds.most_common():
    print("   %-30s %4d" % (name, count))
print("\nfiles inside formatted_tables (count, total GB):")
for name, count in sorted(table_kinds.items()):
    print("   %-50s %4d  %8.2f GB" % (name, count, table_bytes[name] / 1e9))

# companion embeddings repo, top level only
print("\naims-foundations/measurement-db-embed:")
try:
    count = 0
    total = 0
    for entry in api.list_repo_tree("aims-foundations/measurement-db-embed", repo_type="dataset",
                                    recursive=False):
        if isinstance(entry, RepoFile):
            count += 1
            total += entry.size
    print("   files:", count, " total GB: %.2f" % (total / 1e9))
except Exception as error:
    print("   failed:", type(error).__name__, str(error)[:150])
