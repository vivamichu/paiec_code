import json

from huggingface_hub import HfApi, RepoFile
from huggingface_hub import hf_hub_download

api = HfApi()

with open("data/benchmark_inventory.json") as file:
    inventory = json.load(file)

revision = inventory["revision"]

folders = [
    row["folder"]
    for row in inventory["benchmarks"]
    if row["eligible_by_type"]
]

print("Candidate folders:", len(folders))
print("Pinned revision:", revision)

wanted_tables = {
    "benchmarks.parquet",
    "subjects.parquet",
    "items.parquet",
    "responses.parquet",
}

total_bytes = 0

for folder in folders:
    print(f"Checking {folder}...")
    found_tables = set()
    entries = api.list_repo_tree(
        repo_id="aims-foundations/measurement-db",
        repo_type="dataset",
        revision=revision,
        path_in_repo=f"{folder}/formatted_tables",
        recursive=False,
    )

    for entry in entries:
        if isinstance(entry, RepoFile):
            filename = entry.path.rsplit("/", 1)[-1]

            if filename in wanted_tables:
                total_bytes += entry.size
                print(f"  Downloading: {filename} ({entry.size / 1_000_000_000:.2f} GB)")
                local_path = hf_hub_download(
                    repo_id="aims-foundations/measurement-db",
                    repo_type="dataset",
                    revision=revision,
                    filename=entry.path,
                    local_dir="data/measurement-db",
                )
                print(f"  Saved: {local_path}")
                found_tables.add(filename)
    missing_tables = wanted_tables - found_tables
    if missing_tables:
        print(f"Missing tables: {missing_tables}")
print(f"Total: {total_bytes / 1_000_000_000:.2f} GB")