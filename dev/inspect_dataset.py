from dbm import error

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import json
from huggingface_hub.errors import HfHubHTTPError

failures = []
inventory = []
api = HfApi()

info = api.repo_info(
    repo_id="aims-foundations/measurement-db", 
    repo_type="dataset",
    revision="main",
)

print("Dataset revision:", info.sha)
print("Number of files in the dataset:", len(info.siblings))

benchmark_files = [
    entry.rfilename
    for entry in info.siblings
    if entry.rfilename.endswith("benchmarks.parquet") 
]

print("Benchmark metadata files found: ", len(benchmark_files))

# for filename in benchmark_files[:20]:
#     print("Benchmark metadata file:", filename)

from huggingface_hub import RepoFolder

benchmark_folders = [
    entry.path
    for entry in api.list_repo_tree(
        repo_id="aims-foundations/measurement-db",
        repo_type="dataset",
        revision=info.sha,
        recursive=False,
    )
    if isinstance(entry, RepoFolder)
]

print("Top-level folders:", len(benchmark_folders))

for folder in benchmark_folders:
    filename = f"{folder}/formatted_tables/benchmarks.parquet"
    print(f"Downloading {filename}...")
    try:
        metadata_path = hf_hub_download(
            repo_id="aims-foundations/measurement-db",
            repo_type="dataset",
            revision=info.sha,
            filename=filename,
            local_dir="data/measurement-db",
        )
    except HfHubHTTPError as error:
        failures.append({
            "folder": folder,
            "error": str(error),
        })
        print(f"Could not download {folder}: {error}")
        continue    
   
    benchmark_rows = pq.read_table(metadata_path).to_pylist()
    for row in benchmark_rows:
        is_eligible = (
            row.get("response_type") == "binary"
            and row.get("granularity") == "item"
        )

        scale = json.loads(row["response_scale"])

        inventory.append({
            "folder": folder,
            "name": row.get("name"),
            "response_type": row.get("response_type"),
            "granularity": row.get("granularity"),
            "eligible_by_type": is_eligible,
            "direction": scale.get("direction"),
            "meanings": scale.get("meanings"),
            "n_subjects": row.get("n_subjects"),
            "n_items": row.get("n_items"),
            "n_responses": row.get("n_responses"),
        })

        print(f"{folder}: eligible_by_type={is_eligible}")

with open("data/benchmark_inventory.json", "w") as file:
    json.dump(
        {"revision": info.sha, "benchmarks": inventory, "failures": failures},
        file,
        indent=2,
    )
# from huggingface_hub import RepoFile

# table_files = list(
#     api.list_repo_tree(
#         repo_id="aims-foundations/measurement-db",
#         repo_type="dataset",
#         revision=info.sha,
#         path_in_repo="real_webagents/formatted_tables",
#         recursive=False,
#     )
# )

# for entry in table_files:
#     if isinstance(entry, RepoFile):
#         size_mb = entry.size / 1_000_000
#         print(f"{entry.path}: {size_mb:.2f} MB")




