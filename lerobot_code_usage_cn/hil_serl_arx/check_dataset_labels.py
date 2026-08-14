from pathlib import Path
from collections import Counter, defaultdict

import pyarrow.parquet as pq

DATASET_ROOT = Path("/media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30")

print("=" * 80)
print("Fast label check")
print("root:", DATASET_ROOT)

parquet_files = sorted((DATASET_ROOT / "data").glob("**/*.parquet"))
print("parquet files:", len(parquet_files))

if not parquet_files:
    raise FileNotFoundError(f"No parquet files found under {DATASET_ROOT / 'data'}")

wanted_cols = ["episode_index", "is_failure_data", "is_infer_data", "task_index"]

global_counter = {
    "episode_index": Counter(),
    "is_failure_data": Counter(),
    "is_infer_data": Counter(),
    "task_index": Counter(),
}

ep_counter = defaultdict(Counter)

def to_int(x):
    if isinstance(x, list):
        x = x[0]
    return int(x)

total_rows = 0

for p in parquet_files:
    schema_cols = set(pq.read_schema(p).names)
    cols = [c for c in wanted_cols if c in schema_cols]

    table = pq.read_table(p, columns=cols)
    data = table.to_pydict()

    n = table.num_rows
    total_rows += n

    for i in range(n):
        ep = to_int(data["episode_index"][i]) if "episode_index" in data else -1
        fail = to_int(data["is_failure_data"][i]) if "is_failure_data" in data else -1
        infer = to_int(data["is_infer_data"][i]) if "is_infer_data" in data else -1

        if "episode_index" in data:
            global_counter["episode_index"][ep] += 1
        if "is_failure_data" in data:
            global_counter["is_failure_data"][fail] += 1
        if "is_infer_data" in data:
            global_counter["is_infer_data"][infer] += 1
        if "task_index" in data:
            task = to_int(data["task_index"][i])
            global_counter["task_index"][task] += 1

        ep_counter[ep][f"fail={fail}, infer={infer}"] += 1

print("\nTotal rows:", total_rows)

print("\nGlobal label statistics:")
for k, c in global_counter.items():
    if c:
        print(k, dict(sorted(c.items())))

print("\nPer episode statistics:")
for ep, c in sorted(ep_counter.items()):
    print("episode", ep, dict(c))

print("\nLabel mapping for reward classifier:")
print("is_failure_data=1 -> label=0 -> failure")
print("is_failure_data=0 -> label=1 -> success/non-failure")
