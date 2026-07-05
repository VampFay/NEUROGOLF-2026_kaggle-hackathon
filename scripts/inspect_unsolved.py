"""Deep look at unsolved tasks - show input/output pairs for the smallest unsolved tasks."""
import sys, json
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from neurogolf import arc_data

# Load latest results
with open("/home/z/my-project/data/aggressive_results.json") as f:
    d = json.load(f)
unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]

# Group by input shape and number of pairs
groups = {}
for tid in unsolved:
    try:
        task = arc_data.load_task(tid)
        pairs = arc_data.get_pairs(task)
        in_s = pairs[0][0].shape
        out_s = pairs[0][1].shape
        key = (in_s, out_s, len(pairs))
        groups.setdefault(key, []).append(tid)
    except Exception:
        pass

# Print the 10 smallest unsolved tasks
print("=== Smallest unsolved tasks (by input size) ===")
sorted_groups = sorted(groups.items(), key=lambda kv: (kv[0][0][0] * kv[0][0][1], kv[0][2]))
for (in_s, out_s, n_pairs), tids in sorted_groups[:5]:
    print(f"\n--- in={in_s}, out={out_s}, n_pairs={n_pairs}, count={len(tids)} ---")
    for tid in tids[:3]:
        task = arc_data.load_task(tid)
        pairs = arc_data.get_pairs(task)
        fname = arc_data.task_id_to_filename(tid)
        print(f"\n  Task {tid} ({fname}):")
        for i, (inp, out) in enumerate(pairs[:2]):
            print(f"    Pair {i} IN:")
            for row in inp:
                print(f"      " + " ".join(str(int(x)) for x in row))
            print(f"    Pair {i} OUT:")
            for row in out:
                print(f"      " + " ".join(str(int(x)) for x in row))
