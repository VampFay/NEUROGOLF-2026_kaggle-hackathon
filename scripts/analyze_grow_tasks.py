"""
Analyze 20 scale/grow tasks to determine rules.
Prints all training pairs in a readable format.
"""
import sys, os, json
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from neurogolf import arc_data

TASK_IDS = [3, 19, 104, 106, 107, 108, 114, 123, 194, 211, 269, 275, 289, 295, 304, 315, 327, 376, 388, 398]

def grid_str(a):
    """Compact grid string."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return "\n".join("".join(str(int(v)) for v in row) for row in a)

def analyze_task(tid):
    task = arc_data.load_task(tid)
    fname = arc_data.task_id_to_filename(tid)
    print(f"\n{'='*80}")
    print(f"# Task {tid} — {fname}")
    print(f"{'='*80}")
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    for i, p in enumerate(train_pairs):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"\n-- Train {i} --")
        print(f"  IN  {inp.shape}:")
        print(grid_str(inp))
        print(f"  OUT {out.shape}:")
        print(grid_str(out))
    for i, p in enumerate(test_pairs):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"\n-- Test {i} --")
        print(f"  IN  {inp.shape}:")
        print(grid_str(inp))
        print(f"  OUT {out.shape}:")
        print(grid_str(out))


if __name__ == "__main__":
    for tid in TASK_IDS:
        analyze_task(tid)
