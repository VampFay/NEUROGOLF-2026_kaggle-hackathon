"""Re-analyze unsolved tasks with the new v3 transforms."""
import sys, json
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from collections import Counter
from neurogolf import arc_data
from neurogolf.direct_solvers_v2 import (
    AllDihedralSolver, GenericColorMapSolver, BoundingBoxSolver,
    ConstantOutputSolver, CropToSingleColorRegionSolver,
    ReplaceOneColorSolver, RemoveColorSolver, MaskOutColorSolver,
    PadOutputSolver, ScaleUpNearestSolver,
)
from neurogolf.direct_solvers_v3 import (
    KroneckerDiagSolver, KroneckerFullSolver, TileMirrorSolver,
    ColorInvertSolver, ColorRotateSolver,
    FirstRowBroadcastSolver, FirstColBroadcastSolver,
)

# Load latest results
with open("/home/z/my-project/data/aggressive_results.json") as f:
    d = json.load(f)
unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
print(f"Unsolved: {len(unsolved)}")

new_solvers = [
    AllDihedralSolver(), GenericColorMapSolver(), BoundingBoxSolver(),
    ConstantOutputSolver(), CropToSingleColorRegionSolver(),
    ReplaceOneColorSolver(), RemoveColorSolver(), MaskOutColorSolver(),
    PadOutputSolver(), ScaleUpNearestSolver(),
    KroneckerDiagSolver(), KroneckerFullSolver(), TileMirrorSolver(),
    ColorInvertSolver(), ColorRotateSolver(),
    FirstRowBroadcastSolver(), FirstColBroadcastSolver(),
]

matched = {}
for tid in unsolved:
    try:
        task = arc_data.load_task(tid)
    except Exception:
        continue
    for s in new_solvers:
        try:
            model = s.attempt(task)
            if model is not None:
                matched.setdefault(s.name, []).append(tid)
                break
        except Exception:
            continue

print("\n=== Newly matched by new solvers ===")
total = 0
for name, tids in sorted(matched.items(), key=lambda kv: -len(kv[1])):
    print(f"  {name}: {len(tids)} tasks — first 5: {tids[:5]}")
    total += len(tids)
print(f"\nTotal newly matched: {total}")
