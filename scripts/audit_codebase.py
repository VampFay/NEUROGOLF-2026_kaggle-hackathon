"""Audit the neurogolf codebase for gaps and improvement opportunities."""
import sys, json
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from neurogolf import arc_data
from collections import Counter

# Load submission results
with open("/home/z/my-project/data/submission_results.json") as f:
    sub = json.load(f)
results = sub["results"]
solved = [r for r in results if r["eligible"]]
failing = [r for r in results if not r["eligible"]]

# Categorize every failing task in detail
failing_details = []
for r in failing:
    tid = r["task_id"]
    task = arc_data.load_task(tid)
    pairs = arc_data.get_pairs(task)
    inp, out = pairs[0]

    details = {
        "task_id": tid,
        "filename": r["filename"],
        "best_solver": r["solver"],
        "in_shape": inp.shape,
        "out_shape": out.shape,
        "same_shape": inp.shape == out.shape,
        "diff_pct": float((inp != out).sum()) / inp.size * 100 if inp.shape == out.shape else None,
        "in_colors": sorted(set(int(c) for p in pairs for c in np.unique(p[0]))),
        "out_colors": sorted(set(int(c) for p in pairs for c in np.unique(p[1]))),
        "in_color_count": len(set(int(c) for p in pairs for c in np.unique(p[0]))),
        "out_color_count": len(set(int(c) for p in pairs for c in np.unique(p[1]))),
        "out_area_in_area_ratio": (out.shape[0] * out.shape[1]) / (inp.shape[0] * inp.shape[1]),
        "in_square": inp.shape[0] == inp.shape[1],
        "out_square": out.shape[0] == out.shape[1],
        "is_1d": out.shape[0] == 1 or out.shape[1] == 1,
    }

    if details["same_shape"] and details["diff_pct"] is not None and details["diff_pct"] < 30:
        details["family_guess"] = "CA_or_pattern_recolor"
    elif details["same_shape"] and details["diff_pct"] is not None and details["diff_pct"] < 50:
        details["family_guess"] = "complex_color_change"
    elif details["same_shape"]:
        details["family_guess"] = "complex_rearrangement"
    elif details["out_area_in_area_ratio"] == 4:
        details["family_guess"] = "scale_2x_or_kronecker_2x2"
    elif details["out_area_in_area_ratio"] == 9:
        details["family_guess"] = "scale_3x_or_kronecker_3x3"
    elif details["out_area_in_area_ratio"] > 1 and not details["same_shape"]:
        details["family_guess"] = "concat_or_extend"
    elif details["out_area_in_area_ratio"] < 1:
        details["family_guess"] = "crop_or_extract"
    else:
        details["family_guess"] = "other"

    failing_details.append(details)

# Aggregate by family
family_counts = Counter(d["family_guess"] for d in failing_details)
print("=== FAILING TASKS BY FAMILY ===")
for family, count in family_counts.most_common():
    print(f"  {family}: {count}")
    samples = [d for d in failing_details if d["family_guess"] == family][:3]
    for s in samples:
        if s['diff_pct'] is not None:
            print(f"    task {s['task_id']:3d} ({s['filename']}): in={s['in_shape']}, out={s['out_shape']}, diff={s['diff_pct']:.1f}%")
        else:
            print(f"    task {s['task_id']:3d} ({s['filename']}): in={s['in_shape']}, out={s['out_shape']}")

# Identify solvers that exist but never produce results
print("\n=== SOLVERS THAT NEVER SOLVED ANY TASK ===")
all_solvers_tried = set(r["solver"] for r in results if r["solver"] != "none")
print(f"Solvers attempted: {sorted(all_solvers_tried)}")

# Per-solver success rate
print("\n=== SOLVER SUCCESS RATE ===")
solver_attempts = Counter(r["solver"] for r in results)
solver_success = Counter(r["solver"] for r in results if r["eligible"])
for s in sorted(solver_attempts.keys()):
    attempts = solver_attempts[s]
    successes = solver_success.get(s, 0)
    print(f"  {s}: {successes}/{attempts} = {100*successes/attempts:.1f}%")

# Cost & score breakdown of solved tasks
print("\n=== SOLVED TASKS — COST/SCORE BREAKDOWN ===")
for r in solved:
    print(f"  task {r['task_id']:3d} ({r['filename']}): solver={r['solver']:30s} cost={r['cost']:5d} score={r['score']:.2f}")

# Average score by solver (among solved)
print("\n=== AVG SCORE PER SOLVER (solved tasks only) ===")
solver_scores = {}
for r in solved:
    solver_scores.setdefault(r["solver"], []).append(r["score"])
for s, scores in sorted(solver_scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"  {s:30s}: avg={sum(scores)/len(scores):.2f}, n={len(scores)}, total={sum(scores):.2f}")

# Save full details
with open("/home/z/my-project/data/codebase_audit.json", "w") as f:
    json.dump({
        "summary": {
            "total_solved": len(solved),
            "total_failing": len(failing),
            "family_counts": dict(family_counts),
        },
        "failing_details": failing_details,
    }, f, indent=2, default=str)
print(f"\nDetailed audit saved to /home/z/my-project/data/codebase_audit.json")
