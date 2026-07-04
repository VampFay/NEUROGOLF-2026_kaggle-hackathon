"""Test the 5 rebuilt golf solvers on all 400 ARC tasks."""
import sys, json, traceback
sys.path.insert(0, "/home/z/my-project")
from neurogolf.memory_golf import get_batch1_golf_solvers
from neurogolf.arc_data import list_task_files, load_task
from neurogolf.validator import evaluate_model

SOLVERS = get_batch1_golf_solvers()
results = {s.name: [] for s in SOLVERS}

for i, path in enumerate(list_task_files()):
    fname = path.stem
    try:
        task = load_task(fname)
    except Exception:
        continue
    for s in SOLVERS:
        try:
            model = s.attempt(task)
        except Exception as e:
            continue
        if model is None:
            continue
        try:
            ev = evaluate_model(model, task)
        except Exception as e:
            results[s.name].append((fname, "EVAL_ERR", str(e)[:80]))
            continue
        if ev["eligible_for_points"]:
            results[s.name].append((fname, "OK", f"cost={ev['cost']} score={ev['score']:.2f}"))

print("=" * 70)
print("RESULTS: 5 rebuilt golf solvers (batch 1)")
print("=" * 70)
total_solved = 0
for s in SOLVERS:
    n = len(results[s.name])
    total_solved += n
    print(f"\n{s.name}: {n} tasks solved")
    for fname, status, info in results[s.name][:15]:
        print(f"  {fname}: {status} {info}")
print(f"\nTotal solved: {total_solved}")
