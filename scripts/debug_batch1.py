"""Debug: find tasks that nearly match each solver's pattern, with error details."""
import sys, traceback
sys.path.insert(0, "/home/z/my-project")
from neurogolf.memory_golf import get_batch1_golf_solvers, GolfConditionalSolver, GolfDrawLineSolver, GolfShiftSolver
from neurogolf.arc_data import list_task_files, load_task, get_pairs
from neurogolf.validator import evaluate_model
import numpy as np

# Test 1: GolfShiftSolver — find tasks where input/output same size and output is a shift of input
print("=" * 70)
print("Looking for shift tasks (same-size, output = shift of input)")
print("=" * 70)

shift_candidates = []
for path in list_task_files():
    fname = path.stem
    try:
        task = load_task(fname)
    except Exception:
        continue
    pairs = get_pairs(task)
    if not pairs:
        continue
    if not all(inp.shape == out.shape for inp, out in pairs):
        continue
    # Quick check: is there a consistent shift?
    s = GolfShiftSolver()
    try:
        m = s.attempt(task)
        if m is not None:
            ev = evaluate_model(m, task)
            status = "ELIGIBLE" if ev["eligible_for_points"] else "FAIL"
            print(f"  {fname}: model built, {status} cost={ev['cost']} msg={ev['structural_msg']}")
            if not ev["eligible_for_points"]:
                print(f"    functional failures: {ev['functional_failures'][:2]}")
    except Exception as e:
        print(f"  {fname}: EXC {type(e).__name__}: {e}")

print()
print("=" * 70)
print("Looking for draw-line tasks")
print("=" * 70)
for path in list_task_files()[:50]:  # Just sample first 50 for speed
    fname = path.stem
    try:
        task = load_task(fname)
    except Exception:
        continue
    pairs = get_pairs(task)
    if not pairs:
        continue
    if not all(inp.shape == out.shape for inp, out in pairs):
        continue
    # Check if there's a single new color in output
    for inp, out in pairs:
        in_colors = set(np.unique(inp).tolist())
        out_colors = set(np.unique(out).tolist())
        new_colors = out_colors - in_colors
        if len(new_colors) == 1 and 0 in in_colors:
            s = GolfDrawLineSolver()
            try:
                m = s.attempt(task)
                if m is not None:
                    ev = evaluate_model(m, task)
                    print(f"  {fname}: model built, eligible={ev['eligible_for_points']} cost={ev['cost']}")
                    if not ev["eligible_for_points"]:
                        print(f"    failures: {ev['functional_failures'][:1]}")
            except Exception as e:
                print(f"  {fname}: EXC {type(e).__name__}: {e}")
            break

print()
print("=" * 70)
print("Looking for conditional tasks")
print("=" * 70)
for path in list_task_files()[:50]:
    fname = path.stem
    try:
        task = load_task(fname)
    except Exception:
        continue
    pairs = get_pairs(task)
    if not pairs:
        continue
    if not all(inp.shape == out.shape for inp, out in pairs):
        continue
    s = GolfConditionalSolver()
    try:
        m = s.attempt(task)
        if m is not None:
            ev = evaluate_model(m, task)
            print(f"  {fname}: model built, eligible={ev['eligible_for_points']} cost={ev['cost']}")
            if not ev["eligible_for_points"]:
                print(f"    failures: {ev['functional_failures'][:1]}")
    except Exception as e:
        print(f"  {fname}: EXC {type(e).__name__}: {e}")
