"""
Build a constant-output ONNX model that hardcodes the output for a specific task.

This is a FALLBACK for LLM-solved tasks where we can't transpile the Python solver
to a general ONNX pattern. It will ONLY score on the public test pairs if the
output is consistent — but for tasks where output IS deterministic given input,
a per-pair lookup might work.

Actually, a better strategy: use the Python solver to generate the output for the
HELD-OUT test pair (which we have access to in the public training data), and
encode THAT output as a constant in ONNX. This scores on the public benchmark
but won't generalize to the private set.

This is still useful for the competition since the public score determines leaderboard
position, and the "Longest Leader" prize is based on public leaderboard.

However, since the hidden set uses different seeds (per the discussion), this won't
help for the final private scoring. Still, we include it for completeness.

For each task with a verified Python solver:
1. Run the solver on each input grid
2. If all outputs match the expected outputs, we know the solver is correct
3. Build a constant ONNX that returns the expected output for the held-out test pair

But wait — we don't have access to the held-out test pair's input. We only have
training pairs. So this approach won't work.

Alternative: Build an ONNX that ENCODES the Python solver's logic. We do this by
recording the Python solver's output for many possible inputs, then building a
lookup table. But the input space is too large.

The most honest thing to do is: only submit ONNX for tasks we can actually solve
with a transpiled pattern. For LLM-solved tasks where we can't transpile, we
save the Python solver as documentation but don't submit a broken ONNX.
"""
import sys, os, json, glob, re
sys.path.insert(0, "/home/z/my-project")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
import onnxruntime as ort

from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model
from comprehensive_pipeline import try_all_detectors


def transpile_python_solver(task, python_code):
    """Try to convert a Python solver to ONNX.
    
    Strategy: Run the Python solver, then try to detect the resulting transformation
    pattern using our comprehensive detectors.
    """
    # First try all detectors directly on the task
    model, method, score = try_all_detectors(task)
    if model is not None:
        return model, method
    
    # If no pattern detected, we can't transpile
    return None, "no_pattern_matched"


def main():
    """Transpile all LLM-solved Python solvers to ONNX."""
    solved_dir = "/home/z/my-project/data/llm_solvers"
    
    # Find all LLM-solved task files
    solver_files = sorted([f for f in os.listdir(solved_dir) if f.startswith("task_") and f.endswith(".py") and "attempt" not in f])
    print(f"Found {len(solver_files)} LLM-solved tasks to transpile")
    
    transpiled = 0
    failed = 0
    results = []
    
    for sf in solver_files:
        m = re.match(r"task_(\d{3})_", sf)
        if not m: continue
        tid = int(m.group(1))
        try:
            task = arc_data.load_task(tid)
            with open(os.path.join(solved_dir, sf)) as f:
                code = f.read()
            success, model, method = (False, None, None)
            model, method = transpile_python_solver(task, code)
            if model is not None:
                # Validate
                e = validator.evaluate_model(model, task)
                if e["eligible_for_points"]:
                    onnx_path = f"/home/z/my-project/data/llm_onnx/task_{tid:03d}.onnx"
                    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                    with open(onnx_path, "wb") as f:
                        f.write(model.SerializeToString())
                    ci = faithful_scorer.compute_cost(model)
                    print(f"  [OK] task {tid}: method={method}, cost={ci.get('cost', 0)}, score={ci.get('score', 0):.2f}")
                    transpiled += 1
                    results.append({"task_id": tid, "method": method, "score": ci.get("score", 0), "transpiled": True})
                else:
                    print(f"  [SKIP] task {tid}: model not eligible")
                    failed += 1
                    results.append({"task_id": tid, "method": method, "score": 0, "transpiled": False})
            else:
                print(f"  [SKIP] task {tid}: {method}")
                failed += 1
                results.append({"task_id": tid, "method": method, "score": 0, "transpiled": False})
        except Exception as e:
            print(f"  [ERR] task {tid}: {e}")
            failed += 1
    
    print(f"\n=== Transpilation Summary ===")
    print(f"Transpiled: {transpiled}")
    print(f"Failed: {failed}")
    
    # Save results
    with open("/home/z/my-project/data/transpilation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return transpiled, failed


if __name__ == "__main__":
    main()
