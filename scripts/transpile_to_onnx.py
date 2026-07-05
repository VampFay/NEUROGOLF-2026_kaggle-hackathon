"""
ONNX transpiler for Python solve(grid) functions.

Strategy: Since the input is bounded (1, 10, 30, 30) one-hot, and the grid values
are 0-9, we can:

1. If the function is "pure per-cell" (output[r,c] depends only on input[r,c]),
   we can derive a color_map and emit a 1x1 conv.

2. If the function is "spatially uniform" (same transformation everywhere),
   we can use Conv or other patterns.

3. Fallback: Build a "Constant Lookup Table" — but only works for grids up to
   30x30 with at most 10 colors. The state space is too large for direct lookup.

4. Most practical fallback: Use the Python solver to compute outputs for the
   test inputs we know about, then build a per-pair constant ONNX. But this
   won't generalize to the hidden test pairs.

5. Best approach: MANUALLY transpile simple solver patterns:
   - color_map: emit 1x1 conv
   - geometric (flip/rotate/transpose): emit Slice/Transpose
   - crop: emit Slice
   - tile: emit Tile
   - constant: emit Constant

Since most ARC tasks have a single transformation pattern, we'll use a different
approach: Use the LLM to also output a description of the pattern, and we
manually map common pattern descriptions to ONNX.

For tasks we can't transpile directly, we'll skip them (no submission for that
task) — which is better than submitting a wrong answer.

Better strategy: After the LLM solves the task in Python, ask the LLM to also
specify the ONNX pattern. For now, use a simple heuristic: if the LLM's solve
function uses certain numpy ops, map them to ONNX.
"""
import sys, os, json, re
sys.path.insert(0, "/home/z/my-project")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
import onnxruntime as ort

from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model
from neurogolf.direct_solvers_v2 import AllDihedralSolver


# ─────────────────────────────────────────────────────────────────────────────
# Pattern detection — try simple ONNX-able patterns after LLM solves in Python
# ─────────────────────────────────────────────────────────────────────────────

def try_dihedral_with_colormap(task):
    """Try all 8 dihedral transforms × color permutations."""
    pairs = arc_data.get_pairs(task)
    for inp, out in pairs:
        if inp.shape != out.shape: return None
    in_h0, in_w0 = pairs[0][0].shape
    for inp, _ in pairs:
        if inp.shape != (in_h0, in_w0): return None
    transforms = [
        ("identity", lambda x: x),
        ("flip_lr", np.fliplr),
        ("flip_ud", np.flipud),
        ("rot180", lambda x: np.rot90(x, 2)),
        ("rot90", lambda x: np.rot90(x, 1)),
        ("rot270", lambda x: np.rot90(x, 3)),
        ("transpose", lambda x: x.T),
        ("anti_transpose", lambda x: np.rot90(np.rot90(x.T, 1), 2)),
    ]
    for tname, tfn in transforms:
        mapping = {}
        ok = True
        for inp, out in pairs:
            transformed = tfn(inp)
            for c in range(NUM_COLORS):
                in_cells = (transformed == c)
                if not in_cells.any(): continue
                out_at = out[in_cells]
                out_colors = np.unique(out_at)
                if len(out_colors) != 1:
                    ok = False; break
                t = int(out_colors[0])
                if c in mapping and mapping[c] != t:
                    ok = False; break
                mapping[c] = t
            if not ok: break
        if not ok: continue
        if not any(k != v for k, v in mapping.items()): continue
        # Verify
        valid = True
        for inp, out in pairs:
            transformed = tfn(inp)
            mapped = transformed.copy()
            for k, v in mapping.items():
                mapped[transformed == k] = v
            if not np.array_equal(mapped, out):
                valid = False; break
        if not valid: continue
        # Build ONNX
        from neurogolf.direct_solvers_v4 import ColorMapThenDihedralSolver
        solver = ColorMapThenDihedralSolver()
        return solver.attempt(task)
    return None


def try_pure_color_map(task):
    """Try a pure color permutation (no geometric transform)."""
    pairs = arc_data.get_pairs(task)
    for inp, out in pairs:
        if inp.shape != out.shape: return None
    mapping = {}
    for inp, out in pairs:
        for c in range(NUM_COLORS):
            in_cells = (inp == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1: return None
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t: return None
            mapping[c] = t
    if not mapping or not any(k != v for k, v in mapping.items()):
        return None
    # Check if bijective — use Gather exploit
    if set(mapping.keys()) == set(mapping.values()):
        indices = list(range(NUM_COLORS))
        for source, target in mapping.items():
            indices[target] = source
        return _make_model([
            h.make_node("Constant", [], ["i"], value=h.make_tensor("iv", TensorProto.INT64, [NUM_COLORS], indices)),
            h.make_node("Gather", [INPUT_NAME, "i"], [OUTPUT_NAME], axis=1),
        ])
    # Otherwise use 1x1 conv
    from neurogolf.dsl import color_map
    return color_map(mapping)


def try_python_solver_to_onnx(task, python_code):
    """Try multiple ONNX patterns to find one that matches the Python solver's behavior."""
    # First try the simple pattern detectors
    candidates = [
        try_pure_color_map(task),
        try_dihedral_with_colormap(task),
    ]
    for cand in candidates:
        if cand is not None:
            # Validate functional correctness
            e = validator.evaluate_model(cand, task)
            if e["eligible_for_points"]:
                return cand
    return None


def transpile_solver(task, python_code):
    """Transpile a Python solve() function to ONNX.
    
    Returns: (success, model_or_none, method)
    """
    # Try simple pattern detection first
    model = try_python_solver_to_onnx(task, python_code)
    if model is not None:
        return True, model, "pattern_detection"
    
    # If we can't transpile, we can still try a "runtime execution" approach:
    # The Python solver runs in Python at inference time. But ONNX can't run Python.
    # So we must skip this task.
    return False, None, "no_onnx_pattern_matched"


def main():
    """Transpile all LLM-solved Python solvers to ONNX."""
    solved_dir = "/home/z/my-project/data/llm_solvers"
    onnx_dir = "/home/z/my-project/data/llm_onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    
    # Find all solved task files
    solver_files = sorted([f for f in os.listdir(solved_dir) if f.startswith("task_") and f.endswith(".py") and "attempt" not in f])
    print(f"Found {len(solver_files)} LLM-solved tasks")
    
    transpiled = 0
    failed = 0
    for sf in solver_files:
        m = re.match(r"task_(\d{3})_", sf)
        if not m: continue
        tid = int(m.group(1))
        try:
            task = arc_data.load_task(tid)
            with open(os.path.join(solved_dir, sf)) as f:
                code = f.read()
            success, model, method = transpile_solver(task, code)
            if success and model is not None:
                # Save the ONNX
                onnx_path = os.path.join(onnx_dir, f"task_{tid:03d}.onnx")
                with open(onnx_path, "wb") as f:
                    f.write(model.SerializeToString())
                ci = faithful_scorer.compute_cost(model)
                score = ci.get("score", 1.0)
                print(f"  [OK] task {tid}: method={method}, cost={ci.get('cost', 0)}, score={score:.2f}")
                transpiled += 1
            else:
                print(f"  [SKIP] task {tid}: {method}")
                failed += 1
        except Exception as e:
            print(f"  [ERR] task {tid}: {e}")
            failed += 1
    
    print(f"\n=== Transpilation Summary ===")
    print(f"Transpiled: {transpiled}")
    print(f"Failed: {failed}")
    print(f"Output: {onnx_dir}")


if __name__ == "__main__":
    main()
