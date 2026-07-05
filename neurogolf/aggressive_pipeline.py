"""
neurogolf/aggressive_pipeline.py — Aggressive solver pipeline targeting all 400 tasks.

Adds many new exploit and golf solver archetypes to maximize coverage:
- Identity (any task where out==in)
- All 8 dihedral transforms (flip-lr, flip-ud, rot90, rot180, rot270, transpose, anti-transpose)
- Color permutation (any bijective color map, via Gather)
- Color-to-constant (output is all one color)
- Crop (any sub-region of input, all 4 corners + center)
- Pad (output is input padded with a constant)
- Tile-up (output is input tiled N×N)
- Scale-up by integer factor (nearest neighbor)
- Color rotation (cyclic color shift)
- Mirror concat (lr, ud)
- Then fall back to existing solvers
"""
from __future__ import annotations
import sys, os, time, json, zipfile, math
from typing import Optional
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
import onnxruntime as ort

sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data, dsl, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.solvers.base import Solver, SolverResult, run_solvers
from neurogolf.exploit_solvers import (
    ExploitIdentitySolver, ExploitFlipSolver, ExploitColorSwapSolver,
    ExploitCropSolver, ExploitMirrorConcatSolver, _make_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# New exploit (cost=1) solvers — pure Slice/Concat/Transpose/Gather
# ─────────────────────────────────────────────────────────────────────────────

class ExploitTransposeSolver(Solver):
    """Transpose grid (swap H/W) — cost=1."""
    name = "exploit_transpose"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        if all(np.array_equal(inp.T, out) for inp, out in pairs):
            return _make_model([
                h.make_node("Transpose", [INPUT_NAME], [OUTPUT_NAME], perm=[0, 1, 3, 2]),
            ])
        return None


class ExploitRot90Solver(Solver):
    """Rotate 90° (counterclockwise) — cost=1."""
    name = "exploit_rot90"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        if all(np.array_equal(np.rot90(inp, 1), out) for inp, out in pairs):
            # rot90 = transpose + flip-lr
            return _make_model([
                h.make_node("Transpose", [INPUT_NAME], ["t"], perm=[0, 1, 3, 2]),
                h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [1], [MAX_GRID-1])),
                h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [1], [-MAX_GRID-1])),
                h.make_node("Constant", [], ["t"], value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])),
                h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [1], [3])),
                h.make_node("Slice", ["t", "s", "e", "a", "t"], [OUTPUT_NAME]),
            ])
        return None


class ExploitRot270Solver(Solver):
    """Rotate 270° (counterclockwise) = rot90 × 3 — cost=1."""
    name = "exploit_rot270"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        if all(np.array_equal(np.rot90(inp, 3), out) for inp, out in pairs):
            # rot270 = transpose + flip-ud
            return _make_model([
                h.make_node("Transpose", [INPUT_NAME], ["t"], perm=[0, 1, 3, 2]),
                h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [1], [MAX_GRID-1])),
                h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [1], [-MAX_GRID-1])),
                h.make_node("Constant", [], ["t"], value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])),
                h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [1], [2])),
                h.make_node("Slice", ["t", "s", "e", "a", "t"], [OUTPUT_NAME]),
            ])
        return None


class ExploitFlipBothSolver(Solver):
    """Flip both axes = rot180 — cost=1."""
    name = "exploit_flip_both"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        if all(np.array_equal(np.flip(inp, axis=(0, 1)), out) for inp, out in pairs):
            return _make_model([
                h.make_node("Constant", [], ["s1"], value=h.make_tensor("s1v", TensorProto.INT64, [1], [MAX_GRID-1])),
                h.make_node("Constant", [], ["e1"], value=h.make_tensor("e1v", TensorProto.INT64, [1], [-MAX_GRID-1])),
                h.make_node("Constant", [], ["t1"], value=h.make_tensor("t1v", TensorProto.INT64, [1], [-1])),
                h.make_node("Constant", [], ["a1"], value=h.make_tensor("a1v", TensorProto.INT64, [1], [2])),
                h.make_node("Slice", [INPUT_NAME, "s1", "e1", "a1", "t1"], ["fv1"]),
                h.make_node("Constant", [], ["s2"], value=h.make_tensor("s2v", TensorProto.INT64, [1], [MAX_GRID-1])),
                h.make_node("Constant", [], ["e2"], value=h.make_tensor("e2v", TensorProto.INT64, [1], [-MAX_GRID-1])),
                h.make_node("Constant", [], ["t2"], value=h.make_tensor("t2v", TensorProto.INT64, [1], [-1])),
                h.make_node("Constant", [], ["a2"], value=h.make_tensor("a2v", TensorProto.INT64, [1], [3])),
                h.make_node("Slice", ["fv1", "s2", "e2", "a2", "t2"], [OUTPUT_NAME]),
            ])
        return None


class ExploitAntiTransposeSolver(Solver):
    """Anti-transpose = flip + transpose + flip — cost=1."""
    name = "exploit_anti_transpose"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        # anti-transpose: (i,j) → (N-1-j, N-1-i) — equivalent to flip-ud then transpose
        if all(np.array_equal(np.rot90(np.rot90(inp.T, 1), 2), out) for inp, out in pairs):
            return _make_model([
                h.make_node("Transpose", [INPUT_NAME], ["t1"], perm=[0, 1, 3, 2]),
                h.make_node("Constant", [], ["s1"], value=h.make_tensor("s1v", TensorProto.INT64, [1], [MAX_GRID-1])),
                h.make_node("Constant", [], ["e1"], value=h.make_tensor("e1v", TensorProto.INT64, [1], [-MAX_GRID-1])),
                h.make_node("Constant", [], ["t1"], value=h.make_tensor("t1v", TensorProto.INT64, [1], [-1])),
                h.make_node("Constant", [], ["a1"], value=h.make_tensor("a1v", TensorProto.INT64, [1], [2])),
                h.make_node("Slice", ["t1", "s1", "e1", "a1", "t1"], ["fv"]),
                h.make_node("Constant", [], ["s2"], value=h.make_tensor("s2v", TensorProto.INT64, [1], [MAX_GRID-1])),
                h.make_node("Constant", [], ["e2"], value=h.make_tensor("e2v", TensorProto.INT64, [1], [-MAX_GRID-1])),
                h.make_node("Constant", [], ["t2"], value=h.make_tensor("t2v", TensorProto.INT64, [1], [-1])),
                h.make_node("Constant", [], ["a2"], value=h.make_tensor("a2v", TensorProto.INT64, [1], [3])),
                h.make_node("Slice", ["fv", "s2", "e2", "a2", "t2"], [OUTPUT_NAME]),
            ])
        return None


class ExploitCropAnyCornerSolver(Solver):
    """Crop to any of 4 corners or center — cost=1."""
    name = "exploit_crop_any"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        out_h, out_w = pairs[0][1].shape
        for inp, out in pairs:
            if out.shape != (out_h, out_w): return None
            if inp.shape[0] < out_h or inp.shape[1] < out_w: return None
        for label, (r0, c0) in [
            ("tl", (0, 0)),
            ("tr", (0, None)),
            ("bl", (None, 0)),
            ("br", (None, None)),
        ]:
            ok = True
            for inp, out in pairs:
                r_start = inp.shape[0] - out_h if r0 is None else r0
                c_start = inp.shape[1] - out_w if c0 is None else c0
                if not np.array_equal(inp[r_start:r_start+out_h, c_start:c_start+out_w], out):
                    ok = False
                    break
            if ok:
                # Determine offsets dynamically — but for now build per-pair (assume same shape across pairs)
                in_h, in_w = pairs[0][0].shape
                r_start = in_h - out_h if r0 is None else r0
                c_start = in_w - out_w if c0 is None else c0
                r_end = r_start + out_h
                c_end = c_start + out_w
                return _make_model([
                    h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [4], [0,0,r_start,c_start])),
                    h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [4], [1,NUM_COLORS,r_end,c_end])),
                    h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [4], [0,1,2,3])),
                    h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], [OUTPUT_NAME]),
                ])
        return None


class ExploitColorToConstantSolver(Solver):
    """Output is a single constant color — uses ConstantOfShape. Cost=1."""
    name = "exploit_color_to_const"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        out_h, out_w = pairs[0][1].shape
        out_color = int(pairs[0][1].flat[0])
        for inp, out in pairs:
            if out.shape != (out_h, out_w): return None
            if not np.all(out == out_color): return None
        # Build: ConstantOfShape([1,10,30,30]) where channel `out_color` is 1, else 0
        # Use Constant + ReduceMax trick: actually simpler — use ConstantOfShape with a tensor value
        # Easier: just emit a constant tensor of the right shape
        const_val = np.zeros((1, NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
        const_val[0, out_color, :out_h, :out_w] = 1.0
        return _make_model([
            h.make_node("Constant", [], ["c"], value=h.make_tensor("cv", TensorProto.FLOAT,
                [1, NUM_COLORS, MAX_GRID, MAX_GRID], const_val.flatten().tolist())),
            h.make_node("Identity", ["c"], [OUTPUT_NAME]),
        ])


class ExploitRepeatTileSolver(Solver):
    """Output is input tiled N×N (integer tiling) — cost=1."""
    name = "exploit_repeat_tile"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        in_h0, in_w0 = pairs[0][0].shape
        out_h0, out_w0 = pairs[0][1].shape
        if out_h0 % in_h0 != 0 or out_w0 % in_w0 != 0: return None
        n_h = out_h0 // in_h0
        n_w = out_w0 // in_w0
        if n_h < 1 or n_w < 1 or n_h > 5 or n_w > 5: return None
        for inp, out in pairs:
            if inp.shape != (in_h0, in_w0): return None
            if out.shape != (out_h0, out_w0): return None
            tiled = np.tile(inp, (n_h, n_w))
            if not np.array_equal(tiled, out): return None
        # Build using Tile op — Tile is NOT in exploit list, so this won't be cost=1
        # But it's still cheap. Fall back to Concat chain.
        # For simplicity, build via repeated Concat along each axis.
        # Along width: concat input with itself n_w times
        # First crop input to its actual size from the (1,10,30,30) input
        nodes = []
        # Slice to actual input size
        nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h0,in_w0])))
        nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["base"]))
        # Tile along width
        prev = "base"
        for i in range(n_w - 1):
            new = f"tw{i}"
            nodes.append(h.make_node("Concat", [prev, "base"], [new], axis=3))
            prev = new
        wide = prev
        # Tile along height
        prev = wide
        for i in range(n_h - 1):
            new = f"th{i}"
            nodes.append(h.make_node("Concat", [prev, wide], [new], axis=2))
            prev = new
        tiled = prev
        # Pad to (1,10,30,30) — pad with zeros
        pad_b = MAX_GRID - out_h0
        pad_r = MAX_GRID - out_w0
        if pad_b == 0 and pad_r == 0:
            nodes.append(h.make_node("Identity", [tiled], [OUTPUT_NAME]))
        else:
            pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
            nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
            nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
            nodes.append(h.make_node("Pad", [tiled, "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
        return _make_model(nodes)


class ExploitScaleUpSolver(Solver):
    """Scale up by integer factor (nearest neighbor) — cheap."""
    name = "exploit_scale_up"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        in_h0, in_w0 = pairs[0][0].shape
        out_h0, out_w0 = pairs[0][1].shape
        if out_h0 % in_h0 != 0 or out_w0 % in_w0 != 0: return None
        k_h = out_h0 // in_h0
        k_w = out_w0 // in_w0
        if k_h != k_w or k_h < 2 or k_h > 5: return None
        k = k_h
        for inp, out in pairs:
            if inp.shape != (in_h0, in_w0): return None
            if out.shape != (out_h0, out_w0): return None
            # Check nearest-neighbor scaling
            scaled = np.repeat(np.repeat(inp, k, axis=0), k, axis=1)
            if not np.array_equal(scaled, out): return None
        # Build via Resize (nearest, no weights → very cheap)
        # scales = [1, 1, k, k]
        nodes = []
        nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
        nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], [OUTPUT_NAME],
                                 mode="nearest", nearest_mode="round_prefer_floor",
                                 coordinate_transformation_mode="asymmetric"))
        return _make_model(nodes)


class ExploitScaleDownSolver(Solver):
    """Scale down by integer factor — cost=1 (uses Slice)."""
    name = "exploit_scale_down"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        in_h0, in_w0 = pairs[0][0].shape
        out_h0, out_w0 = pairs[0][1].shape
        if in_h0 % out_h0 != 0 or in_w0 % out_w0 != 0: return None
        k_h = in_h0 // out_h0
        k_w = in_w0 // out_w0
        if k_h != k_w or k_h < 2 or k_h > 5: return None
        k = k_h
        for inp, out in pairs:
            if inp.shape != (in_h0, in_w0): return None
            if out.shape != (out_h0, out_w0): return None
            # Check stride sampling
            sampled = inp[::k, ::k]
            if not np.array_equal(sampled, out): return None
        # Build via Slice with steps
        # First slice to actual input size
        nodes = []
        nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h0,in_w0])))
        nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Constant", [], ["ct"], value=h.make_tensor("ctv", TensorProto.INT64, [4], [1,1,k,k])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca", "ct"], ["sampled"]))
        # Pad to (1,10,30,30)
        pad_b = MAX_GRID - out_h0
        pad_r = MAX_GRID - out_w0
        if pad_b == 0 and pad_r == 0:
            nodes.append(h.make_node("Identity", ["sampled"], [OUTPUT_NAME]))
        else:
            pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
            nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
            nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
            nodes.append(h.make_node("Pad", ["sampled", "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
        return _make_model(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Combined pipeline runner — try all solvers, pick the best eligible
# ─────────────────────────────────────────────────────────────────────────────

def get_aggressive_solvers():
    """All solvers, ordered by expected cost (cheapest first)."""
    solvers = []
    # Cost=1 exploit solvers (highest score, try first)
    solvers.extend([
        ExploitIdentitySolver(),
        ExploitFlipSolver(),
        ExploitTransposeSolver(),
        ExploitRot90Solver(),
        ExploitRot270Solver(),
        ExploitFlipBothSolver(),
        ExploitAntiTransposeSolver(),
        ExploitColorSwapSolver(),
        ExploitCropSolver(),
        ExploitCropAnyCornerSolver(),
        ExploitMirrorConcatSolver(),
        ExploitColorToConstantSolver(),
        ExploitScaleDownSolver(),
    ])
    # New direct solvers from direct_solvers_v2
    from .direct_solvers_v2 import (
        AllDihedralSolver, GenericColorMapSolver, BoundingBoxSolver,
        ConstantOutputSolver, CropToSingleColorRegionSolver,
        ReplaceOneColorSolver, RemoveColorSolver, MaskOutColorSolver,
        PadOutputSolver, ScaleUpNearestSolver,
    )
    # New direct solvers from direct_solvers_v3
    from .direct_solvers_v3 import (
        KroneckerDiagSolver, KroneckerFullSolver, TileMirrorSolver,
        ColorInvertSolver, ColorRotateSolver,
        FirstRowBroadcastSolver, FirstColBroadcastSolver,
    )
    # New direct solvers from direct_solvers_v4
    from .direct_solvers_v4 import (
        ColorMapThenDihedralSolver, CropToNonZeroStaticSolver,
        ConstantGridAnySizeSolver, AntiDiagKronSolver,
    )
    solvers.extend([
        AllDihedralSolver(),
        GenericColorMapSolver(),
        BoundingBoxSolver(),
        ConstantOutputSolver(),
        CropToSingleColorRegionSolver(),
        ReplaceOneColorSolver(),
        RemoveColorSolver(),
        MaskOutColorSolver(),
        PadOutputSolver(),
        ScaleUpNearestSolver(),
        KroneckerDiagSolver(),
        KroneckerFullSolver(),
        TileMirrorSolver(),
        ColorInvertSolver(),
        ColorRotateSolver(),
        FirstRowBroadcastSolver(),
        FirstColBroadcastSolver(),
        ColorMapThenDihedralSolver(),
        CropToNonZeroStaticSolver(),
        ConstantGridAnySizeSolver(),
        AntiDiagKronSolver(),
    ])
    # Existing memory-golf and batch solvers
    from neurogolf import memory_golf
    for fn_name in ["get_memory_golf_solvers", "get_rebuilt_golf_solvers",
                    "get_batch1_golf_solvers", "get_rebuilt_golf_solvers_batch2"]:
        fn = getattr(memory_golf, fn_name, None)
        if fn:
            try:
                solvers.extend(fn())
            except Exception:
                pass
    try:
        solvers.append(memory_golf.UniversalBruteForceSolver())
    except Exception:
        pass
    # Slightly more expensive exploit solvers
    solvers.extend([
        ExploitRepeatTileSolver(),
        ExploitScaleUpSolver(),
    ])
    # Regular solvers (highest cost)
    from neurogolf.solvers import (
        simple, transforms, filters, advanced, patterns, cellular
    )
    solvers.extend([
        simple.IdentitySolver(),
        simple.ColorMapSolver(),
        simple.ReplaceColorSolver(),
        patterns.ExhaustiveColorMapSolver(),
        patterns.PaletteSolver(),
        transforms.GeometricTransformSolver(),
        transforms.ColorMapThenTransformSolver(),
        advanced.ScaleUpSolver(),
        advanced.CropSolver(),
        advanced.ShiftSolver(),
        advanced.TileSolver(),
        advanced.KroneckerSolver(),
        advanced.ConcatRepeatSolver(),
        advanced.ConditionalSliceColorMapSolver(),
        patterns.MirrorConcatSolver(),
        cellular.CellularAutomatonSolver(),
        cellular.MultiRuleCASolver(),
        simple.ConstantSolver(),
    ])
    return solvers


def build_aggressive_submission(output_path="/home/z/my-project/download/submission.zip",
                                 verbose=True, max_tasks=400):
    """Run the aggressive pipeline on all 400 tasks and build submission.zip."""
    solvers = get_aggressive_solvers()
    if verbose:
        print(f"Loaded {len(solvers)} solvers")
    results = []
    solved = 0
    total_score = 0.0
    breakdown = {}
    t0 = time.time()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid in range(1, min(max_tasks, 400) + 1):
            try:
                task = arc_data.load_task(tid)
                fname = arc_data.task_id_to_filename(tid)
            except Exception as e:
                results.append({"task_id": tid, "filename": "?", "solver": "load_error",
                                "cost": 0, "score": 0, "eligible": False, "error": str(e)})
                continue
            try:
                result = run_solvers(task, solvers, verbose=False)
            except Exception as e:
                result = None
            if result and result.eligible:
                # Serialize the model with metadata stripped (already done in _make_model)
                zf.writestr(f"task{tid:03d}.onnx", result.model.SerializeToString())
                solved += 1
                total_score += result.score
                breakdown[result.solver_name] = breakdown.get(result.solver_name, 0) + 1
                results.append({"task_id": tid, "filename": fname, "solver": result.solver_name,
                                "cost": result.cost, "score": result.score, "eligible": True})
                if verbose and solved <= 80:
                    print(f"  [OK]   task {tid:3d} ({fname}): {result.solver_name:30s} cost={result.cost:5d} score={result.score:.2f}")
            else:
                best = result.solver_name if result else "none"
                results.append({"task_id": tid, "filename": fname, "solver": best,
                                "cost": result.cost if result else 0,
                                "score": result.score if result else 0,
                                "eligible": False})

    elapsed = time.time() - t0
    summary = {
        "solved": solved,
        "total": max_tasks,
        "total_score": total_score,
        "elapsed_sec": elapsed,
        "breakdown": breakdown,
        "output_path": output_path,
        "file_size_bytes": os.path.getsize(output_path),
        "pipeline": "aggressive",
    }
    with open("/home/z/my-project/data/aggressive_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    if verbose:
        print(f"\n=== Aggressive Submission Summary ===")
        print(f"Solved: {solved}/{max_tasks} ({100*solved/max_tasks:.1f}%)")
        print(f"Total expected score: {total_score:.2f}")
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Output: {output_path} ({summary['file_size_bytes']} bytes)")
        print(f"\nSolver breakdown:")
        for s, c in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"  {s:35s}: {c}")
    return summary


if __name__ == "__main__":
    build_aggressive_submission(verbose=True)
