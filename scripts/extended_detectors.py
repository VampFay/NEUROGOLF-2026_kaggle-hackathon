"""
Extended detectors for specific patterns found in unsolved tasks:
- Quilt (mirror-tile): [[inp, flip_lr], [flip_ud, flip_both]]
- Anti-quilt: [[inp, flip_ud], [flip_lr, flip_both]]
- Kronecker diagonal: each cell → k×k block with cell value on diagonal
- Kronecker anti-diagonal: each cell → k×k block with cell value on anti-diagonal
- Border fill: set border cells to a specific color
- Grid overlay: draw grid lines at intervals
- Color count → dimension: output size depends on count of a color
"""
import sys, os, json, time, zipfile
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model
from dsl_transpiler import (
    Transpiler, py_color_map, py_crop, py_pad_to,
    py_flip_lr, py_flip_ud, py_rot180, py_transpose, py_rot90, py_rot270,
    py_scale_up, py_scale_down, py_tile, py_repeat_rows, py_repeat_cols,
)


def _strip_metadata(model):
    model.ClearField("producer_name")
    model.ClearField("producer_version")
    model.ClearField("doc_string")
    model.ClearField("domain")
    model.ClearField("model_version")
    model.graph.ClearField("doc_string")
    if len(model.graph.name) > 1:
        model.graph.name = "g"
    return model


def py_quilt(grid):
    """[[inp, flip_lr], [flip_ud, flip_both]]"""
    return np.block([[grid, np.fliplr(grid)], [np.flipud(grid), np.flip(np.flip(grid, 0), 1)]])

def py_quilt2(grid):
    """[[flip_lr, inp], [flip_both, flip_ud]]"""
    flr = np.fliplr(grid)
    fud = np.flipud(grid)
    fb = np.flip(np.flip(grid, 0), 1)
    return np.block([[flr, grid], [fb, fud]])

def py_quilt3(grid):
    """[[inp, flip_ud], [flip_lr, flip_both]] - transpose quilt"""
    return np.block([[grid, np.flipud(grid)], [np.fliplr(grid), np.flip(np.flip(grid, 0), 1)]])


def try_quilt(pairs):
    """Try quilt (mirror-tile) patterns. Handles variable input sizes."""
    if not pairs: return None
    # All outputs must be 2x the input size for each pair
    for inp, out in pairs:
        if out.shape[0] != inp.shape[0] * 2 or out.shape[1] != inp.shape[1] * 2:
            return None
    quilts = [("quilt", py_quilt), ("quilt2", py_quilt2), ("quilt3", py_quilt3)]
    for name, fn in quilts:
        if all(np.array_equal(fn(inp), out) for inp, out in pairs):
            # Use the max input size for the static ONNX model
            in_h = max(inp.shape[0] for inp, _ in pairs)
            in_w = max(inp.shape[1] for inp, _ in pairs)
            out_h = in_h * 2
            out_w = in_w * 2
            # Build ONNX
            nodes = []
            initializers = []
            # Slice input to (in_h, in_w)
            nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
            nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
            nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
            nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["base"]))
            # flip_lr
            nodes.append(h.make_node("Constant", [], ["flrs"], value=h.make_tensor("flrsv", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["flre"], value=h.make_tensor("flrev", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["flrt"], value=h.make_tensor("flrtv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["flra"], value=h.make_tensor("flrav", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["base", "flrs", "flre", "flra", "flrt"], ["flr"]))
            # flip_ud
            nodes.append(h.make_node("Constant", [], ["fuds"], value=h.make_tensor("fudsv", TensorProto.INT64, [1], [in_h-1])))
            nodes.append(h.make_node("Constant", [], ["fude"], value=h.make_tensor("fudev", TensorProto.INT64, [1], [-in_h-1])))
            nodes.append(h.make_node("Constant", [], ["fudt"], value=h.make_tensor("fudtv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fuda"], value=h.make_tensor("fudav", TensorProto.INT64, [1], [2])))
            nodes.append(h.make_node("Slice", ["base", "fuds", "fude", "fuda", "fudt"], ["fud"]))
            # flip_both
            nodes.append(h.make_node("Constant", [], ["fbs"], value=h.make_tensor("fbsv", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["fbe"], value=h.make_tensor("fbev", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["fbt"], value=h.make_tensor("fbtv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fba"], value=h.make_tensor("fbav", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["fud", "fbs", "fbe", "fba", "fbt"], ["fb"]))
            # Concat
            if name == "quilt":
                nodes.append(h.make_node("Concat", ["base", "flr"], ["top"], axis=3))
                nodes.append(h.make_node("Concat", ["fud", "fb"], ["bot"], axis=3))
                nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
            elif name == "quilt2":
                nodes.append(h.make_node("Concat", ["flr", "base"], ["top"], axis=3))
                nodes.append(h.make_node("Concat", ["fb", "fud"], ["bot"], axis=3))
                nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
            elif name == "quilt3":
                nodes.append(h.make_node("Concat", ["base", "fud"], ["top"], axis=3))
                nodes.append(h.make_node("Concat", ["flr", "fb"], ["bot"], axis=3))
                nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
            # Pad to MAX_GRID
            pad_b = MAX_GRID - out_h
            pad_r = MAX_GRID - out_w
            if pad_b == 0 and pad_r == 0:
                nodes.append(h.make_node("Identity", ["conc"], [OUTPUT_NAME]))
            else:
                pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
                nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
                nodes.append(h.make_node("Constant", [], ["pv2"], value=h.make_tensor("pv2v", TensorProto.FLOAT, [1], [0.0])))
                nodes.append(h.make_node("Pad", ["conc", "pv", "pv2"], [OUTPUT_NAME], mode="constant"))
            return _make_model(nodes, initializers=initializers)
    return None


def try_quilt_3x3(pairs):
    """Try 3x3 quilt patterns (9 cells, each a flip/rotation of base)."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h != in_h * 3 or out_w != in_w * 3: return None
    # Common 3x3 quilts: all 8 dihedral + center
    # This is complex — skip for now
    return None


def try_kronecker_diagonal(pairs):
    """Each cell c → k×k block with c on main diagonal."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 2 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w) or out.shape != (out_h, out_w): return None
        for r in range(in_h):
            for c in range(in_w):
                val = int(inp[r, c])
                block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                expected = np.zeros((k, k), dtype=np.int64)
                for i in range(k):
                    expected[i, i] = val
                if not np.array_equal(block, expected): return None
    # Build: Resize (nearest) then Mul with diagonal mask
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["up"],
        mode="nearest", nearest_mode="round_prefer_floor",
        coordinate_transformation_mode="asymmetric"))
    mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(MAX_GRID):
        for c in range(MAX_GRID):
            if r % k == c % k:
                mask[0, 0, r, c] = 1.0
    nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
        [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
    nodes.append(h.make_node("Mul", ["up", "m"], [OUTPUT_NAME]))
    return _make_model(nodes)


def try_kronecker_anti_diagonal(pairs):
    """Each cell c → k×k block with c on anti-diagonal."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 2 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w) or out.shape != (out_h, out_w): return None
        for r in range(in_h):
            for c in range(in_w):
                val = int(inp[r, c])
                block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                expected = np.zeros((k, k), dtype=np.int64)
                for i in range(k):
                    expected[i, k-1-i] = val
                if not np.array_equal(block, expected): return None
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["up"],
        mode="nearest", nearest_mode="round_prefer_floor",
        coordinate_transformation_mode="asymmetric"))
    mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(MAX_GRID):
        for c in range(MAX_GRID):
            if (r % k) + (c % k) == k - 1:
                mask[0, 0, r, c] = 1.0
    nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
        [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
    nodes.append(h.make_node("Mul", ["up", "m"], [OUTPUT_NAME]))
    return _make_model(nodes)


def try_kronecker_border(pairs):
    """Each cell c → k×k block with c on border, 0 inside."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 3 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w) or out.shape != (out_h, out_w): return None
        for r in range(in_h):
            for c in range(in_w):
                val = int(inp[r, c])
                block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                expected = np.zeros((k, k), dtype=np.int64)
                expected[0, :] = val
                expected[-1, :] = val
                expected[:, 0] = val
                expected[:, -1] = val
                if not np.array_equal(block, expected): return None
    # Build: Resize then Mul with border mask
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["up"],
        mode="nearest", nearest_mode="round_prefer_floor",
        coordinate_transformation_mode="asymmetric"))
    mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(MAX_GRID):
        for c in range(MAX_GRID):
            ri, ci = r % k, c % k
            if ri == 0 or ri == k-1 or ci == 0 or ci == k-1:
                mask[0, 0, r, c] = 1.0
    nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
        [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
    nodes.append(h.make_node("Mul", ["up", "m"], [OUTPUT_NAME]))
    return _make_model(nodes)


def try_color_map_then_quilt(pairs):
    """Color map then quilt."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h != in_h * 2 or out_w != in_w * 2: return None
    # Derive color map by reverse-quilting
    quilts = [("quilt", py_quilt), ("quilt2", py_quilt2), ("quilt3", py_quilt3)]
    for qname, qfn in quilts:
        mapping = {}
        ok = True
        for inp, out in pairs:
            # Reverse: top-left quadrant of out should be color-mapped input
            sub = out[:in_h, :in_w]
            for c in range(NUM_COLORS):
                in_cells = (inp == c)
                if not in_cells.any(): continue
                out_at = sub[in_cells]
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
        # Verify: color_map then quilt
        valid = True
        for inp, out in pairs:
            mapped = py_color_map(inp, mapping)
            quilted = qfn(mapped)
            if not np.array_equal(quilted, out):
                valid = False; break
        if not valid: continue
        # Build ONNX
        full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
        W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
        for frm, to in full_map.items():
            W[to, frm, 0, 0] = 1.0
        nodes = []
        initializers = [h.make_tensor("w", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist())]
        # color_map
        nodes.append(h.make_node("Conv", [INPUT_NAME, "w"], ["cm"],
            pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
        # crop to (in_h, in_w)
        nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
        nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Slice", ["cm", "cs", "ce", "ca"], ["base"]))
        # flip_lr
        nodes.append(h.make_node("Constant", [], ["flrs"], value=h.make_tensor("flrsv", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["flre"], value=h.make_tensor("flrev", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["flrt"], value=h.make_tensor("flrtv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["flra"], value=h.make_tensor("flrav", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["base", "flrs", "flre", "flra", "flrt"], ["flr"]))
        # flip_ud
        nodes.append(h.make_node("Constant", [], ["fuds"], value=h.make_tensor("fudsv", TensorProto.INT64, [1], [in_h-1])))
        nodes.append(h.make_node("Constant", [], ["fude"], value=h.make_tensor("fudev", TensorProto.INT64, [1], [-in_h-1])))
        nodes.append(h.make_node("Constant", [], ["fudt"], value=h.make_tensor("fudtv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fuda"], value=h.make_tensor("fudav", TensorProto.INT64, [1], [2])))
        nodes.append(h.make_node("Slice", ["base", "fuds", "fude", "fuda", "fudt"], ["fud"]))
        # flip_both
        nodes.append(h.make_node("Constant", [], ["fbs"], value=h.make_tensor("fbsv", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["fbe"], value=h.make_tensor("fbev", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["fbt"], value=h.make_tensor("fbtv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fba"], value=h.make_tensor("fbav", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["fud", "fbs", "fbe", "fba", "fbt"], ["fb"]))
        # quilt concat
        if qname == "quilt":
            nodes.append(h.make_node("Concat", ["base", "flr"], ["top"], axis=3))
            nodes.append(h.make_node("Concat", ["fud", "fb"], ["bot"], axis=3))
            nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
        elif qname == "quilt2":
            nodes.append(h.make_node("Concat", ["flr", "base"], ["top"], axis=3))
            nodes.append(h.make_node("Concat", ["fb", "fud"], ["bot"], axis=3))
            nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
        elif qname == "quilt3":
            nodes.append(h.make_node("Concat", ["base", "fud"], ["top"], axis=3))
            nodes.append(h.make_node("Concat", ["flr", "fb"], ["bot"], axis=3))
            nodes.append(h.make_node("Concat", ["top", "bot"], ["conc"], axis=2))
        # Pad
        pad_b = MAX_GRID - out_h
        pad_r = MAX_GRID - out_w
        if pad_b == 0 and pad_r == 0:
            nodes.append(h.make_node("Identity", ["conc"], [OUTPUT_NAME]))
        else:
            pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
            nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
            nodes.append(h.make_node("Constant", [], ["pv2"], value=h.make_tensor("pv2v", TensorProto.FLOAT, [1], [0.0])))
            nodes.append(h.make_node("Pad", ["conc", "pv", "pv2"], [OUTPUT_NAME], mode="constant"))
        return _make_model(nodes, initializers=initializers)
    return None


EXTENDED_DETECTORS = [
    ("quilt", try_quilt),
    ("kronecker_diagonal", try_kronecker_diagonal),
    ("kronecker_anti_diagonal", try_kronecker_anti_diagonal),
    ("kronecker_border", try_kronecker_border),
    ("color_map_then_quilt", try_color_map_then_quilt),
]


def try_extended_detectors(task):
    """Try all extended detectors."""
    pairs = arc_data.get_pairs(task)
    for name, detector in EXTENDED_DETECTORS:
        try:
            model = detector(pairs)
        except Exception:
            continue
        if model is None: continue
        e = validator.evaluate_model(model, task)
        if e["eligible_for_points"]:
            model = _strip_metadata(model)
            e2 = validator.evaluate_model(model, task)
            if e2["eligible_for_points"]:
                return model, name, e2["score"]
    return None, None, 0


def main():
    with open("/home/z/my-project/data/final_unified_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    solved = 0
    score = 0.0
    breakdown = {}
    
    output_path = "/home/z/my-project/download/submission.zip"
    t0 = time.time()
    
    with zipfile.ZipFile(output_path, "a", zipfile.ZIP_DEFLATED) as zf:
        for tid in unsolved:
            try:
                task = arc_data.load_task(tid)
                model, method, sc = try_extended_detectors(task)
                if model is not None:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    score += sc
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={sc:.2f}")
            except Exception:
                pass
    
    elapsed = time.time() - t0
    print(f"\n=== Extended Detectors Summary ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")


if __name__ == "__main__":
    main()
