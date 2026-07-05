"""Session 7 detectors: count objects, sort rows, draw rectangle, replace with bbox, connect pairs."""
import sys, os, json, time, zipfile, numpy as np
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import onnx
import onnx.helper as h
from onnx import TensorProto
from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model
from extended_detectors import _connected_components, _strip_metadata

def try_count_to_dimension(pairs):
    """Count objects of color X → output grid of size count×count filled with color X."""
    if not pairs: return None
    # For each color, check if output size = count of objects of that color
    for color in range(1, NUM_COLORS):
        ok = True
        for inp, out in pairs:
            components = _connected_components(inp, color)
            count = len(components)
            if count == 0:
                ok = False; break
            if out.shape != (count, count):
                ok = False; break
            # Check output is filled with `color`
            if not np.all(out == color):
                ok = False; break
        if ok:
            # Can't build static ONNX (output size depends on input)
            return None
    return None

def try_sort_rows(pairs):
    """Sort rows by their first non-zero color."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    ok = True
    for inp, out in pairs:
        # Get first non-zero color per row
        keys = []
        for r in range(inp.shape[0]):
            nonzeros = np.where(inp[r] != 0)[0]
            if len(nonzeros) > 0:
                keys.append((int(inp[r, nonzeros[0]]), r))
            else:
                keys.append((999, r))  # empty rows go last
        # Sort by key
        sorted_indices = [k[1] for k in sorted(keys, key=lambda x: x[0])]
        expected = inp[sorted_indices]
        if not np.array_equal(expected, out):
            ok = False; break
    if ok:
        # Can't build static ONNX (sorting is data-dependent)
        return None
    return None

def try_draw_rect_around_color(pairs):
    """Draw rectangle border around bounding box of each object of color X."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    for color in range(1, NUM_COLORS):
        for border_color in range(1, NUM_COLORS):
            ok = True
            for inp, out in pairs:
                expected = inp.copy()
                components = _connected_components(inp, color)
                for cells, bbox in components:
                    r0, c0, r1, c1 = bbox
                    # Draw border around bbox
                    if r0 > 0:
                        expected[r0-1, c0:c1] = border_color
                    if r1 < inp.shape[0]:
                        expected[r1, c0:c1] = border_color
                    if c0 > 0:
                        expected[r0:r1, c0-1] = border_color
                    if c1 < inp.shape[1]:
                        expected[r0:r1, c1] = border_color
                if not np.array_equal(expected, out):
                    ok = False; break
            if ok:
                # Can't build static ONNX (bbox is data-dependent)
                return None
    return None

def try_replace_with_bbox(pairs):
    """Replace each object of color X with a filled rectangle of its bounding box."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    for color in range(1, NUM_COLORS):
        ok = True
        for inp, out in pairs:
            expected = inp.copy()
            components = _connected_components(inp, color)
            for cells, bbox in components:
                r0, c0, r1, c1 = bbox
                expected[r0:r1, c0:c1] = color
            if not np.array_equal(expected, out):
                ok = False; break
        if ok:
            # Can't build static ONNX (bbox is data-dependent)
            return None
    return None

def try_connect_same_color_pairs(pairs):
    """Connect pairs of same-color cells with a line (horizontal or vertical)."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    for color in range(1, NUM_COLORS):
        for line_color in range(1, NUM_COLORS):
            # Try horizontal connections
            ok = True
            for inp, out in pairs:
                expected = inp.copy()
                # Find pairs of same-color cells in the same row
                for r in range(inp.shape[0]):
                    cells = np.where(inp[r] == color)[0]
                    if len(cells) == 2:
                        c0, c1 = cells
                        expected[r, c0:c1+1] = line_color
                if not np.array_equal(expected, out):
                    ok = False; break
            if ok:
                # Can't build static ONNX (data-dependent)
                return None
            # Try vertical connections
            ok = True
            for inp, out in pairs:
                expected = inp.copy()
                for c in range(inp.shape[1]):
                    cells = np.where(inp[:, c] == color)[0]
                    if len(cells) == 2:
                        r0, r1 = cells
                        expected[r0:r1+1, c] = line_color
                if not np.array_equal(expected, out):
                    ok = False; break
            if ok:
                return None
    return None

# Patterns that CAN be built in ONNX

def try_color_to_count_grid(pairs):
    """If output is a small grid where each cell = count of a specific color in input.
    Example: count of color 2 in input → 3 → output is 3×3 grid of 2s.
    """
    if not pairs: return None
    # Check if output is square and filled with one color
    out_h, out_w = pairs[0][1].shape
    if out_h != out_w: return None
    for inp, out in pairs:
        if out.shape != (out_h, out_w): return None
    # Check if output is all one color
    for inp, out in pairs:
        colors = np.unique(out)
        if len(colors) != 1: return None
        fill_color = int(colors[0])
    # Check if output size = count of fill_color in input (consistent across pairs)
    for inp, out in pairs:
        count = int((inp == fill_color).sum())
        if count != out_h:
            # Try: count = out_h * out_w (total cells)
            if count != out_h * out_w:
                return None
    # Can't build static ONNX (output size depends on input)
    return None

def try_zero_except_border(pairs):
    """Output = input but everything except the border is zeroed."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    ok = True
    for inp, out in pairs:
        h, w = inp.shape
        expected = np.zeros_like(inp)
        expected[0, :] = inp[0, :]
        expected[-1, :] = inp[-1, :]
        expected[:, 0] = inp[:, 0]
        expected[:, -1] = inp[:, -1]
        if not np.array_equal(expected, out):
            ok = False; break
    if ok:
        # Build ONNX: multiply input by border mask
        in_h, in_w = pairs[0][0].shape
        mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
        mask[0, 0, 0, :] = 1.0
        mask[0, 0, in_h-1, :] = 1.0
        mask[0, 0, :, 0] = 1.0
        mask[0, 0, :, in_w-1] = 1.0
        nodes = []
        initializers = [h.make_tensor("m", TensorProto.FLOAT, [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())]
        nodes.append(h.make_node("Mul", [INPUT_NAME, "m"], [OUTPUT_NAME]))
        return _make_model(nodes, initializers=initializers)
    return None

def try_extract_border(pairs):
    """Output = only the border cells of input (non-border cells set to 0)."""
    return try_zero_except_border(pairs)

def try_keep_center_block(pairs):
    """Output = center block of input (everything except border set to 0, border preserved)."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    ok = True
    for inp, out in pairs:
        h, w = inp.shape
        expected = np.zeros_like(inp)
        expected[1:-1, 1:-1] = inp[1:-1, 1:-1]
        if not np.array_equal(expected, out):
            ok = False; break
    if ok:
        in_h, in_w = pairs[0][0].shape
        mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
        mask[0, 0, 1:in_h-1, 1:in_w-1] = 1.0
        nodes = []
        initializers = [h.make_tensor("m", TensorProto.FLOAT, [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())]
        nodes.append(h.make_node("Mul", [INPUT_NAME, "m"], [OUTPUT_NAME]))
        return _make_model(nodes, initializers=initializers)
    return None

def try_remove_border(pairs):
    """Output = input with border cells set to 0."""
    if not pairs: return None
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size: return None
    
    ok = True
    for inp, out in pairs:
        h, w = inp.shape
        expected = inp.copy()
        expected[0, :] = 0
        expected[-1, :] = 0
        expected[:, 0] = 0
        expected[:, -1] = 0
        if not np.array_equal(expected, out):
            ok = False; break
    if ok:
        in_h, in_w = pairs[0][0].shape
        # mask = 1 everywhere except border
        mask = np.ones((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
        mask[0, 0, 0, :] = 0.0
        mask[0, 0, in_h-1, :] = 0.0
        mask[0, 0, :, 0] = 0.0
        mask[0, 0, :, in_w-1] = 0.0
        nodes = []
        initializers = [h.make_tensor("m", TensorProto.FLOAT, [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())]
        nodes.append(h.make_node("Mul", [INPUT_NAME, "m"], [OUTPUT_NAME]))
        return _make_model(nodes, initializers=initializers)
    return None


SESSION7_DETECTORS = [
    ("zero_except_border", try_zero_except_border),
    ("keep_center_block", try_keep_center_block),
    ("remove_border", try_remove_border),
    ("count_to_dimension", try_count_to_dimension),
    ("sort_rows", try_sort_rows),
    ("draw_rect_around_color", try_draw_rect_around_color),
    ("replace_with_bbox", try_replace_with_bbox),
    ("connect_same_color_pairs", try_connect_same_color_pairs),
    ("color_to_count_grid", try_color_to_count_grid),
]

def try_session7_detectors(task):
    pairs = arc_data.get_pairs(task)
    for name, detector in SESSION7_DETECTORS:
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
    with open("/home/z/my-project/data/final_comprehensive_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    solved = 0
    score = 0.0
    breakdown = {}
    
    with zipfile.ZipFile("/home/z/my-project/download/submission.zip", "a", zipfile.ZIP_DEFLATED) as zf:
        for tid in unsolved:
            try:
                task = arc_data.load_task(tid)
                model, method, sc = try_session7_detectors(task)
                if model is not None:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    score += sc
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={sc:.2f}")
            except Exception:
                pass
    
    print(f"\n=== Session 7 Detectors Summary ===")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")

if __name__ == "__main__":
    main()
