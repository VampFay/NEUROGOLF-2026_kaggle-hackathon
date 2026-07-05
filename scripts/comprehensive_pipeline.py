"""
Comprehensive pattern detection for ARC-AGI tasks.

Try many pattern types in order of decreasing simplicity:
1. Color permutation (any bijection including rotations/inversions)
2. Geometric transform + color permutation
3. Crop to fixed region
4. Scale up/down by integer
5. Pad output to larger size
6. Constant output
7. Tile/quilt
8. Color isolation (keep only color X, zero others)
9. Color removal (set color X to 0)
10. Color replacement (set color X to Y)
11. Cellular automaton (3x3 conv)
12. Bounding box extraction
13. Object detection (largest connected component)
"""
import sys, os, json, time
sys.path.insert(0, "/home/z/my-project")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
from collections import deque, Counter

from neurogolf import arc_data, validator, faithful_scorer, dsl
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model


# ─────────────────────────────────────────────────────────────────────────────
# Pattern detectors (return ONNX model or None)
# ─────────────────────────────────────────────────────────────────────────────

def detect_color_permutation(pairs):
    """Pure color permutation (bijective)."""
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
    # Bijective → Gather exploit (cost=1, score=25)
    if set(mapping.keys()) == set(mapping.values()):
        indices = list(range(NUM_COLORS))
        for source, target in mapping.items():
            indices[target] = source
        return _make_model([
            h.make_node("Constant", [], ["i"], value=h.make_tensor("iv", TensorProto.INT64, [NUM_COLORS], indices)),
            h.make_node("Gather", [INPUT_NAME, "i"], [OUTPUT_NAME], axis=1),
        ])
    # Non-bijective → 1x1 conv
    return dsl.color_map(mapping)


def detect_dihedral_with_colormap(pairs):
    """Try all 8 dihedral transforms × color permutations."""
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
        # Build ONNX via ColorMapThenDihedralSolver
        from neurogolf.direct_solvers_v4 import ColorMapThenDihedralSolver
        # Need a task to use the solver — but solver.attempt(task) re-runs detection
        # Use the static helper instead
        return _build_colormap_then_dihedral(in_h0, in_w0, tname, mapping)
    return None


def _build_colormap_then_dihedral(in_h, in_w, tname, mapping):
    """Build ONNX for: color_map → dihedral transform → pad back."""
    full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
    W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
    for frm, to in full_map.items():
        W[to, frm, 0, 0] = 1.0
    nodes = []
    nodes.append(h.make_node("Conv", [INPUT_NAME, "w"], ["cm"],
        pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
    nodes.append(h.make_node("Constant", [], ["cs"],
        value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
    nodes.append(h.make_node("Constant", [], ["ce"],
        value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
    nodes.append(h.make_node("Constant", [], ["ca"],
        value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
    nodes.append(h.make_node("Slice", ["cm", "cs", "ce", "ca"], ["cropped"]))
    out_h, out_w = in_h, in_w
    if tname == "identity":
        nodes.append(h.make_node("Identity", ["cropped"], ["transformed"]))
    elif tname == "flip_lr":
        nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["ft"], value=h.make_tensor("ftv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("fav", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["cropped", "fs", "fe", "fa", "ft"], ["transformed"]))
    elif tname == "flip_ud":
        nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_h-1])))
        nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_h-1])))
        nodes.append(h.make_node("Constant", [], ["ft"], value=h.make_tensor("ftv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("fav", TensorProto.INT64, [1], [2])))
        nodes.append(h.make_node("Slice", ["cropped", "fs", "fe", "fa", "ft"], ["transformed"]))
    elif tname == "transpose":
        nodes.append(h.make_node("Transpose", ["cropped"], ["transformed"], perm=[0, 1, 3, 2]))
        out_h, out_w = in_w, in_h
    elif tname == "rot180":
        nodes.append(h.make_node("Constant", [], ["fs1"], value=h.make_tensor("fs1v", TensorProto.INT64, [1], [in_h-1])))
        nodes.append(h.make_node("Constant", [], ["fe1"], value=h.make_tensor("fe1v", TensorProto.INT64, [1], [-in_h-1])))
        nodes.append(h.make_node("Constant", [], ["ft1"], value=h.make_tensor("ft1v", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fa1"], value=h.make_tensor("fa1v", TensorProto.INT64, [1], [2])))
        nodes.append(h.make_node("Slice", ["cropped", "fs1", "fe1", "fa1", "ft1"], ["fv1"]))
        nodes.append(h.make_node("Constant", [], ["fs2"], value=h.make_tensor("fs2v", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["fe2"], value=h.make_tensor("fe2v", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["ft2"], value=h.make_tensor("ft2v", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fa2"], value=h.make_tensor("fa2v", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["fv1", "fs2", "fe2", "fa2", "ft2"], ["transformed"]))
    elif tname == "rot90":
        nodes.append(h.make_node("Transpose", ["cropped"], ["t"], perm=[0, 1, 3, 2]))
        nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["ft2"], value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("av", TensorProto.INT64, [1], [2])))
        nodes.append(h.make_node("Slice", ["t", "fs", "fe", "fa", "ft2"], ["transformed"]))
        out_h, out_w = in_w, in_h
    elif tname == "rot270":
        nodes.append(h.make_node("Transpose", ["cropped"], ["t"], perm=[0, 1, 3, 2]))
        nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_h-1])))
        nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_h-1])))
        nodes.append(h.make_node("Constant", [], ["ft2"], value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("av", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["t", "fs", "fe", "fa", "ft2"], ["transformed"]))
        out_h, out_w = in_w, in_h
    elif tname == "anti_transpose":
        nodes.append(h.make_node("Transpose", ["cropped"], ["t1"], perm=[0, 1, 3, 2]))
        nodes.append(h.make_node("Constant", [], ["s1"], value=h.make_tensor("s1v", TensorProto.INT64, [1], [in_w-1])))
        nodes.append(h.make_node("Constant", [], ["e1"], value=h.make_tensor("e1v", TensorProto.INT64, [1], [-in_w-1])))
        nodes.append(h.make_node("Constant", [], ["t1v2"], value=h.make_tensor("t1vv", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["a1"], value=h.make_tensor("a1v", TensorProto.INT64, [1], [2])))
        nodes.append(h.make_node("Slice", ["t1", "s1", "e1", "a1", "t1v2"], ["fv"]))
        nodes.append(h.make_node("Constant", [], ["s2"], value=h.make_tensor("s2v", TensorProto.INT64, [1], [in_h-1])))
        nodes.append(h.make_node("Constant", [], ["e2"], value=h.make_tensor("e2v", TensorProto.INT64, [1], [-in_h-1])))
        nodes.append(h.make_node("Constant", [], ["t2"], value=h.make_tensor("t2v", TensorProto.INT64, [1], [-1])))
        nodes.append(h.make_node("Constant", [], ["a2"], value=h.make_tensor("a2v", TensorProto.INT64, [1], [3])))
        nodes.append(h.make_node("Slice", ["fv", "s2", "e2", "a2", "t2"], ["transformed"]))
        out_h, out_w = in_w, in_h
    # Pad back
    pad_b = MAX_GRID - out_h
    pad_r = MAX_GRID - out_w
    if pad_b == 0 and pad_r == 0:
        nodes.append(h.make_node("Identity", ["transformed"], [OUTPUT_NAME]))
    else:
        pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
        nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
        nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
        nodes.append(h.make_node("Pad", ["transformed", "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
    initializers = [h.make_tensor("w", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist())]
    return _make_model(nodes, initializers=initializers)


def detect_constant_output(pairs):
    """Output is the same constant grid for all pairs."""
    if len(pairs) < 1: return None
    first_out = pairs[0][1]
    for inp, out in pairs:
        if out.shape != first_out.shape: return None
        if not np.array_equal(out, first_out): return None
    out_h, out_w = first_out.shape
    const_val = np.zeros((1, NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(out_h):
        for c in range(out_w):
            color = int(first_out[r, c])
            const_val[0, color, r, c] = 1.0
    return _make_model([
        h.make_node("Constant", [], ["c"], value=h.make_tensor("cv", TensorProto.FLOAT,
            [1, NUM_COLORS, MAX_GRID, MAX_GRID], const_val.flatten().tolist())),
        h.make_node("Identity", ["c"], [OUTPUT_NAME]),
    ])


def detect_crop_to_static_bbox(pairs):
    """Crop input to a static bounding box."""
    bbox = None
    for inp, out in pairs:
        nz = np.argwhere(inp != 0)
        if len(nz) == 0: return None
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0) + 1
        cur = (int(r0), int(c0), int(r1), int(c1))
        if bbox is None:
            bbox = cur
        elif bbox != cur:
            return None
        if out.shape != (r1 - r0, c1 - c0):
            return None
        if not np.array_equal(inp[r0:r1, c0:c1], out):
            return None
    r0, c0, r1, c1 = bbox
    return _make_model([
        h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [4], [0,0,r0,c0])),
        h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [4], [1,NUM_COLORS,r1,c1])),
        h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [4], [0,1,2,3])),
        h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], [OUTPUT_NAME]),
    ])


def detect_crop_top_left(pairs):
    """Output is top-left HxW of input."""
    out_h, out_w = pairs[0][1].shape
    for inp, out in pairs:
        if out.shape != (out_h, out_w): return None
        if inp.shape[0] < out_h or inp.shape[1] < out_w: return None
        if not np.array_equal(inp[:out_h, :out_w], out): return None
    return _make_model([
        h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [4], [0,0,0,0])),
        h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [4], [1,NUM_COLORS,out_h,out_w])),
        h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [4], [0,1,2,3])),
        h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], [OUTPUT_NAME]),
    ])


def detect_color_isolation(pairs):
    """Output keeps only color X, sets everything else to 0."""
    for inp, out in pairs:
        if inp.shape != out.shape: return None
    # Find the color that's kept (others go to 0)
    kept_color = None
    for inp, out in pairs:
        diff = inp != out
        if not diff.any():
            continue
        # Cells that changed
        changed_in = inp[diff]
        changed_out = out[diff]
        # All changed input cells should be the same color (the one being removed)
        in_colors = np.unique(changed_in)
        # All changed output cells should be 0
        if not np.all(changed_out == 0):
            return None
        # Multiple colors might be zeroed out
        # If only one color is being zeroed, that's our pattern
    # Try each color as "kept"
    for kept in range(NUM_COLORS):
        ok = True
        for inp, out in pairs:
            expected = np.where(inp == kept, inp, 0)
            # But we also need to keep 'kept' cells as 'kept'
            expected = np.where(inp == kept, kept, 0)
            if not np.array_equal(expected, out):
                ok = False; break
        if ok:
            # Build color_map: kept→kept, everything else → 0
            mapping = {c: 0 for c in range(NUM_COLORS)}
            mapping[kept] = kept
            return dsl.color_map(mapping)
    return None


def detect_color_removal(pairs):
    """Output = input with one specific color X removed (set to 0)."""
    for inp, out in pairs:
        if inp.shape != out.shape: return None
    removed = None
    for inp, out in pairs:
        diff = inp != out
        if not diff.any(): continue
        changed_in = inp[diff]
        changed_out = out[diff]
        in_colors = np.unique(changed_in)
        out_colors = np.unique(changed_out)
        if len(in_colors) == 1 and len(out_colors) == 1 and int(out_colors[0]) == 0:
            if removed is None:
                removed = int(in_colors[0])
            elif removed != int(in_colors[0]):
                return None
        else:
            return None
    if removed is None: return None
    mapping = {removed: 0}
    return dsl.color_map(mapping)


def detect_color_replacement(pairs):
    """Output = input with color X replaced by color Y (Y != 0)."""
    for inp, out in pairs:
        if inp.shape != out.shape: return None
    src, dst = None, None
    for inp, out in pairs:
        diff = inp != out
        if not diff.any(): continue
        changed_in = inp[diff]
        changed_out = out[diff]
        in_colors = np.unique(changed_in)
        out_colors = np.unique(changed_out)
        if len(in_colors) == 1 and len(out_colors) == 1:
            s = int(in_colors[0])
            d = int(out_colors[0])
            if src is None:
                src, dst = s, d
            elif src != s or dst != d:
                return None
        else:
            return None
    if src is None: return None
    # Verify
    for inp, out in pairs:
        modified = inp.copy()
        modified[modified == src] = dst
        if not np.array_equal(modified, out): return None
    return dsl.color_map({src: dst})


def detect_scale_up(pairs):
    """Output = input scaled up by integer factor k (nearest neighbor)."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k_h = out_h // in_h
    k_w = out_w // in_w
    if k_h != k_w or k_h < 2 or k_h > 5: return None
    k = k_h
    for inp, out in pairs:
        if inp.shape != (in_h, in_w): return None
        if out.shape != (out_h, out_w): return None
        scaled = np.repeat(np.repeat(inp, k, axis=0), k, axis=1)
        if not np.array_equal(scaled, out): return None
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], [OUTPUT_NAME],
                             mode="nearest", nearest_mode="round_prefer_floor",
                             coordinate_transformation_mode="asymmetric"))
    return _make_model(nodes)


def detect_scale_down(pairs):
    """Output = input scaled down by integer factor k (subsample)."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if in_h % out_h != 0 or in_w % out_w != 0: return None
    k_h = in_h // out_h
    k_w = in_w // out_w
    if k_h != k_w or k_h < 2 or k_h > 5: return None
    k = k_h
    for inp, out in pairs:
        if inp.shape != (in_h, in_w): return None
        if out.shape != (out_h, out_w): return None
        sampled = inp[::k, ::k]
        if not np.array_equal(sampled, out): return None
    nodes = []
    nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
    nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
    nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
    nodes.append(h.make_node("Constant", [], ["ct"], value=h.make_tensor("ctv", TensorProto.INT64, [4], [1,1,k,k])))
    nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca", "ct"], ["sampled"]))
    pad_b = MAX_GRID - out_h
    pad_r = MAX_GRID - out_w
    if pad_b == 0 and pad_r == 0:
        nodes.append(h.make_node("Identity", ["sampled"], [OUTPUT_NAME]))
    else:
        pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
        nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
        nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
        nodes.append(h.make_node("Pad", ["sampled", "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
    return _make_model(nodes)


def detect_tile(pairs):
    """Output = input tiled N×N."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    n_h = out_h // in_h
    n_w = out_w // in_w
    if n_h != n_w or n_h < 2 or n_h > 4: return None
    n = n_h
    for inp, out in pairs:
        if inp.shape != (in_h, in_w): return None
        if out.shape != (out_h, out_w): return None
        tiled = np.tile(inp, (n, n))
        if not np.array_equal(tiled, out): return None
    # Build via Tile op
    nodes = []
    nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
    nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
    nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
    nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["base"]))
    nodes.append(h.make_node("Constant", [], ["reps"], value=h.make_tensor("repsv", TensorProto.INT64, [4], [1, 1, n, n])))
    nodes.append(h.make_node("Tile", ["base", "reps"], ["tiled"]))
    pad_b = MAX_GRID - out_h
    pad_r = MAX_GRID - out_w
    if pad_b == 0 and pad_r == 0:
        nodes.append(h.make_node("Identity", ["tiled"], [OUTPUT_NAME]))
    else:
        pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
        nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
        nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
        nodes.append(h.make_node("Pad", ["tiled", "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
    return _make_model(nodes)


def detect_kronecker_full(pairs):
    """Each cell c → k×k block of color c."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 2 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w): return None
        if out.shape != (out_h, out_w): return None
        upscaled = np.repeat(np.repeat(inp, k, axis=0), k, axis=1)
        if not np.array_equal(upscaled, out): return None
    # Same as scale_up nearest — use Resize
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], [OUTPUT_NAME],
                             mode="nearest", nearest_mode="round_prefer_floor",
                             coordinate_transformation_mode="asymmetric"))
    return _make_model(nodes)


def detect_kronecker_diagonal(pairs):
    """Each cell c → k×k block with c on main diagonal, 0 elsewhere."""
    if not pairs: return None
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if out_h % in_h != 0 or out_w % in_w != 0: return None
    k = out_h // in_h
    if k != out_w // in_w or k < 2 or k > 5: return None
    for inp, out in pairs:
        if inp.shape != (in_h, in_w): return None
        if out.shape != (out_h, out_w): return None
        for r in range(in_h):
            for c in range(in_w):
                val = int(inp[r, c])
                block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                expected = np.zeros((k, k), dtype=np.int64)
                for i in range(k):
                    expected[i, i] = val
                if not np.array_equal(block, expected): return None
    nodes = []
    nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
    nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["upscaled"],
                             mode="nearest", nearest_mode="round_prefer_floor",
                             coordinate_transformation_mode="asymmetric"))
    mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
    for r in range(MAX_GRID):
        for c in range(MAX_GRID):
            if r % k == c % k:
                mask[0, 0, r, c] = 1.0
    nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
        [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
    nodes.append(h.make_node("Mul", ["upscaled", "m"], [OUTPUT_NAME]))
    return _make_model(nodes)


def detect_first_row_broadcast(pairs):
    """Output = first row of input repeated N times."""
    out_h, out_w = pairs[0][1].shape
    for inp, out in pairs:
        if inp.shape[1] != out_w: return None
        if out.shape != (out_h, out_w): return None
        if not np.array_equal(np.tile(inp[:1, :], (out_h, 1)), out): return None
    in_h, in_w = pairs[0][0].shape
    nodes = []
    nodes.append(h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [4], [0,0,0,0])))
    nodes.append(h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [4], [1,NUM_COLORS,1,in_w])))
    nodes.append(h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [4], [0,1,2,3])))
    nodes.append(h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], ["row0"]))
    nodes.append(h.make_node("Constant", [], ["reps"], value=h.make_tensor("repsv", TensorProto.INT64, [4], [1, 1, out_h, 1])))
    nodes.append(h.make_node("Tile", ["row0", "reps"], ["tiled"]))
    pad_b = MAX_GRID - out_h
    pad_r = MAX_GRID - out_w
    if pad_b == 0 and pad_r == 0:
        nodes.append(h.make_node("Identity", ["tiled"], [OUTPUT_NAME]))
    else:
        pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
        nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
        nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
        nodes.append(h.make_node("Pad", ["tiled", "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
    return _make_model(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline — try all detectors in order
# ─────────────────────────────────────────────────────────────────────────────

DETECTORS = [
    ("color_permutation", detect_color_permutation),
    ("dihedral_with_colormap", detect_dihedral_with_colormap),
    ("constant_output", detect_constant_output),
    ("crop_to_static_bbox", detect_crop_to_static_bbox),
    ("crop_top_left", detect_crop_top_left),
    ("color_isolation", detect_color_isolation),
    ("color_removal", detect_color_removal),
    ("color_replacement", detect_color_replacement),
    ("scale_up", detect_scale_up),
    ("scale_down", detect_scale_down),
    ("tile", detect_tile),
    ("kronecker_full", detect_kronecker_full),
    ("kronecker_diagonal", detect_kronecker_diagonal),
    ("first_row_broadcast", detect_first_row_broadcast),
]


def try_all_detectors(task):
    """Try all detectors and return the best (highest score) eligible model."""
    pairs = arc_data.get_pairs(task)
    best_model = None
    best_score = 0
    best_method = None
    for name, detector in DETECTORS:
        try:
            model = detector(pairs)
        except Exception:
            continue
        if model is None: continue
        # Validate functional correctness
        e = validator.evaluate_model(model, task)
        if not e["eligible_for_points"]: continue
        if e["score"] > best_score:
            best_score = e["score"]
            best_model = model
            best_method = name
    return best_model, best_method, best_score


def main():
    """Run all detectors on all 400 tasks and build a complete submission."""
    print("Running comprehensive pattern detection on all 400 tasks...")
    
    # Load latest results
    with open("/home/z/my-project/data/aggressive_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    
    # Also try already-solved tasks to see if we can find better solutions
    all_tasks = list(range(1, 401))
    
    results = []
    solved = 0
    total_score = 0.0
    breakdown = {}
    t0 = time.time()
    
    import zipfile, os
    output_path = "/home/z/my-project/download/submission.zip"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid in all_tasks:
            try:
                task = arc_data.load_task(tid)
                fname = arc_data.task_id_to_filename(tid)
            except Exception as e:
                results.append({"task_id": tid, "filename": "?", "solver": "load_error",
                                "cost": 0, "score": 0, "eligible": False, "error": str(e)})
                continue
            try:
                model, method, score = try_all_detectors(task)
            except Exception as e:
                model, method, score = None, None, 0
            if model is not None:
                # Get accurate cost
                ci = faithful_scorer.compute_cost(model)
                cost = ci.get("cost", 0)
                # Re-validate
                e = validator.evaluate_model(model, task)
                if e["eligible_for_points"]:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    total_score += e["score"]
                    breakdown[method] = breakdown.get(method, 0) + 1
                    results.append({"task_id": tid, "filename": fname, "solver": method,
                                    "cost": cost, "score": e["score"], "eligible": True})
                    if solved <= 80 or tid % 50 == 0:
                        print(f"  [OK]   task {tid:3d} ({fname}): {method:30s} cost={cost:5d} score={e['score']:.2f}")
                else:
                    results.append({"task_id": tid, "filename": fname, "solver": method + "_invalid",
                                    "cost": cost, "score": 0, "eligible": False})
            else:
                results.append({"task_id": tid, "filename": fname, "solver": "none",
                                "cost": 0, "score": 0, "eligible": False})
    
    elapsed = time.time() - t0
    summary = {
        "solved": solved, "total": 400, "total_score": total_score,
        "elapsed_sec": elapsed, "breakdown": breakdown,
        "output_path": output_path, "file_size_bytes": os.path.getsize(output_path),
        "pipeline": "comprehensive_pattern_detection",
    }
    with open("/home/z/my-project/data/comprehensive_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    print(f"\n=== Comprehensive Submission Summary ===")
    print(f"Solved: {solved}/400 ({100*solved/400:.1f}%)")
    print(f"Total expected score: {total_score:.2f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {output_path} ({summary['file_size_bytes']} bytes)")
    print(f"\nSolver breakdown:")
    for s, c in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {s:35s}: {c}")
    return summary


if __name__ == "__main__":
    main()
