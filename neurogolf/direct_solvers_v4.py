"""
neurogolf/direct_solvers_v4.py — Even more solvers focused on common patterns.

Each solver tries a specific pattern and emits ONNX.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
from .solvers.base import Solver
from . import arc_data
from .constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from .exploit_solvers import _make_model


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ColorMapThenDihedralSolver — color permutation + geometric transform
# ─────────────────────────────────────────────────────────────────────────────
class ColorMapThenDihedralSolver(Solver):
    """Try: output = dihedral(color_map(input)). Same-size only, all pairs same shape."""
    name = "direct_colormap_dihedral"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        in_h0, in_w0 = pairs[0][0].shape
        for inp, _ in pairs:
            if inp.shape != (in_h0, in_w0): return None
        # Try all 8 dihedral transforms, deriving color map for each
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
            # Check that the mapping is non-trivial
            if not any(k != v for k, v in mapping.items()): continue
            # Verify: applying transform then color map gives correct output
            valid = True
            for inp, out in pairs:
                transformed = tfn(inp)
                mapped = transformed.copy()
                for k, v in mapping.items():
                    mapped[transformed == k] = v
                if not np.array_equal(mapped, out):
                    valid = False; break
            if not valid: continue
            # Build: color_map then dihedral
            # Use 1x1 conv color_map, then dihedral Slice/Transpose
            from .dsl import color_map, single_layer_conv2d
            cm_model = color_map(mapping)
            # The color_map model takes "input" and produces "output"
            # We need to chain: input → color_map → dihedral → output
            # For simplicity, build inline
            W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
            full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
            for frm, to in full_map.items():
                W[to, frm, 0, 0] = 1.0
            # Build model: Conv (color_map) then crop+transform+pad
            nodes = []
            # Conv color map
            nodes.append(h.make_node("Conv", [INPUT_NAME, "w"], ["cm"],
                pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
            # Then crop to (in_h0, in_w0)
            nodes.append(h.make_node("Constant", [], ["cs"],
                value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
            nodes.append(h.make_node("Constant", [], ["ce"],
                value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h0,in_w0])))
            nodes.append(h.make_node("Constant", [], ["ca"],
                value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
            nodes.append(h.make_node("Slice", ["cm", "cs", "ce", "ca"], ["cropped"]))
            # Apply dihedral transform
            out_h, out_w = in_h0, in_w0
            if tname == "identity":
                nodes.append(h.make_node("Identity", ["cropped"], ["transformed"]))
            elif tname == "flip_lr":
                nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["ft"], value=h.make_tensor("ftv", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("fav", TensorProto.INT64, [1], [3])))
                nodes.append(h.make_node("Slice", ["cropped", "fs", "fe", "fa", "ft"], ["transformed"]))
            elif tname == "flip_ud":
                nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["ft"], value=h.make_tensor("ftv", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("fav", TensorProto.INT64, [1], [2])))
                nodes.append(h.make_node("Slice", ["cropped", "fs", "fe", "fa", "ft"], ["transformed"]))
            elif tname == "transpose":
                nodes.append(h.make_node("Transpose", ["cropped"], ["transformed"], perm=[0, 1, 3, 2]))
                out_h, out_w = in_w0, in_h0
            elif tname == "rot180":
                nodes.append(h.make_node("Constant", [], ["fs1"], value=h.make_tensor("fs1v", TensorProto.INT64, [1], [in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["fe1"], value=h.make_tensor("fe1v", TensorProto.INT64, [1], [-in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["ft1"], value=h.make_tensor("ft1v", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["fa1"], value=h.make_tensor("fa1v", TensorProto.INT64, [1], [2])))
                nodes.append(h.make_node("Slice", ["cropped", "fs1", "fe1", "fa1", "ft1"], ["fv1"]))
                nodes.append(h.make_node("Constant", [], ["fs2"], value=h.make_tensor("fs2v", TensorProto.INT64, [1], [in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["fe2"], value=h.make_tensor("fe2v", TensorProto.INT64, [1], [-in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["ft2"], value=h.make_tensor("ft2v", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["fa2"], value=h.make_tensor("fa2v", TensorProto.INT64, [1], [3])))
                nodes.append(h.make_node("Slice", ["fv1", "fs2", "fe2", "fa2", "ft2"], ["transformed"]))
            elif tname == "rot90":
                # transpose + flip-ud
                nodes.append(h.make_node("Transpose", ["cropped"], ["t"], perm=[0, 1, 3, 2]))
                nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["ft2"], value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("av", TensorProto.INT64, [1], [2])))
                nodes.append(h.make_node("Slice", ["t", "fs", "fe", "fa", "ft2"], ["transformed"]))
                out_h, out_w = in_w0, in_h0
            elif tname == "rot270":
                # transpose + flip-lr
                nodes.append(h.make_node("Transpose", ["cropped"], ["t"], perm=[0, 1, 3, 2]))
                nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["ft2"], value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("av", TensorProto.INT64, [1], [3])))
                nodes.append(h.make_node("Slice", ["t", "fs", "fe", "fa", "ft2"], ["transformed"]))
                out_h, out_w = in_w0, in_h0
            elif tname == "anti_transpose":
                nodes.append(h.make_node("Transpose", ["cropped"], ["t1"], perm=[0, 1, 3, 2]))
                nodes.append(h.make_node("Constant", [], ["s1"], value=h.make_tensor("s1v", TensorProto.INT64, [1], [in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["e1"], value=h.make_tensor("e1v", TensorProto.INT64, [1], [-in_w0-1])))
                nodes.append(h.make_node("Constant", [], ["t1v2"], value=h.make_tensor("t1vv", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["a1"], value=h.make_tensor("a1v", TensorProto.INT64, [1], [2])))
                nodes.append(h.make_node("Slice", ["t1", "s1", "e1", "a1", "t1v2"], ["fv"]))
                nodes.append(h.make_node("Constant", [], ["s2"], value=h.make_tensor("s2v", TensorProto.INT64, [1], [in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["e2"], value=h.make_tensor("e2v", TensorProto.INT64, [1], [-in_h0-1])))
                nodes.append(h.make_node("Constant", [], ["t2"], value=h.make_tensor("t2v", TensorProto.INT64, [1], [-1])))
                nodes.append(h.make_node("Constant", [], ["a2"], value=h.make_tensor("a2v", TensorProto.INT64, [1], [3])))
                nodes.append(h.make_node("Slice", ["fv", "s2", "e2", "a2", "t2"], ["transformed"]))
                out_h, out_w = in_w0, in_h0
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
            # Build model with the conv weight as initializer
            initializers = [h.make_tensor("w", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist())]
            return _make_model(nodes, initializers=initializers)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Solver: CropToBoundingBoxStatic — crop to non-zero region (same bbox across pairs)
# ─────────────────────────────────────────────────────────────────────────────
class CropToNonZeroStaticSolver(Solver):
    """Crop input to bounding box of non-zero cells (requires same bbox across pairs)."""
    name = "direct_crop_bbox_static"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
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


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ConstantGridAnySize — output is a constant grid (any size, any pattern)
# ─────────────────────────────────────────────────────────────────────────────
class ConstantGridAnySizeSolver(Solver):
    """Output is the same constant grid for all pairs (any pattern, not just one color)."""
    name = "direct_const_grid_any"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        if len(pairs) < 1: return None
        first_out = pairs[0][1]
        for inp, out in pairs:
            if out.shape != first_out.shape: return None
            if not np.array_equal(out, first_out): return None
        out_h, out_w = first_out.shape
        # Build a constant tensor of (1, 10, 30, 30) with the output one-hot encoded
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


# ─────────────────────────────────────────────────────────────────────────────
# Solver: DiagKronSolver — kronecker with anti-diagonal pattern
# ─────────────────────────────────────────────────────────────────────────────
class AntiDiagKronSolver(Solver):
    """Each cell c → k×k block with c on anti-diagonal, 0 elsewhere."""
    name = "direct_kron_anti_diag"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        if not pairs: return None
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h != 0 or out_w % in_w != 0: return None
        k = out_h // in_h
        if k != out_w // in_w: return None
        if k < 2 or k > 5: return None
        for inp, out in pairs:
            if inp.shape != (in_h, in_w): return None
            if out.shape != (out_h, out_w): return None
            for r in range(in_h):
                for c in range(in_w):
                    val = int(inp[r, c])
                    block = out[r*k:(r+1)*k, c*k:(c+1)*k]
                    expected = np.zeros((k, k), dtype=np.int64)
                    for i in range(k):
                        expected[i, k-1-i] = val
                    if not np.array_equal(block, expected): return None
        # Build via Resize + Mul(anti-diag mask)
        nodes = []
        nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
        nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["upscaled"],
                                 mode="nearest", nearest_mode="round_prefer_floor",
                                 coordinate_transformation_mode="asymmetric"))
        mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
        for r in range(MAX_GRID):
            for c in range(MAX_GRID):
                if (r % k) + (c % k) == k - 1:
                    mask[0, 0, r, c] = 1.0
        nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
            [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
        nodes.append(h.make_node("Mul", ["upscaled", "m"], [OUTPUT_NAME]))
        return _make_model(nodes)
