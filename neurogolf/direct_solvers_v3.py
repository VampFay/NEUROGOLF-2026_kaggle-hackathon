"""
neurogolf/direct_solvers_v3.py — More direct solvers based on actual unsolved task analysis.

Patterns observed in unsolved tasks:
- Kronecker expansion: each cell becomes a k×k block (with patterns: identity, full, diagonal)
- Tile-with-mirror: tile input but mirror alternating cells
- Diagonal spread: each non-zero in input becomes a diagonal line
- Color invert (replace each color c with N-c)
- Replace specific row/column with another color
- Mask out non-largest-object
- Sort rows or columns by some criterion
- Output = top-K rows of input
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
# Solver: KroneckerExpandSolver — each cell c → k×k block (c on diagonal, 0 elsewhere)
# ─────────────────────────────────────────────────────────────────────────────
class KroneckerDiagSolver(Solver):
    """Each cell c → k×k block with c on main diagonal, 0 elsewhere."""
    name = "direct_kron_diag"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        if not pairs: return None
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h != 0 or out_w % in_w != 0: return None
        k = out_h // in_h
        if k != out_w // in_w: return None
        if k < 2 or k > 5: return None
        # Verify: each k×k block in output is diagonal with input value
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
        # Build via Conv (each output channel derived from input channel via diagonal kernel)
        # For k=2: kernel [[1,0],[0,1]] per channel; output position (r*2+i, c*2+j) = input[r,c] if i==j else 0
        # Easier: use Resize to upscale, then mask off-diagonal cells
        # Actually simplest: Conv with k×k kernel that's identity matrix per channel
        # But that's expensive. Use Resize + Mul(mask).
        # For ONNX: Resize (nearest) gives the upscaled grid. Then multiply by a fixed mask tensor.
        nodes = []
        # Resize input by factor k (nearest)
        nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
        nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], ["upscaled"],
                                 mode="nearest", nearest_mode="round_prefer_floor",
                                 coordinate_transformation_mode="asymmetric"))
        # Build mask: 1 where (i % k == j % k), else 0
        mask = np.zeros((1, 1, MAX_GRID, MAX_GRID), dtype=np.float32)
        for r in range(MAX_GRID):
            for c in range(MAX_GRID):
                if r % k == c % k:
                    mask[0, 0, r, c] = 1.0
        nodes.append(h.make_node("Constant", [], ["m"], value=h.make_tensor("mv", TensorProto.FLOAT,
            [1, 1, MAX_GRID, MAX_GRID], mask.flatten().tolist())))
        nodes.append(h.make_node("Mul", ["upscaled", "m"], [OUTPUT_NAME]))
        return _make_model(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Solver: KroneckerFullSolver — each cell c → k×k block of color c
# ─────────────────────────────────────────────────────────────────────────────
class KroneckerFullSolver(Solver):
    """Each cell c → k×k block where every cell is c."""
    name = "direct_kron_full"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        if not pairs: return None
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h != 0 or out_w % in_w != 0: return None
        k = out_h // in_h
        if k != out_w // in_w: return None
        if k < 2 or k > 5: return None
        # Verify
        for inp, out in pairs:
            if inp.shape != (in_h, in_w): return None
            if out.shape != (out_h, out_w): return None
            upscaled = np.repeat(np.repeat(inp, k, axis=0), k, axis=1)
            if not np.array_equal(upscaled, out): return None
        # Build via Resize (nearest) — this is exactly nearest-neighbor upscaling
        nodes = []
        nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
        nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], [OUTPUT_NAME],
                                 mode="nearest", nearest_mode="round_prefer_floor",
                                 coordinate_transformation_mode="asymmetric"))
        return _make_model(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Solver: TileMirrorSolver — tile input N×N with alternating mirror pattern
# ─────────────────────────────────────────────────────────────────────────────
class TileMirrorSolver(Solver):
    """Tile input N×N where alternating cells are mirrored (quilt pattern)."""
    name = "direct_tile_mirror"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        if not pairs: return None
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h != 0 or out_w % in_w != 0: return None
        n_h = out_h // in_h
        n_w = out_w // in_w
        if n_h != n_w or n_h < 2 or n_h > 4: return None
        n = n_h
        # Try several mirror patterns
        # Pattern 1: standard quilt (mirror on odd rows/cols)
        # Pattern 2: mirror only on rows
        # Pattern 3: mirror only on cols
        patterns = [
            ("quilt", lambda inp, r, c: np.flip(inp, axis=(0,1)) if (r % 2, c % 2) == (1, 1)
                       else np.flip(inp, axis=0) if r % 2 == 1
                       else np.flip(inp, axis=1) if c % 2 == 1
                       else inp),
            ("mirror_rows", lambda inp, r, c: np.flipud(inp) if r % 2 == 1 else inp),
            ("mirror_cols", lambda inp, r, c: np.fliplr(inp) if c % 2 == 1 else inp),
        ]
        for pname, pfn in patterns:
            ok = True
            for inp, out in pairs:
                if inp.shape != (in_h, in_w): return None
                if out.shape != (out_h, out_w): return None
                tiled = np.zeros((out_h, out_w), dtype=inp.dtype)
                for r in range(n):
                    for c in range(n):
                        tiled[r*in_h:(r+1)*in_h, c*in_w:(c+1)*in_w] = pfn(inp, r, c)
                if not np.array_equal(tiled, out):
                    ok = False; break
            if ok:
                # Build ONNX — complex. For now, just handle mirror_cols and mirror_rows
                # via Slice + Concat
                if pname == "mirror_cols":
                    nodes = []
                    # Slice input to actual size
                    nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
                    nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
                    nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
                    nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["base"]))
                    # Flip lr
                    nodes.append(h.make_node("Constant", [], ["fs"], value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_w-1])))
                    nodes.append(h.make_node("Constant", [], ["fe"], value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_w-1])))
                    nodes.append(h.make_node("Constant", [], ["ft"], value=h.make_tensor("ftv", TensorProto.INT64, [1], [-1])))
                    nodes.append(h.make_node("Constant", [], ["fa"], value=h.make_tensor("fav", TensorProto.INT64, [1], [3])))
                    nodes.append(h.make_node("Slice", ["base", "fs", "fe", "fa", "ft"], ["flip"]))
                    # Concat base + flip n_w times
                    prev = "base"
                    for i in range(n - 1):
                        new = f"c{i}"
                        nodes.append(h.make_node("Concat", [prev, "flip" if i % 2 == 0 else "base"],
                                                 [new], axis=3))
                        prev = new
                    # Pad to (1,10,30,30)
                    pad_b = MAX_GRID - out_h
                    pad_r = MAX_GRID - out_w
                    if pad_b == 0 and pad_r == 0:
                        nodes.append(h.make_node("Identity", [prev], [OUTPUT_NAME]))
                    else:
                        pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
                        nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
                        nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
                        nodes.append(h.make_node("Pad", [prev, "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
                    return _make_model(nodes)
                # For other patterns, fall through and skip (too complex for now)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ColorInvertSolver — replace each color c with N-c (N=9 or 5)
# ─────────────────────────────────────────────────────────────────────────────
class ColorInvertSolver(Solver):
    """Output = (N - input) for some constant N. Try N=9 first."""
    name = "direct_color_invert"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        for N in [9, 5, 4, 1, 8, 7, 6, 3, 2]:
            if all(np.array_equal(N - inp, out) for inp, out in pairs):
                # Build color_map
                from .dsl import color_map
                mapping = {c: (N - c) for c in range(NUM_COLORS) if (N - c) >= 0 and (N - c) < NUM_COLORS}
                return color_map(mapping)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ColorRotateSolver — cyclic color shift (c → (c+k) % 10)
# ─────────────────────────────────────────────────────────────────────────────
class ColorRotateSolver(Solver):
    """Output = (input + k) % 10 for some k. Try all k."""
    name = "direct_color_rotate"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        for k in range(1, NUM_COLORS):
            if all(np.array_equal((inp + k) % NUM_COLORS, out) for inp, out in pairs):
                # Build color_map
                from .dsl import color_map
                mapping = {c: (c + k) % NUM_COLORS for c in range(NUM_COLORS)}
                return color_map(mapping)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Solver: FirstRowBroadcastSolver — output = first row of input broadcast to N rows
# ─────────────────────────────────────────────────────────────────────────────
class FirstRowBroadcastSolver(Solver):
    """Output = first row of input repeated N times."""
    name = "direct_first_row_broadcast"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        out_h, out_w = pairs[0][1].shape
        for inp, out in pairs:
            if inp.shape[1] != out_w: return None
            if out.shape != (out_h, out_w): return None
            if not np.array_equal(np.tile(inp[:1, :], (out_h, 1)), out): return None
        # Build via Slice (take row 0) + Tile
        nodes = []
        # Slice to take row 0 of input (full width = in_w)
        in_h, in_w = pairs[0][0].shape
        nodes.append(h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [4], [1,NUM_COLORS,1,in_w])))
        nodes.append(h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], ["row0"]))
        # Tile by (out_h, 1)
        nodes.append(h.make_node("Constant", [], ["reps"], value=h.make_tensor("repsv", TensorProto.INT64, [4], [1, 1, out_h, 1])))
        nodes.append(h.make_node("Tile", ["row0", "reps"], ["tiled"]))
        # Pad to (1,10,30,30)
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
# Solver: FirstColBroadcastSolver — output = first col of input broadcast to N cols
# ─────────────────────────────────────────────────────────────────────────────
class FirstColBroadcastSolver(Solver):
    """Output = first col of input repeated N times."""
    name = "direct_first_col_broadcast"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        out_h, out_w = pairs[0][1].shape
        for inp, out in pairs:
            if inp.shape[0] != out_h: return None
            if out.shape != (out_h, out_w): return None
            if not np.array_equal(np.tile(inp[:, :1], (1, out_w)), out): return None
        in_h, in_w = pairs[0][0].shape
        nodes = []
        nodes.append(h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,1])))
        nodes.append(h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], ["col0"]))
        nodes.append(h.make_node("Constant", [], ["reps"], value=h.make_tensor("repsv", TensorProto.INT64, [4], [1, 1, 1, out_w])))
        nodes.append(h.make_node("Tile", ["col0", "reps"], ["tiled"]))
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
