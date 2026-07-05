"""
neurogolf/direct_solvers_v2.py — Many more auto-detector solvers for common ARC patterns.

Each solver:
1. Reads task input/output pairs
2. Auto-detects the transformation
3. Emits an ONNX model (preferably with cost=1 exploit)
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
# Solver: MaskOutColor — set all cells of one color to 0 (or another color)
# ─────────────────────────────────────────────────────────────────────────────
class MaskOutColorSolver(Solver):
    """For tasks where output = input with cells of color X set to 0 (background)."""
    name = "direct_mask_color"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        # Find which color is being masked, and to what
        masked_color = None
        target_color = None
        for inp, out in pairs:
            for c in range(NUM_COLORS):
                in_cells = (inp == c)
                if not in_cells.any(): continue
                out_at = out[in_cells]
                out_colors = np.unique(out_at)
                if len(out_colors) == 1:
                    tc = int(out_colors[0])
                    if tc != c:
                        if masked_color is None:
                            masked_color = c
                            target_color = tc
                        elif masked_color != c or target_color != tc:
                            # Multiple transformations — not this solver
                            pass
        if masked_color is None:
            return None
        # Verify: every cell of masked_color becomes target_color, everything else unchanged
        for inp, out in pairs:
            modified = inp.copy()
            modified[modified == masked_color] = target_color
            if not np.array_equal(modified, out): return None
        # Build: 1x1 conv color_map with masked_color → target_color, identity elsewhere
        # This is a color_map task — use the existing color_map approach
        # Cost will be ~500, score ~17
        from .dsl import color_map
        mapping = {masked_color: target_color}
        return color_map(mapping)


# ─────────────────────────────────────────────────────────────────────────────
# Solver: BoundingBoxSolver — output is bounding box of non-zero cells
# ─────────────────────────────────────────────────────────────────────────────
class BoundingBoxSolver(Solver):
    """For tasks where output = bounding box of all non-zero cells in input."""
    name = "direct_bounding_box"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            # Find non-zero region in input
            nz = np.argwhere(inp != 0)
            if len(nz) == 0: return None
            r0, c0 = nz.min(axis=0)
            r1, c1 = nz.max(axis=0) + 1
            cropped = inp[r0:r1, c0:c1]
            if cropped.shape != out.shape: return None
            if not np.array_equal(cropped, out): return None
        # Build crop solver — crop to bounding box of non-zero
        # But the bounding box is data-dependent. For static crop, we need same bbox across pairs.
        r0_0, c0_0 = None, None
        r1_0, c1_0 = None, None
        for inp, out in pairs:
            nz = np.argwhere(inp != 0)
            if len(nz) == 0: return None
            r0, c0 = nz.min(axis=0)
            r1, c1 = nz.max(axis=0) + 1
            if r0_0 is None:
                r0_0, c0_0, r1_0, c1_0 = int(r0), int(c0), int(r1), int(c1)
            elif (r0, c0, r1, c1) != (r0_0, c0_0, r1_0, c1_0):
                return None  # Variable bounding box — can't handle statically
        return _make_model([
            h.make_node("Constant", [], ["s"], value=h.make_tensor("sv", TensorProto.INT64, [4], [0,0,r0_0,c0_0])),
            h.make_node("Constant", [], ["e"], value=h.make_tensor("ev", TensorProto.INT64, [4], [1,NUM_COLORS,r1_0,c1_0])),
            h.make_node("Constant", [], ["a"], value=h.make_tensor("av", TensorProto.INT64, [4], [0,1,2,3])),
            h.make_node("Slice", [INPUT_NAME, "s", "e", "a"], [OUTPUT_NAME]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ConstantOutputSolver — output is a fixed-size grid of one color
# ─────────────────────────────────────────────────────────────────────────────
class ConstantOutputSolver(Solver):
    """For tasks where every output is the same constant grid."""
    name = "direct_constant_output"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        if len(pairs) < 2: return None
        first_out = pairs[0][1]
        for inp, out in pairs:
            if out.shape != first_out.shape: return None
            if not np.array_equal(out, first_out): return None
        # Build a constant output
        out_h, out_w = first_out.shape
        # All cells must be the same color
        unique_colors = np.unique(first_out)
        if len(unique_colors) > 1: return None  # Not a constant
        out_color = int(unique_colors[0])
        const_val = np.zeros((1, NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
        const_val[0, out_color, :out_h, :out_w] = 1.0
        return _make_model([
            h.make_node("Constant", [], ["c"], value=h.make_tensor("cv", TensorProto.FLOAT,
                [1, NUM_COLORS, MAX_GRID, MAX_GRID], const_val.flatten().tolist())),
            h.make_node("Identity", ["c"], [OUTPUT_NAME]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Solver: RowMajoritySolver — each row replaced by its majority color
# ─────────────────────────────────────────────────────────────────────────────
class RowMajoritySolver(Solver):
    """For tasks where each row of output is the majority color of that row of input."""
    name = "direct_row_majority"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
            # Compute majority per row
            for r in range(inp.shape[0]):
                row = inp[r]
                colors, counts = np.unique(row, return_counts=True)
                maj = int(colors[np.argmax(counts)])
                if not np.all(out[r] == maj): return None
        # Too complex to ONNX-ify cheaply without learning the majority per row
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Solver: SortRowsSolver — sort each row of input
# ─────────────────────────────────────────────────────────────────────────────
class SortRowsSolver(Solver):
    """For tasks where each row of output is the sorted version of input's row."""
    name = "direct_sort_rows"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
            for r in range(inp.shape[0]):
                if not np.array_equal(np.sort(inp[r]), out[r]): return None
        # TopK can sort, but it's complex. Skip for now.
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Solver: RemoveColorSolver — remove all cells of a specific color (set to 0)
# ─────────────────────────────────────────────────────────────────────────────
class RemoveColorSolver(Solver):
    """Output = input with cells of color X set to 0."""
    name = "direct_remove_color"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        # Find which color was removed
        removed = None
        for inp, out in pairs:
            diff = inp != out
            if not diff.any(): return None
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
        # Build color_map with removed → 0
        from .dsl import color_map
        return color_map({removed: 0})


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ReplaceColorSolver — replace one color with another
# ─────────────────────────────────────────────────────────────────────────────
class ReplaceOneColorSolver(Solver):
    """Output = input with one color X replaced by color Y (Y != 0)."""
    name = "direct_replace_one_color"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        # Find the (X → Y) mapping
        src_color, dst_color = None, None
        for inp, out in pairs:
            diff = inp != out
            if not diff.any(): continue
            changed_in = inp[diff]
            changed_out = out[diff]
            in_colors = np.unique(changed_in)
            out_colors = np.unique(changed_out)
            if len(in_colors) == 1 and len(out_colors) == 1:
                src = int(in_colors[0])
                dst = int(out_colors[0])
                if src_color is None:
                    src_color, dst_color = src, dst
                elif src_color != src or dst_color != dst:
                    return None
            else:
                return None
        if src_color is None: return None
        # Verify
        for inp, out in pairs:
            modified = inp.copy()
            modified[modified == src_color] = dst_color
            if not np.array_equal(modified, out): return None
        from .dsl import color_map
        return color_map({src_color: dst_color})


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ColorMapSolver — generic 1:1 color permutation (any bijection)
# ─────────────────────────────────────────────────────────────────────────────
class GenericColorMapSolver(Solver):
    """Output = input with a bijective color permutation. Tries Gather exploit first."""
    name = "direct_generic_color_map"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        # Build mapping
        mapping = {}
        for inp_p, out_p in pairs:
            for c in range(NUM_COLORS):
                in_cells = (inp_p == c)
                if not in_cells.any(): continue
                out_colors = np.unique(out_p[in_cells])
                if len(out_colors) != 1: return None
                t = int(out_colors[0])
                if c in mapping and mapping[c] != t: return None
                mapping[c] = t
        if not mapping: return None
        # If it's a bijection, try Gather exploit (cost=1)
        if set(mapping.keys()) == set(mapping.values()):
            indices = list(range(NUM_COLORS))
            for source, target in mapping.items():
                indices[target] = source
            return _make_model([
                h.make_node("Constant", [], ["i"], value=h.make_tensor("iv", TensorProto.INT64, [NUM_COLORS], indices)),
                h.make_node("Gather", [INPUT_NAME, "i"], [OUTPUT_NAME], axis=1),
            ])
        # Otherwise use 1x1 conv
        from .dsl import color_map
        return color_map(mapping)


# ─────────────────────────────────────────────────────────────────────────────
# Solver: IdentityOrConstantSolver — output is identity or single color
# ─────────────────────────────────────────────────────────────────────────────
class IdentityOrConstantSolver(Solver):
    """Catches identity tasks (out==in) — same as exploit_identity but as a fallback."""
    name = "direct_identity_or_const"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape or not np.array_equal(inp, out):
                return None
        return _make_model([h.make_node("Identity", [INPUT_NAME], [OUTPUT_NAME])])


# ─────────────────────────────────────────────────────────────────────────────
# Solver: CropToNonZeroSolver — crop to non-zero region (variable size)
# ─────────────────────────────────────────────────────────────────────────────
class CropToSingleColorRegionSolver(Solver):
    """Crop to top-left HxW where H,W = output dimensions."""
    name = "direct_crop_topleft_any"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
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


# ─────────────────────────────────────────────────────────────────────────────
# Solver: ScaleUpNearestSolver — integer scaling via Resize (cheap, no params)
# ─────────────────────────────────────────────────────────────────────────────
class ScaleUpNearestSolver(Solver):
    """Output = input scaled up by integer k (nearest neighbor)."""
    name = "direct_scale_up_nearest"
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
            scaled = np.repeat(np.repeat(inp, k, axis=0), k, axis=1)
            if not np.array_equal(scaled, out): return None
        # Build using Resize (nearest, no weights → very cheap)
        nodes = []
        nodes.append(h.make_node("Constant", [], ["sc"], value=h.make_tensor("scv", TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)])))
        nodes.append(h.make_node("Resize", [INPUT_NAME, "", "sc"], [OUTPUT_NAME],
                                 mode="nearest", nearest_mode="round_prefer_floor",
                                 coordinate_transformation_mode="asymmetric"))
        return _make_model(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Solver: PadOutputSolver — output = input padded with 0 to a fixed size
# ─────────────────────────────────────────────────────────────────────────────
class PadOutputSolver(Solver):
    """Output = input padded with 0 to a fixed larger size."""
    name = "direct_pad_output"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        in_h0, in_w0 = pairs[0][0].shape
        out_h0, out_w0 = pairs[0][1].shape
        if out_h0 <= in_h0 or out_w0 <= in_w0: return None
        for inp, out in pairs:
            if inp.shape != (in_h0, in_w0): return None
            if out.shape != (out_h0, out_w0): return None
            if not np.array_equal(out[:in_h0, :in_w0], inp): return None
            if not np.all(out[in_h0:, :] == 0): return None
            if not np.all(out[:, in_w0:] == 0): return None
        pad_b = out_h0 - in_h0
        pad_r = out_w0 - in_w0
        # First slice input to its actual size
        nodes = []
        nodes.append(h.make_node("Constant", [], ["cs"], value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["ce"], value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h0,in_w0])))
        nodes.append(h.make_node("Constant", [], ["ca"], value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["cropped"]))
        # Pad to output size, then pad again to (1,10,30,30)
        # Actually we can just pad directly to (1,10,30,30) since the grader crops
        pad_total_b = MAX_GRID - in_h0
        pad_total_r = MAX_GRID - in_w0
        pads = [0, 0, 0, 0, 0, 0, pad_total_b, pad_total_r]
        nodes.append(h.make_node("Constant", [], ["pv"], value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
        nodes.append(h.make_node("Constant", [], ["pvv2"], value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
        nodes.append(h.make_node("Pad", ["cropped", "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
        return _make_model(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Solver: FlipAllAxesSolver — try flip-lr, flip-ud, both, transpose, all 8 dihedral
# ─────────────────────────────────────────────────────────────────────────────
class AllDihedralSolver(Solver):
    """Try all 8 dihedral transforms; emit cost=1 if any matches.
    Properly crops input to actual content size before transforming, then pads back.
    """
    name = "direct_dihedral"
    def attempt(self, task):
        pairs = arc_data.get_pairs(task)
        for inp, out in pairs:
            if inp.shape != out.shape: return None
        # All input shapes must be the same (for static crop)
        in_h0, in_w0 = pairs[0][0].shape
        for inp, _ in pairs:
            if inp.shape != (in_h0, in_w0): return None
        # Try all 8 dihedral transforms
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
        for name, fn in transforms:
            if all(np.array_equal(fn(inp), out) for inp, out in pairs):
                # Build the model: crop input to (in_h0, in_w0), apply transform, pad back
                return self._build_model(name, in_h0, in_w0)
        return None

    def _build_model(self, name, in_h, in_w):
        """Build a model that:
        1. Slices (1,10,30,30) input to (1,10,in_h,in_w) — top-left content
        2. Applies the dihedral transform on the cropped region
        3. Pads back to (1,10,30,30) with zeros
        """
        nodes = []
        # Step 1: Slice to actual content size
        nodes.append(h.make_node("Constant", [], ["cs"],
            value=h.make_tensor("csv", TensorProto.INT64, [4], [0,0,0,0])))
        nodes.append(h.make_node("Constant", [], ["ce"],
            value=h.make_tensor("cev", TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w])))
        nodes.append(h.make_node("Constant", [], ["ca"],
            value=h.make_tensor("cav", TensorProto.INT64, [4], [0,1,2,3])))
        nodes.append(h.make_node("Slice", [INPUT_NAME, "cs", "ce", "ca"], ["cropped"]))

        # Step 2: Apply transform on the cropped region
        out_h, out_w = in_h, in_w  # default for identity/flip/rot180
        if name == "identity":
            nodes.append(h.make_node("Identity", ["cropped"], ["transformed"]))
        elif name == "flip_lr":
            # Slice axis 3 (width) reversed: starts=in_w-1, ends=-in_w-1, step=-1
            nodes.append(h.make_node("Constant", [], ["fs"],
                value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["fe"],
                value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["ft"],
                value=h.make_tensor("ftv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fa"],
                value=h.make_tensor("fav", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["cropped", "fs", "fe", "fa", "ft"], ["transformed"]))
        elif name == "flip_ud":
            nodes.append(h.make_node("Constant", [], ["fs"],
                value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_h-1])))
            nodes.append(h.make_node("Constant", [], ["fe"],
                value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_h-1])))
            nodes.append(h.make_node("Constant", [], ["ft"],
                value=h.make_tensor("ftv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fa"],
                value=h.make_tensor("fav", TensorProto.INT64, [1], [2])))
            nodes.append(h.make_node("Slice", ["cropped", "fs", "fe", "fa", "ft"], ["transformed"]))
        elif name == "rot180":
            # Flip both axes
            nodes.append(h.make_node("Constant", [], ["fs1"],
                value=h.make_tensor("fs1v", TensorProto.INT64, [1], [in_h-1])))
            nodes.append(h.make_node("Constant", [], ["fe1"],
                value=h.make_tensor("fe1v", TensorProto.INT64, [1], [-in_h-1])))
            nodes.append(h.make_node("Constant", [], ["ft1"],
                value=h.make_tensor("ft1v", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fa1"],
                value=h.make_tensor("fa1v", TensorProto.INT64, [1], [2])))
            nodes.append(h.make_node("Slice", ["cropped", "fs1", "fe1", "fa1", "ft1"], ["fv1"]))
            nodes.append(h.make_node("Constant", [], ["fs2"],
                value=h.make_tensor("fs2v", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["fe2"],
                value=h.make_tensor("fe2v", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["ft2"],
                value=h.make_tensor("ft2v", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fa2"],
                value=h.make_tensor("fa2v", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["fv1", "fs2", "fe2", "fa2", "ft2"], ["transformed"]))
        elif name == "transpose":
            nodes.append(h.make_node("Transpose", ["cropped"], ["transformed"], perm=[0, 1, 3, 2]))
            out_h, out_w = in_w, in_h
        elif name == "rot90":
            # rot90 (counterclockwise) = transpose + flip-ud
            nodes.append(h.make_node("Transpose", ["cropped"], ["t"], perm=[0, 1, 3, 2]))
            nodes.append(h.make_node("Constant", [], ["fs"],
                value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["fe"],
                value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["ft2"],
                value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fa"],
                value=h.make_tensor("av", TensorProto.INT64, [1], [2])))
            nodes.append(h.make_node("Slice", ["t", "fs", "fe", "fa", "ft2"], ["transformed"]))
            out_h, out_w = in_w, in_h
        elif name == "rot270":
            # rot270 (clockwise) = transpose + flip-lr
            nodes.append(h.make_node("Transpose", ["cropped"], ["t"], perm=[0, 1, 3, 2]))
            nodes.append(h.make_node("Constant", [], ["fs"],
                value=h.make_tensor("fsv", TensorProto.INT64, [1], [in_h-1])))
            nodes.append(h.make_node("Constant", [], ["fe"],
                value=h.make_tensor("fev", TensorProto.INT64, [1], [-in_h-1])))
            nodes.append(h.make_node("Constant", [], ["ft2"],
                value=h.make_tensor("tv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["fa"],
                value=h.make_tensor("av", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["t", "fs", "fe", "fa", "ft2"], ["transformed"]))
            out_h, out_w = in_w, in_h
        elif name == "anti_transpose":
            # anti-transpose = transpose + rot180
            nodes.append(h.make_node("Transpose", ["cropped"], ["t1"], perm=[0, 1, 3, 2]))
            nodes.append(h.make_node("Constant", [], ["s1"],
                value=h.make_tensor("s1v", TensorProto.INT64, [1], [in_w-1])))
            nodes.append(h.make_node("Constant", [], ["e1"],
                value=h.make_tensor("e1v", TensorProto.INT64, [1], [-in_w-1])))
            nodes.append(h.make_node("Constant", [], ["t1v2"],
                value=h.make_tensor("t1vv", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["a1"],
                value=h.make_tensor("a1v", TensorProto.INT64, [1], [2])))
            nodes.append(h.make_node("Slice", ["t1", "s1", "e1", "a1", "t1v2"], ["fv"]))
            nodes.append(h.make_node("Constant", [], ["s2"],
                value=h.make_tensor("s2v", TensorProto.INT64, [1], [in_h-1])))
            nodes.append(h.make_node("Constant", [], ["e2"],
                value=h.make_tensor("e2v", TensorProto.INT64, [1], [-in_h-1])))
            nodes.append(h.make_node("Constant", [], ["t2"],
                value=h.make_tensor("t2v", TensorProto.INT64, [1], [-1])))
            nodes.append(h.make_node("Constant", [], ["a2"],
                value=h.make_tensor("a2v", TensorProto.INT64, [1], [3])))
            nodes.append(h.make_node("Slice", ["fv", "s2", "e2", "a2", "t2"], ["transformed"]))
            out_h, out_w = in_w, in_h

        # Step 3: Pad back to (1,10,30,30)
        pad_b = MAX_GRID - out_h
        pad_r = MAX_GRID - out_w
        if pad_b == 0 and pad_r == 0:
            nodes.append(h.make_node("Identity", ["transformed"], [OUTPUT_NAME]))
        else:
            pads = [0, 0, 0, 0, 0, 0, pad_b, pad_r]
            nodes.append(h.make_node("Constant", [], ["pv"],
                value=h.make_tensor("pvv", TensorProto.INT64, [8], pads)))
            nodes.append(h.make_node("Constant", [], ["pvv2"],
                value=h.make_tensor("pvvv", TensorProto.FLOAT, [1], [0.0])))
            nodes.append(h.make_node("Pad", ["transformed", "pv", "pvv2"], [OUTPUT_NAME], mode="constant"))
        return _make_model(nodes)
