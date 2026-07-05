"""
Restricted DSL of transpilable primitives for ARC-AGI tasks.

Each primitive has:
1. A Python implementation (for verification on training pairs)
2. An ONNX builder (for transpilation)

Solvers written using ONLY these primitives can be mechanically transpiled to ONNX.
"""
import sys, os, json, re, time
sys.path.insert(0, "/home/z/my-project")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
import onnxruntime as ort

from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model


# ============================================================================
# DSL PRIMITIVES — each returns (numpy_result, onnx_nodes, onnx_initializers)
# ============================================================================

# The transpiler tracks the "current tensor" — a symbolic name in the ONNX graph
# Initially it's INPUT_NAME. Each primitive appends nodes and updates the current tensor.

class Transpiler:
    """Mechanical transpiler from DSL calls to ONNX."""
    
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.counter = 0
        self.current = INPUT_NAME  # current tensor name
        self.current_shape = IO_SHAPE  # (1, 10, 30, 30)
    
    def _fresh(self, prefix="t"):
        self.counter += 1
        return f"{prefix}{self.counter}"
    
    def color_map(self, mapping):
        """Apply a color permutation. mapping: dict[int, int]."""
        full_map = {c: mapping.get(c, c) for c in range(NUM_COLORS)}
        W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
        for frm, to in full_map.items():
            W[to, frm, 0, 0] = 1.0
        w_name = self._fresh("w")
        out_name = self._fresh("cm")
        self.initializers.append(h.make_tensor(w_name, TensorProto.FLOAT,
            [NUM_COLORS, NUM_COLORS, 1, 1], W.flatten().tolist()))
        self.nodes.append(h.make_node("Conv", [self.current, w_name], [out_name],
            pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
        self.current = out_name
        return self
    
    def crop(self, r0, c0, r1, c1):
        """Crop to [r0:r1, c0:c1] in spatial dims."""
        s_name = self._fresh("s")
        e_name = self._fresh("e")
        a_name = self._fresh("a")
        out_name = self._fresh("cr")
        self.initializers.append(h.make_tensor(s_name, TensorProto.INT64, [4], [0,0,r0,c0]))
        self.initializers.append(h.make_tensor(e_name, TensorProto.INT64, [4], [1,NUM_COLORS,r1,c1]))
        self.initializers.append(h.make_tensor(a_name, TensorProto.INT64, [4], [0,1,2,3]))
        self.nodes.append(h.make_node("Slice", [self.current, s_name, e_name, a_name], [out_name]))
        self.current = out_name
        self.current_shape = (1, NUM_COLORS, r1-r0, c1-c0)
        return self
    
    def crop_top_left(self, h_size, w_size):
        """Crop to top-left h_size × w_size."""
        return self.crop(0, 0, h_size, w_size)
    
    def pad_to(self, out_h, out_w):
        """Pad with zeros to (out_h, out_w) spatial size."""
        cur_h, cur_w = self.current_shape[2], self.current_shape[3]
        pad_b = out_h - cur_h
        pad_r = out_w - cur_w
        if pad_b == 0 and pad_r == 0:
            return self  # no-op
        if pad_b < 0 or pad_r < 0:
            raise ValueError(f"Cannot pad negative: {pad_b}, {pad_r}")
        p_name = self._fresh("p")
        v_name = self._fresh("v")
        out_name = self._fresh("pd")
        self.initializers.append(h.make_tensor(p_name, TensorProto.INT64, [8], [0,0,0,0,0,0,pad_b,pad_r]))
        self.initializers.append(h.make_tensor(v_name, TensorProto.FLOAT, [1], [0.0]))
        self.nodes.append(h.make_node("Pad", [self.current, p_name, v_name], [out_name], mode="constant"))
        self.current = out_name
        self.current_shape = (1, NUM_COLORS, out_h, out_w)
        return self
    
    def flip_lr(self):
        """Flip horizontally (left-right)."""
        in_h, in_w = self.current_shape[2], self.current_shape[3]
        s_name = self._fresh("s")
        e_name = self._fresh("e")
        t_name = self._fresh("t")
        a_name = self._fresh("a")
        out_name = self._fresh("fl")
        self.initializers.append(h.make_tensor(s_name, TensorProto.INT64, [1], [in_w-1]))
        self.initializers.append(h.make_tensor(e_name, TensorProto.INT64, [1], [-in_w-1]))
        self.initializers.append(h.make_tensor(t_name, TensorProto.INT64, [1], [-1]))
        self.initializers.append(h.make_tensor(a_name, TensorProto.INT64, [1], [3]))
        self.nodes.append(h.make_node("Slice", [self.current, s_name, e_name, a_name, t_name], [out_name]))
        self.current = out_name
        return self
    
    def flip_ud(self):
        """Flip vertically (up-down)."""
        in_h, in_w = self.current_shape[2], self.current_shape[3]
        s_name = self._fresh("s")
        e_name = self._fresh("e")
        t_name = self._fresh("t")
        a_name = self._fresh("a")
        out_name = self._fresh("fu")
        self.initializers.append(h.make_tensor(s_name, TensorProto.INT64, [1], [in_h-1]))
        self.initializers.append(h.make_tensor(e_name, TensorProto.INT64, [1], [-in_h-1]))
        self.initializers.append(h.make_tensor(t_name, TensorProto.INT64, [1], [-1]))
        self.initializers.append(h.make_tensor(a_name, TensorProto.INT64, [1], [2]))
        self.nodes.append(h.make_node("Slice", [self.current, s_name, e_name, a_name, t_name], [out_name]))
        self.current = out_name
        return self
    
    def rot180(self):
        """Rotate 180 degrees."""
        return self.flip_lr().flip_ud()
    
    def transpose(self):
        """Transpose spatial dims (swap H and W)."""
        out_name = self._fresh("tr")
        self.nodes.append(h.make_node("Transpose", [self.current], [out_name], perm=[0, 1, 3, 2]))
        self.current = out_name
        cur_h, cur_w = self.current_shape[2], self.current_shape[3]
        self.current_shape = (1, NUM_COLORS, cur_w, cur_h)
        return self
    
    def rot90(self):
        """Rotate 90 degrees counterclockwise = transpose + flip_ud."""
        return self.transpose().flip_ud()
    
    def rot270(self):
        """Rotate 270 degrees = transpose + flip_lr."""
        return self.transpose().flip_lr()
    
    def scale_up(self, k):
        """Scale up by integer factor k (nearest neighbor)."""
        sc_name = self._fresh("sc")
        out_name = self._fresh("su")
        self.initializers.append(h.make_tensor(sc_name, TensorProto.FLOAT, [4], [1.0, 1.0, float(k), float(k)]))
        self.nodes.append(h.make_node("Resize", [self.current, "", sc_name], [out_name],
            mode="nearest", nearest_mode="round_prefer_floor",
            coordinate_transformation_mode="asymmetric"))
        self.current = out_name
        cur_h, cur_w = self.current_shape[2], self.current_shape[3]
        self.current_shape = (1, NUM_COLORS, cur_h * k, cur_w * k)
        return self
    
    def scale_down(self, k):
        """Scale down by integer factor k (subsample)."""
        in_h, in_w = self.current_shape[2], self.current_shape[3]
        s_name = self._fresh("s")
        e_name = self._fresh("e")
        a_name = self._fresh("a")
        t_name = self._fresh("t")
        out_name = self._fresh("sd")
        self.initializers.append(h.make_tensor(s_name, TensorProto.INT64, [4], [0,0,0,0]))
        self.initializers.append(h.make_tensor(e_name, TensorProto.INT64, [4], [1,NUM_COLORS,in_h,in_w]))
        self.initializers.append(h.make_tensor(a_name, TensorProto.INT64, [4], [0,1,2,3]))
        self.initializers.append(h.make_tensor(t_name, TensorProto.INT64, [4], [1,1,k,k]))
        self.nodes.append(h.make_node("Slice", [self.current, s_name, e_name, a_name, t_name], [out_name]))
        self.current = out_name
        self.current_shape = (1, NUM_COLORS, in_h // k, in_w // k)
        return self
    
    def tile(self, n_h, n_w):
        """Tile the current tensor n_h × n_w times."""
        reps_name = self._fresh("reps")
        out_name = self._fresh("ti")
        self.initializers.append(h.make_tensor(reps_name, TensorProto.INT64, [4], [1, 1, n_h, n_w]))
        self.nodes.append(h.make_node("Tile", [self.current, reps_name], [out_name]))
        self.current = out_name
        cur_h, cur_w = self.current_shape[2], self.current_shape[3]
        self.current_shape = (1, NUM_COLORS, cur_h * n_h, cur_w * n_w)
        return self
    
    def repeat_rows(self, n):
        """Repeat each row n times (vertical scaling, row-doubler)."""
        return self.tile(n, 1)
    
    def repeat_cols(self, n):
        """Repeat each column n times."""
        return self.tile(1, n)
    
    def gather_colors(self, indices):
        """Reorder color channels via Gather. indices: list of 10 ints (permutation)."""
        i_name = self._fresh("i")
        out_name = self._fresh("ga")
        self.initializers.append(h.make_tensor(i_name, TensorProto.INT64, [NUM_COLORS], indices))
        self.nodes.append(h.make_node("Gather", [self.current, i_name], [out_name], axis=1))
        self.current = out_name
        return self
    
    def constant_output(self, grid):
        """Replace with a constant grid (list of lists of ints)."""
        out_h = len(grid)
        out_w = len(grid[0]) if out_h > 0 else 0
        const_val = np.zeros((1, NUM_COLORS, MAX_GRID, MAX_GRID), dtype=np.float32)
        for r in range(out_h):
            for c in range(out_w):
                color = int(grid[r][c])
                const_val[0, color, r, c] = 1.0
        c_name = self._fresh("c")
        out_name = self._fresh("co")
        self.initializers.append(h.make_tensor(c_name, TensorProto.FLOAT,
            [1, NUM_COLORS, MAX_GRID, MAX_GRID], const_val.flatten().tolist()))
        self.nodes.append(h.make_node("Identity", [c_name], [out_name]))
        self.current = out_name
        self.current_shape = (1, NUM_COLORS, MAX_GRID, MAX_GRID)
        return self
    
    def identity(self):
        """No-op."""
        return self
    
    def build(self):
        """Finalize the ONNX model. The last node's output becomes OUTPUT_NAME."""
        # Rename the current tensor to OUTPUT_NAME
        # Add an Identity node to make the connection
        if self.current != OUTPUT_NAME:
            self.nodes.append(h.make_node("Identity", [self.current], [OUTPUT_NAME]))
        return _make_model(self.nodes, initializers=self.initializers)


# ============================================================================
# Python equivalents (for verification) — mirror the DSL primitives
# ============================================================================

def py_color_map(grid, mapping):
    result = grid.copy()
    for k, v in mapping.items():
        result[grid == k] = v
    return result

def py_crop(grid, r0, c0, r1, c1):
    return grid[r0:r1, c0:c1]

def py_pad_to(grid, out_h, out_w):
    h, w = grid.shape
    if h >= out_h and w >= out_w:
        return grid[:out_h, :out_w]
    result = np.zeros((out_h, out_w), dtype=grid.dtype)
    result[:h, :w] = grid
    return result

def py_flip_lr(grid): return np.fliplr(grid)
def py_flip_ud(grid): return np.flipud(grid)
def py_rot180(grid): return np.rot90(grid, 2)
def py_transpose(grid): return grid.T
def py_rot90(grid): return np.rot90(grid, 1)
def py_rot270(grid): return np.rot90(grid, 3)
def py_scale_up(grid, k): return np.repeat(np.repeat(grid, k, axis=0), k, axis=1)
def py_scale_down(grid, k): return grid[::k, ::k]
def py_tile(grid, n_h, n_w): return np.tile(grid, (n_h, n_w))
def py_repeat_rows(grid, n): return np.repeat(grid, n, axis=0)
def py_repeat_cols(grid, n): return np.repeat(grid, n, axis=1)
def py_gather_colors(grid, indices):
    # indices: list of 10 ints. grid values 0-9. New value at position c is indices[c].
    # Actually: gather reorders channels. So new_value[c] = old_value[indices[c]]... 
    # Actually for color permutation: output_color = indices[input_color]
    # Wait — Gather on axis=1 with indices [a,b,c,...] means output[i] = input[indices[i]]
    # So if input channel c is "1", and indices[c]=target, then output channel indices[c] gets the "1"
    # This is a permutation: output[indices[c]] = input[c]
    # For a color map: cell of color c becomes color indices[c]... no wait.
    # If we Gather axis=1 with indices=[i0,i1,...,i9], output channel j = input channel indices[j]
    # So a cell that was color c (channel c is 1) — after gather, channel j is 1 iff indices[j]==c
    # i.e., output channel j is 1 iff indices[j] == input_color
    # For this to be a color map c→t: we need output channel t to be 1 when input is c
    #   → indices[t] == c → indices is the inverse permutation
    # So py_gather_colors(grid, indices): new_color = position where indices[pos] == old_color
    result = grid.copy()
    # Build forward map: old_color → new_color
    forward = {}
    for new_color, old_color in enumerate(indices):
        forward[old_color] = new_color
    for old_c, new_c in forward.items():
        result[grid == old_c] = new_c
    return result

def py_constant_output(grid, const_grid):
    return np.array(const_grid, dtype=grid.dtype)


# ============================================================================
# Combinatorial solver — try all 1-step and 2-step combinations
# ============================================================================

def derive_color_map(pairs):
    """Derive color mapping from pairs (if it's a pure color permutation)."""
    mapping = {}
    for inp, out in pairs:
        if inp.shape != out.shape: return None
        for c in range(NUM_COLORS):
            in_cells = (inp == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1: return None
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t: return None
            mapping[c] = t
    return mapping


def try_combinations(task):
    """Try all 1-step and 2-step combinations of DSL primitives.
    
    Returns (model, method) or (None, None).
    """
    pairs = arc_data.get_pairs(task)
    if not pairs: return None, None
    
    # ====== 1-step combinations ======
    
    # Pure color map
    mapping = derive_color_map(pairs)
    if mapping and any(k != v for k, v in mapping.items()):
        # Verify
        if all(np.array_equal(py_color_map(inp, mapping), out) for inp, out in pairs):
            t = Transpiler()
            t.color_map(mapping)
            return t.build(), "color_map"
    
    # Pure dihedral (same shape)
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    all_same_in = all(inp.shape == pairs[0][0].shape for inp, _ in pairs)
    if same_size and all_same_in:
        dihedral = [
            ("flip_lr", py_flip_lr, lambda t: t.flip_lr()),
            ("flip_ud", py_flip_ud, lambda t: t.flip_ud()),
            ("rot180", py_rot180, lambda t: t.rot180()),
            ("rot90", py_rot90, lambda t: t.rot90()),
            ("rot270", py_rot270, lambda t: t.rot270()),
            ("transpose", py_transpose, lambda t: t.transpose()),
        ]
        in_h, in_w = pairs[0][0].shape
        for name, py_fn, t_fn in dihedral:
            if all(np.array_equal(py_fn(inp), out) for inp, out in pairs):
                t = Transpiler()
                t.crop_top_left(in_h, in_w)
                t_fn(t)
                out_h, out_w = pairs[0][1].shape
                t.pad_to(out_h, out_w)
                # Actually we need to pad to MAX_GRID, not out_h/out_w
                # The grader crops to expected output size
                # So pad to MAX_GRID to be safe
                t2 = Transpiler()
                t2.crop_top_left(in_h, in_w)
                t_fn(t2)
                # Get final shape
                cur_h, cur_w = t2.current_shape[2], t2.current_shape[3]
                t2.pad_to(MAX_GRID, MAX_GRID)
                return t2.build(), f"dihedral_{name}"
    
    # Color map + dihedral
    if same_size and all_same_in:
        in_h, in_w = pairs[0][0].shape
        dihedral = [
            ("identity", lambda x: x, lambda t: t),
            ("flip_lr", py_flip_lr, lambda t: t.flip_lr()),
            ("flip_ud", py_flip_ud, lambda t: t.flip_ud()),
            ("rot180", py_rot180, lambda t: t.rot180()),
            ("rot90", py_rot90, lambda t: t.rot90()),
            ("rot270", py_rot270, lambda t: t.rot270()),
            ("transpose", py_transpose, lambda t: t.transpose()),
        ]
        for dname, py_dfn, t_dfn in dihedral:
            mapping = {}
            ok = True
            for inp, out in pairs:
                transformed = py_dfn(inp)
                for c in range(NUM_COLORS):
                    in_cells = (transformed == c)
                    if not in_cells.any(): continue
                    out_at = out[in_cells]
                    out_colors = np.unique(out_at)
                    if len(out_colors) != 1:
                        ok = False; break
                    tc = int(out_colors[0])
                    if c in mapping and mapping[c] != tc:
                        ok = False; break
                    mapping[c] = tc
                if not ok: break
            if not ok: continue
            if not any(k != v for k, v in mapping.items()): continue
            # Verify
            valid = True
            for inp, out in pairs:
                transformed = py_dfn(inp)
                mapped = py_color_map(transformed, mapping)
                if not np.array_equal(mapped, out):
                    valid = False; break
            if not valid: continue
            # Build
            t = Transpiler()
            t.crop_top_left(in_h, in_w)
            t_dfn(t)
            t.color_map(mapping)
            t.pad_to(MAX_GRID, MAX_GRID)
            return t.build(), f"colormap_then_{dname}"
    
    # ====== Crop variants ======
    
    # Crop to top-left (variable input size, fixed output size)
    out_h, out_w = pairs[0][1].shape
    if all(out.shape == (out_h, out_w) for inp, out in pairs):
        if all(inp.shape[0] >= out_h and inp.shape[1] >= out_w and
               np.array_equal(inp[:out_h, :out_w], out) for inp, out in pairs):
            t = Transpiler()
            t.crop_top_left(out_h, out_w)
            t.pad_to(MAX_GRID, MAX_GRID)
            return t.build(), "crop_top_left"
    
    # Crop to static bbox (same bbox across pairs)
    bbox = None
    valid = True
    for inp, out in pairs:
        nz = np.argwhere(inp != 0)
        if len(nz) == 0:
            valid = False; break
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0) + 1
        cur = (int(r0), int(c0), int(r1), int(c1))
        if bbox is None:
            bbox = cur
        elif bbox != cur:
            valid = False; break
        if out.shape != (r1-r0, c1-c0):
            valid = False; break
        if not np.array_equal(inp[r0:r1, c0:c1], out):
            valid = False; break
    if valid and bbox:
        r0, c0, r1, c1 = bbox
        t = Transpiler()
        t.crop(r0, c0, r1, c1)
        t.pad_to(MAX_GRID, MAX_GRID)
        return t.build(), "crop_bbox"
    
    # ====== Scale up (nearest) ======
    if all_same_in:
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h == 0 and out_w % in_w == 0:
            k_h = out_h // in_h
            k_w = out_w // in_w
            if k_h == k_w and 2 <= k_h <= 5:
                if all(np.array_equal(py_scale_up(inp, k_h), out) for inp, out in pairs):
                    t = Transpiler()
                    t.scale_up(k_h)
                    return t.build(), f"scale_up_{k_h}"
    
    # ====== Scale down (subsample) ======
    if all_same_in:
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if in_h % out_h == 0 and in_w % out_w == 0:
            k_h = in_h // out_h
            k_w = in_w // out_w
            if k_h == k_w and 2 <= k_h <= 5:
                if all(np.array_equal(py_scale_down(inp, k_h), out) for inp, out in pairs):
                    t = Transpiler()
                    t.crop_top_left(in_h, in_w)
                    t.scale_down(k_h)
                    t.pad_to(MAX_GRID, MAX_GRID)
                    return t.build(), f"scale_down_{k_h}"
    
    # ====== Tile (n × n) ======
    if all_same_in:
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h == 0 and out_w % in_w == 0:
            n_h = out_h // in_h
            n_w = out_w // in_w
            if n_h == n_w and 2 <= n_h <= 4:
                if all(np.array_equal(py_tile(inp, n_h, n_h), out) for inp, out in pairs):
                    t = Transpiler()
                    t.crop_top_left(in_h, in_w)
                    t.tile(n_h, n_h)
                    t.pad_to(MAX_GRID, MAX_GRID)
                    return t.build(), f"tile_{n_h}x{n_w}"
    
    # ====== Constant output ======
    if len(pairs) >= 1:
        first_out = pairs[0][1]
        if all(out.shape == first_out.shape and np.array_equal(out, first_out)
               for inp, out in pairs):
            t = Transpiler()
            t.constant_output(first_out.tolist())
            return t.build(), "constant_output"
    
    # ====== Color isolation (keep only color X, zero others) ======
    for kept in range(NUM_COLORS):
        if all(np.array_equal(np.where(inp == kept, kept, 0), out) for inp, out in pairs):
            mapping = {c: 0 for c in range(NUM_COLORS) if c != kept}
            mapping[kept] = kept
            t = Transpiler()
            t.color_map(mapping)
            return t.build(), f"color_isolate_{kept}"
    
    # ====== Color removal (set color X to 0) ======
    for removed in range(NUM_COLORS):
        ok = True
        for inp, out in pairs:
            modified = inp.copy()
            modified[modified == removed] = 0
            if not np.array_equal(modified, out):
                ok = False; break
        if ok:
            t = Transpiler()
            t.color_map({removed: 0})
            return t.build(), f"color_remove_{removed}"
    
    # ====== Color replacement (X → Y) ======
    for src in range(NUM_COLORS):
        for dst in range(NUM_COLORS):
            if src == dst: continue
            ok = True
            for inp, out in pairs:
                modified = inp.copy()
                modified[modified == src] = dst
                if not np.array_equal(modified, out):
                    ok = False; break
            if ok:
                t = Transpiler()
                t.color_map({src: dst})
                return t.build(), f"color_replace_{src}_to_{dst}"
    
    # ====== 2-step: color_map + scale_up ======
    if all_same_in:
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h == 0 and out_w % in_w == 0:
            k_h = out_h // in_h
            k_w = out_w // in_w
            if k_h == k_w and 2 <= k_h <= 5:
                # Try color_map then scale
                mapping = {}
                ok = True
                for inp, out in pairs:
                    scaled_down = py_scale_down(out, k_h)  # reverse of scale_up
                    for c in range(NUM_COLORS):
                        in_cells = (inp == c)
                        if not in_cells.any(): continue
                        out_at = scaled_down[in_cells]
                        out_colors = np.unique(out_at)
                        if len(out_colors) != 1:
                            ok = False; break
                        tc = int(out_colors[0])
                        if c in mapping and mapping[c] != tc:
                            ok = False; break
                        mapping[c] = tc
                    if not ok: break
                if ok and any(k != v for k, v in mapping.items()):
                    # Verify
                    valid = True
                    for inp, out in pairs:
                        mapped = py_color_map(inp, mapping)
                        scaled = py_scale_up(mapped, k_h)
                        if not np.array_equal(scaled, out):
                            valid = False; break
                    if valid:
                        t = Transpiler()
                        t.color_map(mapping)
                        t.scale_up(k_h)
                        return t.build(), f"colormap_then_scale_up_{k_h}"
                # Try scale then color_map
                mapping = {}
                ok = True
                for inp, out in pairs:
                    scaled = py_scale_up(inp, k_h)
                    for c in range(NUM_COLORS):
                        in_cells = (scaled == c)
                        if not in_cells.any(): continue
                        out_at = out[in_cells]
                        out_colors = np.unique(out_at)
                        if len(out_colors) != 1:
                            ok = False; break
                        tc = int(out_colors[0])
                        if c in mapping and mapping[c] != tc:
                            ok = False; break
                        mapping[c] = tc
                    if not ok: break
                if ok and any(k != v for k, v in mapping.items()):
                    valid = True
                    for inp, out in pairs:
                        scaled = py_scale_up(inp, k_h)
                        mapped = py_color_map(scaled, mapping)
                        if not np.array_equal(mapped, out):
                            valid = False; break
                    if valid:
                        t = Transpiler()
                        t.scale_up(k_h)
                        t.color_map(mapping)
                        return t.build(), f"scale_up_{k_h}_then_colormap"
    
    # ====== 2-step: color_map + tile ======
    if all_same_in:
        in_h, in_w = pairs[0][0].shape
        out_h, out_w = pairs[0][1].shape
        if out_h % in_h == 0 and out_w % in_w == 0:
            n_h = out_h // in_h
            n_w = out_w // in_w
            if n_h == n_w and 2 <= n_h <= 4:
                # color_map then tile
                mapping = {}
                ok = True
                for inp, out in pairs:
                    # Reverse tile: take top-left in_h × in_w of out
                    sub = out[:in_h, :in_w]
                    for c in range(NUM_COLORS):
                        in_cells = (inp == c)
                        if not in_cells.any(): continue
                        out_at = sub[in_cells]
                        out_colors = np.unique(out_at)
                        if len(out_colors) != 1:
                            ok = False; break
                        tc = int(out_colors[0])
                        if c in mapping and mapping[c] != tc:
                            ok = False; break
                        mapping[c] = tc
                    if not ok: break
                if ok and any(k != v for k, v in mapping.items()):
                    valid = True
                    for inp, out in pairs:
                        mapped = py_color_map(inp, mapping)
                        tiled = py_tile(mapped, n_h, n_h)
                        if not np.array_equal(tiled, out):
                            valid = False; break
                    if valid:
                        t = Transpiler()
                        t.color_map(mapping)
                        t.crop_top_left(in_h, in_w)
                        t.tile(n_h, n_h)
                        t.pad_to(MAX_GRID, MAX_GRID)
                        return t.build(), f"colormap_then_tile_{n_h}"
                # tile then color_map
                mapping = {}
                ok = True
                for inp, out in pairs:
                    tiled = py_tile(inp, n_h, n_h)
                    for c in range(NUM_COLORS):
                        in_cells = (tiled == c)
                        if not in_cells.any(): continue
                        out_at = out[in_cells]
                        out_colors = np.unique(out_at)
                        if len(out_colors) != 1:
                            ok = False; break
                        tc = int(out_colors[0])
                        if c in mapping and mapping[c] != tc:
                            ok = False; break
                        mapping[c] = tc
                    if not ok: break
                if ok and any(k != v for k, v in mapping.items()):
                    valid = True
                    for inp, out in pairs:
                        tiled = py_tile(inp, n_h, n_h)
                        mapped = py_color_map(tiled, mapping)
                        if not np.array_equal(mapped, out):
                            valid = False; break
                    if valid:
                        t = Transpiler()
                        t.crop_top_left(in_h, in_w)
                        t.tile(n_h, n_h)
                        t.color_map(mapping)
                        t.pad_to(MAX_GRID, MAX_GRID)
                        return t.build(), f"tile_{n_h}_then_colormap"
    
    return None, None


if __name__ == "__main__":
    # Test on unsolved tasks
    with open("/home/z/my-project/data/final_unified_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    solved = 0
    score = 0.0
    breakdown = {}
    
    output_path = "/home/z/my-project/download/submission.zip"
    import zipfile
    
    with zipfile.ZipFile(output_path, "a", zipfile.ZIP_DEFLATED) as zf:
        for tid in unsolved:
            try:
                task = arc_data.load_task(tid)
                model, method = try_combinations(task)
                if model is not None:
                    e = validator.evaluate_model(model, task)
                    if e["eligible_for_points"]:
                        # Strip metadata
                        model.ClearField("producer_name")
                        model.ClearField("producer_version")
                        model.ClearField("doc_string")
                        model.ClearField("domain")
                        model.ClearField("model_version")
                        model.graph.ClearField("doc_string")
                        if len(model.graph.name) > 1:
                            model.graph.name = "g"
                        # Re-verify
                        e2 = validator.evaluate_model(model, task)
                        if e2["eligible_for_points"]:
                            zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                            solved += 1
                            score += e2["score"]
                            breakdown[method] = breakdown.get(method, 0) + 1
                            print(f"  [OK] task {tid}: {method}, score={e2['score']:.2f}")
            except Exception as e:
                pass
    
    print(f"\n=== DSL Transpiler Summary ===")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")
