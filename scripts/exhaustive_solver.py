"""
Exhaustive combinatorial solver — try ALL combinations of DSL operations
to find ones that match the training pairs.

Strategy: For each task, try:
1. All single ops (color_map variants, dihedral, crop, scale, tile, etc.)
2. All 2-op combinations (color_map + dihedral, crop + color_map, etc.)
3. All 3-op combinations for common patterns

This is computationally feasible because:
- Color maps: derive from pairs (1 candidate)
- Dihedral: 8 variants
- Crops: derive from output shape (1 candidate)
- Scale: 2-5 variants
- Tile: 2-4 variants

Total combinations: ~8 * 8 * 5 = 320 per task, each verified in <1ms.
"""
import sys, os, json, time, zipfile, itertools
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


def derive_color_map(pairs):
    """Derive color mapping if pairs differ only by color permutation."""
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


def derive_color_map_after_transform(pairs, transform_fn):
    """Derive color map after applying a transform."""
    mapping = {}
    for inp, out in pairs:
        try:
            transformed = transform_fn(inp)
        except Exception:
            return None
        if transformed.shape != out.shape: return None
        for c in range(NUM_COLORS):
            in_cells = (transformed == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1: return None
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t: return None
            mapping[c] = t
    return mapping


def verify_python(pairs, ops_py):
    """Verify a Python sequence of ops on all pairs."""
    try:
        for inp, out in pairs:
            cur = inp.copy()
            for fn in ops_py:
                cur = fn(cur)
                if cur is None: return False
            if cur.shape != out.shape: return False
            if not np.array_equal(cur, out): return False
        return True
    except Exception:
        return False


def build_onnx(ops_dsl, in_h, in_w):
    """Build ONNX from DSL ops."""
    t = Transpiler()
    t.crop_top_left(in_h, in_w)
    for op_name, op_args in ops_dsl:
        if op_name == "color_map":
            t.color_map(op_args)
        elif op_name == "crop_top_left":
            t.crop_top_left(*op_args)
        elif op_name == "pad_to":
            t.pad_to(*op_args)
        elif op_name == "flip_lr":
            t.flip_lr()
        elif op_name == "flip_ud":
            t.flip_ud()
        elif op_name == "rot90":
            t.rot90()
        elif op_name == "rot180":
            t.rot180()
        elif op_name == "rot270":
            t.rot270()
        elif op_name == "transpose":
            t.transpose()
        elif op_name == "scale_up":
            t.scale_up(*op_args)
        elif op_name == "scale_down":
            t.scale_down(*op_args)
        elif op_name == "tile":
            t.tile(*op_args)
        elif op_name == "repeat_rows":
            t.repeat_rows(*op_args)
        elif op_name == "repeat_cols":
            t.repeat_cols(*op_args)
    t.pad_to(MAX_GRID, MAX_GRID)
    return t.build()


def try_exhaustive(task):
    """Try all combinations of DSL operations."""
    pairs = arc_data.get_pairs(task)
    if not pairs: return None, None
    
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    all_same_in = all(inp.shape == (in_h, in_w) for inp, _ in pairs)
    all_same_out = all(out.shape == (out_h, out_w) for _, out in pairs)
    
    # === Level 1: Single ops ===
    
    # 1a. Pure color map
    mapping = derive_color_map(pairs)
    if mapping and any(k != v for k, v in mapping.items()):
        ops_py = [lambda g, m=mapping: py_color_map(g, m)]
        if verify_python(pairs, ops_py):
            return build_onnx([("color_map", mapping)], in_h, in_w), "color_map"
    
    # 1b. Dihedral (same size, same shape)
    if same_size and all_same_in:
        dihedrals = [
            ("flip_lr", lambda g: py_flip_lr(g)),
            ("flip_ud", lambda g: py_flip_ud(g)),
            ("rot180", lambda g: py_rot180(g)),
            ("rot90", lambda g: py_rot90(g)),
            ("rot270", lambda g: py_rot270(g)),
            ("transpose", lambda g: py_transpose(g)),
        ]
        for name, fn in dihedrals:
            if verify_python(pairs, [fn]):
                return build_onnx([(name, None)], in_h, in_w), name
    
    # 1c. Crop top-left (variable input, fixed output)
    if all_same_out:
        ops_py = [lambda g, h=out_h, w=out_w: py_crop(g, 0, 0, h, w)]
        if verify_python(pairs, ops_py):
            return build_onnx([("crop_top_left", (out_h, out_w))], in_h, in_w), "crop_top_left"
    
    # 1d. Scale up
    if all_same_in and all_same_out and out_h > in_h and out_w > in_w:
        for k in range(2, 6):
            if out_h == in_h * k and out_w == in_w * k:
                ops_py = [lambda g, k=k: py_scale_up(g, k)]
                if verify_python(pairs, ops_py):
                    return build_onnx([("scale_up", (k,))], in_h, in_w), f"scale_up_{k}"
    
    # 1e. Scale down
    if all_same_in and all_same_out and out_h < in_h and out_w < in_w:
        for k in range(2, 6):
            if in_h == out_h * k and in_w == out_w * k:
                ops_py = [lambda g, k=k: py_scale_down(g, k)]
                if verify_python(pairs, ops_py):
                    return build_onnx([("scale_down", (k,))], in_h, in_w), f"scale_down_{k}"
    
    # 1f. Tile
    if all_same_in and all_same_out and out_h > in_h and out_w > in_w:
        for n_h in range(2, 5):
            for n_w in range(2, 5):
                if out_h == in_h * n_h and out_w == in_w * n_w:
                    ops_py = [lambda g, nh=n_h, nw=n_w: py_tile(g, nh, nw)]
                    if verify_python(pairs, ops_py):
                        return build_onnx([("tile", (n_h, n_w))], in_h, in_w), f"tile_{n_h}x{n_w}"
    
    # 1g. Repeat rows
    if all_same_in and all_same_out and out_w == in_w and out_h > in_h and out_h % in_h == 0:
        n = out_h // in_h
        if 2 <= n <= 5:
            ops_py = [lambda g, n=n: py_repeat_rows(g, n)]
            if verify_python(pairs, ops_py):
                return build_onnx([("repeat_rows", (n,))], in_h, in_w), f"repeat_rows_{n}"
    
    # 1h. Repeat cols
    if all_same_in and all_same_out and out_h == in_h and out_w > in_w and out_w % in_w == 0:
        n = out_w // in_w
        if 2 <= n <= 5:
            ops_py = [lambda g, n=n: py_repeat_cols(g, n)]
            if verify_python(pairs, ops_py):
                return build_onnx([("repeat_cols", (n,))], in_h, in_w), f"repeat_cols_{n}"
    
    # === Level 2: Two-op combinations ===
    
    # 2a. Color map + dihedral
    if same_size and all_same_in:
        dihedrals = [
            ("identity", lambda g: g),
            ("flip_lr", lambda g: py_flip_lr(g)),
            ("flip_ud", lambda g: py_flip_ud(g)),
            ("rot180", lambda g: py_rot180(g)),
            ("rot90", lambda g: py_rot90(g)),
            ("rot270", lambda g: py_rot270(g)),
            ("transpose", lambda g: py_transpose(g)),
        ]
        for dname, dfn in dihedrals:
            mapping = derive_color_map_after_transform(pairs, dfn)
            if mapping and any(k != v for k, v in mapping.items()):
                ops_py = [dfn, lambda g, m=mapping: py_color_map(g, m)]
                if verify_python(pairs, ops_py):
                    ops_dsl = []
                    if dname != "identity":
                        ops_dsl.append((dname, None))
                    ops_dsl.append(("color_map", mapping))
                    return build_onnx(ops_dsl, in_h, in_w), f"{dname}_then_colormap"
    
    # 2b. Dihedral + color map
    if same_size and all_same_in:
        dihedrals = [
            ("identity", lambda g: g),
            ("flip_lr", lambda g: py_flip_lr(g)),
            ("flip_ud", lambda g: py_flip_ud(g)),
            ("rot180", lambda g: py_rot180(g)),
            ("rot90", lambda g: py_rot90(g)),
            ("rot270", lambda g: py_rot270(g)),
            ("transpose", lambda g: py_transpose(g)),
        ]
        # First color_map, then dihedral
        mapping = derive_color_map(pairs)
        if mapping and any(k != v for k, v in mapping.items()):
            for dname, dfn in dihedrals:
                ops_py = [lambda g, m=mapping: py_color_map(g, m), dfn]
                if verify_python(pairs, ops_py):
                    ops_dsl = [("color_map", mapping)]
                    if dname != "identity":
                        ops_dsl.append((dname, None))
                    return build_onnx(ops_dsl, in_h, in_w), f"colormap_then_{dname}"
    
    # 2c. Color map + scale_up
    if all_same_in and all_same_out and out_h > in_h:
        for k in range(2, 6):
            if out_h == in_h * k and out_w == in_w * k:
                # color_map then scale
                # Derive color map by reverse-scaling the output
                mapping = {}
                ok = True
                for inp, out in pairs:
                    scaled_down = py_scale_down(out, k)
                    if scaled_down.shape != inp.shape:
                        ok = False; break
                    for c in range(NUM_COLORS):
                        in_cells = (inp == c)
                        if not in_cells.any(): continue
                        out_at = scaled_down[in_cells]
                        out_colors = np.unique(out_at)
                        if len(out_colors) != 1:
                            ok = False; break
                        t = int(out_colors[0])
                        if c in mapping and mapping[c] != t:
                            ok = False; break
                        mapping[c] = t
                    if not ok: break
                if ok and any(k2 != v for k2, v in mapping.items()):
                    ops_py = [lambda g, m=mapping: py_color_map(g, m), lambda g, k=k: py_scale_up(g, k)]
                    if verify_python(pairs, ops_py):
                        return build_onnx([("color_map", mapping), ("scale_up", (k,))], in_h, in_w), f"colormap_then_scale_up_{k}"
                # scale then color_map
                mapping = {}
                ok = True
                for inp, out in pairs:
                    scaled = py_scale_up(inp, k)
                    if scaled.shape != out.shape:
                        ok = False; break
                    for c in range(NUM_COLORS):
                        in_cells = (scaled == c)
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
                if ok and any(k2 != v for k2, v in mapping.items()):
                    ops_py = [lambda g, k=k: py_scale_up(g, k), lambda g, m=mapping: py_color_map(g, m)]
                    if verify_python(pairs, ops_py):
                        return build_onnx([("scale_up", (k,)), ("color_map", mapping)], in_h, in_w), f"scale_up_{k}_then_colormap"
    
    # 2d. Color map + tile
    if all_same_in and all_same_out and out_h > in_h:
        for n_h in range(2, 5):
            for n_w in range(2, 5):
                if out_h == in_h * n_h and out_w == in_w * n_w:
                    # color_map then tile
                    mapping = {}
                    ok = True
                    for inp, out in pairs:
                        sub = py_crop(out, 0, 0, in_h, in_w)
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
                    if ok and any(k2 != v for k2, v in mapping.items()):
                        ops_py = [lambda g, m=mapping: py_color_map(g, m), lambda g, nh=n_h, nw=n_w: py_tile(g, nh, nw)]
                        if verify_python(pairs, ops_py):
                            return build_onnx([("color_map", mapping), ("tile", (n_h, n_w))], in_h, in_w), f"colormap_then_tile_{n_h}x{n_w}"
                    # tile then color_map
                    mapping = {}
                    ok = True
                    for inp, out in pairs:
                        tiled = py_tile(inp, n_h, n_w)
                        for c in range(NUM_COLORS):
                            in_cells = (tiled == c)
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
                    if ok and any(k2 != v for k2, v in mapping.items()):
                        ops_py = [lambda g, nh=n_h, nw=n_w: py_tile(g, nh, nw), lambda g, m=mapping: py_color_map(g, m)]
                        if verify_python(pairs, ops_py):
                            return build_onnx([("tile", (n_h, n_w)), ("color_map", mapping)], in_h, in_w), f"tile_{n_h}x{n_w}_then_colormap"
    
    # 2e. Crop + color_map (crop then recolor)
    if all_same_out and out_h <= in_h and out_w <= in_w:
        # Crop to output size, then color map
        mapping = {}
        ok = True
        for inp, out in pairs:
            cropped = py_crop(inp, 0, 0, out_h, out_w)
            for c in range(NUM_COLORS):
                in_cells = (cropped == c)
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
        if ok and any(k2 != v for k2, v in mapping.items()):
            ops_py = [lambda g, h=out_h, w=out_w: py_crop(g, 0, 0, h, w), lambda g, m=mapping: py_color_map(g, m)]
            if verify_python(pairs, ops_py):
                return build_onnx([("crop_top_left", (out_h, out_w)), ("color_map", mapping)], in_h, in_w), "crop_then_colormap"
    
    # 2f. Color_map + crop (recolor then crop)
    if all_same_out and out_h <= in_h and out_w <= in_w:
        mapping = derive_color_map(pairs)
        if mapping and any(k != v for k, v in mapping.items()):
            ops_py = [lambda g, m=mapping: py_color_map(g, m), lambda g, h=out_h, w=out_w: py_crop(g, 0, 0, h, w)]
            if verify_python(pairs, ops_py):
                return build_onnx([("color_map", mapping), ("crop_top_left", (out_h, out_w))], in_h, in_w), "colormap_then_crop"
    
    # 2g. Scale_down + color_map
    if all_same_in and all_same_out and out_h < in_h:
        for k in range(2, 6):
            if in_h == out_h * k and in_w == out_w * k:
                # scale_down then color_map
                mapping = {}
                ok = True
                for inp, out in pairs:
                    scaled = py_scale_down(inp, k)
                    for c in range(NUM_COLORS):
                        in_cells = (scaled == c)
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
                if ok and any(k2 != v for k2, v in mapping.items()):
                    ops_py = [lambda g, k=k: py_scale_down(g, k), lambda g, m=mapping: py_color_map(g, m)]
                    if verify_python(pairs, ops_py):
                        return build_onnx([("scale_down", (k,)), ("color_map", mapping)], in_h, in_w), f"scale_down_{k}_then_colormap"
    
    # 2h. Repeat_rows + color_map
    if all_same_in and all_same_out and out_w == in_w and out_h > in_h and out_h % in_h == 0:
        n = out_h // in_h
        if 2 <= n <= 5:
            mapping = {}
            ok = True
            for inp, out in pairs:
                repeated = py_repeat_rows(inp, n)
                for c in range(NUM_COLORS):
                    in_cells = (repeated == c)
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
            if ok and any(k2 != v for k2, v in mapping.items()):
                ops_py = [lambda g, n=n: py_repeat_rows(g, n), lambda g, m=mapping: py_color_map(g, m)]
                if verify_python(pairs, ops_py):
                    return build_onnx([("repeat_rows", (n,)), ("color_map", mapping)], in_h, in_w), f"repeat_rows_{n}_then_colormap"
    
    # 2i. Color_map + repeat_rows
    if all_same_in and all_same_out and out_w == in_w and out_h > in_h and out_h % in_h == 0:
        n = out_h // in_h
        if 2 <= n <= 5:
            mapping = derive_color_map(pairs)
            if mapping and any(k != v for k, v in mapping.items()):
                ops_py = [lambda g, m=mapping: py_color_map(g, m), lambda g, n=n: py_repeat_rows(g, n)]
                if verify_python(pairs, ops_py):
                    return build_onnx([("color_map", mapping), ("repeat_rows", (n,))], in_h, in_w), f"colormap_then_repeat_rows_{n}"
    
    return None, None


def main():
    """Run exhaustive solver on all unsolved tasks."""
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
                model, method = try_exhaustive(task)
                if model is not None:
                    model = _strip_metadata(model)
                    e = validator.evaluate_model(model, task)
                    if e["eligible_for_points"]:
                        zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                        solved += 1
                        score += e["score"]
                        breakdown[method] = breakdown.get(method, 0) + 1
                        print(f"  [OK] task {tid}: {method}, score={e['score']:.2f}")
            except Exception as e:
                pass
    
    elapsed = time.time() - t0
    print(f"\n=== Exhaustive Solver Summary ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")
    return solved, score


if __name__ == "__main__":
    main()
