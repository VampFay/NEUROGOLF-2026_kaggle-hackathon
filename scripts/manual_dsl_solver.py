"""Manual DSL solver — I analyze each task and write DSL operations directly."""
import sys, os, json, time, zipfile, numpy as np
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import MAX_GRID, NUM_COLORS
from dsl_transpiler import Transpiler, py_color_map, py_crop, py_pad_to, py_flip_lr, py_flip_ud, py_rot90, py_rot180, py_rot270, py_transpose, py_scale_up, py_scale_down, py_tile, py_repeat_rows, py_repeat_cols

OUTPUT_PATH = "/home/z/my-project/download/submission.zip"

def build_and_verify(ops, in_h, in_w, task, name):
    """Build ONNX from DSL ops, verify, return (model, score) or (None, 0)."""
    try:
        t = Transpiler()
        t.crop_top_left(in_h, in_w)
        for op_name, op_args in ops:
            if op_name == "color_map": t.color_map(op_args)
            elif op_name == "crop_top_left": t.crop_top_left(*op_args)
            elif op_name == "pad_to": t.pad_to(*op_args)
            elif op_name == "flip_lr": t.flip_lr()
            elif op_name == "flip_ud": t.flip_ud()
            elif op_name == "rot90": t.rot90()
            elif op_name == "rot180": t.rot180()
            elif op_name == "rot270": t.rot270()
            elif op_name == "transpose": t.transpose()
            elif op_name == "scale_up": t.scale_up(*op_args)
            elif op_name == "scale_down": t.scale_down(*op_args)
            elif op_name == "tile": t.tile(*op_args)
            elif op_name == "repeat_rows": t.repeat_rows(*op_args)
            elif op_name == "repeat_cols": t.repeat_cols(*op_args)
            elif op_name == "constant_output": t.constant_output(op_args)
        t.pad_to(MAX_GRID, MAX_GRID)
        model = t.build()
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
            return model, e["score"]
    except Exception as e:
        pass
    return None, 0

def verify_py(pairs, ops_py):
    """Verify Python ops on all pairs."""
    try:
        for inp, out in pairs:
            cur = inp.copy()
            for fn in ops_py:
                cur = fn(cur)
            if cur.shape != out.shape: return False
            if not np.array_equal(cur, out): return False
        return True
    except: return False

def try_manual_solutions(task):
    """Try manually identified patterns."""
    pairs = arc_data.get_pairs(task)
    if not pairs: return None, None, 0
    
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    all_same_in = all(inp.shape == (in_h, in_w) for inp, _ in pairs)
    all_same_out = all(out.shape == (out_h, out_w) for _, out in pairs)
    
    # Pattern 1: Output is all zeros
    if same_size:
        all_zero = True
        for inp, out in pairs:
            if not np.all(out == 0):
                all_zero = False; break
        if all_zero:
            model, score = build_and_verify([("constant_output", [[0]*out_w]*out_h)], in_h, in_w, task, "all_zeros")
            if model: return model, "constant_zeros", score
    
    # Pattern 2: Output = input with everything set to 0 except one color
    if same_size:
        for kept in range(NUM_COLORS):
            ok = True
            for inp, out in pairs:
                expected = np.where(inp == kept, kept, 0)
                if not np.array_equal(expected, out):
                    ok = False; break
            if ok:
                mapping = {c: 0 for c in range(NUM_COLORS) if c != kept}
                mapping[kept] = kept
                model, score = build_and_verify([("color_map", mapping)], in_h, in_w, task, f"keep_only_{kept}")
                if model: return model, f"keep_only_{kept}", score
    
    # Pattern 3: Output = input with one color replaced by another, then cropped
    if all_same_out and out_h <= in_h and out_w <= in_w:
        for src in range(NUM_COLORS):
            for dst in range(NUM_COLORS):
                if src == dst: continue
                ops_py = [
                    lambda g, s=src, d=dst: py_color_map(g, {s: d}),
                    lambda g, h=out_h, w=out_w: py_crop(g, 0, 0, h, w)
                ]
                if verify_py(pairs, ops_py):
                    model, score = build_and_verify([("color_map", {src: dst}), ("crop_top_left", (out_h, out_w))], in_h, in_w, task, f"recolor_{src}_to_{dst}_crop")
                    if model: return model, f"recolor_{src}_to_{dst}_crop", score
    
    # Pattern 4: Output = crop + recolor
    if all_same_out and out_h <= in_h and out_w <= in_w:
        for src in range(NUM_COLORS):
            for dst in range(NUM_COLORS):
                if src == dst: continue
                ops_py = [
                    lambda g, h=out_h, w=out_w: py_crop(g, 0, 0, h, w),
                    lambda g, s=src, d=dst: py_color_map(g, {s: d})
                ]
                if verify_py(pairs, ops_py):
                    model, score = build_and_verify([("crop_top_left", (out_h, out_w)), ("color_map", {src: dst})], in_h, in_w, task, f"crop_recolor_{src}_to_{dst}")
                    if model: return model, f"crop_recolor_{src}_to_{dst}", score
    
    # Pattern 5: Scale up + color map (for grow tasks)
    if all_same_in and all_same_out and out_h > in_h:
        for k in range(2, 6):
            if out_h != in_h * k or out_w != in_w * k: continue
            # Try scale_up then color_map
            for src in range(NUM_COLORS):
                for dst in range(NUM_COLORS):
                    if src == dst: continue
                    ops_py = [
                        lambda g, k=k: py_scale_up(g, k),
                        lambda g, s=src, d=dst: py_color_map(g, {s: d})
                    ]
                    if verify_py(pairs, ops_py):
                        model, score = build_and_verify([("scale_up", (k,)), ("color_map", {src: dst})], in_h, in_w, task, f"scale{k}_recolor_{src}_to_{dst}")
                        if model: return model, f"scale{k}_recolor_{src}_to_{dst}", score
            # Try color_map then scale_up
            for src in range(NUM_COLORS):
                for dst in range(NUM_COLORS):
                    if src == dst: continue
                    ops_py = [
                        lambda g, s=src, d=dst: py_color_map(g, {s: d}),
                        lambda g, k=k: py_scale_up(g, k)
                    ]
                    if verify_py(pairs, ops_py):
                        model, score = build_and_verify([("color_map", {src: dst}), ("scale_up", (k,))], in_h, in_w, task, f"recolor_{src}_to_{dst}_scale{k}")
                        if model: return model, f"recolor_{src}_to_{dst}_scale{k}", score
    
    # Pattern 6: Repeat rows + color map
    if all_same_in and all_same_out and out_w == in_w and out_h > in_h and out_h % in_h == 0:
        n = out_h // in_h
        if 2 <= n <= 5:
            # Try repeat_rows then color_map
            for src in range(NUM_COLORS):
                for dst in range(NUM_COLORS):
                    if src == dst: continue
                    ops_py = [
                        lambda g, n=n: py_repeat_rows(g, n),
                        lambda g, s=src, d=dst: py_color_map(g, {s: d})
                    ]
                    if verify_py(pairs, ops_py):
                        model, score = build_and_verify([("repeat_rows", (n,)), ("color_map", {src: dst})], in_h, in_w, task, f"repeat{n}_recolor_{src}_to_{dst}")
                        if model: return model, f"repeat{n}_recolor_{src}_to_{dst}", score
            # Try color_map then repeat_rows
            for src in range(NUM_COLORS):
                for dst in range(NUM_COLORS):
                    if src == dst: continue
                    ops_py = [
                        lambda g, s=src, d=dst: py_color_map(g, {s: d}),
                        lambda g, n=n: py_repeat_rows(g, n)
                    ]
                    if verify_py(pairs, ops_py):
                        model, score = build_and_verify([("color_map", {src: dst}), ("repeat_rows", (n,))], in_h, in_w, task, f"recolor_{src}_to_{dst}_repeat{n}")
                        if model: return model, f"recolor_{src}_to_{dst}_repeat{n}", score

    # Pattern 7: Scale down + color map
    if all_same_in and all_same_out and out_h < in_h:
        for k in range(2, 6):
            if in_h != out_h * k or in_w != out_w * k: continue
            for src in range(NUM_COLORS):
                for dst in range(NUM_COLORS):
                    if src == dst: continue
                    ops_py = [
                        lambda g, k=k: py_scale_down(g, k),
                        lambda g, s=src, d=dst: py_color_map(g, {s: d})
                    ]
                    if verify_py(pairs, ops_py):
                        model, score = build_and_verify([("scale_down", (k,)), ("color_map", {src: dst})], in_h, in_w, task, f"scaledown{k}_recolor_{src}_to_{dst}")
                        if model: return model, f"scaledown{k}_recolor_{src}_to_{dst}", score
    
    # Pattern 8: Multiple color replacements (2 colors changed)
    if same_size:
        for s1 in range(NUM_COLORS):
            for d1 in range(NUM_COLORS):
                if s1 == d1: continue
                for s2 in range(s1+1, NUM_COLORS):
                    for d2 in range(NUM_COLORS):
                        if s2 == d2 or d2 == d1: continue
                        ok = True
                        for inp, out in pairs:
                            modified = inp.copy()
                            modified[modified == s1] = d1
                            modified[modified == s2] = d2
                            if not np.array_equal(modified, out):
                                ok = False; break
                        if ok:
                            mapping = {s1: d1, s2: d2}
                            model, score = build_and_verify([("color_map", mapping)], in_h, in_w, task, f"multi_recolor_{s1}_{d1}_{s2}_{d2}")
                            if model: return model, f"multi_recolor_{s1}_{d1}_{s2}_{d2}", score
    
    # Pattern 9: Dihedral + multi-color replacement
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
            # Try dihedral + 2-color replacement
            for s1 in range(NUM_COLORS):
                for d1 in range(NUM_COLORS):
                    if s1 == d1: continue
                    for s2 in range(s1+1, NUM_COLORS):
                        for d2 in range(NUM_COLORS):
                            if s2 == d2 or d2 == d1: continue
                            ok = True
                            for inp, out in pairs:
                                transformed = dfn(inp)
                                modified = transformed.copy()
                                modified[modified == s1] = d1
                                modified[modified == s2] = d2
                                if not np.array_equal(modified, out):
                                    ok = False; break
                            if ok:
                                ops_py = [dfn, lambda g, m={s1:d1, s2:d2}: py_color_map(g, m)]
                                if verify_py(pairs, ops_py):
                                    ops_dsl = []
                                    if dname != "identity": ops_dsl.append((dname, None))
                                    ops_dsl.append(("color_map", {s1: d1, s2: d2}))
                                    model, score = build_and_verify(ops_dsl, in_h, in_w, task, f"{dname}_multirecolor")
                                    if model: return model, f"{dname}_multirecolor", score
    
    return None, None, 0

def main():
    with open("/home/z/my-project/data/final_comprehensive_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    solved = 0
    score = 0.0
    breakdown = {}
    
    with zipfile.ZipFile(OUTPUT_PATH, "a", zipfile.ZIP_DEFLATED) as zf:
        for tid in unsolved[:60]:  # Try first 60
            try:
                task = arc_data.load_task(tid)
                model, method, sc = try_manual_solutions(task)
                if model is not None:
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    solved += 1
                    score += sc
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={sc:.2f}")
            except Exception:
                pass
    
    print(f"\n=== Manual DSL Summary ===")
    print(f"Newly solved: {solved}")
    print(f"New score: {score:.2f}")
    print(f"Breakdown: {breakdown}")

if __name__ == "__main__":
    main()
