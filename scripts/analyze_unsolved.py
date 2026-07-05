"""Analyze unsolved tasks and try common Python transforms to find patterns."""
import sys, json
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from neurogolf import arc_data

# Load latest results
with open("/home/z/my-project/data/aggressive_results.json") as f:
    d = json.load(f)
solved_ids = {r["task_id"] for r in d["results"] if r.get("eligible")}
unsolved_ids = [r["task_id"] for r in d["results"] if not r.get("eligible")]
print(f"Solved: {len(solved_ids)}, Unsolved: {len(unsolved_ids)}")

# Candidate Python transforms
def T_identity(x): return x
def T_flip_lr(x): return np.fliplr(x)
def T_flip_ud(x): return np.flipud(x)
def T_rot90(x): return np.rot90(x, 1)
def T_rot180(x): return np.rot90(x, 2)
def T_rot270(x): return np.rot90(x, 3)
def T_transpose(x): return x.T
def T_anti_transpose(x): return np.rot90(np.rot90(x.T, 1), 2)
def T_repeat_2(x): return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)
def T_repeat_3(x): return np.repeat(np.repeat(x, 3, axis=0), 3, axis=1)
def T_subsample_2(x): return x[::2, ::2]
def T_subsample_3(x): return x[::3, ::3]
def T_tile_2(x): return np.tile(x, (2, 2))
def T_tile_3(x): return np.tile(x, (3, 3))
def T_unique_rows(x):
    # Return grid with only unique rows preserved
    seen = set()
    out = []
    for r in x:
        key = tuple(r.tolist())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return np.array(out) if out else np.zeros((0, x.shape[1]), dtype=x.dtype)
def T_unique_cols(x):
    return T_unique_rows(x.T).T
def T_bounding_box(x):
    nz = np.argwhere(x != 0)
    if len(nz) == 0: return x[:0, :0]
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return x[r0:r1, c0:c1]
def T_first_row(x): return x[:1, :]
def T_first_col(x): return x[:, :1]
def T_last_row(x): return x[-1:, :]
def T_last_col(x): return x[:, -1:]
def T_sort_rows(x): return np.sort(x, axis=1)
def T_sort_cols(x): return np.sort(x, axis=0)
def T_sort_rows_desc(x): return -np.sort(-x, axis=1)
def T_to_grayscale_zero(x):
    # Replace all non-zero with 0
    return np.zeros_like(x)
def T_color_0_to_5(x):
    y = x.copy(); y[y == 0] = 5; return y
def T_color_5_to_0(x):
    y = x.copy(); y[y == 5] = 0; return y

TRANSFORMS = [
    ("identity", T_identity),
    ("flip_lr", T_flip_lr),
    ("flip_ud", T_flip_ud),
    ("rot90", T_rot90),
    ("rot180", T_rot180),
    ("rot270", T_rot270),
    ("transpose", T_transpose),
    ("anti_transpose", T_anti_transpose),
    ("repeat_2", T_repeat_2),
    ("repeat_3", T_repeat_3),
    ("subsample_2", T_subsample_2),
    ("subsample_3", T_subsample_3),
    ("tile_2", T_tile_2),
    ("tile_3", T_tile_3),
    ("unique_rows", T_unique_rows),
    ("unique_cols", T_unique_cols),
    ("bounding_box", T_bounding_box),
    ("first_row", T_first_row),
    ("first_col", T_first_col),
    ("last_row", T_last_row),
    ("last_col", T_last_col),
    ("sort_rows", T_sort_rows),
    ("sort_cols", T_sort_cols),
    ("sort_rows_desc", T_sort_rows_desc),
    ("to_grayscale_zero", T_to_grayscale_zero),
    ("color_0_to_5", T_color_0_to_5),
    ("color_5_to_0", T_color_5_to_0),
]

# For each unsolved task, try all transforms and report matches
matched = {}
unmatched_shapes = []
for tid in unsolved_ids:
    try:
        task = arc_data.load_task(tid)
        pairs = arc_data.get_pairs(task)
    except Exception:
        continue
    found = None
    for name, fn in TRANSFORMS:
        try:
            ok = all(np.array_equal(fn(inp), out) for inp, out in pairs)
        except Exception:
            ok = False
        if ok:
            found = name
            break
    if found:
        matched.setdefault(found, []).append(tid)
    else:
        # Record shapes for analysis
        in_shape = pairs[0][0].shape
        out_shape = pairs[0][1].shape
        same_size = all(inp.shape == out.shape for inp, out in pairs)
        unmatched_shapes.append({
            "tid": tid, "in_shape": in_shape, "out_shape": out_shape,
            "same_size": same_size,
            "n_pairs": len(pairs),
        })

print("\n=== Matched by transform ===")
for name, tids in sorted(matched.items(), key=lambda x: -len(x[1])):
    print(f"  {name}: {len(tids)} tasks — first 5: {tids[:5]}")

print(f"\n=== Unmatched tasks: {len(unmatched_shapes)} ===")
print("Shape distribution (top 10):")
from collections import Counter
shape_combos = Counter((u["in_shape"], u["out_shape"], u["same_size"]) for u in unmatched_shapes)
for (in_s, out_s, ss), count in shape_combos.most_common(10):
    print(f"  in={in_s} → out={out_s}, same_size={ss}: {count} tasks")

# Save results
with open("/tmp/unsolved_analysis.json", "w") as f:
    json.dump({
        "matched": matched,
        "unmatched_count": len(unmatched_shapes),
        "unmatched_shapes": unmatched_shapes[:50],
    }, f, indent=2)
print(f"\nTotal matched: {sum(len(v) for v in matched.values())} new solvable tasks!")
print(f"Total unmatched: {len(unmatched_shapes)}")
