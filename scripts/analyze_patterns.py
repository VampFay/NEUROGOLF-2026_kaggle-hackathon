"""More aggressive pattern detection — try color maps, combinations, etc."""
import sys, json
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from itertools import permutations, product
from neurogolf import arc_data
from collections import Counter

# Load latest results
with open("/home/z/my-project/data/aggressive_results.json") as f:
    d = json.load(f)
solved_ids = {r["task_id"] for r in d["results"] if r.get("eligible")}
unsolved_ids = [r["task_id"] for r in d["results"] if not r.get("eligible")]
print(f"Unsolved: {len(unsolved_ids)}")

# Try ALL 10! = 3.6M permutations is too many. But we can derive the permutation
# from one pair, then verify on others.
def derive_color_map(pairs):
    """If pairs differ only by a color permutation, derive it."""
    mapping = {}
    for inp, out in pairs:
        if inp.shape != out.shape: return None
        for c in range(10):
            in_cells = (inp == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1: return None
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t: return None
            mapping[c] = t
    return mapping

def derive_color_map_with_geom(pairs, geom_fn):
    """Try: output = geom_fn(input) then color_map."""
    mapping = {}
    for inp, out in pairs:
        if inp.shape != out.shape: return None
        transformed = geom_fn(inp)
        if transformed.shape != out.shape: return None
        for c in range(10):
            in_cells = (transformed == c)
            if not in_cells.any(): continue
            out_at = out[in_cells]
            out_colors = np.unique(out_at)
            if len(out_colors) != 1: return None
            t = int(out_colors[0])
            if c in mapping and mapping[c] != t: return None
            mapping[c] = t
    return mapping

def is_constant_output(pairs):
    """Check if output is the same constant grid for all pairs."""
    if len(pairs) < 1: return None
    first = pairs[0][1]
    for inp, out in pairs:
        if out.shape != first.shape: return None
        if not np.array_equal(out, first): return None
    return first

# More transforms to try (with color_map on top)
GEOM_TRANSFORMS = [
    ("identity", lambda x: x),
    ("flip_lr", np.fliplr),
    ("flip_ud", np.flipud),
    ("rot90", lambda x: np.rot90(x, 1)),
    ("rot180", lambda x: np.rot90(x, 2)),
    ("rot270", lambda x: np.rot90(x, 3)),
    ("transpose", lambda x: x.T),
    ("anti_transpose", lambda x: np.rot90(np.rot90(x.T, 1), 2)),
]

# Stats
matched_summary = Counter()
matched_tasks = {}

for tid in unsolved_ids:
    try:
        task = arc_data.load_task(tid)
        pairs = arc_data.get_pairs(task)
    except Exception:
        continue
    found_pattern = None

    # 1. Pure color map (any permutation, including non-bijective)
    mapping = derive_color_map(pairs)
    if mapping is not None and any(k != v for k, v in mapping.items()):
        # Is it a bijection?
        is_bijective = set(mapping.keys()) == set(mapping.values())
        found_pattern = f"color_map_{'bij' if is_bijective else 'nonbij'}"

    # 2. Geometric transform + color map
    if found_pattern is None:
        for gname, gfn in GEOM_TRANSFORMS:
            try:
                mapping = derive_color_map_with_geom(pairs, gfn)
            except Exception:
                continue
            if mapping is not None and any(k != v for k, v in mapping.items()):
                found_pattern = f"geom_{gname}_+colormap"
                break

    # 3. Constant output
    if found_pattern is None:
        const = is_constant_output(pairs)
        if const is not None:
            unique_colors = np.unique(const)
            if len(unique_colors) == 1:
                found_pattern = f"const_out_c{int(unique_colors[0])}"
            else:
                found_pattern = f"const_out_multi"

    # 4. Output = bounding box of nonzero
    if found_pattern is None:
        try:
            ok = True
            for inp, out in pairs:
                nz = np.argwhere(inp != 0)
                if len(nz) == 0:
                    ok = False; break
                r0, c0 = nz.min(axis=0)
                r1, c1 = nz.max(axis=0) + 1
                if not np.array_equal(inp[r0:r1, c0:c1], out):
                    ok = False; break
            if ok:
                found_pattern = "bbox_nonzero"
        except Exception:
            pass

    if found_pattern:
        matched_summary[found_pattern] += 1
        matched_tasks.setdefault(found_pattern, []).append(tid)

print("\n=== Matched patterns ===")
total_new = 0
for pat, cnt in matched_summary.most_common():
    print(f"  {pat}: {cnt} tasks — first 5: {matched_tasks[pat][:5]}")
    total_new += cnt

print(f"\nTotal newly matched: {total_new}")
print(f"Still unmatched: {len(unsolved_ids) - total_new}")

# Save
with open("/tmp/pattern_analysis.json", "w") as f:
    json.dump({
        "matched_summary": dict(matched_summary),
        "matched_tasks": matched_tasks,
    }, f, indent=2)
