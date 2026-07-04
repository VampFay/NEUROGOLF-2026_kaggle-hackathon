"""Analyze all 400 tasks to determine the optimal solving strategy."""
import sys; sys.path.insert(0, '/home/z/my-project')
import numpy as np
from neurogolf import arc_data

single_op = 0
color_map_count = 0
for tid in range(1, 401):
    task = arc_data.load_task(tid)
    pairs = arc_data.get_pairs(task)
    inp, out = pairs[0]
    solved = False
    for name, fn in [
        ("identity", lambda a: a),
        ("fliplr", lambda a: np.fliplr(a)),
        ("flipud", lambda a: np.flipud(a)),
        ("rot90", lambda a: np.rot90(a)),
        ("rot180", lambda a: np.rot90(a, 2)),
        ("rot270", lambda a: np.rot90(a, 3)),
        ("transpose", lambda a: a.T if a.shape[0]==a.shape[1] else None),
    ]:
        try:
            if fn(inp) is not None and all(np.array_equal(fn(p[0]), p[1]) for p in pairs):
                solved = True; break
        except: pass
    if solved:
        single_op += 1; continue
    if inp.shape == out.shape:
        ok = True; mapping = {}
        for inp_p, out_p in pairs:
            for c in range(10):
                in_cells = (inp_p == c)
                if in_cells.any():
                    out_cs = np.unique(out_p[in_cells])
                    if len(out_cs) != 1: ok = False; break
                    t = int(out_cs[0])
                    if c in mapping and mapping[c] != t: ok = False; break
                    mapping[c] = t
            if not ok: break
        if ok and mapping and any(k != v for k, v in mapping.items()):
            color_map_count += 1

print("Tasks solvable by single numpy op: {}/400".format(single_op))
print("Tasks solvable by color map: {}/400".format(color_map_count))
print("Remaining need complex rules: {}/400".format(400 - single_op - color_map_count))
