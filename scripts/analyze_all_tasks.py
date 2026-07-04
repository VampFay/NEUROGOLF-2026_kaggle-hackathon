import sys; sys.path.insert(0, '/home/z/my-project')
import json, numpy as np
from neurogolf import arc_data

with open('/home/z/my-project/data/submission_results.json') as f:
    sub = json.load(f)
solved_set = {r['task_id'] for r in sub['results'] if r['eligible']}

results = []
for tid in range(1, 401):
    task = arc_data.load_task(tid)
    fname = arc_data.task_id_to_filename(tid)
    pairs = arc_data.get_pairs(task)
    inp, out = pairs[0]
    in_h, in_w = inp.shape
    out_h, out_w = out.shape
    same_size = inp.shape == out.shape
    if same_size:
        diff = (inp != out)
        n_diff = int(diff.sum())
        pct = n_diff / inp.size * 100
    else:
        n_diff = -1
        pct = -1
    in_colors = sorted(set(int(c) for c in np.unique(inp)))
    out_colors = sorted(set(int(c) for c in np.unique(out)))
    pattern = "UNKNOWN"
    if same_size and n_diff == 0:
        pattern = "identity"
    elif same_size and pct < 30:
        is_cm = True
        mapping = {}
        for inp_p, out_p in pairs:
            for c in range(10):
                in_cells = (inp_p == c)
                if in_cells.any():
                    out_cs = np.unique(out_p[in_cells])
                    if len(out_cs) != 1:
                        is_cm = False; break
                    t = int(out_cs[0])
                    if c in mapping and mapping[c] != t:
                        is_cm = False; break
                    mapping[c] = t
            if not is_cm: break
        if is_cm and mapping and any(k != v for k, v in mapping.items()):
            is_perm = set(mapping.keys()) == set(mapping.values())
            pattern = "color_" + ("perm" if is_perm else "map")
        else:
            pattern = "CA_or_conditional"
    elif same_size and pct < 70:
        pattern = "complex_rearrange"
    elif same_size:
        pattern = "major_rearrange"
    else:
        ratio = (out_h * out_w) / (in_h * in_w)
        if ratio > 1.5:
            kh = out_h / in_h if in_h > 0 else 0
            kw = out_w / in_w if in_w > 0 else 0
            if kh == kw and kh.is_integer() and kh > 1:
                pattern = "scale_" + str(int(kh)) + "x"
            elif kh.is_integer() and kw.is_integer() and kh > 1 and kw > 1:
                pattern = "scale_" + str(int(kh)) + "x" + str(int(kw))
            else:
                pattern = "grow_other"
        elif ratio < 0.67:
            pattern = "shrink"
        else:
            pattern = "resize_other"
    solved = tid in solved_set
    results.append({
        'tid': tid, 'fname': fname, 'pattern': pattern,
        'in_shape': str(in_h) + "x" + str(in_w), 'out_shape': str(out_h) + "x" + str(out_w),
        'diff_pct': round(pct, 1) if pct >= 0 else -1,
        'in_colors': in_colors, 'out_colors': out_colors,
        'solved': solved,
    })

print("=== ALL 400 TASKS ===\n")
print("{:>4} {:<12} {:<22} {:>8} {:>8} {:>6} {:>8}".format("TID", "Filename", "Pattern", "In", "Out", "Diff%", "Status"))
print("-" * 80)
for r in results:
    status = "SOLVED" if r['solved'] else "FAIL"
    print("{:>4} {:<12} {:<22} {:>8} {:>8} {:>6} {:>8}".format(
        r['tid'], r['fname'], r['pattern'], r['in_shape'], r['out_shape'], r['diff_pct'], status))

with open('/home/z/my-project/data/task_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

from collections import Counter
pattens = Counter(r['pattern'] for r in results)
print("\n=== Pattern distribution ===")
for p, c in pattens.most_common():
    solved_count = sum(1 for r in results if r['pattern'] == p and r['solved'])
    print("  {:<22} {:>4} total, {:>3} solved, {:>3} unsolved".format(p, c, solved_count, c-solved_count))
