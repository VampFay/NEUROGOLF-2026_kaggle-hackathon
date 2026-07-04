"""Fixed direct solvers v3 — analyzed actual task data."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

def solve_task3(grid):
    result = np.where(grid == 1, 2, grid)
    H, W = grid.shape
    for P in range(1, H):
        if all(np.array_equal(result[i], result[i + P]) for i in range(H - P)):
            break
    else:
        P = H
    out_h = H + H // 2
    return np.array([result[i % P] for i in range(out_h)])

def solve_task7(grid):
    H, W = grid.shape
    nonzero = [(i, j, grid[i, j]) for i in range(H) for j in range(W) if grid[i, j] != 0]
    if not nonzero: return grid
    seen = set(); pattern = []
    for _, j, c in sorted(nonzero, key=lambda x: (x[1], x[0])):
        if c not in seen: pattern.append(int(c)); seen.add(c)
    n = len(pattern); start_col = min(j for _, j, _ in nonzero)
    result = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            result[i, j] = pattern[(j - start_col + i) % n]
    return result

def solve_task13(grid):
    H, W = grid.shape
    col_info = {}
    for j in range(W):
        for i in range(H):
            if grid[i, j] != 0:
                col_info[j] = int(grid[i, j]); break
    if not col_info: return grid
    sorted_cols = sorted(col_info.keys())
    colors_seq = [col_info[c] for c in sorted_cols]
    gap = sorted_cols[1] - sorted_cols[0] if len(sorted_cols) > 1 else 2
    start = sorted_cols[0]; n = len(colors_seq)
    result = np.zeros((H, W), dtype=int)
    for j in range(W):
        if (j - start) % gap == 0 and j >= start:
            result[:, j] = colors_seq[((j - start) // gap) % n]
    return result

def solve_task15(grid):
    result = grid.copy()
    H, W = grid.shape
    frame_map = {2: 4, 1: 7}
    for color, fc in frame_map.items():
        positions = list(zip(*np.where(grid == color)))
        for r, c in positions:
            if color == 2:  # diagonal frame
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                        result[nr, nc] = fc
            elif color == 1:  # orthogonal frame
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                        result[nr, nc] = fc
    return result

def solve_task17(grid):
    H, W = grid.shape
    # Find horizontal period
    for hp in range(1, W + 1):
        ok = True
        for i in range(H):
            for j in range(W):
                if grid[i, j] != 0 and grid[i, j % hp] != 0 and grid[i, j] != grid[i, j % hp]:
                    ok = False; break
            if not ok: break
        if ok: break
    # Find vertical period
    for vp in range(1, H + 1):
        ok = True
        for i in range(H):
            for j in range(W):
                if grid[i, j] != 0 and grid[i % vp, j] != 0 and grid[i, j] != grid[i % vp, j]:
                    ok = False; break
            if not ok: break
        if ok: break
    # Fill missing cells
    result = grid.copy()
    for i in range(H):
        for j in range(W):
            if result[i, j] == 0:
                si, sj = i % vp, j % hp
                if grid[si, sj] != 0:
                    result[i, j] = grid[si, sj]
    return result

def solve_task10(grid):
    # Assign colors to 5-columns by height (tallest=1, shortest=4)
    H, W = grid.shape
    cols_with_5 = sorted(set(j for j in range(W) if (grid[:, j] == 5).any()))
    if not cols_with_5: return grid
    groups = []; current = [cols_with_5[0]]
    for c in cols_with_5[1:]:
        if c == current[-1] + 1: current.append(c)
        else: groups.append(current); current = [c]
    groups.append(current)
    # Sort by height (descending) and assign 1,2,3,4
    heights = [(sum(1 for i in range(H) if grid[i, g[0]] == 5), idx) for idx, g in enumerate(groups)]
    heights.sort(reverse=True)
    color_for_group = {}
    for rank, (_, idx) in enumerate(heights):
        color_for_group[idx] = rank + 1
    result = grid.copy()
    for idx, group in enumerate(groups):
        c = color_for_group[idx]
        for col in group:
            result[grid[:, col] == 5, col] = c
    return result

def verify(name, fn, tid):
    task = arc_data.load_task(tid)
    pairs = arc_data.get_pairs(task)
    for i, (inp, out) in enumerate(pairs):
        try:
            result = np.array(fn(inp.copy()))
            if result.shape != out.shape or not np.array_equal(result, out):
                diffs = int((result != out).sum()) if result.shape == out.shape else -1
                print(f"  {name} task {tid}: pair {i} FAIL ({diffs} diffs)")
                return False
        except Exception as e:
            print(f"  {name} task {tid}: pair {i} ERROR: {e}")
            return False
    print(f"  {name} task {tid}: ALL {len(pairs)} PAIRS OK ✓")
    return True

if __name__ == "__main__":
    solved = 0
    for name, fn, tid in [
        ("task3", solve_task3, 3),
        ("task7", solve_task7, 7),
        ("task10", solve_task10, 10),
        ("task13", solve_task13, 13),
        ("task15", solve_task15, 15),
        ("task17", solve_task17, 17),
    ]:
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/6 solvers verified ===")
