"""Batch 4 v4 — final fixes."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 4: Shift left edge right by 1, keep right edge in place
def solve_task4(grid):
    result = np.zeros_like(grid)
    H, W = grid.shape
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if not positions: continue
        rows = sorted(set(r for r, c in positions))
        rmax = max(rows)
        for r, c in positions:
            if r == rmax:
                result[r, c] = color
            else:
                # Shift right by 1, but only if the cell to the right is empty in the original
                if c + 1 < W and grid[r, c + 1] == 0:
                    result[r, c + 1] = color
                else:
                    result[r, c] = color
    return result

# Task 48: Output 0 if count(2) >= count(8), else 1
def solve_task48(grid):
    c8 = int((grid == 8).sum())
    c2 = int((grid == 2).sum())
    return np.array([[0 if c2 >= c8 else 1]])

# Task 49: Output the color of the SMALLEST non-zero object
def solve_task49(grid):
    best_color = 0; best_area = 999
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if not positions: continue
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if area < best_area:
            best_area = area; best_color = color
    if best_color == 0: return grid
    return np.full((3, 3), best_color, dtype=int)

# Task 13: Two seeds alternate with period=gap (fixed for 3+ seeds)
def solve_task13(grid):
    H, W = grid.shape
    seeds = {}
    for j in range(W):
        for i in range(H):
            if grid[i, j] != 0:
                seeds[j] = int(grid[i, j]); break
    if len(seeds) < 2: return grid
    sorted_cols = sorted(seeds.keys())
    c1 = sorted_cols[0]
    gap = sorted_cols[1] - sorted_cols[0]
    colors = [seeds[c] for c in sorted_cols]
    n = len(colors)
    result = np.zeros((H, W), dtype=int)
    for j in range(W):
        if j >= c1 and (j - c1) % gap == 0:
            idx = ((j - c1) // gap) % n
            result[:, j] = colors[idx]
    return result

# === VERIFY ===
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
            print(f"  {name} task {tid}: pair {i} ERROR: {str(e)[:80]}")
            return False
    print(f"  {name} task {tid}: ALL {len(pairs)} PAIRS OK ✓")
    return True

if __name__ == "__main__":
    solved = 0; total = 0
    for name, fn, tid in [
        ("task4", solve_task4, 4),
        ("task13", solve_task13, 13),
        ("task48", solve_task48, 48),
        ("task49", solve_task49, 49),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
