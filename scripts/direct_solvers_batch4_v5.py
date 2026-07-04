"""Batch 4 v5 — final final fixes."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 4: Top row of each shape shifts right by 1
# Actually: EVERY row shifts right by 1, EXCEPT the bottom row
# Fix: shift ALL cells in non-bottom rows by 1, don't check collision
def solve_task4(grid):
    result = np.zeros_like(grid)
    H, W = grid.shape
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if not positions: continue
        rmax = max(r for r, c in positions)
        for r, c in positions:
            if r == rmax:
                result[r, c] = color
            else:
                result[r, c + 1] = color
    return result

# Task 48: Output the color with the HIGHER count
def solve_task48(grid):
    colors, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(colors) == 0:
        return np.array([[0]])
    winner = int(colors[np.argmax(counts)])
    return np.array([[winner]])

# Task 49: Output the SMALLEST object (by area) as its bounding box
def solve_task49(grid):
    best_color = 0; best_area = 999; best_shape = None
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if not positions: continue
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        area = len(positions)  # count of cells, not bounding box area
        if area < best_area:
            best_area = area; best_color = color
            best_shape = grid[min(rows):max(rows)+1, min(cols):max(cols)+1]
    if best_color == 0 or best_shape is None: return grid
    return best_shape.copy()

# Task 13: Two seeds alternate
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

# === Also include the previously verified solvers ===
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

def solve_task10(grid):
    H, W = grid.shape
    cols_with_5 = sorted(set(j for j in range(W) if (grid[:, j] == 5).any()))
    if not cols_with_5: return grid
    groups = []; current = [cols_with_5[0]]
    for c in cols_with_5[1:]:
        if c == current[-1] + 1: current.append(c)
        else: groups.append(current); current = [c]
    groups.append(current)
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

def solve_task15(grid):
    result = grid.copy()
    H, W = grid.shape
    for color, fc in [(2, 4), (1, 7)]:
        positions = list(zip(*np.where(grid == color)))
        for r, c in positions:
            if color == 2:
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                        result[nr, nc] = fc
            elif color == 1:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                        result[nr, nc] = fc
    return result

def solve_task40(grid):
    result = grid.copy()
    H, W = grid.shape
    left_vals = set(grid[i, 0] for i in range(H) if grid[i, 0] != 0)
    right_vals = set(grid[i, -1] for i in range(H) if grid[i, -1] != 0)
    top_vals = set(grid[0, j] for j in range(W) if grid[0, j] != 0)
    bot_vals = set(grid[-1, j] for j in range(W) if grid[-1, j] != 0)
    left_color = left_vals.pop() if len(left_vals) == 1 else None
    right_color = right_vals.pop() if len(right_vals) == 1 else None
    top_color = top_vals.pop() if len(top_vals) == 1 else None
    bot_color = bot_vals.pop() if len(bot_vals) == 1 else None
    boundary_colors = {c for c in [left_color, right_color, top_color, bot_color] if c is not None}
    if not boundary_colors: return grid
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0 or c in boundary_colors: continue
            if left_color is not None and right_color is not None and left_color != right_color:
                result[i, j] = left_color if j < W/2 else right_color
            elif top_color is not None and bot_color is not None and top_color != bot_color:
                result[i, j] = top_color if i < H/2 else bot_color
    return result

def solve_task45(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        if grid[i, 0] != 0 and grid[i, -1] != 0 and grid[i, 0] == grid[i, -1]:
            result[i, :] = grid[i, 0]
    return result

def solve_task47(grid):
    H, W = grid.shape
    crosses = []
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            cross = np.zeros((H, W), dtype=int)
            cross[i, :] = c; cross[:, j] = c
            crosses.append((c, cross))
    result = grid.copy()
    fill_count = np.zeros((H, W), dtype=int)
    for c, cross in crosses:
        mask = cross != 0
        result[mask] = np.where(fill_count[mask] == 0, c, result[mask])
        fill_count[mask] += 1
    if len(crosses) >= 2:
        overlap = fill_count >= 2
        result[overlap] = 2
    return result

# === VERIFY ALL ===
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
        ("task3", solve_task3, 3),
        ("task4", solve_task4, 4),
        ("task10", solve_task10, 10),
        ("task13", solve_task13, 13),
        ("task15", solve_task15, 15),
        ("task40", solve_task40, 40),
        ("task45", solve_task45, 45),
        ("task47", solve_task47, 47),
        ("task48", solve_task48, 48),
        ("task49", solve_task49, 49),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
    
    if solved > 0:
        import json
        with open("/home/z/my-project/data/verified_solvers.json", "w") as f:
            json.dump({"solved": solved, "total": total}, f)
