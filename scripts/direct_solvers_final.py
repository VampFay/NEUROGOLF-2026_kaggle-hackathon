"""Fixed batch 3 — corrected rules based on actual data analysis."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 47: Cross pattern with intersection=2
def solve_task47(grid):
    H, W = grid.shape
    # Compute crosses separately
    crosses = []
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            cross = np.zeros((H, W), dtype=int)
            cross[i, :] = c
            cross[:, j] = c
            crosses.append((c, cross))
    # Merge: at intersections of different colors, use 2
    result = grid.copy()
    fill_count = np.zeros((H, W), dtype=int)
    for c, cross in crosses:
        mask = cross != 0
        result[mask] = np.where(fill_count[mask] == 0, c, result[mask])
        fill_count[mask] += 1
    # At cells with 2+ fills from different colors, set to 2
    if len(crosses) >= 2:
        overlap = fill_count >= 2
        result[overlap] = 2
    return result

# Task 37: Diagonal lines. Direction: c%3==2 → down-left, else down-right
def solve_task37(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            if c % 3 == 2:  # down-left
                di, dj = 1, -1
            else:  # down-right
                di, dj = 1, 1
            ni, nj = i + di, j + dj
            while 0 <= ni < H and 0 <= nj < W:
                if result[ni, nj] == 0:
                    result[ni, nj] = c
                ni += di; nj += dj
    return result

# Task 40: Markers become nearest boundary color
def solve_task40(grid):
    result = grid.copy()
    H, W = grid.shape
    # Check for horizontal boundaries (left/right columns all same color)
    left_color = None; right_color = None
    if H > 0:
        left_vals = set(grid[i, 0] for i in range(H) if grid[i, 0] != 0)
        right_vals = set(grid[i, -1] for i in range(H) if grid[i, -1] != 0)
        if len(left_vals) == 1: left_color = left_vals.pop()
        if len(right_vals) == 1: right_color = right_vals.pop()
    # Check for vertical boundaries (top/bottom rows all same color)
    top_color = None; bot_color = None
    if W > 0:
        top_vals = set(grid[0, j] for j in range(W) if grid[0, j] != 0)
        bot_vals = set(grid[-1, j] for j in range(W) if grid[-1, j] != 0)
        if len(top_vals) == 1: top_color = top_vals.pop()
        if len(bot_vals) == 1: bot_color = bot_vals.pop()
    
    boundary_colors = {c for c in [left_color, right_color, top_color, bot_color] if c is not None}
    if not boundary_colors: return grid
    
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0 or c in boundary_colors: continue
            # Determine nearest boundary
            if left_color is not None and right_color is not None and left_color != right_color:
                mid = W / 2
                result[i, j] = left_color if j < mid else right_color
            elif top_color is not None and bot_color is not None and top_color != bot_color:
                mid = H / 2
                result[i, j] = top_color if i < mid else bot_color
    return result

# Task 45: Two same-color edge markers → fill row (already working, keep as is)
def solve_task45(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        if grid[i, 0] != 0 and grid[i, -1] != 0 and grid[i, 0] == grid[i, -1]:
            result[i, :] = grid[i, 0]
    return result

# Task 3: Color 1→2 + periodic extension (already working)
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

# Task 10: Color 5 → rank by height (already working)
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

# Task 15: Frame markers (already working)
def solve_task15(grid):
    result = grid.copy()
    H, W = grid.shape
    frame_map = {2: 4, 1: 7}
    for color, fc in frame_map.items():
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
    solved = 0
    total = 0
    for name, fn, tid in [
        ("task3", solve_task3, 3),
        ("task10", solve_task10, 10),
        ("task15", solve_task15, 15),
        ("task37", solve_task37, 37),
        ("task40", solve_task40, 40),
        ("task45", solve_task45, 45),
        ("task47", solve_task47, 47),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
    
    # Save verified solvers
    if solved > 0:
        import json
        with open("/home/z/my-project/data/verified_solvers.json", "w") as f:
            json.dump({"solved": solved, "total": total, 
                       "solvers": [name for name, fn, tid in [
                           ("task3", solve_task3, 3), ("task10", solve_task10, 10),
                           ("task15", solve_task15, 15), ("task37", solve_task37, 37),
                           ("task40", solve_task40, 40), ("task45", solve_task45, 45),
                           ("task47", solve_task47, 47)] if verify(name, fn, tid)]}, f)
