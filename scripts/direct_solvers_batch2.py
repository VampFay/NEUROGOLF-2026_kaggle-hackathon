"""Direct solvers batch 2 — analyzed from task data."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 20: Pattern repeats every 4 cells — fill missing repetitions
def solve_task20(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            if result[i, j] == 0:
                for period in range(1, W):
                    if j >= period and grid[i, j - period] != 0:
                        result[i, j] = grid[i, j - period]
                        break
                    if j + period < W and grid[i, j + period] != 0:
                        result[i, j] = grid[i, j + period]
                        break
    return result

# Task 24: Diagonal cell → fill column, off-diagonal → fill row
def solve_task24(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0:
                c = grid[i, j]
                if i == j:  # diagonal → fill column
                    result[:, j] = c
                else:  # off-diagonal → fill row
                    result[i, :] = c
    return result

# Task 40: Markers become nearest boundary color (left=1, right=2)
def solve_task40(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find boundary colors
    left_color = None
    right_color = None
    for i in range(H):
        if grid[i, 0] != 0:
            left_color = grid[i, 0]
        if grid[i, -1] != 0:
            right_color = grid[i, -1]
    if left_color is None or right_color is None:
        return grid
    mid = W // 2
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0 and grid[i, j] != left_color and grid[i, j] != right_color:
                if j < mid:
                    result[i, j] = left_color
                else:
                    result[i, j] = right_color
    return result

# Task 45: Two same-color edge markers → fill row
def solve_task45(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        if grid[i, 0] != 0 and grid[i, -1] != 0 and grid[i, 0] == grid[i, -1]:
            result[i, :] = grid[i, 0]
    return result

# Task 37: Each marker draws a diagonal line
# Color 2 → down-left, colors 4,6 → down-right
def solve_task37(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            if c == 2:  # down-left
                di, dj = 1, -1
            elif c in [4, 6]:  # down-right
                di, dj = 1, 1
            else:
                continue
            ni, nj = i + di, j + dj
            while 0 <= ni < H and 0 <= nj < W:
                if result[ni, nj] == 0:
                    result[ni, nj] = c
                ni += di
                nj += dj
    return result

# Task 30: Keep only overlapping rows of blocks
def solve_task30(grid):
    H, W = grid.shape
    # Find rows that have non-zero cells
    nonzero_rows = [i for i in range(H) if (grid[i] != 0).any()]
    if not nonzero_rows:
        return grid
    # Find the intersection: rows where ALL block types appear
    # Group consecutive nonzero rows
    result = np.zeros_like(grid)
    # Find the row range where blocks overlap
    # For each color, find its row range
    color_ranges = {}
    for c in range(1, 10):
        rows = [i for i in range(H) if (grid[i] == c).any()]
        if rows:
            color_ranges[c] = (min(rows), max(rows))
    if len(color_ranges) < 2:
        return grid
    # Find overlapping range
    max_start = max(r[0] for r in color_ranges.values())
    min_end = min(r[1] for r in color_ranges.values())
    if max_start > min_end:
        return grid
    # Keep only cells in the overlapping range
    for c, (rmin, rmax) in color_ranges.items():
        for i in range(max_start, min_end + 1):
            for j in range(W):
                if grid[i, j] == c:
                    result[i, j] = c
        # Also shift: place the block in the overlapping range
        # The block shape from original rows
        orig_rows = [i for i in range(H) if (grid[i] == c).any()]
        if orig_rows:
            block = np.array([grid[i] for i in orig_rows])
            for idx, target_row in enumerate(range(max_start, min_end + 1)):
                if idx < len(block):
                    for j in range(W):
                        if block[idx, j] == c:
                            result[target_row, j] = c
    return result

# Task 34: Draw 3-wide diagonal stripe from marker going up-right
def solve_task34(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find the marker (color 4 and 2 adjacent)
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 4:
                # Draw 3-wide diagonal going up-right
                for di in range(H):
                    r = i - di
                    c = j + di
                    if 0 <= r < H and 0 <= c < W:
                        for dc in range(3):
                            cc = c - 1 + dc
                            if 0 <= cc < W and result[r, cc] == 0:
                                result[r, cc] = 4
                    elif r < 0 or c >= W:
                        break
                return result
    return result

# Task 47: Cross pattern — marker fills entire row AND column
def solve_task47(grid):
    result = grid.copy()
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            c = grid[i, j]
            if c == 0: continue
            # Fill row and column
            result[i, :] = np.where(result[i, :] == 0, c, result[i, :])
            result[:, j] = np.where(result[:, j] == 0, c, result[:, j])
    return result

# Task 41: Fill triangle between diagonal lines
def solve_task41(grid):
    result = grid.copy()
    H, W = grid.shape
    # Find 3s forming a triangle
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 3:
                # Check if this is part of a diagonal
                pass
    # Simple approach: fill cells between pairs of 3s on the same row
    for i in range(H):
        cols_with_3 = [j for j in range(W) if grid[i, j] == 3]
        if len(cols_with_3) >= 2:
            for j1, j2 in zip(cols_with_3[:-1], cols_with_3[1:]):
                result[i, j1:j2+1] = 3
    return result

# Task 43: Copy pattern from row 0 to other rows containing a marker
def solve_task43(grid):
    result = grid.copy()
    H, W = grid.shape
    # Row 0 has the pattern
    pattern_row = grid[0]
    # Find rows with a marker at the last column
    for i in range(1, H):
        if grid[i, -1] != 0:
            marker = grid[i, -1]
            # Copy non-zero cells from row 0, replacing the marker color
            for j in range(W):
                if pattern_row[j] != 0:
                    result[i, j] = pattern_row[j]
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
    solved = 0
    for name, fn, tid in [
        ("task20", solve_task20, 20),
        ("task24", solve_task24, 24),
        ("task40", solve_task40, 40),
        ("task45", solve_task45, 45),
        ("task37", solve_task37, 37),
        ("task30", solve_task30, 30),
        ("task34", solve_task34, 34),
        ("task47", solve_task47, 47),
        ("task41", solve_task41, 41),
        ("task43", solve_task43, 43),
    ]:
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/10 solvers verified ===")
