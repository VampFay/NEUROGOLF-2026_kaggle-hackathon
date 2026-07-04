"""Fixed direct solvers after analyzing actual task data."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# === Task 3 (017c7c7b) ===
# Rule: Color 1→2, then extend vertically by continuing the periodic pattern.
# The period is the smallest P where row[i] == row[i+P] for all valid i.
def solve_task3(grid):
    result = np.where(grid == 1, 2, grid)
    H, W = grid.shape
    # Find period
    for P in range(1, H):
        if all(np.array_equal(result[i], result[i + P]) for i in range(H - P)):
            break
    else:
        P = H
    # Extend by continuing the pattern
    out_h = H + H // 2  # output is 1.5x input
    extended = np.zeros((out_h, W), dtype=int)
    for i in range(out_h):
        extended[i] = result[i % P] if i < H else result[i % P]
    return extended

# === Task 7 (05269061) ===
# Rule: Each row of the input is a pattern that tiles cyclically to fill the output.
# The first row "283" tiles horizontally. Each subsequent row shifts the pattern.
def solve_task7(grid):
    H, W = grid.shape
    # Extract the pattern from the first row (non-zero prefix)
    row0 = grid[0]
    # Find the pattern: non-zero elements
    nz = np.where(row0 != 0)[0]
    if len(nz) == 0:
        return grid
    # The pattern is the values at those positions
    pattern = row0[nz[0]:nz[-1]+1]
    n = len(pattern)
    # Build output: each row shifts the pattern left by 1
    result = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            result[i, j] = pattern[(j + i) % n]
    return result

# === Task 10 (08ed6ac7) ===
# Rule: Replace color 5 with different colors for each vertical line.
# The colors are assigned based on column position: left to right gets 2,3,1,4
# But need to verify the assignment from ALL pairs.
def solve_task10(grid):
    # Find columns with 5s
    cols_with_5 = sorted(set(j for j in range(grid.shape[1]) if (grid[:, j] == 5).any()))
    if not cols_with_5:
        return grid
    # Group consecutive columns
    groups = []
    current = [cols_with_5[0]]
    for c in cols_with_5[1:]:
        if c == current[-1] + 1:
            current.append(c)
        else:
            groups.append(current)
            current = [c]
    groups.append(current)
    
    # Assign colors: need to determine from training data
    # From pair 0: 4 groups, colors 2,3,1,4
    # The assignment might be: 2,3,1,4 cycling
    # Let me check if it's: for group i, color = [2,3,1,4][i % 4]
    colors = [2, 3, 1, 4]
    result = grid.copy()
    for i, group in enumerate(groups):
        c = colors[i % len(colors)]
        for col in group:
            result[grid[:, col] == 5, col] = c
    return result

# === Task 13 (0a938d79) ===
# Rule: Each non-zero color in the input defines a column. 
# The columns repeat horizontally with a fixed gap.
def solve_task13(grid):
    H, W = grid.shape
    # Find non-zero colors and their column positions
    col_info = {}
    for j in range(W):
        for i in range(H):
            if grid[i, j] != 0:
                col_info[j] = grid[i, j]
                break
    if not col_info:
        return grid
    sorted_cols = sorted(col_info.keys())
    colors_seq = [col_info[c] for c in sorted_cols]
    gap = sorted_cols[1] - sorted_cols[0] if len(sorted_cols) > 1 else 2
    start = sorted_cols[0]
    
    result = np.zeros((H, W), dtype=int)
    n = len(colors_seq)
    for j in range(W):
        if (j - start) % gap == 0 and j >= start:
            idx = ((j - start) // gap) % n
            result[:, j] = colors_seq[idx]
    return result

# === Task 15 (0ca9ddb6) ===
# Rule: Each single-cell marker gets surrounded by a frame.
# Color 2 → diagonal frame of color 4
# Color 1 → orthogonal frame of color 7
# But the frame colors might vary per pair. Need to detect dynamically.
def solve_task15(grid):
    result = grid.copy()
    H, W = grid.shape
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if len(positions) != 1:
            continue
        r, c = positions[0]
        # Check what frame color appears in the output (we don't have output here)
        # Use fixed mapping from pair 0
        frame_map = {2: 4, 1: 7}
        if color not in frame_map:
            continue
        fc = frame_map[color]
        # Diagonal frame for color 2
        if color == 2:
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                    result[nr, nc] = fc
        # Orthogonal frame for color 1
        elif color == 1:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == 0:
                    result[nr, nc] = fc
    return result

# === Task 17 (0dfd9992) ===
# Rule: The grid has a repeating pattern with some cells missing (0).
# Fill missing cells by finding the period and copying from the pattern.
def solve_task17(grid):
    H, W = grid.shape
    # Find horizontal period
    for hp in range(1, W):
        ok = True
        for i in range(H):
            for j in range(W - hp):
                if grid[i, j] != 0 and grid[i, j + hp] != 0 and grid[i, j] != grid[i, j + hp]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            # Verify: every non-zero cell at j matches cell at j % hp
            for i in range(H):
                for j in range(W):
                    if grid[i, j] != 0 and grid[i, j % hp] != 0 and grid[i, j] != grid[i, j % hp]:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            break
    else:
        hp = W
    
    # Find vertical period
    for vp in range(1, H):
        ok = True
        for i in range(H - vp):
            for j in range(W):
                if grid[i, j] != 0 and grid[i + vp, j] != 0 and grid[i, j] != grid[i + vp, j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break
    else:
        vp = H
    
    # Fill missing cells
    result = grid.copy()
    for i in range(H):
        for j in range(W):
            if result[i, j] == 0:
                # Try to find the value from the periodic pattern
                src_i = i % vp
                src_j = j % hp
                if grid[src_i, src_j] != 0:
                    result[i, j] = grid[src_i, src_j]
                else:
                    # Search all positions in the same period
                    for di in range(vp):
                        for dj in range(hp):
                            if grid[(src_i + di) % vp, (src_j + dj) % hp] != 0:
                                result[i, j] = grid[(src_i + di) % vp, (src_j + dj) % hp]
                                break
                        if result[i, j] != 0:
                            break
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
            print(f"  {name} task {tid}: pair {i} ERROR: {e}")
            return False
    print(f"  {name} task {tid}: ALL {len(pairs)} PAIRS OK ✓")
    return True

if __name__ == "__main__":
    for name, fn, tid in [
        ("task3", solve_task3, 3),
        ("task7", solve_task7, 7),
        ("task10", solve_task10, 10),
        ("task13", solve_task13, 13),
        ("task15", solve_task15, 15),
        ("task17", solve_task17, 17),
    ]:
        verify(name, fn, tid)
