"""Batch 4 FIXED — corrected rules from actual data."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 4: Shift right by 1, but NOT a simple roll — each shape shifts independently
# From data: row 4 has "....6..6.." → "....6.6.." — the SECOND 6 stays, first shifts
# Actually: it's a shift right by 1 of the ENTIRE grid, but only for certain shapes
# Looking more carefully: .666. → ..666, .6..6. → ..6..6., ..6..6. → ...6..6.
# This IS a shift right by 1! But row 4: ...6..6.. → ....6.6.. — wait, the second 6 moves LEFT?
# No: ...6..6.. (cols 3,6) → ....6.6.. (cols 4,5) — the gap closed!
# So it's NOT a simple shift. Let me re-examine:
# The shape is a parallelogram. Each row shifts right by 1 MORE than the previous row.
# Row 1: cols 1-3 → cols 2-4 (shift 1)
# Row 2: cols 1,4 → cols 2,5 (shift 1)
# Row 3: cols 2,5 → cols 3,6 (shift 1)
# Row 4: cols 3,6 → cols 4,5 (shift 1 for col3→4, but col6→5 is shift -1!)
# That doesn't work either. Let me look at the SHAPE:
# The 6-shape is a parallelogram that shifts right by 1. But row 4's second 6
# in input is at col 6, in output at col 5. That's shift LEFT by 1.
# 
# Actually, looking again at output row 4: "....6.6.." = cols 4 and 6.
# Input row 4: "...6..6.." = cols 3 and 6.
# So col 3→4 (shift +1), col 6→6 (no shift). That's NOT a uniform shift.
#
# The real rule: the shape slides right until it touches the right edge of the
# previous row's shape. It's a "gravity" slide.
# Actually the simplest interpretation: shift ALL non-zero right by 1 column.
# Let me check row 4 again: input cols 3,6 → output cols 4,6?
# No, output is "....6.6.." = positions 4,6. So col3→4 (+1), col6→6 (0). Not uniform.
#
# Let me just try: shift right by 1, but don't move cells that would collide
def solve_task4(grid):
    result = np.zeros_like(grid)
    H, W = grid.shape
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0:
                nj = j + 1
                if nj < W:
                    result[i, nj] = grid[i, j]
                else:
                    result[i, j] = grid[i, j]  # keep in place if at edge
    return result

# Task 13: Two seed colors alternate with period = gap between them
# col 5 has color 2, col 7 has color 8. gap = 2.
# Pattern: 2 at cols 5,9,13,17,21 and 8 at cols 7,11,15,19,23
# So: color1 at c1 + k*gap for even k, color2 at c1 + k*gap for odd k
def solve_task13(grid):
    H, W = grid.shape
    seeds = {}
    for j in range(W):
        for i in range(H):
            if grid[i, j] != 0:
                seeds[j] = int(grid[i, j])
                break
    if len(seeds) < 2:
        return grid
    sorted_cols = sorted(seeds.keys())
    c1 = sorted_cols[0]
    gap = sorted_cols[1] - sorted_cols[0]
    colors = [seeds[c] for c in sorted_cols]
    n = len(colors)
    result = np.zeros((H, W), dtype=int)
    for j in range(W):
        if (j - c1) % gap == 0 and j >= c1:
            idx = ((j - c1) // gap) % n
            result[:, j] = colors[idx]
    return result

# Task 49: Output = the 3x3 block of the color that appears in the center of the input
# From data: color 8 is in the center, output is all 8s in 3x3
def solve_task49(grid):
    H, W = grid.shape
    # Find the color at the center of the grid
    center = grid[H//2, W//2]
    if center == 0:
        # Find the most common non-zero color
        colors, counts = np.unique(grid[grid != 0], return_counts=True)
        if len(colors) == 0:
            return grid
        center = int(colors[np.argmax(counts)])
    return np.full((3, 3), int(center), dtype=int)

# Task 48: Output 0 if count of color 8 > count of color 2, else 1
def solve_task48(grid):
    count_8 = int((grid == 8).sum())
    count_2 = int((grid == 2).sum())
    return np.array([[0 if count_8 > count_2 else 1]])

# Task 17: Fill periodic pattern (fixed vertical period detection)
def solve_task17(grid):
    H, W = grid.shape
    # Find horizontal period
    for hp in range(1, W + 1):
        ok = True
        for i in range(H):
            for j in range(W):
                if grid[i, j] != 0:
                    src_j = j % hp
                    if grid[i, src_j] != 0 and grid[i, j] != grid[i, src_j]:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            break
    else:
        hp = W
    # Find vertical period
    for vp in range(1, H + 1):
        ok = True
        for i in range(H):
            for j in range(W):
                if grid[i, j] != 0:
                    src_i = i % vp
                    if grid[src_i, j] != 0 and grid[i, j] != grid[src_i, j]:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            break
    else:
        vp = H
    # Fill
    result = grid.copy()
    for i in range(H):
        for j in range(W):
            if result[i, j] == 0:
                si, sj = i % vp, j % hp
                if grid[si, sj] != 0:
                    result[i, j] = grid[si, sj]
    return result

# Task 19: Scale 2x with checkerboard fill
# From data: input 2x4, output 4x8. 
# Input: .... / .5.. → Output: 8.8.8.8. / .5...5.. / ........ / .5.8.5.8
# The 5 is at (1,1) in input. In output: (1,1) and (1,4) — repeated horizontally
# Color 8 fills alternating cells
def solve_task19(grid):
    H, W = grid.shape
    out_h, out_w = H * 2, W * 2
    result = np.zeros((out_h, out_w), dtype=int)
    # Fill with checkerboard of 8
    for i in range(out_h):
        for j in range(out_w):
            if (i + j) % 2 == 1:
                result[i, j] = 8
    # Place scaled input (each cell → 2x2 block, but only at even positions)
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0:
                result[2*i, 2*j] = grid[i, j]
                result[2*i, 2*j+1] = grid[i, j]
                result[2*i+1, 2*j] = grid[i, j]
                result[2*i+1, 2*j+1] = grid[i, j]
    return result

# Task 7: Circulant tiling (fixed)
def solve_task7(grid):
    H, W = grid.shape
    nonzero = [(i, j, int(grid[i, j])) for i in range(H) for j in range(W) if grid[i, j] != 0]
    if not nonzero:
        return grid
    # Extract unique colors in order of anti-diagonal appearance
    seen = set()
    pattern = []
    for i, j, c in sorted(nonzero, key=lambda x: (x[0] + x[1], x[1])):
        if c not in seen:
            pattern.append(c)
            seen.add(c)
    n = len(pattern)
    if n == 0:
        return grid
    start_col = min(j for _, j, _ in nonzero)
    result = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            result[i, j] = pattern[(j - start_col + i) % n]
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
        ("task7", solve_task7, 7),
        ("task13", solve_task13, 13),
        ("task17", solve_task17, 17),
        ("task19", solve_task19, 19),
        ("task48", solve_task48, 48),
        ("task49", solve_task49, 49),
    ]:
        total += 1
        if verify(name, fn, tid): solved += 1
    print(f"\n=== {solved}/{total} solvers verified ===")
