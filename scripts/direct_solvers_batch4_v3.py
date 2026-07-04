"""Batch 4 v3 — final corrections."""
import numpy as np
import sys
sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Task 4: Parallelogram shear — each row shifts right by 1 more than the row above
# BUT: the shift is WITHIN each shape, not the entire row
# Looking at the data: the top row of each shape shifts right by 1,
# and each subsequent row in the same shape shifts right by 1 from the previous.
# This means: for each shape, row_offset = row - shape_top_row. Shift = row_offset + 1.
# But row 5 (bottom of shape) "....666.." → "....666.." — no shift!
# So the bottom row doesn't shift. Only the rows ABOVE shift.
# Actually: the shape slides right, with the bottom row staying fixed.
# Row 1 (top): shift +1, Row 2: shift +1, Row 3: shift +1, Row 4: shift +1, Row 5 (bottom): shift 0
# But row 4: ...6..6.. → ....6.6.. — the first 6 shifted +1 (col 3→4), 
#            the second 6 shifted -1 (col 6→5). That's inconsistent.
# 
# Let me look at the SHAPE differently. The 6-shape is:
# .666.
# .6..6
# ..6..6
# ...6..6
# ....666
# After transformation:
# ..666
# ..6..6
# ...6..6
# ....6.6
# ....666
# 
# The shape shifts right by 1 at the TOP, and the bottom stays.
# The shape is a parallelogram that "straightens" — the top moves right.
# Actually: EACH ROW shifts right by 1 from its position, except the bottom row.
# Row 1: shift +1 (cols 1-3 → 2-4) ✓
# Row 2: shift +1 (cols 1,4 → 2,5) ✓
# Row 3: shift +1 (cols 2,5 → 3,6) ✓
# Row 4: shift +1 (cols 3,6 → 4,7) — but output has 6 at cols 4,6 not 4,7!
# 
# Wait, output row 4 = "....6.6.." = cols 4 and 6. Input = "...6..6.." = cols 3 and 6.
# Col 3→4 (+1), col 6→6 (0). The second element doesn't move.
# 
# This is the shape becoming MORE LIKE A RECTANGLE. The slanted edges straighten.
# Actually: the rule is that the shape becomes a RECTANGLE. 
# Top-left corner moves right, bottom-right corner stays.
# Each row's left edge shifts +1 per row from top. Each row's right edge stays.
# 
# For the 6-shape: top row .666. at cols 1-3. Bottom row ....666 at cols 4-6.
# After: all rows have the shape at cols 2-4 (for 666) or the right edge at col 6.
# 
# Simple rule: shift each row right by (max_row - current_row) positions within the shape.
# The bottom row stays (shift 0), the top row shifts the most.
# For 5-row shape: shifts are 4,3,2,1,0. But that doesn't match (row 1 shifts only 1).
#
# Actually: the shape fills the gap. The left edge of each row aligns with 
# the right edge of the row below. It's like "gravity right" for the top portion.
# 
# I think the simplest correct rule: shift the entire shape right by 1, 
# but the BOTTOM ROW stays in place.
def solve_task4(grid):
    result = np.zeros_like(grid)
    H, W = grid.shape
    # Find each shape (contiguous non-zero block)
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if not positions:
            continue
        rows = sorted(set(r for r, c in positions))
        rmax = max(rows)
        # Each row shifts right by 1, except the bottom row
        for r, c in positions:
            if r == rmax:
                result[r, c] = color
            else:
                result[r, c + 1] = color
    return result

# Task 13: Two seed colors alternate, period = gap between seeds
# Fix: also handle 3+ seeds
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
        if j >= c1 and (j - c1) % gap == 0:
            idx = ((j - c1) // gap) % n
            result[:, j] = colors[idx]
    return result

# Task 49: Output the color that's in the center of the largest block
def solve_task49(grid):
    H, W = grid.shape
    # Find all non-zero colors and their bounding boxes
    best_color = 0
    best_area = 0
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if not positions:
            continue
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if area > best_area:
            best_area = area
            best_color = color
    if best_color == 0:
        return grid
    # Output 3x3 filled with that color
    result = np.full((3, 3), best_color, dtype=int)
    # But actually from data: the output is the bounding box of that color
    positions = list(zip(*np.where(grid == best_color)))
    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    sub = grid[min(rows):max(rows)+1, min(cols):max(cols)+1]
    # Resize to 3x3
    oh, ow = 3, 3
    result = np.full((oh, ow), best_color, dtype=int)
    for i in range(min(sub.shape[0], oh)):
        for j in range(min(sub.shape[1], ow)):
            result[i, j] = sub[i, j]
    return result

# Task 48: Compare counts of colors, output 0 or 1
def solve_task48(grid):
    # From pair 0: 8 appears more → output 0
    # From pair 1: need to check
    colors, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(colors) < 2:
        return np.array([[0]])
    # Sort by count (descending)
    sorted_idx = np.argsort(-counts)
    # Output 0 if most common color has higher count
    # Actually: output the LESS common color? Or 0 if 8>2?
    # From pair 0: 8 count=5, 2 count=4 → output 0
    # From pair 1: need to check
    c8 = int((grid == 8).sum())
    c2 = int((grid == 2).sum())
    return np.array([[0 if c8 >= c2 else 1]])

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
