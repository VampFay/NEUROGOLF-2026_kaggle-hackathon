# Task 300 (be94b721)
# LLM-solved in 1 attempt(s)

import numpy as np
from collections import Counter

def solve(input_grid):
    """
    Solves the ARC-AGI task by identifying the primary object in the input grid.

    The task involves identifying the primary object, which is typically the
    most frequent non-zero color in the grid, excluding a potential large background
    color. The output is this primary object cropped to its bounding box.

    Steps:
    1. Convert the input grid to a NumPy array.
    2. Count the frequency of each color, ignoring zeros.
    3. Identify the most frequent color. This is assumed to be the primary object.
    4. If there are no non-zero colors, return an empty grid.
    5. Find the bounding box of all cells with the primary object's color.
    6. Create the output grid by cropping the input to this bounding box and
       preserving only the primary object's color, setting all other colors to 0.
    """
    grid = np.array(input_grid)
    rows, cols = grid.shape

    # Count the frequency of each non-zero color
    color_counts = Counter(grid.flatten())
    del color_counts[0] # Remove the background (0)

    if not color_counts:
        # If the grid is all zeros, return an empty grid
        return [[]]

    # Find the most frequent non-zero color
    primary_color = color_counts.most_common(1)[0][0]

    # Find the bounding box of the primary object
    min_r, max_r = rows, -1
    min_c, max_c = cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == primary_color:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # If the primary object was not found (should not happen if color_counts is not empty)
    if min_r > max_r or min_c > max_c:
        return [[]]

    # Crop the input grid to the bounding box of the primary object
    cropped_grid = grid[min_r:max_r+1, min_c:max_c+1]

    # Create the output grid, setting all non-primary colors to 0
    output_grid = np.zeros_like(cropped_grid)
    output_grid[cropped_grid == primary_color] = primary_color

    return output_grid.tolist()