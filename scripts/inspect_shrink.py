"""Inspect 30 shrink tasks - print all pairs as ASCII art."""
import sys
import numpy as np
sys.path.insert(0, '/home/z/my-project')
from neurogolf import arc_data

TASKS = [6, 14, 21, 22, 26, 29, 31, 36, 38, 39, 46, 49, 57, 65, 72, 79, 88, 91, 104, 112, 113, 121, 127, 131, 145, 148, 153, 155, 157, 160]

# ASCII digits for colors 0-9
def render(arr):
    lines = []
    for row in arr:
        line = ''.join(str(int(c)) if c != 0 else '.' for c in row)
        lines.append(line)
    return lines

def print_pair(inp, out, idx):
    in_lines = render(inp)
    out_lines = render(out)
    print(f"  Pair {idx}: in={inp.shape} out={out.shape}")
    # Print side-by-side
    max_h = max(len(in_lines), len(out_lines))
    for i in range(max_h):
        l = in_lines[i] if i < len(in_lines) else ' ' * len(in_lines[0])
        r = out_lines[i] if i < len(out_lines) else ' ' * len(out_lines[0])
        print(f"    {l}  ->  {r}")
    print()

for tid in TASKS:
    fname = arc_data.task_id_to_filename(tid)
    task = arc_data.load_task(tid)
    pairs = arc_data.get_pairs(task)
    print(f"=== Task {tid} ({fname}) ===  n_pairs={len(pairs)}")
    for i, (inp, out) in enumerate(pairs):
        print_pair(inp, out, i)
    print()
