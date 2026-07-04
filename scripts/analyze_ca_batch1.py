"""Analyze 30 CA tasks batch 1 — print pairs as ASCII art for each."""
import sys
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from neurogolf import arc_data

TARGETS = [4, 5, 7, 9, 11, 12, 17, 18, 33, 34, 36, 37, 38, 39, 55, 70, 71, 89, 102, 136, 154, 162, 165, 182, 191, 208, 230, 278, 293, 314]

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def grid_ascii(arr):
    """Convert 2D int array to ASCII art string."""
    lines = []
    for row in arr:
        line = "".join(CHARS[int(v) % len(CHARS)] for v in row)
        lines.append(line)
    return "\n".join(lines)


def analyze_task(tid):
    fname = arc_data.task_id_to_filename(tid)
    task = arc_data.load_task(tid)
    train_pairs = arc_data.get_train_pairs(task)
    test_pairs = arc_data.get_test_pairs(task)

    print(f"\n{'='*70}")
    print(f"TASK {tid} ({fname})")
    print(f"{'='*70}")
    print(f"Train pairs: {len(train_pairs)}  Test pairs: {len(test_pairs)}")

    # Signature
    sig = arc_data.task_signature(task)
    print(f"In sizes:  {sig['in_sizes']}")
    print(f"Out sizes: {sig['out_sizes']}")
    print(f"Same size: {sig['all_same_size']}  In==Out: {sig['in_eq_out_all']}")
    print(f"In colors:  {sig['all_in_colors']}")
    print(f"Out colors: {sig['all_out_colors']}")

    for i, (inp, out) in enumerate(train_pairs):
        print(f"\n--- Train pair {i} ---")
        # Print side-by-side
        inp_lines = grid_ascii(inp).split("\n")
        out_lines = grid_ascii(out).split("\n")
        max_lines = max(len(inp_lines), len(out_lines))
        print(f"  INPUT ({inp.shape})            OUTPUT ({out.shape})")
        for j in range(max_lines):
            l = inp_lines[j] if j < len(inp_lines) else ""
            r = out_lines[j] if j < len(out_lines) else ""
            print(f"  {l:<30s}  {r}")

    for i, (inp, out) in enumerate(test_pairs):
        print(f"\n--- Test pair {i} ---")
        inp_lines = grid_ascii(inp).split("\n")
        out_lines = grid_ascii(out).split("\n")
        max_lines = max(len(inp_lines), len(out_lines))
        print(f"  INPUT ({inp.shape})            OUTPUT ({out.shape})")
        for j in range(max_lines):
            l = inp_lines[j] if j < len(inp_lines) else ""
            r = out_lines[j] if j < len(out_lines) else ""
            print(f"  {l:<30s}  {r}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tids = [int(x) for x in sys.argv[1:]]
    else:
        tids = TARGETS
    for tid in tids:
        analyze_task(tid)
