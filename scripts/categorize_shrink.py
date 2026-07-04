"""Categorize 30 shrink tasks by checking various extraction patterns."""
import sys
import numpy as np
sys.path.insert(0, '/home/z/my-project')
from neurogolf import arc_data

TASKS = [6, 14, 21, 22, 26, 29, 31, 36, 38, 39, 46, 49, 57, 65, 72, 79, 88, 91, 104, 112, 113, 121, 127, 131, 145, 148, 153, 155, 157, 160]

def find_separator_rows(arr):
    """Return list of indices of all-same-nonzero-color rows."""
    seps = []
    for i in range(arr.shape[0]):
        row = arr[i]
        if (row == row[0]).all() and row[0] != 0:
            seps.append((i, int(row[0])))
    return seps

def find_separator_cols(arr):
    seps = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if (col == col[0]).all() and col[0] != 0:
            seps.append((j, int(col[0])))
    return seps

def find_zero_rows(arr):
    """Indices of all-zero rows."""
    return [i for i in range(arr.shape[0]) if (arr[i] == 0).all()]

def find_zero_cols(arr):
    return [j for j in range(arr.shape[1]) if (arr[:, j] == 0).all()]

def check_slice(inp, out):
    """Check if out = inp[a:b, c:d] for some a,b,c,d."""
    H_in, W_in = inp.shape
    H_out, W_out = out.shape
    for a in range(H_in - H_out + 1):
        for b in range(a + H_out, H_in + 1):
            if b - a != H_out: continue
            for c in range(W_in - W_out + 1):
                for d in range(c + W_out, W_in + 1):
                    if d - c != W_out: continue
                    if np.array_equal(inp[a:b, c:d], out):
                        return (a, b, c, d)
    return None

def check_slice_after_colormap(inp, out):
    """Check if out = colormap(inp[a:b, c:d]) for some slice and color map."""
    H_in, W_in = inp.shape
    H_out, W_out = out.shape
    for a in range(H_in - H_out + 1):
        for c in range(W_in - W_out + 1):
            sub = inp[a:a+H_out, c:c+W_out]
            if sub.shape != out.shape: continue
            # Try to find color map
            m = {}
            valid = True
            for c_col in range(10):
                in_cells = (sub == c_col)
                if in_cells.any():
                    out_colors = np.unique(out[in_cells])
                    if len(out_colors) != 1:
                        valid = False; break
                    m[c_col] = int(out_colors[0])
            if valid:
                return (a, c, m)
    return None

def check_subsample(inp, out):
    """Check if out[i,j] = inp[k*i + a, k*j + b] for some k, a, b."""
    H_in, W_in = inp.shape
    H_out, W_out = out.shape
    if H_out == 0 or W_out == 0: return None
    # Try various strides
    for kh in range(1, H_in + 1):
        for kw in range(1, W_in + 1):
            if (H_in - 1) // kh + 1 != H_out: continue
            if (W_in - 1) // kw + 1 != W_out: continue
            for ah in range(kh):
                for aw in range(kw):
                    ok = True
                    for i in range(H_out):
                        for j in range(W_out):
                            r = ah + i * kh
                            c = aw + j * kw
                            if r >= H_in or c >= W_in or inp[r, c] != out[i, j]:
                                ok = False; break
                        if not ok: break
                    if ok:
                        return (kh, kw, ah, aw)
    return None

def check_color_filter_pack(inp, out):
    """Check if out = pack cells of color c to top-left."""
    H_in, W_in = inp.shape
    H_out, W_out = out.shape
    for c in range(1, 10):
        cells = (inp == c)
        n = cells.sum()
        if n == H_out * W_out:
            # Pack in row-major order
            packed = np.full((H_out, W_out), 0, dtype=inp.dtype)
            coords = np.argwhere(cells)
            for k, (r, cc) in enumerate(coords):
                packed[k // W_out, k % W_out] = c
                if out[k // W_out, k % W_out] != c:
                    break
            else:
                if np.array_equal(packed, out):
                    return (c, 'row-major')
            # Pack in col-major
            packed = np.full((H_out, W_out), 0, dtype=inp.dtype)
            for k, (r, cc) in enumerate(coords):
                packed[k % H_out, k // H_out] = c
                if out[k % H_out, k // H_out] != c:
                    break
            else:
                if np.array_equal(packed, out):
                    return (c, 'col-major')
    return None

def check_count(inp, out):
    """Check if out is a count of something."""
    # Hard to verify without more analysis
    return None

for tid in TASKS:
    fname = arc_data.task_id_to_filename(tid)
    task = arc_data.load_task(tid)
    pairs = arc_data.get_pairs(task)
    print(f"=== Task {tid} ({fname}) ===")
    
    # Check separator extraction
    sep_info = []
    for inp, out in pairs:
        sep_rows = find_separator_rows(inp)
        sep_cols = find_separator_cols(inp)
        zero_rows = find_zero_rows(inp)
        zero_cols = find_zero_cols(inp)
        sep_info.append((sep_rows, sep_cols, zero_rows, zero_cols))
        if sep_rows or sep_cols:
            print(f"  pair: in={inp.shape} out={out.shape} sep_rows={sep_rows[:3]} sep_cols={sep_cols[:3]} zero_rows={len(zero_rows)} zero_cols={len(zero_cols)}")
    
    # Check slice
    slice_results = []
    for inp, out in pairs:
        s = check_slice(inp, out)
        slice_results.append(s)
    if all(s is not None for s in slice_results):
        print(f"  SLICE: {slice_results}")
        continue
    
    # Check slice + colormap
    sc_results = []
    for inp, out in pairs:
        s = check_slice_after_colormap(inp, out)
        sc_results.append(s)
    if all(s is not None for s in sc_results):
        print(f"  SLICE+COLORMAP: {sc_results}")
        continue
    
    # Check subsample
    ss_results = []
    for inp, out in pairs:
        s = check_subsample(inp, out)
        ss_results.append(s)
    if all(s is not None for s in ss_results):
        print(f"  SUBSAMPLE: {ss_results}")
        continue
    
    # Check color filter pack
    cf_results = []
    for inp, out in pairs:
        s = check_color_filter_pack(inp, out)
        cf_results.append(s)
    if all(s is not None for s in cf_results):
        print(f"  COLOR_FILTER_PACK: {cf_results}")
        continue
    
    # No simple pattern
    print(f"  NO_SIMPLE_PATTERN: slices={slice_results}, sc={sc_results}, ss={ss_results[:2]}, cf={cf_results}")
