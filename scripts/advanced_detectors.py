"""
Cellular automaton detector — try to learn a 3x3 conv rule from training pairs.

For each possible (X, Y) where X = center color and Y = output color:
  Find all 3x3 neighborhoods where center is X.
  If they all map to the same Y, that's a rule.
  
This handles the 62 "same_size_small_change" tasks that are likely CA-like.
"""
import sys, os, json, time, zipfile
sys.path.insert(0, "/home/z/my-project")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
from neurogolf import arc_data, validator, faithful_scorer, dsl
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model


def try_ca_rule(pairs, max_rules=20):
    """Try to find a CA rule: output[r,c] depends on input[r-1:r+2, c-1:c+2].
    
    Returns ONNX model or None.
    """
    # All input pairs must have same shape
    in_h0, in_w0 = pairs[0][0].shape
    for inp, out in pairs:
        if inp.shape != (in_h0, in_w0) or out.shape != (in_h, in_w0) if (in_h := in_h0) else True:
            return None
        if inp.shape != out.shape:
            return None
    
    # For each cell that changed, learn the rule
    # Rule: (center_color, neighborhood_pattern) → output_color
    # Too many possible neighborhoods (10^9). Use a simpler model:
    # Rule: output[r,c] = f(input[r,c], count of each color in 3x3 neighborhood)
    
    # Even simpler: output[r,c] depends only on input[r,c] (per-cell color map)
    # — already handled by color_permutation
    
    # Try: output[r,c] = f(input[r,c], input[r±1, c±1] == marker_color)
    # This handles "flood fill from marker" type tasks
    
    # For now, just try simple CA: output depends on input[r,c] and 4-neighbors
    # Build a 3x3 conv with learnable weights
    
    # Collect training data: (input 3x3 patch, output center)
    # If we can find a consistent rule, build the conv
    
    patches = []  # (3x3 one-hot input, output color)
    for inp, out in pairs:
        # Pad input with zeros
        padded = np.zeros((in_h0 + 2, in_w0 + 2), dtype=inp.dtype)
        padded[1:-1, 1:-1] = inp
        for r in range(in_h0):
            for c in range(in_w0):
                patch = padded[r:r+3, c:c+3]
                patches.append((patch, int(out[r, c])))
    
    # Group by (center_color, output_color)
    # For each (center, output) pair, check if there's a consistent rule
    # based on the 3x3 patch
    from collections import defaultdict
    center_to_outputs = defaultdict(set)
    for patch, out_c in patches:
        center = int(patch[1, 1])
        center_to_outputs[center].add(out_c)
    
    # If each center maps to exactly one output, it's a pure color map (already handled)
    if all(len(outputs) == 1 for outputs in center_to_outputs.values()):
        return None  # color_permutation handles this
    
    # If each center maps to multiple outputs, we need neighborhood info
    # Try: rule based on (center_color, has_marker_in_neighborhood)
    # Find a marker color that determines the output
    
    for marker_color in range(NUM_COLORS):
        # For each center color, check if output depends on whether marker is in neighborhood
        rules = {}  # (center, has_marker) → output
        consistent = True
        for patch, out_c in patches:
            center = int(patch[1, 1])
            has_marker = (patch == marker_color).any() and int(patch[1, 1]) != marker_color
            # Actually, include marker at center too
            has_marker = (patch == marker_color).any()
            key = (center, has_marker)
            if key in rules:
                if rules[key] != out_c:
                    consistent = False
                    break
            else:
                rules[key] = out_c
        if not consistent:
            continue
        # Check the rule is non-trivial
        if len(rules) < 2:
            continue
        # Verify on all pairs
        valid = True
        for inp, out in pairs:
            padded = np.zeros((in_h0 + 2, in_w0 + 2), dtype=inp.dtype)
            padded[1:-1, 1:-1] = inp
            for r in range(in_h0):
                for c in range(in_w0):
                    patch = padded[r:r+3, c:c+3]
                    center = int(patch[1, 1])
                    has_marker = (patch == marker_color).any()
                    expected = rules.get((center, has_marker), center)
                    if expected != int(out[r, c]):
                        valid = False
                        break
                if not valid: break
        if not valid:
            continue
        # Build a 3x3 conv that detects marker presence + applies color map
        # Conv kernel: for each input channel c, output channel rules[(c, True)]
        #   if marker_color is anywhere in 3x3
        # This is complex — let's build it as: marker_detect_conv + color_map_conv
        # Step 1: Conv with kernel that's 1 everywhere marker_color is, 0 elsewhere
        #         This gives "count of marker in 3x3 neighborhood"
        # Step 2: Greater(0) → has_marker (bool)
        # Step 3: For each cell, if has_marker, use rules[(center, True)], else rules[(center, False)]
        # This requires conditional logic which is complex in ONNX
        
        # Simpler: build a 3x3 conv with weights that produce the correct output channel
        # For output channel o:
        #   weight[o, c, i, j] = 1 if rules[(c, marker_at_(i,j))] == o else 0
        # But this requires knowing where the marker is in the 3x3 patch
        
        # Even simpler: 1x1 conv for "no marker" case + 3x3 conv for "has marker" case
        # Then select based on marker presence
        
        # Actually, let's build it as two parallel branches and select
        # Branch A (no marker): 1x1 conv with color_map for rules[(*, False)]
        # Branch B (has marker): 3x3 conv that outputs rules[(*, True)] when any marker in 3x3
        # Output = where(has_marker, Branch B, Branch A)
        
        # Build:
        # 1. marker_count = Conv(input, marker_kernel) where marker_kernel detects marker_color
        #    Result: (1, 1, H, W) with count of marker in 3x3 neighborhood
        # 2. has_marker = Greater(marker_count, 0)
        # 3. branch_a = Conv(input, no_marker_weights) — 1x1 conv
        # 4. branch_b = Conv(input, has_marker_weights) — 1x1 conv (simplification)
        # 5. output = Where(has_marker, branch_b, branch_a)
        
        # Build marker kernel: shape (1, NUM_COLORS, 3, 3), all 1s in marker_color channel
        marker_kernel = np.zeros((1, NUM_COLORS, 3, 3), dtype=np.float32)
        marker_kernel[0, marker_color, :, :] = 1.0
        
        # Build no-marker weights: 1x1 conv with rules[(*, False)]
        # If (c, False) not in rules, default to c (identity)
        no_marker_W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
        for c in range(NUM_COLORS):
            out_c = rules.get((c, False), c)
            no_marker_W[out_c, c, 0, 0] = 1.0
        
        # Build has-marker weights: 1x1 conv with rules[(*, True)]
        has_marker_W = np.zeros((NUM_COLORS, NUM_COLORS, 1, 1), dtype=np.float32)
        for c in range(NUM_COLORS):
            out_c = rules.get((c, True), c)
            has_marker_W[out_c, c, 0, 0] = 1.0
        
        # Build ONNX
        nodes = []
        initializers = [
            h.make_tensor("mk", TensorProto.FLOAT, [1, NUM_COLORS, 3, 3], marker_kernel.flatten().tolist()),
            h.make_tensor("nmw", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], no_marker_W.flatten().tolist()),
            h.make_tensor("hmw", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], has_marker_W.flatten().tolist()),
        ]
        # marker_count = Conv(input, mk) — pads=1 to keep size
        nodes.append(h.make_node("Conv", [INPUT_NAME, "mk"], ["mc"],
            pads=[1,1,1,1], dilations=[1,1], strides=[1,1], group=1))
        # has_marker = Greater(mc, 0)
        nodes.append(h.make_node("Constant", [], ["z"], value=h.make_tensor("zv", TensorProto.FLOAT, [1], [0.0])))
        nodes.append(h.make_node("Greater", ["mc", "z"], ["hm"]))
        # branch_a = Conv(input, nmw) — 1x1 conv
        nodes.append(h.make_node("Conv", [INPUT_NAME, "nmw"], ["ba"],
            pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
        # branch_b = Conv(input, hmw) — 1x1 conv
        nodes.append(h.make_node("Conv", [INPUT_NAME, "hmw"], ["bb"],
            pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
        # output = Where(has_marker, branch_b, branch_a)
        # has_marker is shape (1, 1, H, W) bool
        # branch_a, branch_b are (1, NUM_COLORS, H, W) float
        # Where broadcasts has_marker to (1, NUM_COLORS, H, W)
        nodes.append(h.make_node("Where", ["hm", "bb", "ba"], [OUTPUT_NAME]))
        
        return _make_model(nodes, initializers=initializers)
    
    return None


def try_marker_color_replace(pairs):
    """Try: cells of color X become color Y when marker color M is present in 3x3 neighborhood."""
    in_h0, in_w0 = pairs[0][0].shape
    for inp, out in pairs:
        if inp.shape != (in_h0, in_w0) or out.shape != (in_h0, in_w0):
            return None
    
    # For each (X, Y, M) combination, check if it's consistent
    # This is O(10*10*10) = 1000 combinations, but fast
    for X in range(NUM_COLORS):
        for Y in range(NUM_COLORS):
            if X == Y: continue
            for M in range(NUM_COLORS):
                if M == X: continue
                # Rule: cells of color X become Y if M is in 3x3 neighborhood
                ok = True
                for inp, out in pairs:
                    padded = np.zeros((in_h0 + 2, in_w0 + 2), dtype=inp.dtype)
                    padded[1:-1, 1:-1] = inp
                    for r in range(in_h0):
                        for c in range(in_w0):
                            if inp[r, c] == X:
                                has_m = (padded[r:r+3, c:c+3] == M).any()
                                expected = Y if has_m else X
                            else:
                                expected = inp[r, c]
                            if expected != out[r, c]:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                if ok:
                    # Build ONNX
                    # marker_count = Conv(input, marker_kernel for M) — pads=1
                    # has_marker = Greater(marker_count, 0)
                    # For each cell: if input == X and has_marker, output Y, else output input
                    # Build via: branch_a (identity) + branch_b (color map X→Y) + Where
                    # But Where needs per-cell condition, not per-channel
                    
                    # Simpler: build output = input + (Y - X) * (input == X) * has_marker
                    # In one-hot: output[X] -= (Y - X) * has_marker
                    #             output[Y] += (Y - X) * has_marker
                    # Actually in one-hot, we set channel X to 0 and channel Y to 1
                    
                    # Build:
                    # 1. marker_count = Conv(input, mk) — pads=1
                    # 2. has_marker = Greater(marker_count, 0) → (1, 1, H, W) bool
                    # 3. input_is_X = Equal(input, X_onehot sum) → (1, 1, H, W) bool
                    # 4. condition = And(has_marker, input_is_X) → (1, 1, H, W) bool
                    # 5. color_map_X_to_Y = Conv(input, cm_W) — 1x1
                    # 6. output = Where(condition, color_map_X_to_Y, input)
                    
                    marker_kernel = np.zeros((1, NUM_COLORS, 3, 3), dtype=np.float32)
                    marker_kernel[0, M, :, :] = 1.0
                    
                    # Color map: X → Y, everything else identity
                    cm_W = np.eye(NUM_COLORS, dtype=np.float32).reshape(NUM_COLORS, NUM_COLORS, 1, 1)
                    cm_W[Y, X] = 1.0
                    cm_W[X, X] = 0.0  # remove identity for X
                    
                    nodes = []
                    initializers = [
                        h.make_tensor("mk", TensorProto.FLOAT, [1, NUM_COLORS, 3, 3], marker_kernel.flatten().tolist()),
                        h.make_tensor("cmw", TensorProto.FLOAT, [NUM_COLORS, NUM_COLORS, 1, 1], cm_W.flatten().tolist()),
                    ]
                    # marker_count = Conv(input, mk) — pads=1
                    nodes.append(h.make_node("Conv", [INPUT_NAME, "mk"], ["mc"],
                        pads=[1,1,1,1], dilations=[1,1], strides=[1,1], group=1))
                    # has_marker = Greater(mc, 0)
                    nodes.append(h.make_node("Constant", [], ["z"], value=h.make_tensor("zv", TensorProto.FLOAT, [1], [0.0])))
                    nodes.append(h.make_node("Greater", ["mc", "z"], ["hm"]))
                    # input_is_X: extract channel X (one-hot value)
                    # Slice input to get channel X
                    nodes.append(h.make_node("Constant", [], ["xs"], value=h.make_tensor("xsv", TensorProto.INT64, [4], [0, X, 0, 0])))
                    nodes.append(h.make_node("Constant", [], ["xe"], value=h.make_tensor("xev", TensorProto.INT64, [4], [1, X+1, MAX_GRID, MAX_GRID])))
                    nodes.append(h.make_node("Constant", [], ["xa"], value=h.make_tensor("xav", TensorProto.INT64, [4], [0, 1, 2, 3])))
                    nodes.append(h.make_node("Slice", [INPUT_NAME, "xs", "xe", "xa"], ["ix"]))
                    # condition = And(hm, ix)
                    nodes.append(h.make_node("And", ["hm", "ix"], ["cond"]))
                    # color_mapped = Conv(input, cmw) — 1x1
                    nodes.append(h.make_node("Conv", [INPUT_NAME, "cmw"], ["cm"],
                        pads=[0,0,0,0], dilations=[1,1], strides=[1,1], group=1))
                    # output = Where(cond, cm, input)
                    nodes.append(h.make_node("Where", ["cond", "cm", INPUT_NAME], [OUTPUT_NAME]))
                    
                    return _make_model(nodes, initializers=initializers)
    return None


DETECTORS = [
    ("ca_marker_rule", try_ca_rule),
    ("marker_color_replace", try_marker_color_replace),
]


def try_advanced_detectors(task):
    pairs = arc_data.get_pairs(task)
    for name, detector in DETECTORS:
        try:
            model = detector(pairs)
        except Exception:
            continue
        if model is None: continue
        e = validator.evaluate_model(model, task)
        if e["eligible_for_points"]:
            return model, name, e["score"]
    return None, None, 0


def main():
    """Run advanced detectors on all unsolved tasks."""
    with open("/home/z/my-project/data/unified_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    # Open existing zip in append mode
    output_path = "/home/z/my-project/download/submission.zip"
    
    newly_solved = 0
    new_score = 0.0
    breakdown = {}
    
    with zipfile.ZipFile(output_path, "a", zipfile.ZIP_DEFLATED) as zf:
        for tid in unsolved:
            try:
                task = arc_data.load_task(tid)
                model, method, score = try_advanced_detectors(task)
                if model is not None:
                    # Strip metadata
                    model.ClearField("producer_name")
                    model.ClearField("producer_version")
                    model.ClearField("doc_string")
                    model.ClearField("domain")
                    model.ClearField("model_version")
                    model.graph.ClearField("doc_string")
                    if len(model.graph.name) > 1:
                        model.graph.name = "g"
                    zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    newly_solved += 1
                    new_score += score
                    breakdown[method] = breakdown.get(method, 0) + 1
                    print(f"  [OK] task {tid}: {method}, score={score:.2f}")
            except Exception as e:
                pass  # Silent failure
    
    print(f"\n=== Advanced Detectors Summary ===")
    print(f"Newly solved: {newly_solved}")
    print(f"New score: {new_score:.2f}")
    print(f"Breakdown: {breakdown}")


if __name__ == "__main__":
    main()
