"""
Final validation + optimization pass for the submission.

1. Load submission.zip
2. For each ONNX file:
   a. Validate with onnx.checker
   b. Load with ORT 1.27 (closest to grader's 1.24.4 we have)
   c. Run on all task pairs and verify correctness
   d. Compute faithful cost via onnx-tool
3. Apply post-processing: strip metadata, shorten names
4. Rebuild submission.zip with optimized files
5. Report final score
"""
import sys, os, json, zipfile, io, math
sys.path.insert(0, "/home/z/my-project")
import numpy as np
import onnx
import onnxruntime as ort
import onnx.helper as h
from onnx import TensorProto

from neurogolf import arc_data
from neurogolf.arc_data import grid_to_onehot, onehot_to_grid, get_pairs
from neurogolf import faithful_scorer

SUBMISSION_ZIP = "/home/z/my-project/download/submission.zip"
OPTIMIZED_ZIP = "/home/z/my-project/download/submission_optimized.zip"

def strip_metadata(model):
    """Strip producer_name, model_version, doc_string, etc. to save bytes."""
    model.ClearField("producer_name")
    model.ClearField("producer_version")
    model.ClearField("doc_string")
    model.ClearField("domain")
    model.ClearField("model_version")
    model.graph.ClearField("doc_string")
    if len(model.graph.name) > 1:
        model.graph.name = "g"
    return model

def shorten_tensor_names(model):
    """Shorten tensor names to single characters where possible."""
    # Build name map
    name_map = {}
    counter = 0
    # Keep input/output names as-is (validator needs them)
    reserved = {"input", "output"}
    # Map initializers
    for init in model.graph.initializer:
        if init.name not in reserved and init.name not in name_map:
            name_map[init.name] = f"i{counter}"
            counter += 1
    # Map node outputs
    for node in model.graph.node:
        for i, out in enumerate(node.output):
            if out not in reserved and out not in name_map and out:
                name_map[out] = f"n{counter}"
                counter += 1
    # Apply renaming
    for init in model.graph.initializer:
        if init.name in name_map:
            init.name = name_map[init.name]
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in name_map:
                node.input[i] = name_map[inp]
        for i, out in enumerate(node.output):
            if out in name_map:
                node.output[i] = name_map[out]
    return model

def validate_and_optimize():
    """Main function: validate submission, optimize, rebuild."""
    if not os.path.exists(SUBMISSION_ZIP):
        print(f"ERROR: {SUBMISSION_ZIP} not found")
        return

    print(f"Loading {SUBMISSION_ZIP}...")
    with zipfile.ZipFile(SUBMISSION_ZIP, "r") as zf:
        files = {name: zf.read(name) for name in zf.namelist()}

    print(f"Found {len(files)} ONNX files")
    print()

    results = []
    total_score_original = 0
    total_score_optimized = 0
    total_size_original = 0
    total_size_optimized = 0
    valid_count = 0
    invalid_count = 0

    with zipfile.ZipFile(OPTIMIZED_ZIP, "w", zipfile.ZIP_DEFLATED) as zf_out:
        for fname, fbytes in sorted(files.items()):
            tid = int(fname.replace("task", "").replace(".onnx", ""))
            try:
                task = arc_data.load_task(tid)
                pairs = get_pairs(task)
            except Exception as e:
                print(f"  [SKIP] {fname}: cannot load task {tid}: {e}")
                invalid_count += 1
                continue

            # Load original model
            try:
                model_orig = onnx.load_model_from_string(fbytes)
            except Exception as e:
                print(f"  [FAIL] {fname}: cannot load ONNX: {e}")
                invalid_count += 1
                continue

            # Validate structurally
            try:
                onnx.checker.check_model(model_orig)
            except Exception as e:
                print(f"  [FAIL] {fname}: onnx.checker failed: {e}")
                invalid_count += 1
                continue

            # Run functional check
            try:
                sess = ort.InferenceSession(fbytes, providers=["CPUExecutionProvider"])
            except Exception as e:
                print(f"  [FAIL] {fname}: ORT load failed: {e}")
                invalid_count += 1
                continue

            all_correct = True
            for i, (inp_arr, exp_arr) in enumerate(pairs):
                inp = grid_to_onehot(inp_arr.tolist())
                try:
                    out = sess.run(None, {"input": inp})[0]
                except Exception as e:
                    all_correct = False
                    break
                H, W = exp_arr.shape
                pred_grid = np.array(onehot_to_grid(out, H, W))
                if not np.array_equal(pred_grid, exp_arr):
                    all_correct = False
                    break

            if not all_correct:
                print(f"  [FAIL] {fname}: functional check failed")
                invalid_count += 1
                continue

            # Compute original cost & score
            ci_orig = faithful_scorer.compute_cost(model_orig)
            cost_orig = ci_orig.get("cost", 0)
            score_orig = max(1.0, 25.0 - math.log(cost_orig)) if cost_orig > 0 else 1.0
            size_orig = len(fbytes)

            # Apply optimizations
            model_opt = strip_metadata(model_orig)
            model_opt = shorten_tensor_names(model_opt)
            fbytes_opt = model_opt.SerializeToString()

            # Re-validate optimized model
            try:
                onnx.checker.check_model(model_opt)
                sess_opt = ort.InferenceSession(fbytes_opt, providers=["CPUExecutionProvider"])
                # Quick functional check on first pair
                inp_arr, exp_arr = pairs[0]
                inp = grid_to_onehot(inp_arr.tolist())
                out = sess_opt.run(None, {"input": inp})[0]
                H, W = exp_arr.shape
                pred_grid = np.array(onehot_to_grid(out, H, W))
                if not np.array_equal(pred_grid, exp_arr):
                    print(f"  [WARN] {fname}: optimized model failed functional check, keeping original")
                    fbytes_opt = fbytes
                    model_opt = model_orig
            except Exception as e:
                print(f"  [WARN] {fname}: optimized model failed validation ({e}), keeping original")
                fbytes_opt = fbytes
                model_opt = model_orig

            # Compute optimized cost & score
            ci_opt = faithful_scorer.compute_cost(model_opt)
            cost_opt = ci_opt.get("cost", 0)
            score_opt = max(1.0, 25.0 - math.log(cost_opt)) if cost_opt > 0 else 1.0
            size_opt = len(fbytes_opt)

            # Write to optimized zip
            zf_out.writestr(fname, fbytes_opt)

            total_score_original += score_orig
            total_score_optimized += score_opt
            total_size_original += size_orig
            total_size_optimized += size_opt
            valid_count += 1

            results.append({
                "task_id": tid, "filename": fname,
                "cost_orig": cost_orig, "score_orig": score_orig, "size_orig": size_orig,
                "cost_opt": cost_opt, "score_opt": score_opt, "size_opt": size_opt,
            })

            delta = score_opt - score_orig
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            print(f"  [OK]   {fname}: cost={cost_orig:5d}→{cost_opt:5d}, score={score_orig:.2f}→{score_opt:.2f} ({delta_str}), size={size_orig}→{size_opt}")

    print()
    print(f"=== Final Validation + Optimization Summary ===")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {invalid_count}")
    print(f"Total score (original):   {total_score_original:.2f}")
    print(f"Total score (optimized):  {total_score_optimized:.2f}")
    print(f"Score gain from optimization: +{total_score_optimized - total_score_original:.2f}")
    print(f"Total size (original):    {total_size_original} bytes")
    print(f"Total size (optimized):   {total_size_optimized} bytes")
    print(f"Size reduction: {total_size_original - total_size_optimized} bytes ({100*(1 - total_size_optimized/total_size_original):.1f}%)")
    print(f"Optimized submission: {OPTIMIZED_ZIP}")

    # Save results
    with open("/home/z/my-project/data/final_validation.json", "w") as f:
        json.dump({
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "total_score_original": total_score_original,
            "total_score_optimized": total_score_optimized,
            "total_size_original": total_size_original,
            "total_size_optimized": total_size_optimized,
            "results": results,
        }, f, indent=2)

    # Replace the main submission.zip with the optimized version
    import shutil
    shutil.copy(OPTIMIZED_ZIP, SUBMISSION_ZIP)
    print(f"\n✓ Replaced {SUBMISSION_ZIP} with optimized version")

    return total_score_optimized


if __name__ == "__main__":
    validate_and_optimize()
