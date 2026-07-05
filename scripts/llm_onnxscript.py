"""
LLM-powered ONNX synthesis using onnxscript.

Strategy: Ask the LLM to write Python code using onnxscript's @script decorator
that compiles directly to ONNX. The LLM writes Pythonic code with op.Conv, op.Slice,
etc., and onnxscript converts it to a valid ONNX model.

This bypasses the "Python solver → ONNX pattern match" pipeline and lets the LLM
express ONNX operations directly.
"""
import sys, os, json, time, re, subprocess, tempfile
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import numpy as np
import onnx
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed

from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from llm_solve_tasks import call_zai_chat, extract_python_code, grid_to_ascii


def build_onnxscript_prompt(task_id, pairs):
    """Build a prompt asking the LLM to write onnxscript code."""
    fname = arc_data.task_id_to_filename(task_id)
    prompt_parts = []
    prompt_parts.append(f"""You are solving ARC-AGI task {fname} by writing ONNX directly.

The ONNX model takes input shape (1, 10, 30, 30) float32 (one-hot encoded grid, 10 colors 0-9, max 30x30 padded with 0).
It must produce output shape (1, 10, 30, 30) float32 (one-hot encoded).
The validator takes argmax over channels to get the grid, then crops to expected output dimensions.

Use onnxscript to write the model. Here's the template:

```python
import numpy as np
import onnxscript
from onnxscript.onnx_opset import opset17 as op

@onnxscript.script()
def solve(input):
    # input is (1, 10, 30, 30) float32
    # Write your transformation here using op.Conv, op.Slice, op.Transpose, etc.
    # Return (1, 10, 30, 30) float32
    return input  # placeholder

# Build and save model
model = solve.to_model()
```

Available ONNX ops (opset 17):
- op.Conv(x, w, b=None, pads=[...], strides=[...], dilations=[...], group=1)
- op.Slice(x, starts, ends, axes, steps)
- op.Concat(*inputs, axis=N)
- op.Transpose(x, perm=[...])
- op.Reshape(x, shape)
- op.Gather(x, indices, axis=N)
- op.Where(condition, x, y)
- op.Equal(x, y), op.Greater(x, y), op.Less(x, y)
- op.Add, op.Sub, op.Mul, op.Div
- op.Constant(value=tensor)
- op.Identity(x)
- op.Resize(x, roi, scales, mode='nearest')
- op.Pad(x, pads, mode='constant', value=...)
- op.Tile(x, repeats)
- op.ArgMax(x, axis=N)
- op.ReduceSum(x, axes=[...], keepdims=0)
- op.Max, op.Min (element-wise)

Helper: To create a tensor constant, use:
  import numpy as np
  w = np.array([...], dtype=np.float32)
  Then pass w directly to op.Conv as the weight.

For color maps: use a 1x1 Conv where W[to, from, 0, 0] = 1.

Training pairs:
""")
    for i, (inp, out) in enumerate(pairs):
        prompt_parts.append(f"=== PAIR {i+1} ===\n")
        prompt_parts.append(f"INPUT ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_ascii(inp)}\n\n")
        prompt_parts.append(f"OUTPUT ({out.shape[0]}x{out.shape[1]}):\n{grid_to_ascii(out)}\n\n")
    
    prompt_parts.append("""
=== INSTRUCTIONS ===

Output ONLY a Python code block (```python ... ```) that:
1. Defines an `@onnxscript.script()` function called `solve` taking `input` argument
2. Builds the ONNX model via `model = solve.to_model()`
3. The model MUST produce correct output for ALL training pairs

CRITICAL: 
- The input is one-hot encoded (1, 10, 30, 30). Channel c is 1.0 where grid==c, else 0.0.
- The output must also be one-hot encoded.
- For color maps: W[to, from, 0, 0] = 1.0 in a 1x1 conv
- For geometric transforms: use op.Slice with steps for flips, op.Transpose for transpose
- For cropping: use op.Slice to extract the top-left HxW region
- For scaling up: use op.Resize with nearest mode

Analyze the pattern carefully. Respond with ONLY the Python code block.
""")
    return "".join(prompt_parts)


def execute_onnxscript_code(code, input_grid):
    """Execute onnxscript code and return the model + output for given input.
    
    Returns (success, model_or_error, output_or_none).
    """
    try:
        # Create namespace
        ns = {"np": np, "onnxscript": __import__("onnxscript")}
        # Add onnx opset
        from onnxscript.onnx_opset import opset17 as op
        ns["op"] = op
        # Execute the code
        exec(code, ns)
        if "solve" not in ns or "model" not in ns:
            return False, "missing solve or model", None
        model = ns["model"]
        # Run the model on the input
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        from neurogolf.arc_data import grid_to_onehot, onehot_to_grid
        inp_oh = grid_to_onehot(input_grid.tolist() if hasattr(input_grid, 'tolist') else input_grid)
        out = sess.run(None, {"input": inp_oh})[0]
        return True, model, out
    except Exception as e:
        return False, str(e), None


def verify_onnxscript_solution(code, pairs):
    """Verify the onnxscript solution on all pairs."""
    try:
        ns = {"np": np, "onnxscript": __import__("onnxscript")}
        from onnxscript.onnx_opset import opset17 as op
        ns["op"] = op
        exec(code, ns)
        if "model" not in ns:
            return False, f"no model defined", None
        model = ns["model"]
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        from neurogolf.arc_data import grid_to_onehot, onehot_to_grid
        for i, (inp_arr, exp_arr) in enumerate(pairs):
            inp = grid_to_onehot(inp_arr.tolist())
            out = sess.run(None, {"input": inp})[0]
            H, W = exp_arr.shape
            pred_grid = np.array(onehot_to_grid(out, H, W))
            if not np.array_equal(pred_grid, exp_arr):
                return False, f"pair {i}: mismatch", model
        return True, "OK", model
    except Exception as e:
        return False, f"exception: {e}", None


def solve_task_with_onnxscript(task_id, max_attempts=2):
    """Use LLM to write onnxscript code that solves the task."""
    try:
        task = arc_data.load_task(task_id)
        pairs = arc_data.get_pairs(task)
    except Exception as e:
        return {"task_id": task_id, "success": False, "error": f"load failed: {e}"}
    
    last_code = None
    last_error = None
    for attempt in range(max_attempts):
        prompt = build_onnxscript_prompt(task_id, pairs)
        if attempt > 0:
            prompt += f"\n\nPrevious attempt failed: {last_error}\nPlease try a different approach."
        response = call_zai_chat(prompt)
        if not response: continue
        code = extract_python_code(response)
        if not code: continue
        last_code = code
        ok, msg, model = verify_onnxscript_solution(code, pairs)
        if ok and model is not None:
            return {"task_id": task_id, "success": True, "model": model, "code": code, "attempts": attempt + 1}
        last_error = msg
    return {"task_id": task_id, "success": False, "error": last_error or "all attempts failed", "last_code": last_code}


def main():
    """Run onnxscript LLM solver on unsolved tasks."""
    with open("/home/z/my-project/data/final_unified_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    # Process in batches of 20 to avoid rate limits
    BATCH = 20
    MAX_WORKERS = 2
    
    output_path = "/home/z/my-project/download/submission.zip"
    newly_solved = 0
    new_score = 0.0
    breakdown = {}
    
    t0 = time.time()
    
    for batch_start in range(0, len(unsolved), BATCH):
        batch = unsolved[batch_start:batch_start + BATCH]
        print(f"\n=== Batch {batch_start//BATCH + 1}/{(len(unsolved)-1)//BATCH + 1} ===")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(solve_task_with_onnxscript, tid): tid for tid in batch}
            results_batch = []
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"task_id": tid, "success": False, "error": str(e)}
                results_batch.append(result)
            
            # Process results and add successful ones to submission.zip
            with zipfile.ZipFile(output_path, "a", zipfile.ZIP_DEFLATED) as zf:
                for result in results_batch:
                    if result["success"]:
                        model = result["model"]
                        tid = result["task_id"]
                        # Strip metadata
                        model.ClearField("producer_name")
                        model.ClearField("producer_version")
                        model.ClearField("doc_string")
                        model.ClearField("domain")
                        model.ClearField("model_version")
                        model.graph.ClearField("doc_string")
                        if len(model.graph.name) > 1:
                            model.graph.name = "g"
                        # Re-verify after stripping
                        try:
                            task = arc_data.load_task(tid)
                            e = validator.evaluate_model(model, task)
                            if e["eligible_for_points"]:
                                ci = faithful_scorer.compute_cost(model)
                                zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                                newly_solved += 1
                                new_score += e["score"]
                                breakdown["llm_onnxscript"] = breakdown.get("llm_onnxscript", 0) + 1
                                print(f"  [OK] task {tid}: cost={ci.get('cost', 0)}, score={e['score']:.2f}")
                            else:
                                print(f"  [SKIP] task {tid}: not eligible after stripping")
                        except Exception as e:
                            print(f"  [ERR] task {tid}: {e}")
                    else:
                        err = result.get("error", "unknown")[:60]
                        # print(f"  [FAIL] task {result['task_id']}: {err}")
        
        elapsed = time.time() - t0
        print(f"  --- Progress: newly_solved={newly_solved}, elapsed={elapsed:.0f}s ---")
        
        # Stop after 4 minutes to leave time for final validation
        if elapsed > 240:
            print("  Stopping after 4 minutes to leave time for validation.")
            break
    
    print(f"\n=== LLM-onnxscript Stage Summary ===")
    print(f"Newly solved: {newly_solved}")
    print(f"New score: {new_score:.2f}")
    print(f"Breakdown: {breakdown}")
    return newly_solved, new_score


if __name__ == "__main__":
    main()
