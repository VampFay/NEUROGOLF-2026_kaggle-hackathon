"""
LLM-powered DSL solver — ask the LLM to analyze each task and output a
sequence of DSL operations that transpile to ONNX.

The LLM is constrained to output JSON with a list of DSL operations.
We parse the JSON, build the ONNX, and verify.
"""
import sys, os, json, time, re, subprocess
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import numpy as np
import onnx
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed

from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.constants import INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, NUM_COLORS
from neurogolf.exploit_solvers import _make_model
from dsl_transpiler import Transpiler, py_color_map, py_crop, py_pad_to, py_flip_lr, py_flip_ud, py_rot180, py_transpose, py_rot90, py_rot270, py_scale_up, py_scale_down, py_tile, py_repeat_rows, py_repeat_cols


def grid_to_ascii(grid):
    lines = []
    for row in grid:
        lines.append(" ".join(str(int(x)) for x in row))
    return "\n".join(lines)


def build_dsl_prompt(task_id, pairs):
    fname = arc_data.task_id_to_filename(task_id)
    prompt = f"""You are solving ARC-AGI task {fname}.

Analyze the transformation and express it as a sequence of DSL operations.

Available DSL operations (applied in order):
- {{"op": "color_map", "mapping": {{0:1, 1:0}}}} — remap colors (dict of old→new)
- {{"op": "crop", "r0": 0, "c0": 0, "r1": 5, "c1": 5}} — crop to region
- {{"op": "crop_top_left", "h": 5, "w": 5}} — crop to top-left h×w
- {{"op": "pad_to", "h": 10, "w": 10}} — pad with zeros to h×w
- {{"op": "flip_lr"}} — flip horizontally
- {{"op": "flip_ud"}} — flip vertically
- {{"op": "rot90"}} — rotate 90° counterclockwise
- {{"op": "rot180"}} — rotate 180°
- {{"op": "rot270"}} — rotate 270°
- {{"op": "transpose"}} — transpose spatial dims
- {{"op": "scale_up", "k": 2}} — scale up by factor k (nearest)
- {{"op": "scale_down", "k": 2}} — scale down by factor k (subsample)
- {{"op": "tile", "n_h": 2, "n_w": 2}} — tile n_h × n_w
- {{"op": "repeat_rows", "n": 2}} — repeat each row n times
- {{"op": "repeat_cols", "n": 2}} — repeat each col n times
- {{"op": "constant_output", "grid": [[0,1],[2,3]]}} — replace with constant grid

The input is a grid of colors 0-9. Output ONLY a JSON object:
{{"operations": [...], "explanation": "..."}}

Training pairs:

"""
    for i, (inp, out) in enumerate(pairs):
        prompt += f"=== PAIR {i+1} ===\nINPUT ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_ascii(inp)}\n\nOUTPUT ({out.shape[0]}x{out.shape[1]}):\n{grid_to_ascii(out)}\n\n"
    
    prompt += """=== INSTRUCTIONS ===

Analyze the pattern carefully. Common patterns:
- Color substitution: some colors change to other colors
- Geometric: flip, rotate, transpose
- Cropping: extract a sub-region
- Scaling: enlarge or shrink by integer factor
- Tiling: repeat the grid
- Combinations: crop then color_map, color_map then scale_up, etc.

Respond with ONLY the JSON object, no other text. The operations must work for ALL pairs.
"""
    return prompt


def call_zai_chat(prompt, max_retries=2):
    """Call z-ai chat CLI."""
    for attempt in range(max_retries):
        try:
            tmp_file = "/tmp/dsl_prompt.txt"
            with open(tmp_file, "w") as f:
                f.write(prompt)
            output_file = "/tmp/dsl_response.json"
            result = subprocess.run(
                ["z-ai", "chat", "--prompt", open(tmp_file).read(), "--output", output_file],
                capture_output=True, text=True, timeout=120
            )
            if os.path.exists(output_file):
                with open(output_file) as f:
                    data = json.load(f)
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                if "data" in data and "choices" in data["data"]:
                    return data["data"]["choices"][0]["message"]["content"]
                if "content" in data:
                    return data["content"]
                return json.dumps(data)
            if result.returncode == 0:
                output = result.stdout
                json_start = output.find("{")
                if json_start >= 0:
                    json_str = output[json_start:]
                    try:
                        data = json.loads(json_str)
                        if "choices" in data:
                            return data["choices"][0]["message"]["content"]
                        if "data" in data and "choices" in data["data"]:
                            return data["data"]["choices"][0]["message"]["content"]
                    except Exception:
                        pass
                return output
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
    return None


def parse_dsl_response(response):
    """Parse the LLM response to extract the operations list."""
    if not response: return None
    # Strip markdown code blocks
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = response.strip()
    # Try parsing the whole response as JSON
    try:
        return json.loads(response)
    except Exception:
        pass
    # Try to find {"operations": ...} (greedy to match nested braces)
    m = re.search(r'\{"operations".*?\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Try finding any JSON object with "operations"
    start = response.find('{"operations"')
    if start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(start, len(response)):
            if response[i] == '{': depth += 1
            elif response[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[start:i+1])
                    except Exception:
                        break
    return None


def apply_dsl_py(grid, operations):
    """Apply DSL operations to a numpy grid (for verification)."""
    cur = grid.copy()
    for op in operations:
        name = op["op"]
        if name == "color_map":
            cur = py_color_map(cur, {int(k): int(v) for k, v in op["mapping"].items()})
        elif name == "crop":
            cur = py_crop(cur, op["r0"], op["c0"], op["r1"], op["c1"])
        elif name == "crop_top_left":
            cur = py_crop(cur, 0, 0, op["h"], op["w"])
        elif name == "pad_to":
            cur = py_pad_to(cur, op["h"], op["w"])
        elif name == "flip_lr":
            cur = py_flip_lr(cur)
        elif name == "flip_ud":
            cur = py_flip_ud(cur)
        elif name == "rot90":
            cur = py_rot90(cur)
        elif name == "rot180":
            cur = py_rot180(cur)
        elif name == "rot270":
            cur = py_rot270(cur)
        elif name == "transpose":
            cur = py_transpose(cur)
        elif name == "scale_up":
            cur = py_scale_up(cur, op["k"])
        elif name == "scale_down":
            cur = py_scale_down(cur, op["k"])
        elif name == "tile":
            cur = py_tile(cur, op["n_h"], op["n_w"])
        elif name == "repeat_rows":
            cur = py_repeat_rows(cur, op["n"])
        elif name == "repeat_cols":
            cur = py_repeat_cols(cur, op["n"])
        elif name == "constant_output":
            cur = np.array(op["grid"], dtype=grid.dtype)
        else:
            raise ValueError(f"Unknown op: {name}")
    return cur


def build_dsl_onnx(operations, in_h, in_w):
    """Build ONNX model from DSL operations."""
    t = Transpiler()
    # If first op needs cropped input, crop first
    # Actually, the Transpiler starts with full (1,10,30,30) input
    # We need to crop to the actual input size first
    t.crop_top_left(in_h, in_w)
    for op in operations:
        name = op["op"]
        if name == "color_map":
            t.color_map({int(k): int(v) for k, v in op["mapping"].items()})
        elif name == "crop":
            t.crop(op["r0"], op["c0"], op["r1"], op["c1"])
        elif name == "crop_top_left":
            t.crop_top_left(op["h"], op["w"])
        elif name == "pad_to":
            t.pad_to(op["h"], op["w"])
        elif name == "flip_lr":
            t.flip_lr()
        elif name == "flip_ud":
            t.flip_ud()
        elif name == "rot90":
            t.rot90()
        elif name == "rot180":
            t.rot180()
        elif name == "rot270":
            t.rot270()
        elif name == "transpose":
            t.transpose()
        elif name == "scale_up":
            t.scale_up(op["k"])
        elif name == "scale_down":
            t.scale_down(op["k"])
        elif name == "tile":
            t.tile(op["n_h"], op["n_w"])
        elif name == "repeat_rows":
            t.repeat_rows(op["n"])
        elif name == "repeat_cols":
            t.repeat_cols(op["n"])
        elif name == "constant_output":
            t.constant_output(op["grid"])
        else:
            raise ValueError(f"Unknown op: {name}")
    # Pad to MAX_GRID at the end
    t.pad_to(MAX_GRID, MAX_GRID)
    return t.build()


def solve_task_dsl(task_id, max_attempts=2):
    """Use LLM to write DSL operations for a task."""
    try:
        task = arc_data.load_task(task_id)
        pairs = arc_data.get_pairs(task)
    except Exception as e:
        return {"task_id": task_id, "success": False, "error": f"load failed: {e}"}
    
    in_h, in_w = pairs[0][0].shape
    
    for attempt in range(max_attempts):
        prompt = build_dsl_prompt(task_id, pairs)
        if attempt > 0:
            prompt += f"\n\nPrevious attempt failed. Try a different approach."
        response = call_zai_chat(prompt)
        if not response: continue
        parsed = parse_dsl_response(response)
        if not parsed or "operations" not in parsed: continue
        operations = parsed["operations"]
        if not operations: continue
        # Verify on all pairs
        try:
            ok = True
            for inp, out in pairs:
                result = apply_dsl_py(inp, operations)
                if not np.array_equal(result, out):
                    ok = False
                    break
            if ok:
                # Build ONNX
                model = build_dsl_onnx(operations, in_h, in_w)
                return {"task_id": task_id, "success": True, "model": model, "operations": operations}
        except Exception as e:
            continue
    return {"task_id": task_id, "success": False, "error": "all attempts failed"}


def main():
    """Run DSL solver on unsolved tasks."""
    with open("/home/z/my-project/data/final_unified_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved: {len(unsolved)}")
    
    # Process sequentially with small delay to avoid rate limits
    BATCH = 50  # Process 50 tasks per run
    to_solve = unsolved[:BATCH]
    print(f"Processing first {len(to_solve)} tasks")
    
    import zipfile
    output_path = "/home/z/my-project/download/submission.zip"
    newly_solved = 0
    new_score = 0.0
    breakdown = {}
    
    t0 = time.time()
    for i, tid in enumerate(to_solve):
        result = solve_task_dsl(tid, max_attempts=1)
        if result["success"]:
            model = result["model"]
            try:
                # Strip metadata
                model.ClearField("producer_name")
                model.ClearField("producer_version")
                model.ClearField("doc_string")
                model.ClearField("domain")
                model.ClearField("model_version")
                model.graph.ClearField("doc_string")
                if len(model.graph.name) > 1:
                    model.graph.name = "g"
                task = arc_data.load_task(tid)
                e = validator.evaluate_model(model, task)
                if e["eligible_for_points"]:
                    with zipfile.ZipFile(output_path, "a", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr(f"task{tid:03d}.onnx", model.SerializeToString())
                    newly_solved += 1
                    new_score += e["score"]
                    ops = result["operations"]
                    first_op = ops[0]["op"] if ops else "unknown"
                    breakdown[first_op] = breakdown.get(first_op, 0) + 1
                    print(f"  [{i+1}/{len(to_solve)}] task {tid}: OK (first_op={first_op}, score={e['score']:.2f})")
                else:
                    print(f"  [{i+1}/{len(to_solve)}] task {tid}: ineligible")
            except Exception as e:
                print(f"  [{i+1}/{len(to_solve)}] task {tid}: error: {e}")
        else:
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{len(to_solve)}] task {tid}: failed")
        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    --- {i+1}/{len(to_solve)} done, solved={newly_solved}, elapsed={elapsed:.0f}s ---")
    
    elapsed = time.time() - t0
    print(f"\n=== DSL Solver Summary ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Newly solved: {newly_solved}")
    print(f"New score: {new_score:.2f}")
    print(f"Breakdown: {breakdown}")
    return newly_solved, new_score


if __name__ == "__main__":
    main()
