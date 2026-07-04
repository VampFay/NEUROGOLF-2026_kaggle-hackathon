"""
neurogolf/llm_pipeline.py — LLM-driven per-task solver pipeline.

STRATEGY: "Compile, don't train"
1. For each unsolved task, ask the LLM to write a Python function solve(grid) -> grid
2. Execute the function on all training pairs to verify correctness
3. If correct, analyze the Python operations used
4. Transpile to minimal ONNX (using exploit ops where possible for cost=1 → score 25)

This is the approach top teams use (per souldrive's "Compile, Don't Train" post).
"""
import sys, os, json, subprocess, tempfile, time, re, traceback
import numpy as np

sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data, validator, faithful_scorer

SYSTEM_PROMPT = """You are an ARC-AGI task solver. You write Python functions that transform input grids to output grids.

You will be shown (input, output) pairs from a single ARC task. Write a Python function `solve(grid)` that takes a 2D numpy array (the input grid) and returns a 2D numpy array (the output grid).

CRITICAL RULES:
- The function must work for ALL pairs, not just memorize one
- Use ONLY numpy array operations — NO Python loops, NO list comprehensions, NO if/else on individual cells
- Allowed: np.fliplr, np.flipud, np.rot90, np.tile, np.repeat, np.where, np.zeros_like, np.ones_like, np.concatenate, np.pad, np.roll, np.max, np.min, np.sum, np.equal, np.isin, boolean indexing, arithmetic (+,-,*,/), np.clip, np.unique
- For neighbor-based rules: use np.roll to shift the grid and compare (e.g., count neighbors = (np.roll(grid,1,0)==c).astype(int) + ...)
- For color maps: use np.where(np.isin(grid, [colors]), new_colors, grid) or build a lookup table
- Think about what GENERAL RULE transforms input to output

Return ONLY the Python code in a ```python block. The function must be named `solve` and take one argument (grid as 2D numpy array)."""

USER_TEMPLATE = """ARC task: {filename}

Training pairs (input -> output):
{pairs_text}

Write the `solve(grid)` function. Think step by step about the GENERAL rule, then write numpy-only code (no loops).

```python
import numpy as np

def solve(grid):
    # Your code here
    return output_grid
```"""


def grid_to_ascii(grid, label=""):
    lines = []
    if label: lines.append(label + ":")
    for row in grid:
        lines.append("".join("." if c == 0 else str(c) for c in row))
    return "\n".join(lines)


def pairs_to_text(pairs, max_pairs=4):
    lines = []
    for i, (inp, out) in enumerate(pairs[:max_pairs]):
        lines.append("--- Pair {} ---".format(i+1))
        lines.append(grid_to_ascii(inp.tolist(), "Input"))
        lines.append(grid_to_ascii(out.tolist(), "Output"))
        lines.append("")
    return "\n".join(lines)


def call_llm(system, user, timeout=90):
    """Call z-ai LLM CLI and return response text."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        out_path = f.name
    try:
        result = subprocess.run(
            ["z-ai", "chat", "-p", user[:12000], "-s", system[:8000], "-o", out_path],
            capture_output=True, text=True, timeout=timeout
        )
        with open(out_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ["content", "response", "text", "message"]:
                if key in data: return data[key]
            if "choices" in data:
                return data["choices"][0].get("message", {}).get("content", "")
        elif isinstance(data, str):
            return data
        return None
    except Exception as e:
        return None
    finally:
        try: os.unlink(out_path)
        except: pass


def extract_code(text):
    """Extract Python code from LLM response."""
    if not text: return None
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    if "def solve" in text:
        start = text.index("def solve")
        return text[start:]
    return None


def verify_python_solver(code, pairs):
    """Execute the Python solver and verify it produces correct output for all pairs."""
    try:
        namespace = {"np": np}
        exec(code, namespace)
        if "solve" not in namespace:
            return False, "No solve function found"
        solve_fn = namespace["solve"]
        for i, (inp, out) in enumerate(pairs):
            result = solve_fn(inp.copy())
            if result is None:
                return False, "Pair {}: returned None".format(i)
            result = np.array(result)
            if result.shape != out.shape:
                return False, "Pair {}: shape mismatch {} vs {}".format(i, result.shape, out.shape)
            if not np.array_equal(result, out):
                return False, "Pair {}: output mismatch".format(i)
        return True, "All {} pairs correct".format(len(pairs))
    except Exception as e:
        return False, "Execution error: {}".format(str(e)[:200])


def solve_task_with_llm(task_id, max_retries=2):
    """Try to solve a single task using the LLM pipeline."""
    task = arc_data.load_task(task_id)
    fname = arc_data.task_id_to_filename(task_id)
    pairs = arc_data.get_pairs(task)
    pairs_text = pairs_to_text(pairs, max_pairs=4)
    user_prompt = USER_TEMPLATE.format(filename=fname, pairs_text=pairs_text)

    for attempt in range(max_retries + 1):
        response = call_llm(SYSTEM_PROMPT, user_prompt, timeout=90)
        if not response:
            continue
        code = extract_code(response)
        if not code:
            continue
        ok, msg = verify_python_solver(code, pairs)
        if ok:
            return {"task_id": task_id, "filename": fname, "solved": True,
                    "code": code, "message": msg, "attempts": attempt + 1}
        else:
            # Feed error back for revision
            if attempt < max_retries:
                user_prompt = USER_TEMPLATE.format(filename=fname, pairs_text=pairs_text) + \
                    "\n\nYour previous attempt failed: {}\nPlease fix the code.".format(msg)
    return {"task_id": task_id, "filename": fname, "solved": False, "code": None,
            "message": "Failed after {} attempts".format(max_retries + 1)}


def run_batch(task_ids, output_path="/home/z/my-project/data/llm_solvers.json"):
    """Run LLM pipeline on a batch of tasks."""
    results = []
    solved = 0
    t0 = time.time()

    for tid in task_ids:
        elapsed = time.time() - t0
        print("  Task {} ({:.0f}s elapsed, {}/{} solved)...".format(
            tid, elapsed, solved, len(results)), end="", flush=True)

        result = solve_task_with_llm(tid, max_retries=1)
        results.append(result)

        if result["solved"]:
            solved += 1
            print(" SOLVED (attempt {})".format(result["attempts"]))
            # Save the working code
            with open("/home/z/my-project/data/llm_solver_task{:03d}.py".format(tid), "w") as f:
                f.write(result["code"])
        else:
            print(" FAILED: {}".format(result["message"][:80]))

    # Save all results
    with open(output_path, "w") as f:
        json.dump({"solved": solved, "total": len(task_ids), "results": results}, f, indent=2)

    print("\n=== Batch Summary ===")
    print("Solved: {}/{}".format(solved, len(task_ids)))
    print("Time: {:.1f}s".format(time.time() - t0))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1, help="Start task ID")
    ap.add_argument("--count", type=int, default=10, help="Number of tasks")
    ap.add_argument("--unsolved", action="store_true", help="Only unsolved tasks")
    args = ap.parse_args()

    if args.unsolved:
        with open("/home/z/my-project/data/submission_results.json") as f:
            sub = json.load(f)
        task_ids = [r["task_id"] for r in sub["results"] if not r["eligible"]]
        task_ids = task_ids[args.start-1:args.start-1+args.count]
    else:
        task_ids = list(range(args.start, args.start + args.count))

    print("Running LLM pipeline on {} tasks: {}".format(len(task_ids), task_ids))
    run_batch(task_ids)
