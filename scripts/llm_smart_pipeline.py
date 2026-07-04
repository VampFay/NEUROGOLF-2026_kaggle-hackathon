"""
Smart LLM pipeline for solving all 400 ARC tasks.

Strategy:
1. Run LLM calls in PARALLEL (10 at a time) — 400 tasks in ~20 min instead of hours
2. Better prompt: ask LLM to describe the rule FIRST, then write code
3. Verify each Python solver instantly on all pairs
4. Transpile working solvers to ONNX by pattern-matching numpy operations
"""
import sys, os, json, subprocess, tempfile, time, re, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, "/home/z/my-project")
from neurogolf import arc_data

# Output directory for working solvers
SOLVER_DIR = "/home/z/my-project/data/llm_solvers"
os.makedirs(SOLVER_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an ARC-AGI task solver. You write Python functions that transform input grids.

You will see (input, output) pairs from one ARC task. Write `solve(grid)` that takes a 2D numpy array and returns a 2D numpy array.

CRITICAL RULES:
- Must work for ALL pairs (generalize, don't memorize)
- Use ONLY numpy array operations — NO Python loops, NO list comprehensions
- Allowed: np.fliplr, np.flipud, np.rot90, np.tile, np.repeat, np.where, np.zeros_like, np.ones_like, np.concatenate, np.pad, np.roll, np.max, np.min, np.sum, np.equal, np.isin, boolean indexing, arithmetic
- For neighbor rules: use np.roll to shift grid (e.g., neighbors = (np.roll(grid,1,0)==c).astype(int) + ...)
- For color maps: use lookup table (lut = np.arange(10); lut[old]=new; result=lut[grid])
- For flood fill: use iterative np.roll propagation from border

Think step by step:
1. Describe the transformation rule in one sentence
2. Write the code

Return ONLY python code in a markdown code block."""

USER_TEMPLATE = """ARC task: {filename}

{pairs_text}

Step 1 - Rule: (describe in one sentence)
Step 2 - Code:

```python
import numpy as np

def solve(grid):
    # implement the rule
    return result
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
    except:
        return None
    finally:
        try: os.unlink(out_path)
        except: pass


def extract_code(text):
    if not text: return None
    if "```python" in text:
        s = text.index("```python") + len("```python")
        e = text.index("```", s)
        return text[s:e].strip()
    if "```" in text:
        s = text.index("```") + 3
        e = text.index("```", s)
        return text[s:e].strip()
    if "def solve" in text:
        return text[text.index("def solve"):]
    return None


def verify_solver(code, pairs):
    try:
        ns = {"np": np}
        exec(code, ns)
        if "solve" not in ns: return False, "no solve function"
        fn = ns["solve"]
        for i, (inp, out) in enumerate(pairs):
            result = fn(inp.copy())
            if result is None: return False, "pair {} None".format(i)
            result = np.array(result)
            if result.shape != out.shape:
                return False, "pair {} shape {} vs {}".format(i, result.shape, out.shape)
            if not np.array_equal(result, out):
                return False, "pair {} mismatch".format(i)
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


def solve_one_task(task_id):
    """Solve a single task using LLM. Returns dict with results."""
    task = arc_data.load_task(task_id)
    fname = arc_data.task_id_to_filename(task_id)
    pairs = arc_data.get_pairs(task)
    pairs_text = pairs_to_text(pairs, max_pairs=4)
    user_prompt = USER_TEMPLATE.format(filename=fname, pairs_text=pairs_text)

    for attempt in range(2):  # max 2 attempts
        response = call_llm(SYSTEM_PROMPT, user_prompt, timeout=90)
        if not response: continue
        code = extract_code(response)
        if not code: continue
        ok, msg = verify_solver(code, pairs)
        if ok:
            # Save the working solver
            with open(os.path.join(SOLVER_DIR, "task{:03d}.py".format(task_id)), "w") as f:
                f.write(code)
            return {"task_id": task_id, "filename": fname, "solved": True,
                    "attempts": attempt + 1, "message": msg}
        # Revision: feed error back
        if attempt == 0:
            user_prompt = USER_TEMPLATE.format(filename=fname, pairs_text=pairs_text) + \
                "\n\nYour code failed: {}. Fix it.".format(msg)

    return {"task_id": task_id, "filename": fname, "solved": False,
            "attempts": 2, "message": "failed"}


def run_parallel_batch(task_ids, max_workers=10):
    """Run LLM on multiple tasks in parallel."""
    results = []
    solved = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_one_task, tid): tid for tid in task_ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["solved"]:
                    solved += 1
                    print("  OK task {:3d} ({}) attempt {}".format(
                        tid, result["filename"], result["attempts"]))
                else:
                    print("  -- task {:3d} ({}) failed".format(tid, result["filename"]))
            except Exception as e:
                print("  !! task {:3d} error: {}".format(tid, str(e)[:80]))
                results.append({"task_id": tid, "solved": False, "message": str(e)[:80]})

    elapsed = time.time() - t0
    print("\n=== Batch Summary ===")
    print("Solved: {}/{} in {:.1f}s ({:.1f}s/task avg)".format(
        solved, len(task_ids), elapsed, elapsed/max(len(task_ids),1)))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--unsolved", action="store_true")
    args = ap.parse_args()

    if args.unsolved:
        results_path = "/home/z/my-project/data/submission_results.json"
        if os.path.exists(results_path):
            with open(results_path) as f:
                sub = json.load(f)
            task_ids = [r["task_id"] for r in sub["results"] if not r["eligible"]]
        else:
            task_ids = list(range(1, 401))
    else:
        task_ids = list(range(args.start, min(args.start + args.count, 401)))

    print("Running LLM pipeline on {} tasks with {} parallel workers".format(
        len(task_ids), args.workers))
    print("Tasks: {}".format(task_ids[:20]), "..." if len(task_ids) > 20 else "")
    print()

    results = run_parallel_batch(task_ids, max_workers=args.workers)

    # Save results
    with open("/home/z/my-project/data/llm_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    solved_ids = [r["task_id"] for r in results if r["solved"]]
    print("\nSolved task IDs: {}".format(solved_ids))
    print("Total solved: {}/{}".format(len(solved_ids), len(task_ids)))
