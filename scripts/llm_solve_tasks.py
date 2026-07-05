"""
LLM-based ARC-AGI solver using z-ai CLI.

Strategy:
1. For each unsolved task, build a structured prompt with:
   - The transformation description
   - All training pairs as ASCII grids
   - Instruction to write a Python solve(grid) function
2. Use z-ai chat to get a Python solve(grid) function
3. Verify the function on all training pairs
4. If verified, save it for ONNX transpilation
"""
import sys, os, json, subprocess, time, re, traceback
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from neurogolf import arc_data

OUTPUT_DIR = "/home/z/my-project/data/llm_solvers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def grid_to_ascii(grid):
    """Convert grid (numpy 2D) to compact ASCII representation."""
    lines = []
    for row in grid:
        lines.append(" ".join(str(int(x)) for x in row))
    return "\n".join(lines)

def build_prompt(task_id, pairs):
    """Build a comprehensive prompt for the LLM."""
    fname = arc_data.task_id_to_filename(task_id)
    prompt_parts = []
    prompt_parts.append(f"""You are solving ARC-AGI task {fname}.

ARC-AGI tasks are grid transformations. You are given input/output grid pairs (colors 0-9).
Your job: write a Python function `solve(input_grid)` that takes a 2D list-of-lists of ints (0-9) and returns the transformed 2D list-of-lists of ints.

The function must work for ALL training pairs AND generalize to held-out test pairs.

Here are the training pairs for this task:

""")
    for i, (inp, out) in enumerate(pairs):
        prompt_parts.append(f"=== TRAINING PAIR {i+1} ===\n")
        prompt_parts.append(f"INPUT ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_ascii(inp)}\n\n")
        prompt_parts.append(f"OUTPUT ({out.shape[0]}x{out.shape[1]}):\n{grid_to_ascii(out)}\n\n")
    
    prompt_parts.append("""
=== INSTRUCTIONS ===

Output ONLY a Python code block (```python ... ```) containing:
1. Helper functions if needed
2. A `solve(input_grid)` function that returns the output grid (as list-of-lists of ints)

Rules:
- input_grid is list[list[int]] (0-9 colors)
- Return list[list[int]] (0-9 colors), same shape as the rule requires
- Do NOT include any code that runs on import (no print, no file I/O, no test calls)
- Use ONLY standard Python + numpy (import numpy as np)
- The function MUST be deterministic and pure (no randomness)
- Analyze the pattern carefully before writing code

Look at the patterns: count colors, find objects, look for symmetry, check if size changes, etc.

Respond with ONLY the Python code block, no other text.
""")
    return "".join(prompt_parts)

def call_zai_chat(prompt, max_retries=3):
    """Call z-ai chat CLI with the given prompt."""
    for attempt in range(max_retries):
        try:
            # Write prompt to a temp file to avoid shell escaping issues
            tmp_file = "/tmp/llm_prompt.txt"
            with open(tmp_file, "w") as f:
                f.write(prompt)
            # Use --output flag to get clean JSON
            output_file = "/tmp/llm_response.json"
            result = subprocess.run(
                ["z-ai", "chat", "--prompt", open(tmp_file).read(), "--thinking", "--output", output_file],
                capture_output=True, text=True, timeout=180
            )
            if os.path.exists(output_file):
                with open(output_file) as f:
                    data = json.load(f)
                # Try multiple response formats
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                if "data" in data and "choices" in data["data"]:
                    return data["data"]["choices"][0]["message"]["content"]
                if "content" in data:
                    return data["content"]
                # Fallback: return raw JSON string
                return json.dumps(data)
            # Fallback to stdout
            if result.returncode == 0:
                output = result.stdout
                # Try to extract JSON from stdout (might have SDK init lines)
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
            else:
                print(f"    attempt {attempt+1} failed: {result.stderr[:200]}")
                time.sleep(2 * (attempt + 1))
        except subprocess.TimeoutExpired:
            print(f"    attempt {attempt+1} timed out")
        except Exception as e:
            print(f"    attempt {attempt+1} error: {e}")
            time.sleep(2 * (attempt + 1))
    return None

def extract_python_code(response):
    """Extract Python code from LLM response."""
    if not response:
        return None
    # Try ```python ... ``` first
    m = re.search(r"```python\s*\n(.*?)\n```", response, re.DOTALL)
    if m:
        return m.group(1)
    # Try ``` ... ```
    m = re.search(r"```\s*\n(.*?)\n```", response, re.DOTALL)
    if m:
        return m.group(1)
    # Try to find code starting with "def solve" or "import"
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith("import ") or line.startswith("def solve") or line.startswith("from "):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)
    return None

def verify_solver(code, pairs):
    """Verify the solver code on all training pairs.
    Returns (success, error_message)."""
    try:
        # Create a namespace for execution
        ns = {"np": np}
        exec(code, ns)
        if "solve" not in ns:
            return False, "no solve() function defined"
        solve = ns["solve"]
        for i, (inp, out) in enumerate(pairs):
            inp_list = inp.tolist()
            result = solve(inp_list)
            # Convert result to numpy
            if isinstance(result, list):
                result_arr = np.array(result, dtype=np.int64)
            else:
                result_arr = np.array(result, dtype=np.int64)
            expected = out.astype(np.int64)
            if result_arr.shape != expected.shape:
                return False, f"pair {i}: shape mismatch {result_arr.shape} vs {expected.shape}"
            if not np.array_equal(result_arr, expected):
                return False, f"pair {i}: value mismatch"
        return True, "OK"
    except Exception as e:
        return False, f"exception: {e}"

def build_revision_prompt(task_id, pairs, previous_code, error_msg):
    """Build a prompt that asks the LLM to revise a failed solution."""
    fname = arc_data.task_id_to_filename(task_id)
    prompt_parts = []
    prompt_parts.append(f"""You are solving ARC-AGI task {fname}.

Your previous solution FAILED with this error: {error_msg}

Here are the training pairs:

""")
    for i, (inp, out) in enumerate(pairs):
        prompt_parts.append(f"=== TRAINING PAIR {i+1} ===\n")
        prompt_parts.append(f"INPUT ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_ascii(inp)}\n\n")
        prompt_parts.append(f"OUTPUT ({out.shape[0]}x{out.shape[1]}):\n{grid_to_ascii(out)}\n\n")
    
    prompt_parts.append(f"""
=== YOUR PREVIOUS (FAILED) SOLUTION ===
```python
{previous_code}
```

=== INSTRUCTIONS ===

The previous solution failed. Analyze WHY it failed by tracing through each pair.

Output ONLY a Python code block (```python ... ```) containing:
1. A CORRECTED `solve(input_grid)` function
2. Helper functions if needed

Rules:
- input_grid is list[list[int]] (0-9 colors)
- Return list[list[int]] (0-9 colors)
- Use ONLY standard Python + numpy (import numpy as np)
- Must work for ALL training pairs
- Must be deterministic and pure

Look more carefully at the pattern. Consider: count colors, find objects, check symmetry, look at positions relative to edges/boundaries, check if size changes.

Respond with ONLY the Python code block, no other text.
""")
    return "".join(prompt_parts)


def solve_task_llm(task_id, max_attempts=3):
    """Use LLM to solve a single task. Returns (success, code, error)."""
    try:
        task = arc_data.load_task(task_id)
        pairs = arc_data.get_pairs(task)
    except Exception as e:
        return False, None, f"load failed: {e}"
    
    last_code = None
    last_error = None
    for attempt in range(max_attempts):
        if attempt == 0:
            prompt = build_prompt(task_id, pairs)
        else:
            prompt = build_revision_prompt(task_id, pairs, last_code, last_error)
        response = call_zai_chat(prompt)
        if not response:
            continue
        code = extract_python_code(response)
        if not code:
            continue
        last_code = code
        ok, msg = verify_solver(code, pairs)
        if ok:
            return True, code, "OK"
        last_error = msg
        # Save failed attempt for debugging
        with open(f"{OUTPUT_DIR}/task_{task_id:03d}_attempt_{attempt+1}.py", "w") as f:
            f.write(f"# Task {task_id}\n# Error: {msg}\n\n{code}")
    return False, last_code, last_error or "all attempts failed"

def main():
    # Load unsolved tasks
    with open("/home/z/my-project/data/aggressive_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    print(f"Unsolved tasks: {len(unsolved)}")
    
    # Load already-solved (skip them)
    already_solved = set()
    solved_file = f"{OUTPUT_DIR}/solved.json"
    if os.path.exists(solved_file):
        with open(solved_file) as f:
            already_solved = set(json.load(f).get("solved", []))
    print(f"Already LLM-solved: {len(already_solved)}")
    
    to_solve = [tid for tid in unsolved if tid not in already_solved]
    print(f"To solve: {len(to_solve)}")
    
    results = {"solved": list(already_solved), "failed": []}
    
    for i, tid in enumerate(to_solve):
        print(f"\n[{i+1}/{len(to_solve)}] Task {tid}...", flush=True)
        t0 = time.time()
        success, code, msg = solve_task_llm(tid)
        elapsed = time.time() - t0
        if success:
            fname = arc_data.task_id_to_filename(tid)
            with open(f"{OUTPUT_DIR}/task_{tid:03d}_{fname}.py", "w") as f:
                f.write(f"# Task {tid} ({fname})\n# LLM-solved in {elapsed:.1f}s\n\n{code}")
            results["solved"].append(tid)
            print(f"  ✓ SOLVED in {elapsed:.1f}s")
        else:
            results["failed"].append({"task_id": tid, "error": msg})
            print(f"  ✗ FAILED: {msg}")
        # Save progress periodically
        if (i + 1) % 5 == 0:
            with open(solved_file, "w") as f:
                json.dump(results, f, indent=2)
    
    # Final save
    with open(solved_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Final Results ===")
    print(f"Total solved: {len(results['solved'])}")
    print(f"Total failed: {len(results['failed'])}")

if __name__ == "__main__":
    main()
