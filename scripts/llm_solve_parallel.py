"""Run LLM solver on unsolved tasks in batches, with parallel calls."""
import sys, os, json, subprocess, time, re, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/home/z/my-project")
import numpy as np
from neurogolf import arc_data

sys.path.insert(0, "/home/z/my-project/scripts")
from llm_solve_tasks import (
    build_prompt, build_revision_prompt, call_zai_chat,
    extract_python_code, verify_solver, grid_to_ascii
)

OUTPUT_DIR = "/home/z/my-project/data/llm_solvers"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def solve_one_task(task_id, max_attempts=2):
    """Solve a single task with LLM. Returns dict with results."""
    try:
        task = arc_data.load_task(task_id)
        pairs = arc_data.get_pairs(task)
    except Exception as e:
        return {"task_id": task_id, "success": False, "error": f"load failed: {e}"}
    
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
            return {"task_id": task_id, "success": True, "code": code, "attempts": attempt + 1}
        last_error = msg
    return {"task_id": task_id, "success": False, "error": last_error or "all attempts failed", "last_code": last_code}


def main():
    # Load unsolved tasks
    with open("/home/z/my-project/data/aggressive_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    
    # Load already-solved
    solved_file = f"{OUTPUT_DIR}/solved.json"
    already_solved = set()
    if os.path.exists(solved_file):
        with open(solved_file) as f:
            already_solved = set(json.load(f).get("solved", []))
    
    to_solve = [tid for tid in unsolved if tid not in already_solved]
    print(f"Total unsolved: {len(unsolved)}")
    print(f"Already LLM-solved: {len(already_solved)}")
    print(f"To solve: {len(to_solve)}")
    
    if not to_solve:
        print("Nothing to do.")
        return
    
    results = {"solved": list(already_solved), "failed": []}
    
    # Run with 4 parallel workers (don't overwhelm the API)
    MAX_WORKERS = 4
    print(f"Running with {MAX_WORKERS} parallel workers")
    print(f"Estimated time: {len(to_solve) * 8 / MAX_WORKERS / 60:.1f} minutes")
    print()
    
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(solve_one_task, tid): tid for tid in to_solve}
        for future in as_completed(futures):
            tid = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                result = {"task_id": tid, "success": False, "error": str(e)}
            
            if result["success"]:
                fname = arc_data.task_id_to_filename(tid)
                code = result["code"]
                with open(f"{OUTPUT_DIR}/task_{tid:03d}_{fname}.py", "w") as f:
                    f.write(f"# Task {tid} ({fname})\n# LLM-solved in {result['attempts']} attempt(s)\n\n{code}")
                results["solved"].append(tid)
                print(f"  [{completed}/{len(to_solve)}] task {tid}: ✓ SOLVED ({result['attempts']} attempts)")
            else:
                results["failed"].append({"task_id": tid, "error": result.get("error", "unknown")})
                print(f"  [{completed}/{len(to_solve)}] task {tid}: ✗ FAILED ({result.get('error', 'unknown')[:80]})")
            
            # Save progress every 10 tasks
            if completed % 10 == 0:
                with open(solved_file, "w") as f:
                    json.dump(results, f, indent=2)
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(to_solve) - completed) / rate if rate > 0 else 0
                print(f"    --- Progress: {completed}/{len(to_solve)} solved={len(results['solved'])-len(already_solved)} elapsed={elapsed:.0f}s rate={rate:.2f}/s ETA={remaining:.0f}s ---")
    
    # Final save
    with open(solved_file, "w") as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - t0
    newly_solved = len(results["solved"]) - len(already_solved)
    print(f"\n=== Final Results ===")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Newly solved: {newly_solved}/{len(to_solve)}")
    print(f"Total LLM-solved: {len(results['solved'])}")
    print(f"Total failed: {len(results['failed'])}")


if __name__ == "__main__":
    main()
