"""Run LLM solver on unsolved tasks in small batches to avoid rate limits."""
import sys, os, json, time, re
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from neurogolf import arc_data
from llm_solve_tasks import (
    build_prompt, build_revision_prompt, call_zai_chat,
    extract_python_code, verify_solver
)

OUTPUT_DIR = "/home/z/my-project/data/llm_solvers"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def solve_one_task(task_id, max_attempts=2):
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
        if not response: continue
        code = extract_python_code(response)
        if not code: continue
        last_code = code
        ok, msg = verify_solver(code, pairs)
        if ok:
            return {"task_id": task_id, "success": True, "code": code, "attempts": attempt + 1}
        last_error = msg
    return {"task_id": task_id, "success": False, "error": last_error or "all attempts failed", "last_code": last_code}


def main():
    # Load unsolved
    with open("/home/z/my-project/data/unified_results.json") as f:
        d = json.load(f)
    unsolved = [r["task_id"] for r in d["results"] if not r.get("eligible")]
    
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
    MAX_WORKERS = 2  # Lower to avoid rate limits
    BATCH_SIZE = 30  # Process in batches
    
    t0 = time.time()
    completed = 0
    
    for batch_start in range(0, len(to_solve), BATCH_SIZE):
        batch = to_solve[batch_start:batch_start + BATCH_SIZE]
        print(f"\n=== Batch {batch_start//BATCH_SIZE + 1}/{(len(to_solve)-1)//BATCH_SIZE + 1} ===")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(solve_one_task, tid): tid for tid in batch}
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
                    print(f"  [{completed}/{len(to_solve)}] task {tid}: SOLVED ({result['attempts']} attempts)")
                else:
                    results["failed"].append({"task_id": tid, "error": result.get("error", "unknown")})
                    err = result.get('error', 'unknown')[:60]
                    print(f"  [{completed}/{len(to_solve)}] task {tid}: FAILED ({err})")
        
        # Save progress after each batch
        with open(solved_file, "w") as f:
            json.dump(results, f, indent=2)
        elapsed = time.time() - t0
        print(f"  --- Batch done. Total solved so far: {len(results['solved'])-len(already_solved)}/{len(to_solve)}, elapsed={elapsed:.0f}s ---")
    
    elapsed = time.time() - t0
    newly_solved = len(results["solved"]) - len(already_solved)
    print(f"\n=== Final Results ===")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Newly solved: {newly_solved}/{len(to_solve)}")
    print(f"Total LLM-solved: {len(results['solved'])}")


if __name__ == "__main__":
    main()
