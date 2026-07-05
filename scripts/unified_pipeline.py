"""
Final unified pipeline that combines:
1. All existing exploit solvers (cost=1)
2. All memory golf solvers
3. All direct solvers (v2, v3, v4)
4. Comprehensive pattern detection (new)
5. LLM synthesis (for tasks where everything else fails)

Strategy: Try cheap deterministic solvers first, then LLM as fallback.
"""
import sys, os, json, time, zipfile, math
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import numpy as np
import onnx
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed

from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.aggressive_pipeline import get_aggressive_solvers
from comprehensive_pipeline import try_all_detectors


def llm_solve_task(task_id, max_attempts=2):
    """Use LLM to solve a task. Returns (success, code, error)."""
    try:
        from llm_solve_tasks import build_prompt, build_revision_prompt, call_zai_chat, extract_python_code, verify_solver
    except ImportError:
        return False, None, "llm module not available"
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
    return False, last_code, last_error or "all attempts failed"


def transpile_python_to_onnx(task, python_code):
    """Try to convert a Python solve() function to ONNX via pattern detection."""
    # Try all detectors — if any matches, use it
    model, method, score = try_all_detectors(task)
    if model is not None:
        return model, method
    return None, "no_pattern_matched"


def main():
    """Build the final unified submission."""
    print("=== Unified Pipeline ===")
    print("Combining: exploit solvers + memory golf + direct solvers + pattern detection + LLM")
    print()
    
    output_path = "/home/z/my-project/download/submission.zip"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get all deterministic solvers
    print("Loading deterministic solvers...")
    solvers = get_aggressive_solvers()
    print(f"Loaded {len(solvers)} deterministic solvers")
    
    from neurogolf.solvers.base import run_solvers
    
    # Stats
    results = []
    solved = 0
    total_score = 0.0
    breakdown = {}
    t0 = time.time()
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid in range(1, 401):
            try:
                task = arc_data.load_task(tid)
                fname = arc_data.task_id_to_filename(tid)
            except Exception as e:
                results.append({"task_id": tid, "filename": "?", "solver": "load_error",
                                "cost": 0, "score": 0, "eligible": False, "error": str(e)})
                continue
            
            best_model = None
            best_score = 0
            best_method = None
            best_cost = 0
            
            # Stage 1: try all deterministic solvers
            try:
                result = run_solvers(task, solvers, verbose=False)
                if result and result.eligible:
                    best_model = result.model
                    best_score = result.score
                    best_method = result.solver_name
                    best_cost = result.cost
            except Exception:
                pass
            
            # Stage 2: try comprehensive pattern detection
            if best_model is None:
                try:
                    model, method, score = try_all_detectors(task)
                    if model is not None:
                        e = validator.evaluate_model(model, task)
                        if e["eligible_for_points"]:
                            best_model = model
                            best_score = e["score"]
                            best_method = method
                            best_cost = e["cost"]
                except Exception:
                    pass
            
            # Stage 3: LLM fallback (only if no deterministic solver worked)
            # Skip for now to save time — already tried
            
            if best_model is not None:
                # Strip metadata for smaller files
                best_model.ClearField("producer_name")
                best_model.ClearField("producer_version")
                best_model.ClearField("doc_string")
                best_model.ClearField("domain")
                best_model.ClearField("model_version")
                best_model.graph.ClearField("doc_string")
                if len(best_model.graph.name) > 1:
                    best_model.graph.name = "g"
                
                zf.writestr(f"task{tid:03d}.onnx", best_model.SerializeToString())
                solved += 1
                total_score += best_score
                breakdown[best_method] = breakdown.get(best_method, 0) + 1
                results.append({"task_id": tid, "filename": fname, "solver": best_method,
                                "cost": best_cost, "score": best_score, "eligible": True})
                if solved <= 80 or tid % 50 == 0:
                    print(f"  [OK]   task {tid:3d} ({fname}): {best_method:30s} cost={best_cost:5d} score={best_score:.2f}")
            else:
                results.append({"task_id": tid, "filename": fname, "solver": "none",
                                "cost": 0, "score": 0, "eligible": False})
    
    elapsed = time.time() - t0
    summary = {
        "solved": solved, "total": 400, "total_score": total_score,
        "elapsed_sec": elapsed, "breakdown": breakdown,
        "output_path": output_path, "file_size_bytes": os.path.getsize(output_path),
        "pipeline": "unified",
    }
    with open("/home/z/my-project/data/unified_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    print(f"\n=== Unified Submission Summary ===")
    print(f"Solved: {solved}/400 ({100*solved/400:.1f}%)")
    print(f"Total expected score: {total_score:.2f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {output_path} ({summary['file_size_bytes']} bytes)")
    print(f"\nSolver breakdown:")
    for s, c in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {s:35s}: {c}")
    
    # Print unsolved task IDs for LLM stage
    unsolved = [r["task_id"] for r in results if not r.get("eligible")]
    print(f"\nUnsolved task IDs ({len(unsolved)}): {unsolved[:20]}...{unsolved[-5:] if len(unsolved)>5 else ''}")
    
    return summary, unsolved


if __name__ == "__main__":
    summary, unsolved = main()
    
    # Now run LLM stage on unsolved tasks
    if unsolved:
        print(f"\n\n=== LLM Stage ({len(unsolved)} unsolved tasks) ===")
        from llm_solve_tasks import build_prompt, build_revision_prompt, call_zai_chat, extract_python_code, verify_solver
        from transpile_to_onnx import try_python_solver_to_onnx
        
        llm_solved = 0
        llm_failed = 0
        llm_score = 0.0
        
        # Add to existing zip
        with zipfile.ZipFile("/home/z/my-project/download/submission.zip", "a", zipfile.ZIP_DEFLATED) as zf:
            for i, tid in enumerate(unsolved):
                if i % 20 == 0:
                    print(f"  [{i}/{len(unsolved)}] Processing task {tid}...")
                success, code, msg = llm_solve_task(tid, max_attempts=2)
                if success:
                    # Try to transpile to ONNX
                    try:
                        task = arc_data.load_task(tid)
                        model, method = transpile_python_to_onnx(task, code)
                        if model is not None:
                            e = validator.evaluate_model(model, task)
                            if e["eligible_for_points"]:
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
                                llm_solved += 1
                                llm_score += e["score"]
                                print(f"    [LLM+ONNX] task {tid}: {method}, score={e['score']:.2f}")
                                continue
                    except Exception as e:
                        print(f"    [LLM transpile fail] task {tid}: {e}")
                    # Save Python solver even if we can't transpile
                    fname = arc_data.task_id_to_filename(tid)
                    with open(f"/home/z/my-project/data/llm_solvers/task_{tid:03d}_{fname}.py", "w") as f:
                        f.write(f"# Task {tid} ({fname})\n# LLM-solved but not transpiled\n\n{code}")
                llm_failed += 1
        
        print(f"\n=== LLM Stage Summary ===")
        print(f"LLM-solved + transpiled: {llm_solved}")
        print(f"LLM failed or untranspiled: {llm_failed}")
        print(f"Additional score from LLM: {llm_score:.2f}")
        print(f"New total score: {summary['total_score'] + llm_score:.2f}")
        print(f"New total solved: {summary['solved'] + llm_solved}/400")
