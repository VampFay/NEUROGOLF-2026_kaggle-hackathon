"""
Final submission builder — combines ALL solvers in priority order.

Strategy:
1. Run all deterministic solvers (exploit + memory golf + direct + comprehensive + advanced)
2. For unsolved tasks, try LLM with onnxscript (LLM writes Python that compiles to ONNX)
3. Validate every ONNX file
4. Apply memory golf optimizations
5. Build final submission.zip
"""
import sys, os, json, time, zipfile, math, re
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/scripts")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
import onnxruntime as ort

from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.aggressive_pipeline import get_aggressive_solvers
from neurogolf.solvers.base import run_solvers
from comprehensive_pipeline import try_all_detectors
from advanced_detectors import try_advanced_detectors


def strip_metadata(model):
    """Strip all optional metadata to minimize file size."""
    model.ClearField("producer_name")
    model.ClearField("producer_version")
    model.ClearField("doc_string")
    model.ClearField("domain")
    model.ClearField("model_version")
    model.graph.ClearField("doc_string")
    if len(model.graph.name) > 1:
        model.graph.name = "g"
    return model


def main():
    """Build the final submission combining all solvers."""
    print("=== FINAL SUBMISSION BUILDER ===")
    print("Combining: exploit + memory golf + direct + comprehensive + advanced detectors")
    print()
    
    output_path = "/home/z/my-project/download/submission.zip"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get all deterministic solvers
    print("Loading deterministic solvers...")
    solvers = get_aggressive_solvers()
    print(f"Loaded {len(solvers)} deterministic solvers")
    
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
            except Exception:
                results.append({"task_id": tid, "filename": "?", "solver": "load_error",
                                "cost": 0, "score": 0, "eligible": False})
                continue
            
            best_model = None
            best_score = 0
            best_method = None
            best_cost = 0
            
            # Stage 1: deterministic solvers from aggressive_pipeline
            try:
                result = run_solvers(task, solvers, verbose=False)
                if result and result.eligible:
                    best_model = result.model
                    best_score = result.score
                    best_method = result.solver_name
                    best_cost = result.cost
            except Exception:
                pass
            
            # Stage 2: comprehensive pattern detection
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
            
            # Stage 3: advanced CA/marker detectors
            if best_model is None:
                try:
                    model, method, score = try_advanced_detectors(task)
                    if model is not None:
                        best_model = model
                        best_score = score
                        best_method = method
                        best_cost = 0  # will compute below
                except Exception:
                    pass
            
            if best_model is not None:
                # Strip metadata
                best_model = strip_metadata(best_model)
                # Get accurate cost
                ci = faithful_scorer.compute_cost(best_model)
                best_cost = ci.get("cost", 0)
                # Re-validate after stripping
                e = validator.evaluate_model(best_model, task)
                if e["eligible_for_points"]:
                    zf.writestr(f"task{tid:03d}.onnx", best_model.SerializeToString())
                    solved += 1
                    total_score += e["score"]
                    breakdown[best_method] = breakdown.get(best_method, 0) + 1
                    results.append({"task_id": tid, "filename": fname, "solver": best_method,
                                    "cost": best_cost, "score": e["score"], "eligible": True})
                    if solved <= 80 or tid % 50 == 0:
                        print(f"  [OK]   task {tid:3d} ({fname}): {best_method:30s} cost={best_cost:5d} score={e['score']:.2f}")
                else:
                    results.append({"task_id": tid, "filename": fname, "solver": best_method + "_invalid_post_strip",
                                    "cost": best_cost, "score": 0, "eligible": False})
            else:
                results.append({"task_id": tid, "filename": fname, "solver": "none",
                                "cost": 0, "score": 0, "eligible": False})
    
    elapsed = time.time() - t0
    summary = {
        "solved": solved, "total": 400, "total_score": total_score,
        "elapsed_sec": elapsed, "breakdown": breakdown,
        "output_path": output_path, "file_size_bytes": os.path.getsize(output_path),
        "pipeline": "final_unified",
    }
    with open("/home/z/my-project/data/final_unified_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    print(f"\n=== Final Unified Submission Summary ===")
    print(f"Solved: {solved}/400 ({100*solved/400:.1f}%)")
    print(f"Total expected score: {total_score:.2f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {output_path} ({summary['file_size_bytes']} bytes)")
    print(f"\nSolver breakdown:")
    for s, c in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {s:35s}: {c}")
    
    unsolved = [r["task_id"] for r in results if not r.get("eligible")]
    print(f"\nUnsolved: {len(unsolved)}")
    
    return summary, unsolved


if __name__ == "__main__":
    summary, unsolved = main()
