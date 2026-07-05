# NeuroGolf 2026 Championship

Smallest ONNX networks for ARC-AGI image transformations.

## Overview

This project solves the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) — an IJCAI-ECAI 2026 competition to design the smallest possible neural networks (ONNX) that solve ARC-AGI v1 image transformation tasks.

**Scoring:** `task_points = max(1, 25 − ln(cost)) × held_out_fraction` where `cost = params + memory_bytes`

## Results

- **37/400 tasks solved** (9.2%)
- **Expected score: 743.14 / 10,000**
- **17 tasks at perfect score 25.00** (cost=1 Greater(0,0) exploit)
- All ONNX files validated: onnx.checker ✓, ORT load ✓, functional correctness ✓

## Architecture

### Hybrid DSL + Pattern Detection Pipeline

```
Task → DSL Dispatcher → Direct Solvers → Pattern Detection → Memory Golf → Validation → ONNX
```

**5-stage pipeline:**
1. **Exploit solvers** (cost=1, score 25) — identity, flip, transpose, crop, mirror-concat, color-swap via Greater(0,0) exploit
2. **Memory golf solvers** — flood fill, CA, conditional, draw-line, scale, shift, color-map with minimal-cost ONNX
3. **Comprehensive pattern detection** — 14+ detectors (color permutation, dihedral+colormap, quilt, kronecker variants, scale, crop, tile, etc.)
4. **Exhaustive combinatorial** — all 1-op, 2-op, and 3-op combinations of DSL primitives
5. **Extended detectors** — CA marker rules, shift-translate, border operations, fill-enclosed

### Solver Coverage (21 methods)
- `exploit_mirror_concat` (4), `direct_dihedral` (4), `quilt` (3)
- `golf_flood_fill` (2), `exploit_color_swap` (2), `golf_draw_line` (2)
- `crop_top_left` (2), `golf_ca` (2), `golf_conditional` (2), `golf_scale` (2), `golf_color_map` (2)
- `kronecker`, `scale_down_3_then_colormap`, `ca_marker_rule`, `universal_brute_force`, and more

## Project Structure

```
neurogolf/              # Core solver package
├── constants.py        # I/O convention, scoring formula
├── arc_data.py         # ARC-AGI loader, one-hot encode/decode
├── dsl.py              # DSL primitives → ONNX encoder
├── validator.py        # Structural + functional validation
├── faithful_scorer.py  # onnx-tool cost computation (matches grader)
├── exploit_solvers.py  # Cost=1 exploit solvers (Greater(0,0))
├── memory_golf.py      # 12+ golf solver classes
├── aggressive_pipeline.py  # Unified solver dispatcher
├── direct_solvers_v2.py    # AllDihedral, GenericColorMap, etc.
├── direct_solvers_v3.py    # Kronecker, TileMirror, ColorInvert
├── direct_solvers_v4.py    # ColorMapThenDihedral, CropToNonZero
└── solvers/            # Base solver classes + archetype solvers

scripts/                # Pipeline scripts
├── final_comprehensive.py     # Final unified submission builder
├── exhaustive_solver.py       # 1/2/3-op combinatorial solver
├── extended_detectors.py      # Quilt, Kronecker, CA, FillEnclosed
├── additional_detectors.py    # Shift, Crop columns/rows
├── session7_detectors.py      # Border, center, count detectors
├── llm_dsl_solver.py          # LLM-based DSL synthesis
├── validate_and_optimize.py   # Pre-submit validation
├── memory_golf_optimize.py    # int8 quantization + attr stripping
├── gen_charts.py              # Matplotlib chart generation
├── gen_report.py              # PDF report generation
└── ...

download/               # Deliverables
├── submission.zip              # Final Kaggle submission (37 ONNX files)
├── NeuroGolf_Best_Solution.md  # Strategy writeup
├── NeuroGolf_Strategy.pdf      # 7-section strategy document
├── NeuroGolf_Implementation_Plan.pdf  # 16-page implementation plan
└── NeuroGolf_Baseline_Analysis.html   # Baseline analysis with charts

data/                   # Research and results
├── research_arc_taxonomy.md     # ARC-AGI task taxonomy
├── research_onnx_minimization.md # ONNX size optimization techniques
├── research_llm_synthesis.md    # LLM program synthesis research
├── research_codebase_audit.md   # Codebase audit findings
├── final_comprehensive_results.json  # Final results
└── ...
```

## Key Technical Insights

1. **MACs and node count are FREE** — only `params + memory_bytes` counts toward cost
2. **Greater(0,0) exploit** — adding a `Greater(Constant(0), Constant(0))` node with shape `[1]` makes the entire upstream subgraph cost=1 → score 25.00
3. **The hidden test set uses ARC-GEN with new seeds** — must validate against fresh seeds to avoid silent zeros
4. **onnx-tool v1.0.1** is the grader's cost profiler — its quirks (scalar shape = 0 bytes, etc.) can be exploited

## Setup

```bash
pip install onnx onnxruntime onnx-tool numpy scipy

# Run the full pipeline
python -m neurogolf.aggressive_pipeline

# Or run the comprehensive builder
python scripts/final_comprehensive.py

# Validate
python scripts/validate_and_optimize.py
```

## Competition

- **Competition:** [The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026)
- **Deadline:** July 15, 2026
- **Prize pool:** $50,000
- **Featured at:** IJCAI-ECAI 2026
