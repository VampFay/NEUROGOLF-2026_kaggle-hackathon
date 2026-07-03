# NeuroGolf Codebase Audit Report

**Date:** 2026-07-03
**Scope:** `/home/z/my-project/neurogolf/` and supporting scripts
**Current state:** 16/400 tasks solved, total score ≈ 300 (leader has ~8002)

---

## Executive Summary

The codebase is cleanly structured and the I/O convention is correctly reverse-engineered, but **only 16 of 400 tasks are solved** because the solver library has three confirmed bugs, several stub solvers that always return `None`, one solver module (`filters.py`) that is never wired into the dispatcher, and **no coverage for the four largest failing task families** (CA-style fills, sub-grid extraction, asymmetric scaling, pattern-based scaling). There are also clear cost-inefficiencies in the CA solvers that are leaving 1–1.5 points per task on the table.

Closing the confirmed bugs alone would unlock ~5 more tasks (mirror-concat family). Adding a flood-fill solver, a bounding-box extractor, and an asymmetric scaler would plausibly unlock 80–120 more tasks. Pruning CA weight tensors would add ~3–5 points to existing solves.

---

## 1. Codebase Strengths

| Strength | Evidence |
|---|---|
| **Clean module separation** | `constants.py`, `arc_data.py`, `dsl.py`, `validator.py`, `solvers/{base,simple,transforms,filters,advanced,patterns,cellular}.py` — each <700 LOC, single responsibility. |
| **Correct I/O convention** | `(1, 10, 30, 30)` one-hot float32, argmax-over-channels decode, top-left crop — matches competition example. Confirmed by `validator.functional_check` passing on all 16 solved tasks. |
| **Composable DSL primitives** | `identity`, `single_layer_conv2d`, `color_map`, `chain`, `argmax_over_channels`, `mask_apply`, `replace_color`, `conv_stack`, `constant_grid` — all return `onnx.ModelProto` and compose via `chain()`. |
| **Local validator mirrors competition** | `structural_check` (dynamic shapes, banned ops, 1.44 MB limit, `onnx.checker`) + `functional_check` (runs `onnxruntime` on every train+test pair). `evaluate_model` returns params/size/cost/score/eligible. |
| **Cost-aware dispatcher** | `run_solvers` returns the lowest-cost eligible model; `SolverResult.__lt__` sorts by eligibility then cost. Cost = `#params + #bytes` per competition spec. |
| **Fast full pipeline** | 400 tasks in ~50 s (worklog Task 1). Leaves plenty of headroom for more solvers. |
| **Reproducible** | Deterministic, no randomness. `submission_results.json` captures per-task solver/cost/score. |
| **`task_signature` helper** | `arc_data.task_signature` already computes size/color stats — a good foundation for a signature-based dispatcher (currently unused). |

---

## 2. Codebase Weaknesses (Bugs & Code Smells)

### 2.1 Confirmed bug: `MirrorConcatSolver` produces a model that always fails functional check

`patterns.MirrorConcatSolver.attempt()` matches the pattern on the *actual grid* (e.g. `np.concatenate([a, np.fliplr(a)], axis=1)`) and returns a model. But `_mirror_concat_model` builds the flip on the **30×30 padded input**, so when concatenated the padding region produces garbage that doesn't match the expected output.

Reproduced with a synthetic `lr_in_flip` task:
```
Testing MirrorConcatSolver on lr_in_flip pattern:
  eligible=False cost=589 note=OK | ['Pair 0: mismatch', 'Pair 1: mismatch']
```
**Impact:** 5 tasks attempted, 0 solved. The model is generated (cost ~589) but never eligible. Tasks 116, 164, 172, 210 are flagged as `mirror_concat` "best" but actually fail.

**Fix:** The model must know the actual grid dimensions. Either (a) bake the expected output dims into the model as constants (since the test pair's output shape is known at build time), or (b) detect the bounding box of non-zero cells first via `ReduceMax` + `Where` + `Slice`. Option (a) is simplest and cheapest.

### 2.2 Confirmed issue: `ScaleUpSolver` emits a malformed output shape

`advanced.ScaleUpSolver` produces a `Resize` node that scales the 30×30 input by `k`, yielding a (1, 10, 30k, 30k) output, but the model's output `ValueInfo` declares (1, 10, 30, 30). onnxruntime emits a warning and falls back to lenient merge:
```
[W:onnxruntime:, graph.cc:122 MergeShapeInfo] Error merging shape info for output.
'output' source:{1,10,60,60} target:{1,10,30,30}. Falling back to lenient merge.
```
It happens to pass our local validator because `onehot_to_grid` crops to expected dims, but **the competition validator may not be lenient**. Every `scale_up` solve (tasks 223, 307) is at risk.

**Fix:** Add a final `Slice` to crop the resized tensor back to (1, 10, 30, 30) before output.

### 2.3 Dead code in `dsl.chain()` (lines 250–261 of `dsl.py`)

```python
new_inputs = []
for inp in list(new_node.input):
    if inp == INPUT_NAME: ...
    elif inp in seen_init_names: ...
    elif any(init.name == inp for init in m.graph.initializer): ...
    else: ...
new_inputs = [prefix + inp if any(init.name == inp ...) else
              (current_input if inp == INPUT_NAME else inp)
              for inp in new_node.input]  # <-- overwrites the loop above
```
The first loop computes `new_inputs` and is then **immediately overwritten** by the list comprehension. Functionally equivalent for the cases tested, but misleading and a likely source of bugs if someone edits only one branch. Verified `chain([color_map(0->5), identity])` does work on a synthetic task.

### 2.4 Dead loop in `transforms.ColorMapThenTransformSolver.attempt()` (lines 192–194)

```python
for inp, out in pairs:
    # Try each transform; if any works, store the color-mapped version
    pass
```
A `for` loop with only `pass` — pure dead code. The real logic is in the loop below. No functional impact, but suggests incomplete refactoring.

### 2.5 `ReplaceColorSolver` is a strict subset of `ColorMapSolver` but reports as "best" on 90 failing tasks

`simple.ReplaceColorSolver` builds a color map of only the colors that change. For tasks where the same color appears in both changed and unchanged cells (the common case), the model is incorrect. Because it always returns *a* model (never `None` when shapes match), it shows up as the "best" solver for 90 failing tasks, **masking the real signal** — we can't tell from the JSON whether the task is genuinely unsolvable or whether ReplaceColor just happened to produce the smallest incorrect model.

**Fix:** Either delete `ReplaceColorSolver` (ColorMapSolver already covers its valid cases) or have it return `None` when the color map is incomplete (i.e., when any cell of a "changed" color also appears unchanged in another pair).

### 2.6 `filters.py` is never imported by the dispatcher

`build_submission.get_all_solvers()` lists 18 solvers but **`filters.ConvFilterSolver` and `filters.ColorSubstitutionSolver` are not among them**. The entire `filters.py` module (161 LOC) is dead code. Additionally, `ConvFilterSolver._conv_onehot` is implemented with 6-deep nested Python for-loops (~810k ops per pair) — unusably slow even if wired up. Should be reimplemented with `scipy.ndimage.convolve` or vectorized `np.pad` + slicing.

### 2.7 Three stub solvers always return `None`

| Solver | File | Status |
|---|---|---|
| `ColorCountSolver._color_count_model` | `patterns.py:194` | Returns `None` with comment "Skip for now" |
| `FillBorderSolver._fill_border_model` | `patterns.py:316` | Returns `None` with comment "Skip for now" |
| `MaxColorSolver.attempt` | `patterns.py:391` | Returns `None` with comment "Too complex" |
| `BiasColorMapSolver.attempt` | `simple.py:92` | Returns `None` with comment "Disabled — color_map handles all cases" |

These contribute zero solves. `ColorCountSolver` in particular would unlock many 1-D output tasks (the audit shows tasks with `out.shape = (1, N)` or `(N, 1)`).

### 2.8 `KroneckerSolver` is overly restrictive

`advanced.KroneckerSolver.attempt()` requires `inp.shape[0] == inp.shape[1] == k` (input must be a k×k square). Many "conditional tiling" tasks have non-square templates (e.g., a 3×2 pattern tiled into a 6×4 grid). The solver bails on these.

### 2.9 `ExhaustiveColorMapSolver` is a duplicate of `ColorMapSolver`

`patterns.ExhaustiveColorMapSolver.attempt()` is line-for-line identical to `simple.ColorMapSolver.attempt()` (both build the same `mapping` dict and call `dsl.color_map(mapping)`). It's listed in the dispatcher after `ColorMapSolver`, so it never runs (ColorMap wins ties). Wasted cycles and confusing.

---

## 3. Solver Gaps — Task Families With No Coverage

The audit script categorised all 384 failing tasks by shape/diff signature:

| Family | Count | Coverage | Sample failing task |
|---|---:|---|---|
| `CA_or_pattern_recolor` (same shape, <30% diff) | **189** | Partial (3 solved) — `CellularAutomatonSolver` only handles single-rule "X with ≥K Y-neighbors → Z"; `MultiRuleCASolver` only handles "empty cell with non-zero neighbor → Z" | task 2 `00d62c1b`: 6×6, 5.6% diff, color 0 → color 4 (enclosed fill) |
| `crop_or_extract` (out smaller than in) | **99** | Minimal — only `ConditionalSliceColorMapSolver` (separator line + color map) | task 6 `0520fde7`: 3×7 → 3×3 (slice right of separator) |
| `complex_color_change` (same shape, 30–50% diff) | **34** | None | task 10 `08ed6ac7`: 9×9, 32.1% diff |
| `complex_rearrangement` (same shape, >50% diff) | **29** | None | task 52 `25d8a9c8`: 3×3, 100% diff |
| `concat_or_extend` (out larger, not integer scale) | **16** | Partial — `ConcatRepeatSolver` only handles "input + first K rows"; `MirrorConcatSolver` is buggy (§2.1) | task 3 `017c7c7b`: 6×3 → 9×3 |
| `scale_2x_or_kronecker_2x2` (4× area) | **14** | Partial — `ScaleUpSolver` only handles uniform nearest-neighbor; `KroneckerSolver` requires k×k input | task 19 `10fcaaa3`: 2×4 → 4×8 |
| `scale_3x_or_kronecker_3x3` (9× area) | **3** | Same as above | task 104 `4522001f`: 3×3 → 9×9 |

### 3.1 Missing: Flood-Fill / Enclosed-Region Solver

**Target:** the 189 CA-family failing tasks. Inspecting samples:
- task 2: color 0 → color 4 (fill enclosed hole with surrounding color)
- task 5: color 0 → colors {2, 3} (fill based on which region)
- task 9: color 0 → colors {2, 3}
- task 17: color 0 → colors {1, 2, 3, 4, 5, 6} (fill each enclosed region with the color of its border)

`MultiRuleCASolver` only handles "empty cell with ≥1 non-zero neighbor → Z" (single-step fill). It cannot handle:
- Multi-step flood fill (propagate through connected empty regions)
- "Fill with the color of the enclosing border" (requires knowing the border color, which varies per region)

**Approach:** A multi-iteration CA where each step propagates the nearest non-zero color into empty cells. With 30×30 grids, worst case needs 30 iterations. Each iteration is a 3×3 conv. Total cost: 30 × (10×10×3×3) = 27,000 params — too expensive. Better: a single 3×3 conv with weights `W[c, c, dh, dw] = 1` for all (dh, dw) in 4-neighbors, run iteratively. But ONNX has no `Scan`/`Loop` (banned). Alternative: unroll 30 conv layers — still 27,000 params. **Cheaper alternative:** detect that the task is "fill enclosed" and bake the *specific* fill color per region into a constant (since the test pair's expected output is known at build time... but that's overfitting and won't generalise to the held-out test input).

This is the single highest-value solver to design. Even a partial solution (e.g., handle the case where all enclosed regions share one fill color) would unlock many tasks.

### 3.2 Missing: Bounding-Box / Object-Extraction Solver

**Target:** the 99 `crop_or_extract` tasks. Sub-families:
- "Extract the largest connected component" (e.g., task 14: 21×21 → 10×10)
- "Extract the region containing a marker color" (e.g., task 22: 11×11 → 3×3)
- "Split the grid into sub-grids by separator lines" (e.g., task 26: 5×7 → 5×3 — split into two halves)
- "Subsample every Nth row/column" (e.g., task 36: 30×30 → 5×3 — take every 6th row, every 10th col)

**Approach:** The current `ConditionalSliceColorMapSolver` handles a single separator column/row. We need:
- `SeparatorGridSolver`: detect a grid of separator lines (e.g., every 5th row and every 5th column are color 8), slice one sub-region
- `BoundingBoxSolver`: compute `ReduceMin`/`ReduceMax` of non-zero mask per axis → 4 indices → `Slice`
- `SubsampleSolver`: `Slice` with `steps=[1, 1, N, M]`

`SubsampleSolver` is trivial (a single `Slice` node with non-trivial steps) and likely unlocks 10–20 tasks on its own.

### 3.3 Missing: Asymmetric Scale Solver

**Target:** 14 `scale_2x` + 3 `scale_3x` + at least 3 concat tasks (task 211: 3×2 → 9×4 = 3×2× scale).

`ScaleUpSolver` requires `kh == kw`. Tasks with `kh != kw` (e.g., 3×2 → 9×4) fail. Fix: relax the constraint and pass `[1.0, 1.0, kh, kw]` to `Resize`. One-line change.

Also missing: **pattern-based scaling** (each cell becomes a fixed k×k block that depends on the cell's color). This is a `Tile` + `color_map` composition. The Kronecker solver is a special case (input IS the block). Generalise to "block lookup table" where each of the 10 colors maps to a k×k block.

### 3.4 Missing: Color-Map + Geometric-Transform Composition (Both Directions)

`ColorMapThenTransformSolver` applies color_map then transform. But many tasks need transform *then* color_map (e.g., flip the grid, then recolor). The current code claims these commute, but they don't always (e.g., a horizontal flip then color swap that depends on position). Add `TransformThenColorMapSolver`.

### 3.5 Missing: Line-Drawing Solver

Common ARC task: "draw a line of color C between two marker cells". No solver. Would need `Where` + `CumSum` (allowed) or a hand-built mask.

### 3.6 Missing: Count-And-Draw Solver

Common ARC task: "count the number of objects of color X in the input, draw that many dots of color Y in the output". The stub `ColorCountSolver` exists but returns `None`. Implementation requires `ReduceSum` + `Range` + `Less` + `OneHot` — all allowed ops.

### 3.7 Missing: Learned-Conv Solver (the "hybrid" in "hybrid DSL+learned")

The strategy PDF mentions "hybrid DSL + learned" but there is **no learned component** in the codebase. For tasks where no DSL rule matches, we could fit a small 3×3 (or 5×5) conv by gradient descent on the train pairs. Even 10–20 tasks captured this way would add ~180 points. Use `torch` + export to ONNX, or directly optimise the weight tensor with `scipy.optimize` and emit via `dsl.single_layer_conv2d`.

---

## 4. Cost Inefficiencies (Bytes/Params Left on the Table)

Score = `max(1, 25 - ln(cost))` where `cost = #params + #bytes`. Every halving of cost adds ~0.69 points.

| Solver | Current cost | Current score | Optimisation | Optimised cost | Optimised score | Δ per task |
|---|---:|---:|---|---:|---:|---:|
| `multi_rule_ca` | 4777 | 16.53 | Prune W to used colors only (typically 4 → 4×4×3×3 = 36 params instead of 900) | ~1100 | ~18.0 | **+1.5** |
| `cellular_automaton` | 1427 | 17.74 | Replace 3-conv + 13-node graph with single 3×3 conv + bias (threshold encoded in bias) | ~700 | ~18.9 | **+1.1** |
| `kronecker` | 1123 | 17.98 | Replace `Resize`+`Tile`+`Slice`+`Pad` with single `Tile` + `Mul` (gate via channel-0 mask) | ~600 | ~18.1 | +0.1 |
| `color_map` | 753 | 18.38 | Prune W to used colors (e.g., 4 colors → 4×4×1×1 = 16 params instead of 100) + use `Constant`+`Gather` | ~250 | ~19.4 | **+1.0** |
| `scale_up` | 341 | 19.17 | Already minimal (Resize + 2 Constants); add Slice to fix shape (§2.2) | ~360 | ~19.1 | -0.1 |
| `crop_top_left` | 130 | 20.13 | Already optimal (Identity) | 130 | 20.13 | 0 |
| `geom_transform` | 146 | 20.02 | Already optimal (Slice/Transpose, no params) | 146 | 20.02 | 0 |

**Total score upside on existing 16 solves:** ~10 points (from CA + color_map pruning alone).

**Bigger picture:** if we extend CA-family solvers to even 30 of the 189 CA tasks (current cost 1427 → optimised 700, score 18.9), that's **30 × 18.9 = 567 points** vs current 3 × 17.74 = 53 points. The pruning makes each solve worth ~1.2 more points.

### 4.1 No constant-folding / simplification pass

Generated models contain many `Constant` nodes (for slice indices, pads, etc.) that could be folded. Running `onnxsim` (or a custom simplifier) on every model before serialisation would shrink byte count 10–30%. Not installed in the venv.

### 4.2 No INT8 / FLOAT16 quantisation

All initializers are FLOAT32. For models with large weight tensors (CA, conv stacks), quantising to INT8 with a scale factor would halve the byte contribution. Params count unchanged, but bytes drop. Net win for any model where bytes dominate (e.g., `multi_rule_ca`).

---

## 5. Validator Limitations

### 5.1 Cannot detect overfitting to the test pair

`validator.functional_check` runs the model on **all** train + test pairs (because the JSON contains both). Since solvers build models *using* the test pair's expected output (e.g., `ColorMapSolver` derives the mapping from all pairs including test), the model is guaranteed to pass the test pair. **This is a form of train/test leakage in our local validation.** The competition's hidden benchmark will use a *different* test input, so a solver that memorised the test pair will fail.

**Mitigation:** Add a `strict_functional_check` that derives the model from train pairs only, then evaluates on the test pair. Solvers that need the test pair's output shape (e.g., `MirrorConcatSolver` fix in §2.1) should be flagged as "shape-only" leaks (acceptable, since the competition gives output dims) vs "content" leaks (unacceptable).

### 5.2 Cannot test the competition's exact ONNX runtime

We use `onnxruntime 1.27` with `CPUExecutionProvider`. The competition may use a different version with different op support (e.g., `Resize` nearest_mode quirks, `OneHot` axis behaviour). The `ScaleUpSolver` shape-mismatch warning (§2.2) is one example of a tolerance that may not extend to the competition.

### 5.3 Cannot test the private benchmark suite

The competition mentions "a small private benchmark suite beyond the public tasks". We have zero visibility into this. Solvers that overfit to ARC-AGI-v1 training tasks (e.g., by hardcoding color mappings) may fail on the private suite.

### 5.4 `structural_check` doesn't validate opset compatibility

`_empty_model` hardcodes `opset 17`. If the competition runtime only supports opset 16, models using opset-17-only ops (e.g., `Resize` with `antialias`) will fail. No check for this.

### 5.5 No batch-dim validation

All models assume batch=1. If the competition runs batches (unlikely but possible), our static-shape models would fail. No test for this.

---

## 6. Performance Bottlenecks

### 6.1 `filters.ConvFilterSolver._conv_onehot` is O(810k) per pair

6-deep nested Python for-loops (c_out × c_in × dh × dw × i × j = 10×10×3×3×30×30). Even if wired up, it would take seconds per task. Should be replaced with `scipy.ndimage.convolve` or `np.pad` + `np.einsum`.

### 6.2 `CellularAutomatonSolver` brute-forces (X, Y, Z, threshold) = 10 × 11 × 10 × 8 = 8800 combinations

Each combination runs `_neighbor_count` (numpy, fast) on all pairs. For 400 tasks × 8800 combos × 3 pairs × ~5ms = ~17 minutes. Currently fast enough (50s total) because most tasks bail early on shape mismatch, but as we add more CA variants this will become a bottleneck.

**Fix:** Precompute `_neighbor_count` once per (inp, color) pair, then test all (X, Y, Z, threshold) combos against the cached counts.

### 6.3 `MultiRuleCASolver._verify` is O(H × W × neighbors × pairs)

30×30×8×3 = 21,600 ops per pair, in pure Python (nested for-loops with `range`). For 400 tasks × 3 neighbor sets × ~3 pairs = ~78M ops. Numpy vectorisation would speed this up 100×.

### 6.4 `run_solvers` has no early-exit or signature-based pruning

Every solver is tried on every task, even when the task signature rules out most solvers (e.g., shape-change tasks skip `ColorMapSolver` trivially, but we still call it). For 18 solvers × 400 tasks = 7200 solver calls, most of which fail fast. With 50+ solvers (planned), this becomes 20,000+ calls. Add a `supports(signature) -> bool` method to `Solver` and filter before calling `attempt`.

### 6.5 No solver-level caching

If two solvers both compute `_neighbor_count` on the same input, the work is duplicated. A per-task memo dict would help.

---

## 7. Missing Infrastructure

### 7.1 No test suite

There are no unit tests. `scripts/test_dsl.py` exists but is a manual smoke script, not a pytest suite. We cannot detect regressions when solvers are modified. **Critical gap.**

**Needed tests:**
- Each DSL primitive produces a structurally-valid model
- `chain([A, B])` is functionally equivalent to running A then B
- Each solver solves its canonical task (1 synthetic task per solver) with cost ≤ threshold
- `validator.functional_check` rejects a deliberately-wrong model
- Round-trip: `grid_to_onehot` → `onehot_to_grid` is identity for all 10 colors

### 7.2 No regression baseline

We have `baseline_results.json` (12 tasks) and `submission_results.json` (16 tasks) but no automated diff. A regression is invisible until someone manually compares the two.

**Fix:** `scripts/regression_check.py` that compares `submission_results.json` against a checked-in `golden_results.json` and reports any task that went from eligible → ineligible.

### 7.3 No per-task failure profiler

When a solver returns `None`, we don't log *why*. Was it shape mismatch? Inconsistent color map? Failed functional check? Without this, diagnosing the 189 CA-family failures requires re-running solvers with `verbose=True` one at a time.

**Fix:** Add a `failure_reason: str` field to `SolverResult` (populated even when `eligible=False`), aggregated into `submission_results.json`.

### 7.4 No cost/score regression dashboard

No way to see "average score per solver over time" or "which solvers regressed". The HTML baseline analysis is a static snapshot.

### 7.5 No CI for submission.zip integrity

No automated check that `submission.zip` contains 400 `.onnx` files (or fewer, with explicit skip list), that each file is <1.44 MB, that each loads in onnxruntime. A bad submission could silently upload 0 files.

### 7.6 No solver-ordering strategy beyond "cheap first"

`get_all_solvers()` is a hardcoded list. There's no mechanism to (a) reorder based on historical success rate, (b) skip solvers that have never solved any task, (c) parallelise solver attempts across tasks. For 400 tasks × 50 solvers, parallelism would cut wall-clock from minutes to seconds.

### 7.7 No "smallest equivalent model" post-processing

After a solver produces a model, we don't run `onnxsim` or any simplifier. Constants aren't folded, identity ops aren't removed, unused initializers aren't pruned. This leaves 10–30% of bytes on the table.

### 7.8 No logging / observability

`build_submission` prints to stdout but doesn't log to a file. `submission_results.json` captures the final state but not the per-solver attempts (which solver returned None, which returned an ineligible model, etc.). Debugging requires re-running with `verbose=True`.

---

## 8. Prioritised Next Actions

Ranked by (expected score gain) / (implementation effort):

| # | Action | Score gain | Effort | Section |
|---|---|---:|---|---|
| 1 | **Fix `MirrorConcatSolver`** — bake expected output dims into the model as Slice constants | +5 tasks (~95 pts) | 2 h | §2.1 |
| 2 | **Fix `ScaleUpSolver` output shape** — add final `Slice` to (1,10,30,30) | protects 2 tasks (~38 pts) | 30 min | §2.2 |
| 3 | **Add `SubsampleSolver`** — `Slice` with steps=[1,1,N,M] | +10–20 tasks (~190 pts) | 1 h | §3.2 |
| 4 | **Relax `ScaleUpSolver` to asymmetric `kh != kw`** | +3–5 tasks (~57 pts) | 30 min | §3.3 |
| 5 | **Prune `multi_rule_ca` weight tensor** to used colors only | +1.5 pts × 1 task now, more later | 1 h | §4 |
| 6 | **Prune `color_map` weight tensor** to used colors only | +1.0 pts × 4 tasks = +4 pts | 1 h | §4 |
| 7 | **Implement `FloodFillSolver`** for enclosed-region fills (single-step variant first) | +30–80 tasks (~600 pts) | 1 day | §3.1 |
| 8 | **Implement `BoundingBoxExtractSolver`** via `ReduceMin`/`ReduceMax` + `Slice` | +20–40 tasks (~380 pts) | 4 h | §3.2 |
| 9 | **Wire up `filters.py`** (after rewriting `_conv_onehot` with scipy) | +5–10 tasks (~95 pts) | 3 h | §2.6 |
| 10 | **Implement `ColorCountSolver`** (currently stub) | +5–10 tasks (~95 pts) | 4 h | §2.7 |
| 11 | **Add regression test suite** (pytest, 1 test per solver + DSL primitives) | protects all gains | 4 h | §7.1 |
| 12 | **Add `failure_reason` field** to `SolverResult` | debugging | 2 h | §7.3 |
| 13 | **Add signature-based dispatcher pruning** | 2× speedup | 3 h | §6.4 |
| 14 | **Implement `LearnedConvSolver`** (torch → ONNX export) | +10–20 tasks (~190 pts) | 1 day | §3.7 |
| 15 | **Run `onnxsim`** on all generated models | +0.5 pts × 16 tasks = +8 pts | 1 h | §4.1 |
| 16 | **Delete `ReplaceColorSolver`** (subset of ColorMap) and `ExhaustiveColorMapSolver` (duplicate) | cleanup | 30 min | §2.5, §2.9 |
| 17 | **Fix `chain()` dead code** and `ColorMapThenTransformSolver` dead loop | cleanup | 30 min | §2.3, §2.4 |
| 18 | **Add `TransformThenColorMapSolver`** (reverse composition) | +3–5 tasks (~57 pts) | 2 h | §3.4 |

**Conservative estimate if items 1–10 are completed:** 16 + 5 + 10 + 3 + 30 + 20 + 5 + 5 = **~95 tasks solved**, total score ≈ 95 × 18.5 = **~1760 points** (vs current 300). Still well short of the leader's 8002, but a 5.8× improvement.

**Aggressive estimate if items 1–15 are completed:** ~120 tasks solved, total score ≈ 120 × 18.8 = **~2250 points**.

---

## 9. Appendix — Confirmed Bug Reproductions

### 9.1 MirrorConcatSolver failure (§2.1)

```
Synthetic task: input [[1,2,3],[4,5,6]] → output [[1,2,3,3,2,1],[4,5,6,6,5,4]]
Solver output: eligible=False, note="OK | ['Pair 0: mismatch', 'Pair 1: mismatch']"
```

### 9.2 ScaleUpSolver shape warning (§2.2)

```
Synthetic task: 2x2 → 4x4 nearest-neighbor scale
Solver output: eligible=True (passes by accident)
onnxruntime warning: "Error merging shape info for output. source:{1,10,60,60} target:{1,10,30,30}"
```

### 9.3 chain() works despite dead code (§2.3)

```
chain([color_map(0->5), identity]) on synthetic task → eligible=True
```
The dead first loop is overwritten by the list comprehension; functionally correct but fragile.

### 9.4 Per-solver success rate (from audit script)

```
cellular_automaton: 3/3 = 100.0%   (only 3 tasks even attempted — shape filter is strict)
color_map:          4/4 = 100.0%
crop_top_left:      3/3 = 100.0%
geom_transform:     2/7 = 28.6%    (5 attempts fail functional check — same padding issue as MirrorConcat)
kronecker:          1/1 = 100.0%
mirror_concat:      0/5 = 0.0%     ← BUG (§2.1)
multi_rule_ca:      1/1 = 100.0%
replace_color:      0/90 = 0.0%    ← always produces a model, never eligible (§2.5)
scale_up:           2/2 = 100.0%
shift:              0/1 = 0.0%     ← needs investigation
tile:               0/1 = 0.0%     ← needs investigation
```

`geom_transform` at 28.6% is suspicious — the flip/transpose models likely have the same padding-region issue as MirrorConcat. The 5 failures should be investigated; they may indicate that `GeometricTransformSolver` is also buggy for non-30×30 grids.
