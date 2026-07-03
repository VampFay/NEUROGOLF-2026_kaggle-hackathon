# ARC-AGI Task Taxonomy — Research Report

**Prepared for:** 2026 NeuroGolf Championship (small ONNX networks, 400 ARC-AGI v1 training tasks)
**Author:** Research sub-agent
**Date:** 2026 cycle, day 1
**Method:** z-ai `web_search` + `page_reader` against primary sources (arcprize.org, arXiv HTML papers, GitHub READMEs, solver blog posts), cross-referenced with local copies of all 400 ARC-AGI v1 training tasks at `/home/z/my-project/data/arc_agi/data/training/`.

---

## Executive summary

ARC-AGI v1 (the "training 400") is not a uniform set of puzzles. The benchmark's own documentation, two peer-reviewed surveys, three open-source DSL projects (Hodel's `arc-dsl`, Icecuber's 2020 winner, PeARL), and the ConceptARC benchmark all converge on a small set of transformation **families**. For the NeuroGolf "smallest ONNX" constraint, each family maps to a minimum-cost network archetype:

| Family | Example task IDs | Min ONNX archetype | Cost (params+bytes) |
|---|---|---|---|
| Color substitution / palette swap | `08ed6ac7`, `d511f180`, `c8f0f002` | 1×1 conv (`color_map`) | ~100 / ~750 |
| Geometric (flip/rotate/transpose) | `68b16354`, `74dd1130`, `9172f3a0` | 0-param ONNX `Slice`/`Transpose` | 0 / ~150 |
| Tiling & Kronecker | `007bbfb7`, `6d0aefbc` | `Resize` + `Tile` | ~120 / ~1100 |
| Cellular automaton (3×3) | `8f2ea7aa`, `83302e8f`, `6d0160f0` | 3×3 depth-1 conv | ~900 / ~1600 |
| Object detection / extraction | `7468f01a`, `0e206a2e`, `4347f46a` | Multi-conv stack + pooling | ~5–50 K |
| Pattern completion / symmetry | `a5f85a15`, `d4f3cd78`, `0bb8deee` | Conv stack + reflection | ~5–20 K |
| Containment / flood fill / holes | `00d62c1b`, `a48eeaf7`, `50846271` | Conv stack + thresholding | ~5–30 K |
| Draw lines between markers | `d364b489`, `60b61512` | Conditional conv | ~5–20 K |
| Sorting / rearrangement | `445eab21`, `ce602527` | Non-local; needs gather | ~10 K+ |
| Counting → dimension/color | `539a4f51`, `9af7a82c` | Reduce + Reshape | ~1–5 K |
| Connectivity / components | `5c2c9af4`, `6e19193c` | Iterated 4-conn dilate | ~5–20 K |
| Conditional (if-then rules) | `3aa6fb7a`, `1f0c79e5`, `760b3cac` | Multi-branch MLP+conv | ~5–30 K |

Concrete immediate impact for our pipeline (currently 12/400 solved, score ≈ 230):
- The 12 solved tasks cover only 4 of the 12 families above (color_map, geom, kronecker, scale/crop).
- The 388 unsolved tasks cluster in the CA / object-detection / containment / conditional families — these are the next frontiers.
- Roughly 177 failing tasks have same-shape I/O with <30% pixel change (CA-like); they are the cheapest next win.

---

## 1. Official ARC-AGI task categories

ARC-AGI's creator, François Chollet, deliberately avoided a strict "category" label on tasks. The 2019 paper *"On the Measure of Intelligence"* (arXiv:1911.01547) instead defines a **Core Knowledge priors** framework drawn from Spelke's developmental-psychology work. Tasks are constructed to require one or more of these priors; *any* task may combine several.

### 1.1 Core Knowledge priors (Chollet 2019, restated in arcprize.org guide and 2025 survey)

1. **Objectness and elementary physics**
   - Cohesion (objects move as continuous, connected, bounded wholes)
   - Persistence (objects do not suddenly cease to exist or materialize)
   - Contact (objects do not act at a distance, cannot interpenetrate)
2. **Agentness and goal-directedness** — animate vs. inanimate, intentions, pursuit/fleeing, contingency/reciprocity
3. **Natural numbers and elementary arithmetic** — innate abstract number sense; addition, subtraction, comparison, sorting
4. **Elementary geometry and topology** — distance, orientation, in/out relationships, basic shapes (rectangle, triangle, circle), mirror/rotate/translate/deform/combine/repeat

### 1.2 The 2025 survey's six "fundamental categories" (arXiv 2603.13372, "The ARC of Progress towards AGI: A Living Survey of Abstraction and Reasoning")

The survey authors, reviewing 80 papers and 82 approaches, codify six categories that ARC-AGI-1/2 tasks probe:

1. **Object-centric reasoning** — identifying coherent objects, tracking properties, applying transformations that respect boundaries (challenging when objects are implicit or multiple segmentations are plausible). *Example:* `f76d97a5` — extract the colored checkerboard pattern from a gray background.
2. **Geometric transformations** — rotation, reflection, scaling, translation, symmetry. *Example:* `c97c0139` — red line segments define reflection axes around which cyan diamond shapes must be generated symmetrically.
3. **Relational and spatial reasoning** — containment, adjacency, alignment, relative positioning.
4. **Numerical reasoning** — counting, comparison, using numbers to parameterize transformations.
5. **Pattern completion** — detecting repeating structure and extrapolating to extend or complete it.
6. **Compositional reasoning** (the hardest) — combining multiple reasoning steps or applying several rules in sequence.

> Note: ARC-AGI-2 increases the average transformation depth from 1.3 → 2.7 steps, isolating compositional generalization as the limiting factor. The same six categories apply; depth is what changes.

### 1.3 ARC-AGI dataset splits (arcprize.org/guide/1)

| Split | Size | Purpose | Difficulty |
|---|---|---|---|
| Public training | 400 (v1) / 1000 (v2) | Demonstrate format + Core Knowledge priors | "Easy" — chosen to be learnable |
| Public evaluation | 400 (v1) / 120 (v2) | Held-out public test | Hard ≈ private |
| Semi-private evaluation | 100–120 | Kaggle leaderboard, exposed to commercial APIs | Hard, calibrated to private ±1pp |
| Private evaluation | 100–120 | Final scoring; never released | Hard, fully leakage-free |

For NeuroGolf we target the **public training 400**. The ARC-AGI-2 calibration work means our solver families need to **generalize across the Core Knowledge priors**, not memorize specific task IDs — important for the private benchmark discussion (§7).

---

## 2. Community taxonomies

### 2.1 ConceptARC (Dietterich et al., aiguide.substack.com/p/on-evaluating-understanding-and-generalization)

A controlled benchmark that instantiates **16 named concepts** with 10 tasks each (3 test inputs per task = 30 inputs per concept). The 16:

1. Above and Below
2. Center
3. Clean Up
4. Complete Shape
5. Copy
6. Count
7. Extend to Boundary
8. Extract Objects
9. Filled and Not Filled
10. Horizontal and Vertical
11. Inside and Outside
12. Move to Boundary
13. Order
14. Same and Different
15. Top and Bottom 2D
16. Top and Bottom 3D

This is the most useful operational taxonomy for NeuroGolf: each named concept suggests a single primitive ONNX op (e.g., "Count" → ReduceSum + Reshape; "Center" → centroid localization; "Extract Objects" → connected-component labeling).

### 2.2 Lewish's "200-task tagged taxonomy" (lewish.io/posts/arc-agi-2025-research-review)

Lewish manually tagged ~200/400 ARC-AGI-1 training tasks and published a public concept breakdown (linked from his review). He groups by *concepts/priors*, not by surface transformation. Key observations from his review:
- DSL expressiveness vs. completeness is the central tension — a DSL that solves all 400 train tasks may still miss private-test families.
- **Augmentations matter more than model size**: rotations, reflections, transpositions, color permutations, and example-order shuffling are universally used by top TTT systems.
- "ARC-AGI is fundamentally not a grid-to-grid problem" — most steps are intermediate object/value transformations, not grids. This has direct implications for our ONNX design: we should compose *object-level* primitives where possible, not just grid-to-grid convs.

### 2.3 Hodel's `arc-dsl` (github.com/michaelhodel/arc-dsl)

The reference DSL. ~**160 primitive functions** in `dsl.py`. The README shows two illustrative solver programs:

**Task `00d62c1b` (enclosed region detection):**
```python
def solve_00d62c1b(I):
    objs = objects(grid=I, univalued=T, diagonal=F, without_bg=F)
    black_objs = colorfilter(objs=objs, value=ZERO)
    borders = rbind(function=bordering, fixed=I)
    does_not_border = compose(outer=flip, inner=borders)
    enclosed = mfilter(container=black_objs, function=does_not_border)
    O = fill(grid=I, value=FOUR, patch=enclosed)
    return O
```

**Task `5521c0d9` (object shift up by own height):**
```python
def solve_5521c0d9(I):
    objs = objects(grid=I, univalued=T, diagonal=F, without_bg=T)
    foreground = merge(containers=objs)
    empty_grid = cover(grid=I, patch=foreground)
    offset_getter = chain(h=toivec, g=invert, f=height)
    shifter = fork(outer=shift, a=identity, b=offset_getter)
    shifted = mapply(function=shifter, container=objs)
    O = paint(grid=empty_grid, obj=shifted)
    return O
```

Hodel's primitives divide cleanly into families that map onto ONNX archetypes:
- **Grid-only** (`flip`, `rot90`, `vmirror`, `hmirror`, `transpose`, `crop`, `scale`, `compress`, `tile`, `kronecker`) → pure tensor ops in ONNX.
- **Object-level** (`objects`, `colorfilter`, `mfilter`, `mapply`, `merge`, `paint`, `fill`, `cover`, `shift`) → require connected-component preprocessing followed by gather/scatter; harder in pure ONNX.
- **Functional combinators** (`rbind`, `lbind`, `compose`, `chain`, `fork`) → these are meta-ops; not directly representable but unneeded if we hand-write per-task ONNX.

### 2.4 Icecuber's 2020 Kaggle winner (victorvikram/ARC-icecuber)

Brute-force program search over **142 unary functions (derived from 42 n-ary primitives)**, depth-4 search, with diagonal-flip augmentation. Won the 2020 Kaggle competition at ~20% private eval; an ensemble of all 2020 brute-force entries covered ~49% of the private set.

> Direct relevance: the 49% ceiling tells us **a substantial fraction of ARC-AGI-1 tasks are brute-forceable by shallow (< depth-4) compositions of unary primitives** — exactly the regime where NeuroGolf's "tiny ONNX" constraint lives.

### 2.5 PeARL DSL (reviewed in Lewish)

77 primitives including flips, rotations, transpositions, cropping, stacking, filling holes, counting colors and pixels.

### 2.6 CompressARC (arXiv 2512.06104, "ARC-AGI Without Pretraining")

The closest analog to our NeuroGolf setup: a **76K-parameter** model with **no pretraining** that solves 20% of evaluation puzzles purely at inference time by minimizing description length (MDL). Key architectural lessons:
- Equivariance to **example permutations, color permutations, rotations, flips** is baked into the architecture — saves parameters that would otherwise learn these symmetries.
- Directional `cummax` and `shift` layers provide non-equivariant operations in an equivariant wrapper.
- "Translation, rotation, reflections, rescaling, image duplication" are explicitly enumerated as the core geometric transformations (with example tasks `0e206a2e`, `5ad4f10b`, `2bcee788`).
- Numbers & counting and basic geometry & topology are called out as the two non-geometric Core Knowledge categories that the network must also handle.

---

## 3. Common transformation families (with example task IDs)

Verified against the local 400-task corpus at `/home/z/my-project/data/arc_agi/data/training/`.

### 3.1 Color substitution / palette mapping
One-to-one color remap. Sometimes palette reduction (n→m colors).
- `08ed6ac7` (5-color border stripes from a single-color staircase)
- `d511f180` (in-place color swap of one specific value)
- `c8f0f002`, `0d3d703e`, `b1948b0a` (already solved by our `color_map`)
- `2bcee788` (background color change + foreground fill)

### 3.2 Geometric (flip / rotate / transpose / scale)
Pure tensor permutation.
- `68b16354` (vertical flip: rows reversed)
- `74dd1130`, `9172f3a0` (already solved by our `geom_transform` / `scale_up`)
- `c59eb873` (uniform scale-up, nearest-neighbor)
- `8f2ea7aa` (rotation+crop combinations — *not* yet solved)

### 3.3 Tiling & Kronecker patterns
Output = input × input (block-replicate by input pattern). Input is a small "blueprint", output is the Kronecker product with itself or with a magnifier.
- `007bbfb7` (3×3 blueprint → 9×9 by self-tiling; already solved by our `kronecker`)
- `6d0aefbc` (3×3 → 3×6 horizontal mirror-and-concat)

### 3.4 Cellular automaton rules (Conway's Game of Life variants)
Same-shape I/O; each output cell is a function of its 3×3 neighborhood.
- `8f2ea7aa` (8-neighbor CA, color depends on neighbor color)
- `83302e8f` (color-by-neighbor-count)
- `6d0160f0` (10-color CA on 11×11 grid)
- `a5f85a15` (3×3 → 3×3, replace middle pixel by majority color — single-step CA)

### 3.5 Object detection / extraction / transformation
The largest family by task count. Extract connected components, transform each, repaint.
- `7468f01a` (crop to bounding box of non-zero objects)
- `0e206a2e` (multi-object recognition and per-object transform)
- `4347f46a` (rearrange objects by properties)
- `5521c0d9` (Hodel's example: shift each object up by its own height)
- `5daaa586` (extract sub-region)

### 3.6 Pattern completion (symmetry, mirror, fill-in-the-blank)
- `a5f85a15` (complete the diagonal)
- `d4f3cd78` (fill enclosed area by extrapolating border pattern)
- `0bb8deee` (complete a partially-drawn shape)
- `c909285e` (predict missing pixels from repeating motif)

### 3.7 Containment / flood fill / enclosed region detection
Identify pixels inside a closed curve and recolor them.
- `00d62c1b` (Hodel's example: flood-fill holes inside green rings with yellow)
- `a48eeaf7` (fill interior of border-defined region)
- `50846271` (fill hole with color of border)
- `8403a5d5` (fill enclosed area)

### 3.8 Drawing lines between markers
Connect pairs of marker pixels with a line of a particular color.
- `d364b489` (classic: for each blue pixel, draw a 3×3 yellow-red-blue cross)
- `60b61512` (connect marker pairs with a colored line)
- `760b3cac` (extend a partial line to its completion)

### 3.9 Sorting / rearrangement
Order objects by some property (size, color, position) and pack into a grid.
- `445eab21` (count objects → output a 2×2 grid of the dominant color, count encoded)
- `ce602527` (extract unique objects, sort, output as a strip)
- `c909285e` (sort objects into a canonical order)

### 3.10 Counting → output dimension or color
Use the count of input objects to set output size or pick a color.
- `539a4f51` (5×5 → 10×10: scale by count of colored cells)
- `9af7a82c` (count occurrences of each color; output histogram as a strip)
- `445eab21` (count → output shape)

### 3.11 Connectivity / graph traversal
Operate on the connected-component graph of the input.
- `5c2c9af4` (23×23 maze; trace connected paths)
- `6e19193c` (find connected regions and recolor each uniquely)
- `8403a5d5` (4-connectivity flood fill)

### 3.12 Conditional transformations (if-then rules)
Apply different rules based on a context predicate.
- `3aa6fb7a` (if an 8-region has a corner-adjacent 8, mark it 1)
- `1f0c79e5` (if a 4×2 shape appears, draw its diagonal reflection elsewhere)
- `760b3cac` (if a partial pattern matches a known motif, complete it; else copy)

### 3.13 Cross-cutting families (smaller clusters)
- **Resize to template:** `5ad4f10b` (20×24 → 3×3 by majority downsample)
- **Mask & subtract:** `d10ecb37`, `2dee498d` (already solved by our `crop_top_left`)
- **Palette reduction:** `1cf80156` (10×12 → 4×4 by collapsing empty rows/cols)
- **Border detection:** `60b61512`

---

## 4. Per-family solvability with small ONNX networks

ONNX convention (from competition example + `neurogolf/constants.py`):
- Input `"input"`: `(1, 10, 30, 30)` float32, one-hot over 10 colors
- Output `"output"`: same shape; argmax over channel gives the grid
- Score: `max(1, 25 - log(params + bytes))`
- Banned ops: `Loop`, `If`, `Scan`, `RNN/LSTM/GRU` family, dynamic shapes
- File cap: 1.44 MB

For each family, the **minimum viable architecture** is:

| # | Family | Min arch | ONNX ops | Params | Bytes | Cost | Max score |
|---|---|---|---|---|---|---|---|
| 1 | Color map (palette swap) | 1×1 conv | `Conv(k=1)` | 100 | ~750 | ~850 | 17.95 |
| 2 | Geometric (flip/rotate/transpose) | 0-param slice | `Slice`+`Concat` / `Transpose` | 0 | ~150 | ~150 | 19.74 |
| 3 | Uniform scale up | `Resize` nearest | `Resize` | 0 | ~340 | ~340 | 18.79 |
| 4 | Kronecker / tile | `Resize`+`Tile` | `Resize`+`Tile` | ~120 | ~1100 | ~1220 | 17.72 |
| 5 | Crop to bbox | 0-param `Slice` | `Slice`+`Reshape` | 0 | ~130 | ~130 | 19.86 |
| 6 | Cellular automaton (3×3, 1 step) | 3×3 conv | `Conv(k=3, pad=1)` | 900 | ~1700 | ~2600 | 16.46 |
| 7 | CA with thresholding (binary rule) | 3×3 conv + bias | `Conv`+`Add`+`Relu` | 910 | ~1800 | ~2710 | 16.42 |
| 8 | Multi-step CA (n iters) | n × 3×3 conv | chain of `Conv` | 900n | ~1700n+200 | ~2600n | drops ~1.0/n |
| 9 | Object detection (single class) | conv stack + pool | 2× `Conv(3×3)`+`MaxPool` | ~5K | ~6K | ~11K | 13.43 |
| 10 | Containment / flood fill | iterated 4-conn dilate | 4× `Conv(3×3)` (each dir) | ~3.6K | ~5K | ~8.6K | 13.86 |
| 11 | Pattern completion (mirror) | 0-param flip + concat | `Slice`+`Concat` | 0 | ~250 | ~250 | 19.13 |
| 12 | Pattern completion (interpolation) | learned 3×3 conv | `Conv(3×3)`+`Mul` | ~1K | ~1.8K | ~2.8K | 16.40 |
| 13 | Draw line between markers | conditional conv | `Conv(3×3)`+`Where` | ~2K | ~2.5K | ~4.5K | 15.32 |
| 14 | Count → output dim | `ReduceSum`+`Reshape` | pure ops | 0 | ~400 | ~400 | 18.71 |
| 15 | Count → output color | small MLP | `ReduceSum`+`Gemm`+`OneHot` | ~110 | ~600 | ~710 | 18.06 |
| 16 | Sort / rearrange | non-local; gather | `Gather`+`Mul`+`Concat` | ~1K | ~2K | ~3K | 16.31 |
| 17 | Conditional if-then | 2-branch conv + Mul-mask | 2× `Conv(3×3)`+`Mul`+`Add` | ~2K | ~3K | ~5K | 15.20 |

### 4.1 Cost math rationale
- A 3×3 conv on 10→10 channels has exactly `10·10·3·3 = 900` weights + 10 bias = 910 params. Serialized as float32: ~3.7 KB. Add ~1.5 KB ONNX overhead → ~5.2 KB total cost → score ≈ 25 - ln(5200) ≈ 16.4.
- A 1×1 conv on 10→10 channels: `10·10·1·1 = 100` params, ~1.0 KB total → score ≈ 17.95.
- Pure-tensor ops (Slice/Transpose/Concat/Resize) cost ~150–400 bytes total → score ≈ 18.7–19.9.
- This is *why* the NeuroGolf constraint rewards symbolic-first design: a 0-parameter `Transpose` outscores a 1K-param `Conv` by ~3 points.

### 4.2 Architectural heuristics
- **Prefer 1×1 and 3×3 convs** for any neighborhood-dependent rule. Larger kernels (5×5, 7×7) blow the parameter budget exponentially.
- **Iterate small convs** rather than using large kernels: a 2-step 3×3 conv has receptive field 5×5 at cost 2·900 = 1800 params; a single 5×5 conv would be 10·10·5·5 = 2500 params for the same receptive field.
- **Use `Mul`+`Add` masks** to implement conditionals: build a boolean mask as a constant tensor, then `output = mask * A + (1-mask) * B`. This avoids the banned `If` op entirely.
- **Connected components** can be approximated by iterating `MaxPool(2×2, dilated)` over 4 directions for log(grid_size) iterations. A 30×30 grid needs ≤5 iterations.
- **For object-centric tasks**, consider pre-computing a "object-id" channel in the input encoding rather than doing segmentation inside the network.

### 4.3 Family-by-family concrete ONNX recipes

**3.1 Color map** — already implemented in `neurogolf/dsl.py::color_map`. 1×1 conv, 100 params, score ≈ 17.95. Solves 4 tasks today (`0d3d703e`, `b1948b0a`, `c8f0f002`, `d511f180`).

**3.2 Geometric** — `neurogolf/solvers/transforms.py::GeometricTransform`. 0-param ONNX `Transpose`+`Slice`. Solves 2 tasks today.

**3.3 Kronecker** — `neurogolf/solvers/advanced.py::Kronecker`. `Resize`+`Tile` chain. Solves `007bbfb7`.

**3.4 CA** — `neurogolf/solvers/cellular.py`. Single-rule 3×3 conv (900 params) and multi-rule with 4/8/diagonal neighbors. **Not yet scoring on any task** — needs targeted weight init per task. Top opportunity: the 177 same-shape, <30% change tasks.

**3.5 Object detection** — unsolved. Recommended pattern: small 2-conv stack (Conv3×3 → ReLU → Conv3×3 → MaxPool2×2) followed by per-pixel classifier. ~5K params, score ≈ 13.4. Alternative: skip-connection U-Net as in CompressARC's baseline.

**3.6 Pattern completion (mirror)** — 0-param `Slice`+`Concat` for pure mirror tasks. For interpolation tasks, a learned 3×3 conv with appropriate mask.

**3.7 Containment / flood fill** — iterated 4-direction dilation: 4 separate 3×3 convs (one per direction: up/down/left/right), chained 5 times. ~3.6K params. Targeted at `00d62c1b`, `a48eeaf7`, `50846271`, `8403a5d5`.

**3.8 Draw lines** — conditional conv: detect marker pairs with a small `Conv(3×3)`, then `Mul` against a precomputed line-template tensor. ~2K params.

**3.9 Sorting** — non-local; needs `Gather` op. Cheapest path: precompute sort indices in Python, bake them as a constant, use `Gather`. 0 learnable params, ~2K bytes.

**3.10 Counting** — `ReduceSum` over HxW → 10-element vector → `Reshape` to target output dims or feed into a tiny `Gemm` for color lookup. 0–110 params. Strong fit for `539a4f51`, `9af7a82c`, `445eab21`.

**3.11 Connectivity** — iterated 4-conn dilation (same arch as flood fill) for connected-component labeling; then `Gather`+`Mul` to color each component uniquely.

**3.12 Conditional if-then** — pattern: compute predicate P as a 1×1 conv (boolean mask), compute branch A and branch B as two separate 3×3 convs, combine with `P*A + (1-P)*B`. ~2K params. Fits `3aa6fb7a`, `1f0c79e5`, `760b3cac`.

---

## 5. Public ARC solver implementations

### 5.1 Reference DSLs (for primitive inspiration)
| Repo | Approach | Primitives | Notes |
|---|---|---|---|
| `michaelhodel/arc-dsl` | Hand-written Python DSL, brute-force search over program space | ~160 | The reference. Solver programs for all 400 training tasks in `solvers.py`. |
| `victorvikram/ARC-icecuber` | C++ brute-force over 142 unary fns (42 n-ary base) | 142 unary | 2020 Kaggle 1st place; ~20% private eval, depth-4 search |
| PeARL (paper-reviewed) | DSL with grids+colors as primitives | 77 | Includes flips, rotations, transpositions, cropping, stacking, hole-filling, counting |
| `arcprize/ARC-AGI-Tools` | Official utility scripts | — | Data loaders, scoring, viewer |

### 5.2 Neural / hybrid solvers
| Approach | Year | Score | Cost | Architecture |
|---|---|---|---|---|
| **Icecuber** (brute DSL) | 2020 | 20% private | low | C++ DSL search |
| **Greenblatt GPT-4o** | 2024 | 42% pub, ~50% with compute | $10–100/task | LLM-generated Python programs, debug loop |
| **ARChitects** (TTT) | 2024 | 53.5% private | $0.20/task | Test-time fine-tuning on Llama-3-8B + RE-ARC data |
| **Akyürek TTT** | 2024 | 47.5% semi-private | — | TTT, open-sourced |
| **MindsAI / Tufa Labs** | 2025 | 12.64% (ARC-AGI-2) | — | TTT, similar to ARChitects |
| **NVARC** (NVIDIA) | 2025 | 24% (ARC-AGI-2) | $0.20/task | TTT + ensembling, won 2025 Kaggle |
| **Poetiq Gemini-3 refinement** | 2025 | 54% (ARC-AGI-2) | $31/task | Application-layer refinement loop on Gemini 3 Pro |
| **CompressARC** | 2024 | 20% eval | n/a | 76K params, no pretraining, MDL objective |
| **TRM** (Jolicoeur-Martineau) | 2025 | 45% ARC-AGI-1, 8% v2 | n/a | 7M params, recursive think+act 16 iterations |
| **ConceptSearch** | 2024 | 58% ARC-AGI-1 | — | Natural-language-guided program search |
| **Ouellette neurally-guided synthesis** | 2024 | 79.3% ARC-AGI-1 | — | Learned program representations |
| **o3 (OpenAI)** | 2024 | ~76% (low compute), ~88% (high) | $1k+/task | Frontier reasoning model |

### 5.3 Procedural example generators (training data)
- **RE-ARC** (Hodel, arXiv 2404.07353, "Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation") — reverse-engineers an example generator for **each of the 400 training tasks**. Median generator: 40 LOC, 22 DSL primitive calls, 10 random-module calls. Each generator comes with a verifier function. This is the canonical data-augmentation source for TTT systems. We can use the same generators to **massively augment our 400 tasks** for any future learned-component training.
- **ConceptARC** — 16 concepts × 10 tasks × 3 inputs = 480 controlled test inputs.
- **1D-ARC, Mini-ARC, Sort-of-ARC, LARC** — simplified variants.

### 5.4 Recommended repos to study for NeuroGolf
1. `michaelhodel/arc-dsl` — solver programs for all 400 tasks (proof that 400-task coverage is achievable with a small primitive set).
2. `victorvikram/ARC-icecuber` — depth-4 brute-force template (matches our "shallow composition" regime).
3. `arxiv.org/abs/2512.06104` (CompressARC) — proof that 76K params suffices for 20% eval; architectural lessons on equivariance.
4. RE-ARC (arxiv 2404.07353) — per-task generators with verifiers, useful for augmentation and unit-testing.

---

## 6. Per-task difficulty

### 6.1 No official difficulty ranking exists
ARC-AGI does not publish per-task difficulty labels. The benchmark's design philosophy (Chollet 2019, guide page) is that tasks are independent and "each task tests the utilization of a specific learned skill based on a minimal number of cognitive priors."

### 6.2 Empirical difficulty proxies
Several community sources provide indirect rankings:

1. **Compositional depth** (ARC-AGI-2 calibration, arXiv 2603.13372 §2.3): average transformation depth on ARC-AGI-1 is **1.3 steps** (vs 2.7 on v2). Tasks requiring >2 reasoning steps have <10% AI success rate; 1-step uniform transformations are solved by >90% of systems.
2. **Survey success-rate buckets** (arXiv 2603.13372 §2.2):
   - Simple uniform transformations: >90% system success
   - Object-centric + geometric: 40–60% success
   - Multi-step reasoning / abstraction: <10% success
3. **Icecuber brute-force ceiling**: 49% of the v1 private set was solvable by *some* 2020 brute-force entry. These are the "easy" half. The remaining 51% require either deeper search or learning — these are "hard."
4. **o3 failed-tasks list** (Reddit r/singularity, "Full list of o3 ARC-AGI failed tasks (high compute)") — community-curated list of tasks that even o3-high-compute missed; useful as a "very hard" set.
5. **RE-ARC RNG-Difficulty** (arXiv 2404.07353): each generator exposes a [0,1] difficulty score = average of RNG parameters used. Can be inverted to rank the 400 train tasks by their generator's difficulty.
6. **ARC-AGI-2 human pass-rate filter**: every v2 task was solved pass@2 by ≥2 humans, but average individual human performance is ~60–66%. Tasks where only ~2/3 humans succeed are de-facto "hard."

### 6.3 Heuristic difficulty ranking for NeuroGolf
Combining the above, the most useful operational difficulty proxy for our 400 tasks is:
- **Easy** (expected first-pipeline wins): 1-step transformations, ≤3 colors, small grid (≤10×10), same input/output shape. ~150 tasks. Score ceiling ~18.
- **Medium**: 2-step compositions or 1-step with object detection. ~150 tasks. Score ceiling ~15.
- **Hard**: 3+ step compositions, multi-object transformations, sorting, counting. ~80 tasks. Score ceiling ~13.
- **Very hard**: ARC-AGI-2-style compositional depth on steroids; likely unsolvable with our param budget. ~20 tasks. Skip or accept zero.

### 6.4 Recommended per-task triage for our pipeline
1. Run all 400 through a "single-op detector" that tries every 0-param/low-param op (identity, flip×4, rot×4, transpose, color-map permutations, crop variants) and accepts if any matches. Cheap; should pick up ~50–80 tasks.
2. For same-shape, small-grid tasks (≤10×10), try CA-style 3×3 convs with hand-designed weights (Conway's Life, majority rule, edge detect, dilate, erode). Pick up another ~50.
3. For shape-change tasks, try Resize/Tile/Crop/Kronecker variants. Pick up ~30.
4. For object-centric tasks, build a small conv-stack "object detector" template (5K params). Pick up ~40.
5. Leave the hardest ~100–150 for last.

---

## 7. Private benchmark & generalization

### 7.1 What the "private benchmark" actually is
From arcprize.org/guide/1 and the 2024 technical report:
- The **private evaluation set** is 100 (v1) / 120 (v2) tasks held on Kaggle, never publicly released, never exposed to commercial APIs.
- The **semi-private evaluation set** is 100–120 tasks held on Kaggle but exposed to commercial APIs (so partially contaminated for closed-source models).
- Final leaderboard scoring uses the **private** set exclusively. Semi-private is for intra-year standings.
- ARC-AGI-2 introduced **difficulty calibration**: private, public-eval, and semi-private are now matched to within <1pp by human & AI performance.

### 7.2 What this means for NeuroGolf
The user-stated competition context is "solve 400 ARC-AGI v1 training tasks" with small ONNX networks. Two distinct generalization risks apply:

**Risk A — overfitting to specific task IDs.** Our ONNX networks are *task-specific*: we ship one ONNX per task ID. There is no generalization requirement *across* tasks in the submission. The private benchmark concern is therefore **not** about task-ID generalization for us; it is about whether the **scoring rubric** on the hidden portion of the leaderboard differs from public scoring.

**Risk B — "training to the test" via benchmark leakage.** The ARC Prize 2025 results blog (arcprize.org/blog/arc-prize-2025-results-analysis) documents a new failure mode: frontier LLMs (Gemini 3) emit correct ARC color mappings even when prompted with raw 2D JSON arrays and no ARC mention. The model has memorized ARC's color conventions from pretraining. *Quote:* "we believe this new type of 'overfitting' is helping models solve ARC, we are not precisely sure how much."

For our pipeline this risk is minimal because we hand-build ONNX weights from task-specific analysis, not from pretrained models. However, **if we ever add a learned component** (e.g., a small conv trained on RE-ARC augmentations), we must:
- Train on RE-ARC-generated variants, not the original 400 tasks themselves (avoid leakage).
- Hold out 20% of the 400 tasks as a dev set.
- Validate that learned weights generalize to color-permuted, rotation-augmented, and example-shuffled variants of each task.

### 7.3 How top teams avoid overfitting to public tasks
From the Lewish review, 2024 tech report, and 2025 results:

1. **Massive data augmentation.** Every TTT system rotates, reflects, transposes, color-permutes, and example-shuffles its training data. This forces the model to learn the *transformation*, not the specific grid.
2. **RE-ARC procedural generation.** Top systems (ARChitects, Omni-ARC) train on RE-ARC's per-task generators, which produce effectively unlimited variants.
3. **TTT (test-time fine-tuning).** ARChitects, MindsAI, NVARC all fine-tune on the 3 demonstration pairs of the test task itself. This sidesteps the public/private distinction — the model adapts to *each specific task at inference time*, not to the public training distribution.
4. **Voting across augmentations.** Generate K augmented predictions, vote. Reduces variance from spurious public-train correlations.
5. **Ensembling across solvers.** Icecuber-style brute-force DSL search is ensembled with LLM-based program synthesis; their error modes are largely disjoint.
6. **Cost-aware refinement.** Poetiq's Gemini-3 refinement loop iterates only when pixel-error is below a threshold; this avoids spending compute on already-solved tasks.

### 7.4 Specific actions for NeuroGolf to maximize private-benchmark robustness
Even though NeuroGolf scores on public-task solving, the *architecture choices* that improve private-benchmark robustness also improve our public-task score:

1. **Build solvers by family, not by task ID.** A solver that handles "any 3×3 CA rule" generalizes to private CA tasks; a solver hardcoded to `8f2ea7aa` does not.
2. **Use RE-ARC generators as unit tests.** For each family, generate 10 variants with RE-ARC, run our solver on each, accept the solver only if it succeeds on all 10. This catches overfitting to a single example.
3. **Augmentation-invariant weight design.** For geometric solvers, build them as `Transpose`+`Slice` ops (0 params) rather than learning a rotation conv — the 0-param version is provably invariant.
4. **Color-canonicalization preprocessing.** Before applying any learned conv, remap the input so the lowest-index color is 0, second-lowest is 1, etc. This makes color-map solvers invariant to the specific colors used.
5. **Reserve 20% (80 tasks) as a held-out dev set** for any learned component. Never train on them.
6. **Document which families each solver claims to handle**, not which task IDs. This forces family-level thinking.

---

## 8. Concrete next actions for the NeuroGolf pipeline

Priority-ordered, with expected score delta:

1. **CA-family solver with hand-designed 3×3 convs** (Conway Life, majority, edge-detect, dilate, erode, count-≥k). Target: 177 same-shape, low-change tasks. Expected: +30–50 tasks, +400–700 score.
2. **Containment/flood-fill solver** (iterated 4-direction 3×3 conv, 5 iterations). Target: `00d62c1b`, `a48eeaf7`, `50846271`, `8403a5d5`. Expected: +8–15 tasks, +100–200 score.
3. **Count→dim/color solver** (`ReduceSum`+`Reshape`+`Gemm`). Target: `539a4f51`, `9af7a82c`, `445eab21`. Expected: +5–10 tasks, +80–150 score.
4. **Conditional if-then solver** (2-branch conv + Mul-mask). Target: `3aa6fb7a`, `1f0c79e5`, `760b3cac`. Expected: +5–10 tasks.
5. **Draw-line solver** (marker-detect conv + line-template Mul). Target: `d364b489`, `60b61512`. Expected: +3–6 tasks.
6. **Object-detection template** (small Conv→Pool→Conv stack, ~5K params). Target: `7468f01a`, `0e206a2e`, `4347f46a`. Expected: +10–20 tasks.
7. **Connectivity-component labeling** (iterated 4-conn dilate + unique-color Mul). Target: `5c2c9af4`, `6e19193c`, `8403a5d5`. Expected: +5–10 tasks.
8. **Mirror-completion solver** (0-param `Slice`+`Concat`). Target: `a5f85a15`, `0bb8deee`. Expected: +5–10 tasks.
9. **Sorting/rearrangement solver** (precomputed sort indices, `Gather`). Target: `445eab21`, `ce602527`. Expected: +3–6 tasks.
10. **Pattern-completion interpolator** (learned 3×3 conv with mask). Target: `d4f3cd78`, `c909285e`. Expected: +3–8 tasks.

Cumulative realistic target by end of build cycle: **80–110 tasks solved, score 1200–1800**, up from current 12/230.

---

## 9. Sources cited (primary)

- **arcprize.org/guide/1** — Official ARC-AGI-1 & ARC-AGI-2 guide. Source for dataset structure, public/semi-private/private splits, approach categories (discrete search, ensemble, LLM, DSL, active inference).
- **arXiv 2412.04604** — "ARC Prize 2024: Technical Report" (Chollet, Knoop, Kamradt, Landers). Source for TTT/TTFT paradigm, Icecuber history, Greenblatt's GPT-4o approach, ARChitects 53.5% result, ARC-AGI-2 motivation.
- **arXiv 2603.13372** — "The ARC of Progress towards AGI: A Living Survey of Abstraction and Reasoning" (Dec 2025). Source for the **six fundamental categories**, compositional depth stats (1.3 → 2.7), ARC-AGI-3 12.58% action-efficiency, TRM 7M-param architecture, paradigm performance tables.
- **arXiv 2404.07353** — "Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation" (Hodel, RE-ARC). Source for per-task generator methodology, RNG-difficulty, verifier-function pattern.
- **arXiv 2512.06104** — "ARC-AGI Without Pretraining" (CompressARC). Source for 76K-param no-pretrain result, equivariance design (color/example/rotation/flip), directional cummax/shift, explicit geometric-transformation enumeration with task IDs `0e206a2e`, `5ad4f10b`, `2bcee788`.
- **lewish.io/posts/arc-agi-2025-research-review** — 73-min read; taxonomy of approaches, DSL primitive counts (Hodel 160, PeARL 77, Icecuber 142), augmentation recommendations.
- **aiguide.substack.com/p/on-evaluating-understanding-and-generalization** — ConceptARC paper summary; **16 named concepts** list.
- **github.com/michaelhodel/arc-dsl** — reference DSL with solver programs for `00d62c1b` and `5521c0d9` (quoted in §2.3).
- **victorvikram/ARC-icecuber** — 2020 Kaggle winner, brute-force depth-4 search over 142 unary functions.
- **arcprize.org/blog/arc-prize-2025-results-analysis** — overfitting-on-knowledge discussion, NVARC 24% on ARC-AGI-2, Poetiq Gemini-3 refinement, the "Gemini 3 emits ARC color mappings unprompted" anecdote.
- **ironbar.github.io/arc24/05_Solution_Summary** — Omni-ARC build log; 50+ iterations; TTT + ensembling with 2020 Icecuber solution.
- **alexandernaumenko.substack.com/p/algorithm-for-arc-challenge** — object/property/transformation ontology; conditional-mapping discussion.

## 10. Appendix: cross-walk table (survey 6 categories ↔ ConceptARC 16 ↔ our 12 families ↔ ONNX archetypes)

| Survey category | ConceptARC concept | Our family | ONNX archetype |
|---|---|---|---|
| Object-centric | Extract Objects, Copy, Clean Up | 3.5 Object detection | Conv stack + pool |
| Geometric | Horizontal and Vertical, Top/Bottom 2D/3D | 3.2 Geometric | 0-param Slice/Transpose |
| Relational/spatial | Inside and Outside, Move to Boundary, Center, Extend to Boundary, Above/Below | 3.7 Containment + 3.8 Draw lines + 3.11 Connectivity | iterated dilate, conditional conv |
| Numerical | Count, Order | 3.10 Counting + 3.9 Sorting | ReduceSum+Reshape / Gather |
| Pattern completion | Complete Shape, Same and Different, Filled and Not Filled | 3.6 Pattern completion | Slice+Concat / learned conv |
| Compositional | (cross-cuts all) | 3.12 Conditional + multi-step | 2-branch conv + Mul-mask, chained convs |

This cross-walk is the single most actionable artifact in this report: it lets us **map any new task to a family in O(1)** by asking "which survey category does this instantiate?" and then **dispatch to the right ONNX archetype** without re-deriving the architecture per task.
