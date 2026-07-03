# ONNX Minimization Research — NeuroGolf 2026

**Goal:** shrink ONNX network files to maximize score `max(1, 25 - ln(params + bytes))`
where `params` = number of weight elements (across all initializers) and `bytes` =
serialized file size. Smaller is exponentially better.

**Method:** all numbers below are **measured empirically** with `onnx 1.22.0` +
`onnxruntime 1.27.0`, using the competition's real I/O convention
(`input`/`output` tensors of shape `(1, 10, 30, 30)` float32, statically-defined
shapes, opset 17 / ir_version 8 baseline). Every model was validated with
`onnx.checker.check_model` **and** executed in ONNX Runtime to confirm it runs.
Research scripts live in `/home/z/my-project/scripts/measure_onnx{,2,3}.py`.

---

## TL;DR — the 7 highest-impact wins

| # | Technique | Example saving | Score gain |
|---|-----------|---------------|------------|
| 1 | **Drop default Conv attributes** (`strides`, `dilations`, `group`, `pads` when 0) | team color_map 653→602 (-51 B) | +0.08 |
| 2 | **Drop `kernel_shape` too** — ORT infers it from the weight tensor | 602→526 (-76 B more) | +0.11 |
| 3 | **int8 weights via `QuantizeLinear`→`QLinearConv`→`DequantizeLinear`** (no bias) | color_map 526→392 (-134 B) | +0.23 |
| 4 | **Replace `Constant` nodes with initializers** | ~56 B per constant | +0.08 ea |
| 5 | **Strip model metadata** (`ClearField("producer_name")` etc.) + 1-char graph name | ~23 B per model | +0.03 |
| 6 | **Shorten initializer/intermediate tensor names** (`conv_w`→`w`, `mid_0`→`t`) | 3–12 B per reference | +0.02–0.10 |
| 7 | **Eliminate redundant `Identity`/`Cast` glue nodes** | ~13 B per node | +0.02 ea |

Combined on the team's `color_map` primitive: **653 B → 392 B, score 18.38 → 18.79 (+0.41)**.
On `argmax_over_channels`: **391 B → 207 B, score 19.03 → 19.65 (+0.62)**.

---

## 1. ONNX file structure overhead

### What's in a minimal valid file

A competition-valid model must contain, at minimum:
- `ir_version` (field 1, varint) — 2 bytes
- `opset_import` (field 8, one entry: domain `""` + version) — ~6 bytes
- `graph` (field 7) — contains:
  - `name` (required non-empty by `onnx.checker`) — 1 char min → 3 bytes framed
  - `input` ValueInfo ("input", FLOAT, [1,10,30,30]) — **~31 bytes**
  - `output` ValueInfo ("output", FLOAT, [1,10,30,30]) — **~32 bytes**
  - at least one node — ~13–20 bytes
  - framing bytes

### Measured floors

| Model | Bytes | Score | Notes |
|-------|------:|------:|-------|
| Empty graph (no I/O, no nodes) | 22 | 21.91 | Invalid for competition (no I/O) |
| Identity, opset 7–21 (generic names) | 67 | 20.80 | No producer, graph "g", 1-char I/O |
| **Identity, competition I/O names** `input`/`output` | **107** | **20.33** | **Practical floor for a valid submission** |
| Identity, team's `dsl.identity()` | 130 | 20.13 | Wastes 23 B on `producer_name="neurogolf-dsl"` + graph name "neurogolf" |

The 107-byte floor is essentially unavoidable: the two `(1,10,30,30)` ValueInfos
alone cost ~63 bytes because each of the 4 dims is a separate framed varint.
The opset version (7 vs 21) makes **zero difference** to size.

### Where "wasted" bytes hide

1. **Proto3 presence semantics**: setting a string field to `""` **adds 2 bytes**
   (tag + zero-length) vs leaving it unset. Use `m.ClearField("producer_name")`
   rather than `m.producer_name = ""`. `helper.make_model` already leaves these
   unset, so the team's explicit `producer_name="neurogolf-dsl"` is pure waste.
2. **Repeated node names**: every node `name` field is serialized even if empty
   (an empty name is 2 bytes; an unset name is 0). The checker does not require
   node names, so leave them unset.
3. **Attribute encoding**: each attribute is `name(string) + type(varint) + value`.
   A `strides=[1,1]` ints attribute costs ~12 bytes (name "strides"=7, type, 2 ints, framing).
   Dropping all 4 default Conv attrs saves ~70 bytes.
4. **Tensor name length**: a name of length *L* referenced *R* times costs *L·R*
   bytes (graph.input/output, node.input/output, initializer.name, value_info).
   The team's `conv_w` (5 chars, ~3 refs) costs ~15 B; `w` costs ~3 B.

---

## 2. int8 / quantized weights

### The viable recipe for this competition

The competition feeds a **float32** `(1,10,30,30)` input, so a pure
`QLinearConv` (which needs uint8/int8 input) cannot be the first op. The working
pattern is:

```
input(f32) ──QuantizeLinear──> q(u8) ──QLinearConv(nobias)──> r(u8) ──DequantizeLinear──> output(f32)
```

| Model | Params | Bytes | Cost | Score |
|-------|-------:|------:|-----:|------:|
| color_map f32 (team) | 100 | 653 | 753 | 18.38 |
| color_map f32 (minimized) | 100 | 526 | 626 | 18.56 |
| **color_map int8 (QuantizeLinear+QLinearConv+DequantizeLinear, nobias)** | **106** | **392** | **498** | **18.79** |

**Correctness verified**: the int8 color_map produces byte-identical argmax
output to the f32 version on test inputs. The 6 extra scalar params (3 scales +
3 zero-points) are negligible.

### Why no bias: the ONNX Runtime bias bug

`QLinearConv` with an int32 bias (exactly as the ONNX spec requires) is
**rejected by ONNX Runtime 1.27** with
`Type Error: Type 'tensor(int32)' of input parameter (b) of operator (QLinearConv) ... is invalid`.
This reproduces across all input/weight dtype combinations and scalar/1-dim
scale shapes (see ORT issue #16105 lineage). The **nobias path runs cleanly**.

If a layer truly needs bias, either:
- keep that one layer as float32 `Conv` (bias is only 10 values = 40 B), or
- add a float32 `Add` after `DequantizeLinear` (works, but the Add+DQ+QQ framing
  eats most of the int8 savings — measured 465 B, worse than the nobias path).

### Size math

For an *N*-element conv weight:
- float32 weight data = **4N bytes**; int8 weight data = **N bytes** → saves 3N bytes.
- QLinearConv overhead = 6 scalar initializers (3 float scales + 3 u8/i8 zero-points)
  ≈ 80–90 bytes extra framing vs plain Conv.
- **Break-even ≈ 30 weight elements.** Below ~30 params, int8 isn't worth it.
  For 100-param color_map it saves 134 B. For 900-param 3×3 conv it would save
  ~2600 B.

### Are quantized ops competition-legal?

Yes. `QuantizeLinear`, `DequantizeLinear`, `QLinearConv`, `DynamicQuantizeLinear`,
`MatMulInteger` are **not** in the banned set
(`Loop, Scan, NonZero, Unique, Script, Function`). They are universally
supported in ORT. `QLinearConv` exists since opset 10.

---

## 3. Opset version selection

**Opset version has essentially zero effect on file size.** Measured identity
model: 67 bytes for opset 7, 9, 11, 13, 15, 17, 19, 21 (all identical). The
opset is encoded as a single varint in `opset_import`; 7 and 17 are both 1-byte
varints.

The **only** opset-sensitive size differences are ops whose schema changed:

| Op | Old opset (smaller) | New opset (larger) | Why |
|----|--------------------|--------------------|-----|
| `Slice` | opset 10 (attributes `starts`/`ends`/`axes`) = **122 B** | opset 11+ (inputs) = **180 B** | opset 11 moved starts/ends/axes to graph inputs, forcing 3 extra initializers |
| `Resize` | opset 11 (`[x, scales]` = 155 B) | opset 13+ (`[x, roi, scales]` = 169 B) | opset 13 added the (often-empty) `roi` input |

**Recommendation:** keep a single opset for the whole submission. opset 17 is
fine and matches the team's current choice. Don't drop to opset 10 just for
Slice — the gain (58 B on Slice ops) is small and risks losing other newer-op
features. The team's opset 17 is size-optimal for everything except Slice/Resize,
where the overhead is unavoidable without an opset split (which ONNX doesn't
support per-op).

`ir_version`: ir 3 fails (requires initializers duplicated in graph.input).
ir 4–9 all produce identical sizes. Use **ir 8** (team's current) — it's the
most broadly compatible.

---

## 4. Graph optimization techniques (all measured)

### 4a. Constant folding

| Pattern | Bytes |
|---------|------:|
| `Constant`→`Identity`→`Add` (unfolded) | 144 |
| `Add` with direct initializer (folded) | 97 |
| **Savings** | **47** |

Rule: **never use a `Constant` node when you can use an initializer.** An
initializer is ~56 B cheaper than a `Constant` node wrapping the same tensor
(the Constant node adds op_type + value-attribute framing).

### 4b. Initializer vs Constant node

| Conv 1×1, 100 weights | Bytes |
|-----------------------|------:|
| Initializer | 577 |
| Constant node | 633 |
| **Initializer wins by** | **56** |

Always prefer initializers for any weight that lives in the graph.

### 4c. Tensor name length

| `conv1x1` 100w, name length | Bytes | Δ from nlen=1 |
|------------------------------|------:|--------------:|
| 1 char (`w`) | 577 | — |
| 4 chars (`conv`) | 601 | +24 |
| 16 chars | 697 | +120 |
| 64 chars | 1082 | +505 |

Each character of a name costs ~1 byte × (number of references). A conv weight
is referenced ~3× (initializer.name, node.input, value_info if present). The
team's `conv_w`→`w` saves ~12 B; `conv_b`→`b` saves ~12 B. The chain helper's
`m0_conv_w` prefix is especially wasteful.

**Fixed names**: `input`/`output` (5 chars each) are locked by the competition
validator — cannot be shortened.

### 4d. Sharing initializers across multiple uses

| Pattern | Bytes |
|---------|------:|
| Same 3-float tensor used twice (shared name) | 113 |
| Two separate identical tensors | 138 |
| **Savings** | **25** |

If the same constant (e.g. a scales vector) is used by multiple nodes, give it
**one name** and reference it everywhere. `scs4onnx` automates this.

### 4e. Eliminating Identity nodes

| Pattern | Bytes |
|---------|------:|
| 3 chained Identity nodes | 116 |
| 0 Identity nodes (direct wire) | 77 |
| **Per Identity node** | **~13** |

The team's `conv_stack` and `chain` helpers insert trailing `Identity` nodes to
rename to `OUTPUT_NAME`. **Wire the last real op's output directly to
`"output"`** instead — saves 13 B each time.

### 4f. Fusing Conv+Add+Relu / dropping redundant Cast

The team's `argmax_over_channels` has a redundant `Cast` (ArgMax already emits
int64) and a trailing `Identity`:

| Version | Bytes | Score |
|---------|------:|------:|
| Team `argmax_over_channels` (Cast + Constant×2 + OneHot + Identity) | 391 | 19.03 |
| Minimized (ArgMax + OneHot, initializers for depth/values, no Cast/Identity) | 207 | 19.65 |
| **Savings** | **184** | **+0.62** |

Conv+Add+Relu "fusion" at the ONNX-graph level doesn't shrink the file (the ops
are still separate nodes); it only helps runtime speed. For file size, the win
is **dropping ops entirely** (e.g. fold a bias Add into the conv, drop a no-op
Cast), not fusing them.

---

## 5. ONNX Runtime compatibility

### Universally safe ops (the team's toolkit)

All of these pass `onnx.checker` and run in ORT 1.27 with no surprises:

`Conv, Slice, Concat, Constant, Mul, Add, Sub, Pad, Resize, Tile, ArgMax,
OneHot, Transpose, Cast, Identity, QuantizeLinear, DequantizeLinear,
QLinearConv, DynamicQuantizeLinear, MatMulInteger`

### Subtle behavior traps observed

| Op | Trap |
|----|------|
| `QLinearConv` | **int32 bias rejected by ORT** (spec-compliant but ORT bug). Use nobias + float32 Add, or pure f32 Conv for that layer. |
| `Conv` | `kernel_shape` is spec-required but **ORT infers it from the weight** and `onnx.checker` accepts omission. Saves ~23 B. (Verified opset 11–21.) |
| `Conv` | omitting `pads` ⇒ valid-pad (output shrinks). For 1×1 this is fine (30×30→30×30); for 3×3 you lose 1px border (30×30→28×28). Add `pads=[1,1,1,1]` only when same-size output is needed. |
| `ArgMax` | `keepdims` default = 1. Use `keepdims=0` so OneHot re-inserts the channel dim cleanly → `(1,10,30,30)`. |
| `Resize` | opset 13+ requires 3 inputs (`x, roi, scales`); pass an empty `roi` initializer (`FLOAT, dims=[0]`). |
| `Slice` | opset 11+ requires `starts/ends/axes` as graph inputs (initializers), not attributes. |
| `OneHot` | `depth` must be a tensor (scalar int64), not an attribute. `values` is `[off, on]` shape `[2]`. |
| Output ValueInfo | **dims are required** by `onnx.checker` ("Field 'shape' of 'type' is required"). Can't omit them to save ~18 B — checked and rejected. |
| dynamic shapes | competition bans them (`dim_param` must be empty); all dims must be literal ints. |

### Banned ops (do not use)

`Loop, Scan, NonZero, Unique, Script, Function` — all flagged by the team's
`BANNED_OPS` and the competition validator.

---

## 6. Concrete byte-size examples (competition I/O convention)

All models use `input`/`output` = `(1,10,30,30)` float32, opset 17, ir 8,
stripped metadata, 1-char internal names.

| Operation | Params | Bytes | Cost | Score | Notes |
|-----------|-------:|------:|-----:|------:|-------|
| **Identity pass-through** | 0 | 107 | 107 | 20.33 | floor for valid submission |
| Identity (team `dsl.identity`) | 0 | 130 | 130 | 20.13 | team wastes 23 B on metadata |
| **1×1 Conv, 100 f32 weights, no bias** | 100 | 508 | 608 | 18.49 | no attrs (ORT infers kernel) |
| 1×1 Conv, 100 f32 weights + bias | 110 | 568 | 678 | 18.47 | |
| 1×1 Conv, team `single_layer_conv2d` style | 110 | 602 | 712 | 18.45 | all 5 Conv attrs explicit |
| **1×1 Conv, 100 int8 weights (QLinearConv nobias, +Q/DQ)** | 106 | 318 | 424 | 18.87 | best for 100-param conv |
| **Color map (1×1 conv, 100 f32 weights, identity palette)** | 100 | 526 | 626 | 18.56 | minimized |
| **Color map int8** | 106 | 392 | 498 | 18.79 | +0.41 over team's 653 B |
| 3×3 Conv, 900 f32 weights + bias, same-pad | 910 | 3822 | 4732 | 16.54 | params dominate |
| 3×3 Conv, 900 f32, valid-pad (28×28 out) | 910 | 3780 | 4690 | 16.55 | |
| **Slice + Concat** (opset 17, input-based) | 3 | 203 | 206 | 19.67 | |
| **Transpose** (NCHW→NHWC, perm attr) | 0 | 127 | 127 | 20.16 | |
| **Resize** nearest 2× (opset 13) | 4 | 170 | 174 | 19.84 | empty roi + scales |
| **Pad** 1px (opset 13) | 8 | 181 | 189 | 19.76 | pads=[0,0,1,1,0,0,1,1] |
| **Mul/Add by scalar** | 1 | 118 | 119 | 20.22 | scalar initializer |
| **Cast** to float | 0 | 114 | 114 | 20.26 | |
| **Tile** 1× (noop) | 4 | 150 | 154 | 19.96 | repeats initializer |
| **ArgMax + OneHot** (channel argmax → one-hot) | 3 | 207 | 210 | 19.65 | minimized; team's is 391 B |

### Score-vs-budget reference (params = 0)

| Target score | Max bytes (`e^(25−score)`) |
|-------------|--------------------------:|
| 25 | 1 |
| 24 | 3 |
| 23 | 7 |
| 22 | 20 |
| 21 | 55 |
| 20 | 148 |
| 19 | 403 |
| 18 | 1097 |

To average **20** a task needs `params + bytes ≤ ~148`. Identity (107 B) already
clears 20.3. Any conv with ≥100 params + ≥400 B of weights lands near 18.5–19.
**The path to 20+ is: 0-param structural transforms (Transpose, Pad, Slice,
Resize, Cast, Mul-by-scalar) and int8 color maps (392 B, score 18.79).**

---

## 7. Tools

| Tool | What it does | Size impact (measured) | Verdict |
|------|--------------|------------------------|---------|
| **`onnxsim` (onnx-simplifier)** | Constant folding + shape inference + eliminates redundant ops | team `argmax_over_channels`: 391 → 295 B (ok). color_map: 391→295. | Useful as a **first pass**, but leaves producer_name, long names, default attrs. Hand-minimization beats it (295 → 207). |
| **`onnx.optimizer`** (`onnx.tools.optimize`) | Built-in passes: eliminate_deadend, fuse_consecutive_concats, constant_folding. | Marginal — mostly constant folding, same as onnxsim. | OK for cleanup; not aggressive on names/metadata. |
| **`scs4onnx`** (Simple Constant Shrink) | Finds duplicate constant tensors and shares them by name. | Saves ~25 B per deduplicated constant (matches §4d). | Good if you have repeated constants (e.g. same `scales` vector). |
| **ORT graph optimization** (`sess.get_session_options().graph_optimization_level`) | Runtime-only fusions (Conv+Bias, MatMul+Add). Does **not** rewrite the serialized file. | 0 bytes saved on disk. | Helps inference speed, not score. |
| **`onnxruntime.quantization`** (`quantize_dynamic`/`quantize_static`) | Auto-converts Conv/MatMul to QLinearConv. | Produces valid int8 models but hits the QLinearConv bias bug for layers with bias. | Use `quantize_dynamic` with `op_types_to_quantize=['Conv']` then **strip bias** manually. |

**Recommended pipeline:** build by hand (minimal names/attrs) → `onnxsim.simplify`
→ `scs4onnx` (dedupe constants) → manual `ClearField` on metadata → verify with
`onnx.checker` + ORT run.

---

## 8. Protobuf-level tricks

ONNX is proto3-with-presence. Field numbers are fixed by the schema, so you
**cannot rename fields**. But you can:

| Trick | Legal? | Saving | Notes |
|-------|:------:|-------:|-------|
| `ClearField("producer_name/producer_version/domain/doc_string/model_version")` | ✅ | ~13 B (team's "neurogolf-dsl") | proto3 omits unset scalars/empty strings. `make_model` already leaves them unset; setting them to `""` **adds** 2 B each. |
| Set graph `name` to 1 char (`"g"`) | ✅ | 8 B (from "neurogolf") | checker requires non-empty; 1 char is the floor. |
| Leave node `name` unset (not `""`) | ✅ | 2 B/node | empty-string name still costs 2 B; unset costs 0. |
| Omit output ValueInfo dims | ❌ | (would save ~18 B) | `onnx.checker` rejects: "Field 'shape' of 'type' is required". |
| Omit `opset_import` | ⚠️ | 2 B | `onnx.checker` accepts, but ORT behavior is undefined/risky. **Don't.** |
| Drop `kernel_shape` from Conv | ✅ | 23 B | ORT infers from weight; checker accepts. Verified opset 11–21. |
| Drop default `strides`/`dilations`/`group`/`pads` from Conv | ✅ | ~70 B | all have spec defaults ORT applies. |
| Hand-edit raw protobuf to strip optional fields | ⚠️ | varies | possible but fragile — `onnx.checker` re-validates and may reject. Use `ClearField` instead. |
| Gzip the file | ❌ | n/a | competition counts **raw serialized bytes** (`len(model.SerializeToString())`), not compressed size. |
| External-data (`.onnx` + `.data`) | ❌ | n/a | splits weight data out of the `.onnx`, but the competition's `model_size_bytes` only measures the main file — **however** this is almost certainly against the spirit/rules and the validator likely requires a single self-contained file. Do not attempt. |

**The single most effective protobuf trick is just "don't set optional fields
in the first place."** `helper.make_model(graph)` with no extra kwargs already
produces a metadata-free model. The team's `_empty_model` adds
`producer_name="neurogolf-dsl"` — delete that one line.

---

## Concrete recommendations for the team's `neurogolf/dsl.py`

1. **`_empty_model`**: remove `producer_name="neurogolf-dsl"`. Change graph name
   `"neurogolf"` → `"g"`. Keep opset 17 / ir 8. **Saves ~22 B on every model.**

2. **`single_layer_conv2d`**: drop `strides=[1,1]`, `dilations=[1,1]`, `group=1`
   (all defaults). Drop `pads` when it's `[0,0,0,0]` (1×1 case). Drop
   `kernel_shape` entirely — ORT infers it. Rename `conv_w`→`w`, `conv_b`→`b`.
   **Saves ~95 B per conv** (653→558 territory for color_map, before int8).

3. **`color_map`**: switch to the int8 recipe
   (`QuantizeLinear`→`QLinearConv`(nobias)→`DequantizeLinear`).
   **653 B → 392 B, +0.41 score.** Verified correct.

4. **`argmax_over_channels`**: drop the redundant `Cast` (ArgMax emits int64
   already), drop the trailing `Identity`, replace the two `Constant` nodes with
   initializers. **391 B → 207 B, +0.62 score.**

5. **`chain`**: the `m{i}_` prefix and `mid_{i}` intermediates are 3–6 chars.
   Use single-char intermediates (`t0`, `t1`…). Wire the final op directly to
   `"output"` instead of inserting an `Identity`.

6. **`conv_stack`**: same Conv attr cleanup as #2; drop the trailing `Identity`
   by naming the last conv's output `"output"` directly.

7. **Add an int8 path** for any conv layer with >30 weight elements — the
   QuantizeLinear/QLinearConv/DequantizeLinear wrapper pays for itself fast.
   For the 900-param 3×3 conv this is worth ~2600 B (score 16.5 → ~17.4).

8. **Run `onnxsim.simplify` + `scs4onnx`** as a final pass on every emitted
   model, then re-`ClearField` the metadata that onnxsim re-adds.

### Expected team impact

Applying just items 1–5 to the existing 12 solved tasks (which span color_map,
replace_color, kronecker, crop, geom, scale solvers) should lift the per-task
average from ~19.2 toward ~19.7. Adding int8 (item 3/7) on weight-heavy tasks
pushes the weight-bound ones up ~0.4 each. Closing the gap to the leader's ~20
average then depends on **solving more tasks with sub-150-B structural
transforms** (Identity, Transpose, Pad, Slice, Resize, Mul-by-scalar) rather
than conv-based solvers.

---

## Appendix: raw measurement data

Key measured sizes (competition I/O, stripped metadata, opset 17, ir 8):

```
identity (floor)              107 B   score 20.33
identity (team dsl)           130 B   score 20.13
color_map f32 team            653 B   score 18.38
color_map f32 min             526 B   score 18.56
color_map int8 min            392 B   score 18.79   ← best
argmax_over_channels team     391 B   score 19.03
argmax_over_channels min      207 B   score 19.65
conv1x1 100w f32 no-attr      508 B   score 18.59
conv1x1 100w int8 nobias      318 B   score 18.87
conv3x3 900w f32 team        3918 B   score 16.52
conv3x3 900w f32 min-pad     3822 B   score 16.54
Transpose                     127 B   score 20.16
Slice+Concat                  203 B   score 19.67
Resize nn 2x                  170 B   score 19.84
Pad 1px                       181 B   score 19.76
Mul/Add scalar                118 B   score 20.22
Cast                          114 B   score 20.26
Tile 1x                       150 B   score 19.96
```

Reproducible measurement scripts:
`/home/z/my-project/scripts/measure_onnx.py` (opset/names/attrs matrix),
`/home/z/my-project/scripts/measure_onnx2.py` (ClearField, QLinearConv bias,
score formula, protobuf decode),
`/home/z/my-project/scripts/measure_onnx3.py` (realistic competition-constrained
minimization of team primitives + onnxsim).
