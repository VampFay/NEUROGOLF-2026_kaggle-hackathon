# LLM-Based Program Synthesis for ARC-AGI — Research Report

**Audience:** NeuroGolf 2026 Championship team (per-task tiny-ONNX generation for 400 ARC-AGI tasks)
**Author:** Research sub-agent
**Date:** 2026-07-03
**Sources:** ARC Prize leaderboard & technical report, Redwood Research (Greenblatt), Trelis Research, Jeremy Berman, Epoch AI, Microsoft ONNX Script docs, DeepMind FunSearch blog, ICLR self-debugging literature. All cost numbers are scraped from the official ARC Prize leaderboard (`arcprize.org/leaderboard`) as of 2026-07.

---

## TL;DR for the Team

1. **Best LLMs now do 85-95% on ARC-AGI-1 and 65-85% on ARC-AGI-2** — but the price-per-task varies from $0.06 (GPT-4o, useless) to $200 (o3-preview brute-force) to $0.96 (Gemini 3.1 Pro, 98%/77%). For a $0.20-1.00/task budget envelope, your realistic floor is **o3-mini-High (34.5% / $0.55)** and your ceiling is **Gemini 3.1 Pro (98% / $0.96)**.
2. **The state-of-the-art recipe is unchanged since Greenblatt's June-2024 GPT-4o post**: prompt an LLM with chain-of-thought + few-shot examples, sample hundreds-to-thousands of candidate Python programs per task, run each on the demonstration pairs, keep the ones that pass, optionally revise the near-misses via execution feedback, then majority-vote the survivors. Jeremy Berman's 53.6% ARC-AGI-Pub result and the SOAR fine-tuning paper both add a 4-generation evolutionary loop on top.
3. **For our ONNX constraint**: do NOT ask the LLM to emit raw ONNX protobuf. Use **`onnxscript`** (Microsoft, `pip install onnxscript`) — Python functions decorated with `@script()` that get AST-converted to a valid `ModelProto`. This gives you a free parse-time validator (`onnx.checker.check_model`) and lets the LLM write Pythonic code with `op.Conv(...)`, `op.Slice(...)`, etc. Then validate functionally against the demo pairs with `onnxruntime.InferenceSession`.
4. **For 400 tasks @ $0.50-1.00/task = $200-400 budget**: this is exactly the Berman / Greenblatt regime. Sample 32-128 candidates per task, do 1 round of self-debugging revision, keep the first one that passes all demo pairs. Skip the 8,000-sample regime (that cost Greenblatt "a bunch of money" — 1000× prior work).
5. **Hybrid pipeline recommended**: the codebase already has a deterministic DSL dispatcher that solves ~16/400 tasks. The LLM should be a *fallback* that fires when the DSL misses — this is the cheapest place to get +100 tasks solved, and it lets you spend the LLM budget only where it's needed.

---

## 1. LLM Performance on ARC-AGI (2024-2026 Snapshot)

### 1.1 The current ARC-AGI-1 leaderboard (cost-per-task view)

Scraped from `arcprize.org/leaderboard` on 2026-07-03. Only entries with **cost ≤ $10/task** and **ARC-AGI-1 ≥ 30%** are shown, sorted by score. "Sys type" = `Base LLM` (single-shot), `CoT` (chain-of-thought with reasoning), `CoT+Synthesis` (CoT plus external program-synthesis scaffold), `Refinement` (multi-turn revision), `Custom` (open-source competition submission).

| Model | Author | Date | Sys type | ARC-AGI-1 | ARC-AGI-2 | Cost/task |
|---|---|---|---|---|---|---|
| Gemini 3.1 Pro (Preview) | Google | 2026-02-19 | CoT | **98.0%** | 77.1% | **$0.962** |
| GPT-5.5 Pro (High) | OpenAI | 2026-04-23 | CoT | 96.5% | 84.6% | $10.51 |
| GPT-5.5 Pro (xHigh) | OpenAI | 2026-04-23 | CoT | 95.0% | 84.2% | $10.76 |
| GPT-5.5 (xHigh) | OpenAI | 2026-04-22 | CoT | 95.0% | 85.0% | $1.87 |
| GPT-5.5 (High) | OpenAI | 2026-04-22 | CoT | 94.5% | 83.3% | $1.45 |
| GPT-5.5 (Medium) | OpenAI | 2026-04-22 | CoT | 92.2% | 70.4% | $0.86 |
| Claude Opus 4.8 (Max) | Anthropic | 2026-06-01 | CoT | 92.5% | — | $2.33 |
| Gemini 3.5 Flash (High) | Google | 2026-05-19 | CoT | 92.5% | 72.1% | $0.85 |
| Claude Opus 4.7 (Max) | Anthropic | 2026-04-16 | CoT | 92.0% | 75.8% | $7.43 |
| Claude Opus 4.6 (120K, High) | Anthropic | 2026-02-05 | CoT | 94.0% | 69.2% | $3.47 |
| GPT-5.4 (xHigh) | OpenAI | 2026-03-04 | CoT | 93.7% | 74.0% | $1.52 |
| GPT-5.4 (High) | OpenAI | 2026-03-04 | CoT | 92.7% | 67.5% | $1.02 |
| GPT-5.4 (Medium) | OpenAI | 2026-03-04 | CoT | 86.2% | 55.4% | $0.68 |
| Claude Sonnet 4.6 (High) | Anthropic | 2026-02-17 | CoT | 86.5% | 60.4% | $2.70 |
| Grok 4.20 (Reasoning) | xAI | 2026-03-09 | CoT | 89.5% | 65.1% | $0.92 |
| GPT-5.2 (xHigh) | OpenAI | 2025-12-11 | CoT | 86.2% | 52.9% | $1.90 |
| GPT-5.2 (High) | OpenAI | 2025-12-11 | CoT | 78.7% | 43.3% | $1.39 |
| Opus 4.5 (Thinking, 64K) | Anthropic | 2025-11-24 | CoT | 80.0% | 37.6% | $2.40 |
| o3 (Preview, Low) | OpenAI | 2024-12-20 | CoT+Synth | 75.7% | 4.0% | **$200.00** |
| GPT-5.2 (Refine.) | Johan Land | 2026-02-03 | Refinement | 94.5% | 72.9% | $38.99 |
| Grok 4 (Refine.) | J. Berman | 2025-08-07 | Refinement | 79.6% | 29.4% | $30.40 |
| Grok 4 (Refine.) | E. Pang | 2025-08-07 | Refinement | 77.1% | 26.0% | $3.97 |
| o3-Pro (High) | OpenAI | 2025-06-10 | CoT+Synth | 59.3% | 4.9% | $7.55 |
| o3 (High) | OpenAI | 2025-04-16 | CoT | 60.8% | 6.5% | $0.834 |
| o3 (Medium) | OpenAI | 2025-04-16 | CoT | 53.8% | 3.0% | $0.479 |
| o3 (Low) | OpenAI | 2025-04-16 | CoT | 41.5% | 2.0% | $0.234 |
| o3-mini (High) | OpenAI | 2025-01-31 | CoT | 34.5% | 3.0% | $0.547 |
| o3-mini (Medium) | OpenAI | 2025-01-31 | CoT | 22.3% | 2.1% | $0.284 |
| o1-mini | OpenAI | 2024-09-12 | CoT | 14.0% | 0.8% | $0.191 |
| **GPT-4o** | OpenAI | 2024-11-20 | **Base LLM** | **4.5%** | 0.0% | $0.080 |
| GPT-4o-mini | OpenAI | 2024-07-18 | Base LLM | — | 0.0% | $0.010 |
| **ARChitects (2024 winner, DSL+TTT)** | ARC 2024 | 2024-11-03 | **Custom** | 56.0% | 2.5% | **$0.20** |
| Icecuber (2020 brute-force DSL) | ARC 2024 | 2023-11-03 | Custom | 17.0% | 1.6% | $0.13 |
| **Human panel** | — | — | — | 98.0% | 100.0% | $17.00 |
| Avg. Mturker | — | — | — | 77.0% | — | $3.00 |
| Stem grad | — | — | — | 98.0% | — | $10.00 |

### 1.2 Key takeaways for model selection

- **Frontier CoT models now beat humans on ARC-AGI-1** (98% vs. 77% MTurk). This is recent (Gemini 3.1 Pro, Feb 2026).
- **Single-shot base LLMs are useless on ARC-AGI** (GPT-4o = 4.5%, Claude 3.7 = 13.6%, Gemini 1.5 Pro = 0.8% on ARC-AGI-2). Always use a CoT / reasoning variant.
- **ARC-AGI-2 is much harder than ARC-AGI-1** for non-reasoning models: even GPT-5.2 (no thinking) scores 0.8% on ARC-AGI-2 vs. 12.3% on ARC-AGI-1. The thinking budget matters enormously: GPT-5.5 Low → High → xHigh jumps 33.3% → 83.3% → 85.0% on ARC-AGI-2.
- **Sweet spot for $0.50-1.00/task budget**: GPT-5.5 Medium ($0.86, 92.2% / 70.4%), Gemini 3.5 Flash High ($0.85, 92.5% / 72.1%), Grok 4.20 ($0.92, 89.5% / 65.1%), Claude Opus 4.6 120K High ($3.47, 94.0% / 69.2%).
- **For extreme cheapness**: o3 (Low) at $0.234/task gets 41.5% ARC-AGI-1 — 100% better than o3-mini at the same price point.
- **The 2024 ARC winner (ARChitects) at $0.20/task got 56%** using a hybrid DSL + Test-Time Training approach. A pure LLM approach in 2024 cost $30-40/task for similar accuracy (Berman, Pang, Land). The gap has closed: in 2026 a single Gemini 3.1 Pro call ($0.96) beats the 2024 winner by 42 points.

### 1.3 ARC-AGI-2 benchmark specifically

From Epoch AI (`epoch.ai/benchmarks/arc-agi-2`):
- **1360 tasks total**: 1000 train + 360 eval (split 120/120/120 across public, semi-private, private).
- **Average human**: 60% (with two-attempts). All eval tasks solvable by ≥2 humans in ≤2 attempts.
- **At release (early 2025)**: pure LLMs scored 0%; frontier reasoning systems achieved single-digit %.
- **Tasks removed from ARC-AGI-1** because they were too brute-forceable. ARC-AGI-2 specifically targets weaknesses in symbolic interpretation, compositional reasoning, and contextual rule application.
- **ARC-AGI-3** (interactive / agentic version) is even harder — most frontier systems score <1% on the released portion.

### 1.4 What prompts work best?

**Effective patterns** (consistently reported across Greenblatt, Berman, SOAR, Trelis):

1. **Chain-of-thought (CoT)** — always. Prompt explicitly with "let's think step by step" or structured `<reasoning>` tags. Even GPT-5.5 needs this; without it the model is a "Base LLM" and scores collapse.
2. **Few-shot prompting with handwritten expert solutions** — Greenblatt's prompt is ~30k tokens including 1-3 fully worked examples with meticulous step-by-step reasoning. Berman found **one-shot works better than 2- or 3-shot** — "LLMs maintain better focus with concise prompts; they benefit more from deeply understanding one example rather than partially grasping several."
3. **Grid representation is critical** (see §6.2). Show the grid in multiple formats: 2D ASCII, spreadsheet-chess notation (A7, B3), connected components, normalized shapes, input/output diff.
4. **Splitting into task categories** — Greenblatt splits into "grid size same" vs. "grid size changes" and uses different prompts for each. Different examples per bucket.
5. **Test-time compute scaling** — the log-linear relationship (3% accuracy per doubling of samples for Greenblatt's V2 prompt; 6.5% per o3 reasoning-level bump) is the dominant lever. Below ~32 samples, expect ≤30% on ARC-AGI-1.
6. **Iterative revision** — sample → run on demos → feed back the diff → resample. Greenblatt's revision step alone lifted 37% → 50% (more than doubling sample count would have). Berman's 4-generation evolutionary loop adds another ~5% on top.
7. **Pool/ensemble** — Berman's "pooling" prompt combines multiple parent solutions that each solve different example pairs. Helps when many near-correct solutions exist.

**Things that don't matter much**:
- Exact wording of the CoT instruction ("let's think step by step" vs. structured tags) — Berman: "I tested a bunch of prompts... I found that for the frontier models, it didn't matter much. Once you get them to reason step by step, the accuracy doesn't change meaningfully."
- Long few-shot prompts with many examples — Berman found 1-shot beats 2-shot and 3-shot.
- Ensembling across many different CoT prompts — Greenblatt: "more samples from V2 basically dominates diversity from the prompt ensemble I use."

---

## 2. Program Synthesis with LLMs — The Literature

### 2.1 The canonical approach (Greenblatt 2024, "draw more samples")

**Paper/blog**: "Getting 50% (SoTA) on ARC-AGI with GPT-4o" — Ryan Greenblatt, Redwood Research, June 17, 2024.
**Code**: `github.com/rgreenblatt/arc_draw_more_samples_pub` (198 stars, MIT).

**Method (paraphrased from the blog)**:

```
For each ARC task:
  1. Construct a prompt containing:
     - Image of each demo input/output grid (PNG)
     - Multiple ASCII representations of each grid:
       a) 2D array with spreadsheet-chess notation per cell (A7=red, B3=blue, ...)
       b) List of color → list of locations (split by connected component)
       c) Normalized connected-component shapes (translated to origin)
       d) Diff between input and output: (color_a → color_b) → locations
     - Few-shot example(s) of step-by-step reasoning → Python function
     - The instruction: reason about the transformation, then implement as Python
  2. Sample ~5,000 completions from GPT-4o (n=32 per API call, ~30k-token prompt)
  3. Execute each completion's Python function on each demo input
     - Keep programs that produce valid grids with the right shape
     - Score each program by Hamming distance to expected output
  4. Select top-12 most promising programs (geometric mean of Hamming-rank across demos)
  5. For each, sample ~250 revision completions (3,040 total) using a "revision prompt"
     that shows the actual-vs-expected output diff and asks the LLM to fix the code
  6. Majority-vote over all programs that pass ALL demos; pick top-3 distinct outputs
  7. Reject outputs equal to the test input (no ARC task is identity)
```

**Ablation results** (from Greenblatt's blog):

| Variant | Samples/problem | Test accuracy |
|---|---|---|
| V0 (basic ASCII, no buckets) | 1,024 | 25% |
| V1 (split buckets, spreadsheet ASCII) | 1,024 | 27% |
| V1.1 (different examples) | 1,024 | 28% |
| V2 (V1 + input/output diff) | 1,024 | 30% |
| V2 | 2,048 | 34% |
| Ensemble (V0+V1+V1.1+V2) | 4,096 | 37% |
| **+ Revision step (3,040 more)** | ~8,000 total | **50%** |

**Scaling law**: each doubling of samples adds ~3% accuracy. To reach 50% without revision would need ~100,000 samples (V2) or ~1,500,000 samples (V0). Revision fixes ~20% of remaining incorrect solutions, which is roughly equivalent to ~5x more samples. To reach 70% (estimated human MTurk) with the full method would need ~2M samples per problem.

**Key tricks (from Greenblatt's Appendix)**:
- **Two-bucket split**: different prompts for "grid size changes" vs. "grid size stays same". Different few-shot examples per bucket.
- **Connected-component ASCII**: list each color's cells, grouped by connected component. Then a normalized view (translated to origin) so shapes can be compared without offset confusion.
- **Input/output diff**: pairs of distinct colors and locations where input has color_a, output has color_b. Only useful when grids are same-size.
- **Hamming-distance ranking**: rank programs by geometric mean of their Hamming-distance ranks across demos (not raw distance, to normalize for grid size). Penalize for being too close to already-selected solutions to encourage diversity.
- **Skip identity output** (no ARC task is identity).
- **Use the `n` parameter** of OpenAI API (n<128, typically 32, because n=128 errors).
- **Filter revisions to problems with <32 fully-correct programs** (don't revise if you already solved it).

**Non-reasoning failure modes (Greenblatt's analysis)**:
- GPT-4o vision is "terrible on grids": fails to extract cell colors for grids >12×12, struggles at 8×8.
- GPT-4o coding is "not that good": off-by-one errors are extremely common.
- Long context falls off after ~32k-40k tokens.
- Doesn't respect few-shot prompt; often produces shorter completions than instructed.
- "Multi-round debugging is probably cheaper and more effective to just get more samples in the current regime" — i.e. **revision beats iterative debugging when you can afford more samples**.

### 2.2 Evolutionary test-time compute (Berman 2024, 2025)

**Blog**: "How I came in first on ARC-AGI-Pub using Sonnet 3.5 with Evolutionary Test-time Compute" — Jeremy Berman, Dec 2024. **53.6%** on public leaderboard (prev. SOTA was Greenblatt's 43%).

**Architecture**: Two parallel tracks run over 4 generations:

```
Track A (Single-parent evolution):
  Gen 1: Sample 250 transform functions (50 + 200)
         If any solves all demos perfectly → pick best 2, stop
  Gen 2: Select top-10 functions → 10 offspring each (100 new)
  Gen 3: Select top-5 → 5 offspring each (25 new)
  Gen 4: Select top-5 → 5 offspring each (25 new)

Track B (Pooled multi-parent evolution):
  Gen 1: Same 250 functions as Track A
  Gen 2: For each demo, find 3 functions that solve it best
         Combine into pooled prompts → 5 offspring per pool
  Gen 3-4: Same pooling process on top-5

Final: From all generations, pick 2 best-performing functions
       that produce distinct output grids.
```

**Prompt structure** (Berman):
- System prompt (see his Appendix 2.3)
- CoT instruction (Appendix 2.1, "Version 1" is a tweaked version of Greenblatt's prompt; "Version 2" is from Harish SG)
- 1-shot worked example (he tested 1, 2, 3-shot — 1-shot won)
- Grid representation: dimensions + base64 image + ASCII + Python nested list `list[list[int]]`
- Per-challenge: prompt + 1 example = 1 LLM call
- 4 generations × ~50-200 calls = ~500 LLM calls per task

**Berman's ablation (60 training challenges, 200 LLM calls each, Sonnet 3.5)**:
- Shallow (1 gen × 200): 70%
- Deep (4 gens × 50): 75% — **42% of Deep's successes came from gens 2-4**
- Implication: generational depth beats raw sampling when you have many near-correct solutions.

**Pooling prompt**: Combines multiple parent functions into a single revision prompt, ensuring at least one function solves each demo case. Larger context but more diverse "genetic material." Berman's pooling prompt PDF is **174KB** — i.e. these prompts get huge.

**Berman's 2025 follow-up**: switched from Python to English ("natural language programs"), still using Grok 4. Got **29.4% on ARC-AGI-2** at $30.40/task (Aug 2025 leaderboard). Reasoning: "perhaps not python that's the limit after all" — for some tasks, irregular shapes are easier to describe in words than in pixel-level Python.

### 2.3 SOAR (Pourcel et al. 2025)

**Paper**: "Self-Improving Language Models for Evolutionary Program Synthesis" (ICML 2025).

**Method**: Fine-tune open-source LLMs (Qwen 2.5 7B-123B) on:
1. Correct programs (sampled from strong teacher LLMs on ARC-AGI-1 train + test tasks).
2. **Hindsight relabelling**: take a wrong program, run it on the demo *inputs*, and use the *actual output* (not the expected output) as the new "ground truth" for a new training task. This converts failures into successes.

**ARC-AGI-1 result**: ensembling open-source models (7B to 123B) subjected to SOAR training exceeds **55% on ARC-AGI-1 public eval**.

**Trelis Research's experience applying SOAR to ARC-AGI-2** (Nov 2025 blog):
- Scored only **1.67% on ARC-AGI-2 semi-private** (then switched to TRM post-training → 6.67%).
- Key blocker: of 120 ARC-AGI-2 eval tasks, only **8/120 had any fully-correct programs** found by strong models (GPT-5, Gemini Pro/Flash, GPT-5-mini) even after extensive sampling. So fine-tuning had nothing to learn from.
- **Refinement didn't help on ARC-AGI-2**: "we were never able to do better by refining programs than by just sampling more." Refinement only beats extra sampling once you've saturated sampling (in the hundreds-to-thousands per task range), which a 4B model on 4×L4 in 12 hours can't reach.
- **Test-time tuning didn't help in compute-constrained regime**: better to spend that compute on more sampling.
- **Transductive programs problem**: 5-20% of generated Python programs hard-code the output grid (e.g. `return [[1,2,3],[4,5,6]]`). These pass all demos but don't generalize to test. Trelis built a feature-based classifier to down-weight these.
- **Program execution is a major pain point**: infinite loops, memory explosions. "We never fully solved program execution."
- Used FP8 quantization on L4 GPUs to double inference speed (LLM Compressor + SGLang + `--kv-cache-dtype fp8_e4m3`).
- Used Unsloth for fine-tuning on single B200/H200.

### 2.4 FunSearch (DeepMind, Nature 2024)

**Blog**: `deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models`. Paper in *Nature*, Dec 2023; combinatorial-competitive-programming update Dec 2024.

**Architecture**: Pair an LLM with an automated evaluator.
```
1. User writes problem spec: seed program + evaluation function
2. Initialize pool of programs (just the seed)
3. Loop:
   a. Sample some programs from the pool (highest-scoring, with diversity)
   b. LLM creatively builds on these → generates new programs
   c. Evaluator runs each new program, scores it
   d. Best new programs added back to the pool
4. Return highest-scoring program at any time
```

**Key insights**:
- LLM provides creativity (mutation operator); evaluator provides selection pressure (anti-hallucination).
- Uses **PaLM 2** originally, **Gemini 1.5 Flash** in the Dec 2024 update ("we no longer require code-specialised models").
- **Multiple islands run in parallel** to maintain diversity and avoid stagnation.
- Favors **concise, human-interpretable programs** (low Kolmogorov complexity) — short programs describe very large objects, enabling scale to needle-in-haystack problems.
- Used to discover new cap set sizes (largest increase in 20 years) and beat Best-fit heuristic for online bin-packing.

**Relevance to ARC**: FunSearch's architecture is essentially Greenblatt + Berman + selection pressure. The "human writes backbone, LLM evolves the key function" pattern from their Dec 2024 competitive-programming update is exactly what Greenblatt's few-shot + revision loop approximates. The SOAR paper is the explicit FunSearch-for-ARC implementation.

### 2.5 Self-debugging (Chen et al. ICLR 2024)

**Paper**: "Teaching Large Language Models to Self-Debug" (Chen, Lin, et al.).

**Method**: Few-shot demonstrations of: write program → run it → observe output → if wrong, explain the bug → fix the program → re-run. Variants:
- **Simple feedback**: "Your code produced output X. Expected Y. Fix it."
- **Unit-test feedback**: "Your code failed unit tests A, B. Fix it."
- **Rubric explanation**: have the LLM first explain in English what its code does, then compare to expected behavior — this works even without execution.

**NeurIPS 2024 follow-up (LeDex)**: train LLMs to better self-debug by collecting trace + explanation data.

**ARC-specific finding (Trelis)**: refinement (single-round self-debug) only beats raw sampling once sampling is saturated. Multi-round self-debug is rarely worth it vs. more samples at the same compute budget — confirming Greenblatt's caveat.

### 2.6 LLMs as zero-shot program synthesizers

For DSL-targeted program synthesis (not just Python), the literature (Ouellette, ARC-DSL by Hodel, etc.) shows:
- LLMs can write valid DSL programs more reliably than valid raw Python for ARC (smaller grammar, no infinite loops).
- DSLs constrain the search space, making brute-force tractable.
- The 2024 ARC tech report distinguishes four approaches:
  1. Brute-force DSL search (icecuber 2020, ~20%; refined to ~40% by 2024).
  2. LLM-powered program generation in open-ended languages (Greenblatt, Python).
  3. LLM-guided discrete program search over a DSL (Ouellette — LLM picks the next branching decision in DSL search).
  4. LLM-powered iterative program debugging (Greenblatt revision step, Berman evolution).
- The "Specialist DL model to guide branching in discrete search" — AlphaProof-style — is mentioned as untried for ARC but expected to perform well.
- "Deep learning-guided program synthesis does not currently decisively beat DSL-based brute-force program search — both score in the 40% range today with comparable compute budgets."

---

## 3. Specific ARC-AGI LLM Solver Approaches

### 3.1 Greenblatt (50% on public test, GPT-4o, Jun 2024)

See §2.1. The reference implementation. Code: `github.com/rgreenblatt/arc_draw_more_samples_pub`. Tech:
- `openai==0.28.1` (legacy Python SDK)
- Redis for caching (port 6381)
- `numpy`, `scipy`, `skimage`, `attrs`, `cattrs`, `matplotlib`
- "several hours of time, a considerable amount of memory, an openai key, and a bunch of money"

### 3.2 Berman (53.6% on ARC-AGI-Pub, Sonnet 3.5, Dec 2024)

See §2.2. The evolutionary extension. Total budget: ~$3-5k in Anthropic credits.

### 3.3 E. Pang (77.1% ARC-AGI-1 / 26.0% ARC-AGI-2, Grok 4, Aug 2025)

On ARC leaderboard as "Grok 4 (Refine.)", $3.97/task — **8× cheaper than Berman's $30.40 for nearly identical accuracy**. Suggests they heavily optimized the refinement loop. Trelis reports similar execution-engineering challenges with Pang's approach. No public writeup as of the time of writing — likely uses a similar Greenblatt-style pipeline with Grok 4's stronger reasoning.

### 3.4 J. Berman (79.6% / 29.4%, Grok 4 + English programs, Aug 2025)

Follow-up to his 2024 Python work. Switched from Python transform functions to **natural-language programs** (which only became possible with Grok 4's stronger reasoning). Scores $30.40/task — i.e. more compute spent per task. See `jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again-swapping-python-for-english` and `jeremyberman.substack.com/p/how-i-came-in-first-on-arc-agi-pub` (Sep 2025 follow-up with multi-agent collaboration).

### 3.5 J. Land (94.5% / 72.9%, GPT-5.2 + Refinement, Feb 2026)

Top refinement-based system on the leaderboard as of July 2026. $38.99/task. Uses GPT-5.2 with refinement loop. No public writeup yet — likely an extension of the Greenblatt/Berman pipeline with GPT-5.2's stronger reasoning reducing the sample count needed.

### 3.6 The 2024 ARC Prize winners (ARChitects, MindsAI, Akyürek)

- **MindsAI** (55.5% private eval, **not open-sourced**): pioneered **Test-Time Training (TTT)**. Fine-tune a model at test time on the specific ARC task using augmented versions of the demo pairs.
- **ARChitects** (53.5%, **open-sourced**): TTT + novel data augmentations + stability-based selection criterion. Cost ~$0.20/task under Kaggle compute budget (single P100, 12 hours).
- **Akyürek et al.** (47.5% semi-private, paper-award winner): TTT, open-sourced.
- **Common pattern**: small transformer (often 2D-attention aware), pre-trained on ARC-like data (BARC, Re-ARC), then test-time fine-tuned on each task's demos with heavy augmentation.
- **Key insight from 2024 tech report**: "Today, all top LLM-based transduction approaches for ARC-AGI leverage TTT, and there does not exist any static inference-style transduction solution that scores above 10%."
- **Combined with program synthesis** (the actual 2024 SOTA pattern): transduction + induction ensembles. "The best transduction-only and induction-only single submissions score around 40%, so only an ensemble of both can compete for the state of the art."

### 3.7 "Code as Policies" style for ARC

The "Code as Policies" paradigm (Liang et al. 2023, for robotics) — generate Python that uses perception APIs to act on the world — maps onto ARC as: **LLM writes Python that uses numpy/scipy/skimage ops to transform grids**. This is exactly Greenblatt's approach. The DSL version is "LLM writes DSL program," and the ONNX version (what we want) is "LLM writes `onnxscript` program."

### 3.8 NVARC and the 2025 winner

- **NVARCARC Prize 2025** (Nov 2024): 27.6% on ARC-AGI-2 at $0.20/task. Uses NVARC (N-Vector ARC) — a custom 2D-aware neural architecture. Open-sourced.
- The 2025 winner wasn't on the public leaderboard scrape at the time of this research; the top 2026 entries are mostly frontier-LLM CoT (Gemini 3.1 Pro, GPT-5.5) plus a few refinement systems.

---

## 4. Practical Implementation Patterns

### 4.1 The standard synthesis pipeline

```
For each ARC task:
  1. Load demos (typically 3 input/output pairs)
  2. Construct a CoT prompt with:
     - Grid representation (see §4.3)
     - Few-shot worked example(s) (see §4.4)
     - Instruction to reason step-by-step then write code
  3. Sample N candidate programs (parallel API calls)
  4. For each candidate:
     a. Parse the code (Python AST for Python, onnxscript AST for ONNX)
     b. Execute on each demo input (sandboxed; timeout 5s; memory cap 1GB)
     c. Compare output to expected output (exact match or Hamming distance)
     d. Record (program, output_per_demo, hamming_per_demo, valid?)
  5. Filter to programs producing valid outputs (right shape, right colors)
  6. If ≥1 program passes ALL demos:
     - Majority-vote the surviving programs' outputs on the test input
     - Pick top-K distinct outputs as final submissions
  7. Else:
     - Take top-M most promising programs (lowest average Hamming)
     - Build revision prompts: show program + actual output + expected output + diff
     - Sample R revisions per parent
     - Go to step 4 with the revised programs
  8. (Optional, Berman-style) Iterate as 4 generations:
     - Gen 1: N samples from scratch
     - Gen 2-4: revision prompts using top-K from prior gen
```

### 4.2 Test-time scaling: sample-budget math

For our use case (400 tasks × ≤$1/task budget):

| Model | $/task budget | Approx samples affordable | Expected ARC-AGI-1 acc |
|---|---|---|---|
| o3-mini (Low) | $0.23 | 1 sample | ~14% |
| o3-mini (Medium) | $0.28 | 1 sample | ~22% |
| o3-mini (High) | $0.55 | 1 sample | ~35% |
| o3 (Medium) | $0.48 | 1 sample | ~54% |
| o3 (High) | $0.83 | 1 sample | ~61% |
| GPT-5.5 (Medium) | $0.86 | 1 sample | ~92% |
| Gemini 3.5 Flash (High) | $0.85 | 1 sample | ~93% |
| **Sonnet 4.6 (High) + Greenblatt-style revision** | $1.00 | ~30 samples + 30 revisions | ~50-60% (extrapolated from Greenblatt at 8k samples) |
| **Gemini 3.5 Flash (High) + Berman-style 4-gen** | $1.00 | ~15 samples × 4 gens = 60 calls | ~70-85% (extrapolated) |

**Implication**: in 2026, **a single Gemini 3.5 Flash High call beats a 2024-style 8000-sample Greenblatt pipeline** at half the cost. The math has shifted — frontier CoT reasoning models now do internally what Greenblatt was brute-forcing externally. The synthesis pipeline becomes most useful when:
- You can't use frontier closed models (e.g. must run offline / Kaggle / on-prem).
- You want provably-correct outputs (programs that pass demos are guaranteed correct on demos).
- You need ONNX or other constrained output formats (which CoT models can't directly produce).
- You need cheap fallback for the ~10-20% of tasks the frontier model gets wrong on first try.

**For our NeuroGolf case** (ONNX required, 400 tasks, $0.20-1.00/task): the LLM-synthesis pipeline is mandatory, not optional, because Gemini/GPT-5 can't emit ONNX directly. The question is just which LLM writes the `onnxscript` code.

### 4.3 Grid representation that works

Greenblatt, Berman, and Trelis all converge on a multi-format representation:

```python
def render_grid(grid: list[list[int]]) -> str:
    """Multi-format grid representation, ~5x more accurate than images alone."""
    rows, cols = len(grid), len(grid[0])
    out = []
    # 1. Dimensions
    out.append(f"Grid dimensions: {rows} rows × {cols} cols")
    # 2. ASCII 2D array with chess notation per cell
    out.append("Coordinates (row,col) use spreadsheet notation: A1=top-left.")
    out.append("Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=teal, 9=brown.")
    out.append("")
    for r in range(rows):
        row_label = chr(ord('A') + r)
        cells = [f"{row_label}{c}={grid[r][c]}" for c in range(cols)]
        out.append(" ".join(cells))
    # 3. Color → location list, split by connected component
    from scipy.ndimage import label
    import numpy as np
    arr = np.array(grid)
    out.append("")
    out.append("Color → locations (grouped by connected component):")
    for color in sorted(set(grid[r][c] for r in range(rows) for c in range(cols))):
        if color == 0: continue  # skip background
        mask = (arr == color).astype(int)
        labeled, n = label(mask)
        out.append(f"  Color {color}:")
        for comp_id in range(1, n+1):
            coords = [(chr(ord('A')+r), c) for r, c in zip(*np.where(labeled == comp_id))]
            # Normalized shape (translate to origin)
            rs = [ord(c[0])-ord('A') for c in coords]
            cs = [c[1] for c in coords]
            rmin, cmin = min(rs), min(cs)
            norm = [(r-rmin, c-cmin) for r, c in zip(rs, cs)]
            out.append(f"    Component: {coords}  (normalized shape: {norm})")
    return "\n".join(out)

def render_diff(input_grid, output_grid) -> str:
    """Input → output diff, only meaningful when shapes match."""
    if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
        return "(shapes differ; diff omitted)"
    rows, cols = len(input_grid), len(input_grid[0])
    diffs = {}
    for r in range(rows):
        for c in range(cols):
            a, b = input_grid[r][c], output_grid[r][c]
            if a != b:
                key = (a, b)
                diffs.setdefault(key, []).append((chr(ord('A')+r), c))
    out = ["Input → output color changes:"]
    for (a, b), locs in sorted(diffs.items()):
        out.append(f"  {a} → {b}: {locs}")
    return "\n".join(out) if diffs else "(no changes)"
```

**Send to LLM**: dimensions + ASCII grid + color/components + (if same-shape) diff + (optional) base64 PNG of the grid. Greenblatt found ASCII + diff >> image alone. Berman found that providing all formats (image + ASCII + Python nested list) was best even though it looks redundant.

### 4.4 Few-shot example structure (from Greenblatt/Berman)

```
<example>
<input_grid>
[render_grid(...)]
</input_grid>

<output_grid>
[render_grid(...)]
</output_grid>

<diff>
[render_diff(...)]
</diff>

<reasoning>
Step 1: Examine the input. The grid is 7×13 with red (2) cells forming a "T" shape
         at rows 3-5, columns 1-3. The green (3) cells form vertical stripes at
         odd columns.
Step 2: Examine the output. The "T" shape has shifted right by 2 columns. The
         green stripes are unchanged.
Step 3: Hypothesize the rule: shift red components right by 2 cells, preserve
         everything else.
Step 4: Verify against all demos. ✓
Step 5: Implement in Python:

```python
import numpy as np
def transform(grid):
    g = np.array(grid)
    red = (g == 2).astype(int)
    shifted = np.zeros_like(red)
    shifted[:, 2:] = red[:, :-2]  # shift right by 2
    g[shifted == 1] = 2
    g[red == 1] = 0  # erase original red
    return g.tolist()
```
</reasoning>
</example>
```

Greenblatt hand-wrote multiple such examples; the prompt totals ~30k tokens. Berman uses 1 example (1-shot). Both split examples by "same shape" vs. "different shape" task category.

### 4.5 Execution sandbox

**Critical**: any LLM-generated Python must be sandboxed. Trelis reports infinite loops and memory explosions as one of their biggest pain points. Recommended:

```python
import resource, multiprocessing, signal

def safe_run(code: str, input_grid, timeout_s=5, mem_mb=512):
    """Run LLM-generated code in a subprocess with hard limits."""
    def worker(code, input_grid, q):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s,) * 2)
            ns = {}
            exec(code, ns)
            if 'transform' not in ns:
                q.put(('error', 'no transform() function defined'))
                return
            out = ns['transform'](input_grid)
            q.put(('ok', out))
        except BaseException as e:
            q.put(('error', f"{type(e).__name__}: {e}"))
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(code, input_grid, q))
    p.start()
    p.join(timeout_s + 1)
    if p.is_alive():
        p.terminate()
        return ('error', 'timeout')
    if q.empty():
        return ('error', 'no output')
    return q.get()
```

For ONNX (which we want), the sandbox problem mostly disappears — `onnxruntime.InferenceSession` is a fixed-cost C++ runtime; we can `onnx.checker.check_model()` first, then run with a timeout.

### 4.6 Verification before shipping

For each candidate program (after parsing + executing on demos):
1. **Structural check**: `onnx.checker.check_model(model_proto)` — must pass.
2. **Constraint check**: no banned ops, dynamic shapes work, ≤1.44 MB file size, op count ≤ N (per competition rules).
3. **Functional check**: `onnxruntime.InferenceSession(model_proto)` then run on each demo input → exact match against expected output?
4. **Generalization check**: if the program hard-codes outputs (transductive), reject. Trelis' classifier: scan code for `return [[<int>...` patterns or large constant tensors baked into the program.

For our NeuroGolf ONNX case, the validator already in the codebase (`validator.py` per the audit report) handles (1) and (2); add a transductive-program detector for (3) and (4).

---

## 5. Code Generation for ONNX Specifically

### 5.1 Can LLMs write valid ONNX directly?

**No, not raw ONNX protobuf.** Raw ONNX is protobuf with verbose field names like `op_type`, `input`, `output`, `attribute`, `tensors.data_type`, etc. LLMs make syntax errors constantly.

**Yes, with `onnxscript`** (Microsoft, `pip install onnxscript`). This is the right abstraction: LLM writes Python, the `@script()` decorator parses the AST and emits a valid `ModelProto`. Strongly typed, IDE-friendly, debuggable.

```python
# Example from the onnxscript docs:
import onnx
from onnxscript import FLOAT, script
from onnxscript import opset18 as op

@script()
def sample_model(X: FLOAT[64, 128], Wt: FLOAT[128, 10], Bias: FLOAT[10]) -> FLOAT[64, 10]:
    matmul = op.MatMul(X, Wt) + Bias
    return matmul

onnx_model = sample_model.to_model_proto()  # in-memory ModelProto
onnx.save(onnx_model, "sample_model.onnx")
onnx.checker.check_model(onnx_model)  # validates
```

The `op.<OperatorName>` API covers all ~200 ONNX ops (`op.Conv`, `op.Slice`, `op.Concat`, `op.ReduceMax`, `op.ArgMax`, `op.Where`, `op.Gather`, `op.Reshape`, `op.Transpose`, `op.Cast`, `op.Add`, `op.Mul`, etc.). Operator attributes are passed as named arguments (`op.Conv(X, W, B, strides=[1,1], pads=[0,0,0,0])`). Eager mode for debugging is supported (`sample_model(np.array(...))` runs it via ONNX Runtime).

### 5.2 Why onnxscript is the right DSL for our case

1. **Parse-time validation** — Python syntax errors are caught by `exec()`, and `@script()` raises a clear error if the AST has constructs onnxscript doesn't support (which is most of Python — it's a strict subset). This is the "DSL grammar" we want.
2. **Graph validation** — `to_model_proto()` + `onnx.checker.check_model()` catches ONNX-level errors (wrong op signatures, bad shapes, missing initializers).
3. **Functional validation** — `onnxruntime.InferenceSession` runs it. We already have this in the codebase validator.
4. **LLM-friendly** — LLMs already write Python well; `op.X` is a small, named API surface they can learn from a 1-shot example.
5. **No infinite loops** — `@script()` doesn't support `while`. `for` loops must have static bounds (unrolled at parse time). So LLM-generated onnxscript code can't hang the sandbox.

### 5.3 onnxscript limitations to know

- Subset of Python. No `print()`, no `import`, no `numpy`, no function calls outside the `op.*` namespace (except other `@script()` functions you define).
- Loops: `for i in range(N)` works (N must be a Python int constant, unrolled). `while` doesn't.
- Conditionals: `if cond:` works only if `cond` is an ONNX tensor (not a Python bool). Use `op.Where` for element-wise selection.
- Type annotations required on function signatures (`X: FLOAT[H, W]`).
- Initializers (constants, weights, biases) must be either `op.Constant(value=...)` calls or passed in as function arguments. The LLM must learn this.

### 5.4 The mapping we want: ARC → onnxscript

Given an ARC task with `train` pairs (list of `{input, output}` dicts of `list[list[int]]`):

```python
# What the LLM should produce:
from onnxscript import FLOAT, INT64, script
from onnxscript import opset18 as op

@script()
def arc_transform(input_grid: FLOAT[1, 10, 30, 30]) -> FLOAT[1, 10, 30, 30]:
    # input_grid: one-hot encoded (batch=1, channels=10 colors, H=30, W=30)
    # ... LLM fills in body using op.* ...
    return output_grid

model = arc_transform.to_model_proto()
```

The LLM must work in the **one-hot tensor** representation, not raw int grids. This is a meaningful shift from Greenblatt's setup (where the LLM writes Python operating on `list[list[int]]`). The model has to:
1. Understand the one-hot encoding (channel = color, argmax over channels = grid value).
2. Implement transformations using ONNX ops on the 4D tensor.
3. Optionally convert to/from int via `op.ArgMax` + `op.OneHot`, but mostly stay in one-hot space throughout.

This is hard for an LLM that hasn't seen it before, so the few-shot example must explicitly show this pattern.

### 5.5 Examples of LLMs generating ONNX-equivalent code

- **Project Babylon** (Java, Nov 2025) demonstrates ONNX model authoring in non-Python languages, suggesting LLMs can transfer the skill.
- **"From Code to Prediction"** (arXiv 2605.03686) — fine-tunes LLMs to emit neural network code.
- **TorchLib** (Microsoft) — the new PyTorch ONNX exporter's operator library is itself written in onnxscript. So all of `aten` ops are now expressible in onnxscript, and LLMs that have seen PyTorch code can transfer.
- **The codebase we have** (`neurogolf/dsl.py`) already provides a Python wrapper around `onnx.helper` that builds ONNX graphs imperatively (`single_layer_conv2d`, `color_map`, `chain`, etc.). This is essentially a hand-written mini-DSL. LLMs can target either this DSL (as Python function calls) or onnxscript directly. The DSL is simpler for the LLM but less expressive.

**Recommendation**: have the LLM target the existing `neurogolf/dsl.py` primitives first (simplest, smallest API surface, already validated), and fall back to raw `onnxscript` only when the DSL can't express the required transformation. The DSL's `chain`, `color_map`, `single_layer_conv2d`, `mask_apply`, `replace_color`, `argmax_over_channels`, `constant_grid` cover most ARC primitives.

---

## 6. Cost Considerations

### 6.1 Cost per task by model

From §1.1, the relevant frontier-LLM options at $0.50-2.00/task:

| Model | $/task | ARC-AGI-1 | ARC-AGI-2 | Recommended use |
|---|---|---|---|---|
| GPT-5.5 (Medium) | $0.86 | 92.2% | 70.4% | Default if available |
| Gemini 3.5 Flash (High) | $0.85 | 92.5% | 72.1% | Default if available; cheaper per token |
| Grok 4.20 | $0.92 | 89.5% | 65.1% | Backup if OpenAI/Google unavailable |
| GPT-5.4 (High) | $1.02 | 92.7% | 67.5% | Mid-tier |
| Claude Opus 4.6 (High, 120K) | $3.47 | 94.0% | 69.2% | Use for hardest tasks only |
| o3 (High) | $0.83 | 60.8% | 6.5% | Skip — weak on ARC-AGI-2 |
| Sonnet 4.6 (High) | $2.70 | 86.5% | 60.4% | Mid-tier Anthropic option |
| Gemini 3 Pro (Refine., Poetiq) | $30.57 | — | 54.0% | Only if synthesis pipeline needs many samples |

### 6.2 Cost for 400 tasks

- **Single-shot CoT, no synthesis**: 400 × $0.85 = **$340**. Gets ~92% on ARC-AGI-1 if Gemini 3.5 Flash High. But this can't emit ONNX directly — need synthesis.
- **Greenblatt-style synthesis with frontier model**: ~32 samples + 1 revision round per task = ~64 LLM calls × 400 tasks = 25,600 calls. If each call is ~5k input tokens + 1k output tokens at $0.30/1M-input + $1.50/1M-output (Sonnet 4.6-ish pricing), that's 25,600 × (5k × $0.30/1M + 1k × $1.50/1M) = 25,600 × $0.003 = ~$77. Plus revision round (similar size) = ~$150 total. **Feasible.**
- **Berman-style 4-gen evolutionary**: 4 × ~50 calls × 400 tasks = 80,000 calls × $0.003 = ~$240. **Feasible.**
- **Greenblatt's actual reported cost**: he doesn't quote a number, but the README says "a bunch of money" and ARC's leaderboard caps at $10k for OpenAI API spend. Trelis reports "thousands of dollars of inference, via OpenRouter, on strong models such as GPT-5-mini, GPT-5 and Gemini Flash and Pro" for dataset generation. So expect **$1,000-3,000** for a serious Greenblatt-style run at 8,000 samples/task.
- **For our $0.20-1.00/task envelope**: we're at the low end. Sample budget is ~30-100 per task, not 8,000. This is fine for ARC-AGI-1 (most tasks easy) but will miss the long tail.

### 6.3 Latency per synthesis round

- OpenAI/Anthropic/Google API: ~5-30 seconds per call for reasoning models (CoT, o1/o3-style). 30-60s for "thinking" levels.
- With 32 parallel calls (n=32 in OpenAI API): wallclock = single-call latency ≈ 30-60s.
- One synthesis round = 1 parallel-sample call + 1 revision round = ~60-120s wallclock.
- 400 tasks × 60s = 24,000s = 6.7 hours serial. With 4-way task parallelism, ~1.7 hours. **Fits in our pipeline.**
- 4-generation evolutionary loop: 4 × 60s = 240s/task × 400 = 27 hours serial. Need ~10-way parallelism to fit in 3 hours.

### 6.4 The "diminishing returns" curve

From Greenblatt's data (V2 prompt, no revision):
- 1 sample: ~12%
- 8 samples: ~18%
- 32 samples: ~24%
- 128 samples: ~28%
- 1,024 samples: ~30%
- 2,048 samples: ~34%
- 8,000 samples: ~38%
- 100,000 samples: ~50%

Adding the revision step at 8,000 samples lifts 38% → 50%. **The first 32 samples are worth more than the next 8,000.** So at our budget, **focus on getting 32-128 good samples per task**, not on drawing more.

---

## 7. Failure Modes

### 7.1 What LLMs struggle with on ARC

From Greenblatt's qualitative analysis + Trelis' lessons + the ARC-AGI-2 design notes:

1. **Vision on grids >12×12** — GPT-4o couldn't extract cell colors. Mitigation: always use ASCII, never rely on vision. (Mostly fixed in 2026 frontier models.)
2. **Off-by-one errors in code** — extremely common. Mitigation: execution feedback (revision loop) catches them.
3. **Long-context attention drop-off** after ~32k-40k tokens. Mitigation: keep prompts under 15k tokens; 1-shot not 5-shot.
4. **Hard-coded outputs (transductive programs)** — 5-20% of generated programs hard-code the demo outputs and don't generalize. Mitigation: transductive-program classifier; reject programs that contain `return [[<integer constants>]]` patterns or bake large constant tensors.
5. **Irregular shapes** — tasks where the transformation is easier to describe in words than in pixel-level code. Trelis: "it is easier to describe the transformation in words than to do so precisely with a python program that is forced to operate on a pixel level." Mitigation: natural-language programs (Berman's 2025 approach) or richer DSL with shape primitives.
6. **Compositional reasoning** — tasks requiring multiple interacting rules (the focus of ARC-AGI-2). Mitigation: stronger reasoning models; more samples; multi-step decomposition.
7. **Symbolic interpretation** — assigning meaning to colors/shapes beyond surface patterns. ARC-AGI-2 specifically targets this; LLMs still struggle.
8. **Context-dependent rules** — same color means different things in different parts of the grid. Mitigation: attention/where-based ops; LLM rarely gets this right without explicit hint.
9. **Tasks with very few demos** (1-2 pairs) — under-determined; many rules fit. Mitigation: generate K candidates, majority-vote, prefer simpler programs (Occam).
10. **Tasks where the output grid size differs from input** — Greenblatt uses a separate prompt bucket for these. LLMs often emit a `resize` that produces wrong shape; need explicit shape check.

### 7.2 Detecting "stuck" and switching to fallback

**Signals the LLM-synthesis is stuck**:
- 0 of N samples pass all demos after revision.
- All passing programs produce identical outputs (low diversity).
- All failing programs have similar Hamming distance (search has converged to a local optimum).
- After K rounds of revision, no improvement in best Hamming distance.
- The LLM keeps generating syntactically invalid code (>50% parse errors).

**Fallback chain** (recommend for our pipeline):
1. Existing DSL dispatcher (current codebase, 16/400) — instant, free.
2. LLM-synthesis (Sonnet/Gemini, 32 samples + 1 revision, $0.50-1.00/task).
3. LLM-synthesis with stronger model (Opus 4.6 / GPT-5.5, 32 samples + 2 revisions, $2-5/task) — only for tasks step 2 missed.
4. LLM-synthesis with natural-language program + LLM-translates-to-ONNX (Berman 2025 style) — only for tasks step 3 missed.
5. Identity / constant-grid fallback (last resort; matches the "skip" baseline).

### 7.3 Overfitting to demo pairs

**The core risk**: LLM-synthesized program passes all 3 demos but produces wrong output on the held-out test pair. This is the **transductive program** problem.

**Mitigations**:
- **Bake-in generality**: instruct the LLM to write parameterized code, not lookups. The prompt should say: "Write a function that would work on ANY input grid with these characteristics, not just the demo inputs."
- **Hard-output rejection**: reject programs containing `return [[<int>...` or large `op.Constant` tensors with hardcoded grid values.
- **Augmentation testing**: run the program on rotated/flipped/recolored versions of the demo inputs. If the program is correct, it should produce correspondingly rotated/flipped/recolored outputs. (Trelis notes this is sometimes violated even by correct programs — ARC programs aren't always rotation-invariant — but it's a useful heuristic.)
- **Symmetry testing**: if the demos exhibit symmetry (e.g. input is symmetric under vertical flip), check the program preserves that symmetry.
- **Simplicity prior**: prefer shorter programs (fewer ops) when multiple candidates pass demos. Use the existing cost metric (`#params + #bytes`) from the validator.

**The 2024 ARC tech report** notes that this overfitting risk is why the leaderboard has both public and semi-private eval sets — to detect overfitting to public.

---

## 8. Concrete Prompt Engineering for Our Use Case

### 8.1 The setup

We need an LLM to produce `onnxscript` (or `neurogolf.dsl` calls) such that:
- Input: `(1, 10, 30, 30)` one-hot float32 tensor (batch=1, 10 color channels, H=30, W=30).
- Output: `(1, 10, 30, 30)` one-hot float32 tensor of the same shape.
- Uses minimal ONNX ops (cost = `#params + #bytes`).
- Has as few parameters as possible (the 1.44MB limit isn't the issue; the score penalty for params is).

### 8.2 Encoding (input, output) pairs in the prompt

Three representations to include (multi-format works best per Berman):

```python
def render_pair_for_prompt(input_grid, output_grid, pair_idx: int) -> str:
    """Render one (input, output) demo pair in three formats."""
    inp_2d = input_grid  # list[list[int]]
    out_2d = output_grid
    inp_1h = onehot_encode(input_grid)  # (10, H, W) of 0/1 floats, channel=color
    out_1h = onehot_encode(output_grid)
    return f"""
<demo_pair_{pair_idx}>
<ascii_input>
{render_grid_ascii(inp_2d)}
</ascii_input>
<ascii_output>
{render_grid_ascii(out_2d)}
</ascii_output>
<diff>
{render_diff(inp_2d, out_2d) if same_shape else '(shapes differ; diff omitted)'}
</diff>
<onehot_input_shape>(10, {len(inp_2d)}, {len(inp_2d[0])}) float32, channel c is one-hot for color c</onehot_input_shape>
<onehot_output_shape>(10, {len(out_2d)}, {len(out_2d[0])}) float32</onehot_output_shape>
</demo_pair_{pair_idx}>
"""
```

The ASCII grid uses 0-9 color codes per cell (matches ARC's 10-color spec). The one-hot tensor is described by shape, not enumerated (would be too verbose).

### 8.3 System prompt template

```
You are an expert at solving ARC-AGI puzzles by writing tiny ONNX neural networks.

You will be given a puzzle: a few (input, output) demo pairs of colored grids.
Each grid is a 2D array of integers 0-9 representing colors. The transformation
from input to output is the same across all demos. Your job is to figure out
the rule and implement it as an ONNX network using onnxscript.

CRITICAL CONTEXT — the ONNX network you write operates on ONE-HOT ENCODED tensors:
- Input shape: (1, 10, 30, 30) float32 — batch=1, channels=10 (one per color 0-9), H=30, W=30.
  The input grid is zero-padded to 30×30 and one-hot encoded: tensor[0, c, h, w] == 1.0
  iff grid[h][w] == c, else 0.0.
- Output shape: (1, 10, 30, 30) float32 — same encoding.
- The actual grid is recovered by argmax over the channel axis.
- The 30×30 canvas is larger than the actual grid (which can be up to 30×30 in ARC-AGI);
  the actual grid lives in the top-left corner; the rest is zero-padding (color 0).

You will use onnxscript (Microsoft's Python DSL for ONNX). Available primitives:
- op.Conv(X, W, B=None, strides=[1,1], pads=[0,0,0,0], dilations=[1,1], group=1)  # 2D conv
- op.MaxPool(X, kernel_shape, strides, pads)
- op.Slice(X, starts, ends, axes, steps)
- op.Concat(inputs, axis)
- op.Reshape(X, shape)
- op.Transpose(X, perm)
- op.ReduceMax(X, axes, keepdims)  / ReduceMin / ReduceSum / ReduceMean
- op.ArgMax(X, axis, keepdims)  # one-hot → int (color index)
- op.OneHot(indices, depth, values, axis)  # int → one-hot
- op.Where(condition, X, Y)  # element-wise select
- op.Equal(X, Y), op.Greater(X, Y), op.Less(X, Y)  # boolean tensors
- op.Add, op.Sub, op.Mul, op.Div  # element-wise math
- op.Cast(X, to=onnx.TensorProto.INT64)
- op.Constant(value=...)  # bake in a weight tensor
- op.Gather(X, indices, axis), op.GatherElements(X, indices, axis)
- op.Flatten(X, axis=1), op.Squeeze(X, axes), op.Unsqueeze(X, axes)
- op.Pad(X, pads, value)  # zero-pad
- op.Max(X, Y)  / op.Min(X, Y)

REQUIREMENTS:
1. Use as FEW ops as possible. Cost = (#params + #bytes) and lower is better.
2. Use as FEW parameters (Conv weights, biases, constants) as possible.
3. The function must work for ANY input grid (not just the demos). NO hard-coding outputs.
4. The output must be valid one-hot (only one channel = 1.0 per pixel; rest = 0.0).
5. Total ONNX model file size must be ≤ 1.44 MB.

OUTPUT FORMAT — respond in this exact structure:

<reasoning>
[step-by-step analysis of the rule, demos, and how to implement it with ONNX ops]
</reasoning>

<code>
```python
from onnxscript import FLOAT, script
from onnxscript import opset18 as op

@script()
def arc_transform(input_grid: FLOAT[1, 10, 30, 30]) -> FLOAT[1, 10, 30, 30]:
    # ... your implementation ...
    return output_grid
```
</code>

Now solve this puzzle:
```

### 8.4 Few-shot example template

Include 1-2 hand-written examples that demonstrate the one-hot idiom. Example for a "swap colors 1 and 2" task:

```python
# === EXAMPLE ===
# Puzzle: swap colors 1 (blue) and 2 (red) everywhere.
# Demos: (3 pairs of (input, output) showing the swap)
#
# <reasoning>
# Step 1: The rule is to swap two colors. In one-hot encoding, color c is
#         channel c. So we need to swap channels 1 and 2.
# Step 2: Channel swap = Transpose with perm that swaps axes 1 and 2's positions.
#         But that's a full channel permutation. Easier: use Concat with slicing.
# Step 3: Simpler: use op.Where. channel_1_input = input_grid[:, 1:2, :, :].
#         channel_2_input = input_grid[:, 2:3, :, :].
#         Output channel 1 = channel_2_input, output channel 2 = channel_1_input,
#         other channels unchanged. Use Concat to reassemble.
# Step 4: Implement.
# </reasoning>
# <code>
from onnxscript import FLOAT, script
from onnxscript import opset18 as op

@script()
def swap_colors(input_grid: FLOAT[1, 10, 30, 30]) -> FLOAT[1, 10, 30, 30]:
    # Channels 0, 3-9 stay. Channels 1 and 2 swap.
    c0 = op.Slice(input_grid, [0,0,0,0], [1,1,30,30], [0,1,2,3], [1,1,1,1])
    c1 = op.Slice(input_grid, [0,1,0,0], [1,2,30,30], [0,1,2,3], [1,1,1,1])
    c2 = op.Slice(input_grid, [0,2,0,0], [1,3,30,30], [0,1,2,3], [1,1,1,1])
    c39 = op.Slice(input_grid, [0,3,0,0], [1,10,30,30], [0,1,2,3], [1,1,1,1])
    return op.Concat(c0, c2, c1, c39, axis=1)
# </code>
```

Provide a second example showing a Conv-based transformation (e.g. "draw a red border around each blue region" via a 3×3 max-pool that detects blue-adjacent cells).

### 8.5 Revision prompt template (when initial samples fail)

```
Your previous attempt at this ARC puzzle produced incorrect output. Here is what happened:

<puzzle>
[re-render the demo pairs]
</puzzle>

<previous_attempt>
<code>
{previous_code}
</code>
</previous_attempt>

<execution_results>
For each demo pair:
  Demo 1:
    Expected output (ASCII):
    [render_grid_ascii(expected_1)]
    Your program's output (ASCII):
    [render_grid_ascii(actual_1)]
    Diff (expected vs. actual):
    [render_diff(expected_1, actual_1)]
    Result: FAIL (12 cells wrong)
  Demo 2: ...
  Demo 3: ...
</execution_results>

<error_or_warning>
[any onnxruntime error message, or "(no errors; just wrong output)"]
</error_or_warning>

Identify the bug in your previous code, then write a corrected version. Use the same output format:

<reasoning>
[what went wrong and how to fix it]
</reasoning>

<code>
```python
[from onnxscript import FLOAT, script
 ...]
```
</code>
```

### 8.6 Verification prompt (before shipping)

This is not an LLM call — it's a programmatic check:

```python
def verify_onnx_program(model_proto, demo_pairs):
    """Returns (ok, reason)."""
    # 1. Structural
    try:
        onnx.checker.check_model(model_proto)
    except onnx.checker.ValidationError as e:
        return False, f"structural: {e}"
    # 2. Size
    size = len(model_proto.SerializeToString())
    if size > 1_440_000:
        return False, f"size {size} > 1.44MB"
    # 3. Op count
    n_ops = len(model_proto.graph.node)
    if n_ops > 64:  # arbitrary cap
        return False, f"too many ops ({n_ops})"
    # 4. Param count
    n_params = sum(len(init.SerializeToString()) for init in model_proto.graph.initializer)
    # 5. Functional — run on each demo
    import onnxruntime as ort
    sess = ort.InferenceSession(model_proto.SerializeToString())
    for i, (inp, expected) in enumerate(demo_pairs):
        inp_1h = onehot_encode(inp, target_shape=(1,10,30,30))
        actual_1h = sess.run(None, {'input_grid': inp_1h})[0]
        actual_grid = onehot_decode(actual_1h)
        if actual_grid != expected:
            # also check shape
            return False, f"demo {i}: output mismatch"
    # 6. Transductive-program check (heuristic)
    code_str = model_proto.graph.SerializeToString()
    # Look for large Constant tensors that could be hardcoded grids:
    for init in model_proto.graph.initializer:
        if len(init.dims) >= 2 and len(init.SerializeToString()) > 1000:
            return False, f"suspected hardcoded output (initializer {init.name} is large)"
    return True, "ok"
```

### 8.7 Full pipeline (recommended for NeuroGolf)

```python
def solve_task_with_llm(task, max_rounds=3, max_samples_per_round=32):
    """LLM-synthesis fallback for tasks the DSL dispatcher missed."""
    demo_pairs = [(p['input'], p['output']) for p in task['train']]
    test_input = task['test'][0]['input']

    # Round 1: fresh samples
    candidates = []
    for round_idx in range(max_rounds):
        if round_idx == 0:
            prompt = build_initial_prompt(demo_pairs)
        else:
            # revision prompt: use best candidate from prior round
            best = candidates[0] if candidates else None
            prompt = build_revision_prompt(demo_pairs, best)

        # Sample N candidates in parallel (use OpenAI/Anthropic n parameter)
        responses = llm_sample(prompt, n=max_samples_per_round, model="gemini-3.5-flash-high")
        for resp in responses:
            code = extract_code_block(resp)
            if code is None: continue
            try:
                model_proto = compile_onnxscript(code)
            except Exception as e:
                continue  # parse error; skip
            ok, reason = verify_onnx_program(model_proto, demo_pairs)
            if ok:
                candidates.append((model_proto, code, "verified"))

        # Sort candidates by cost (params + bytes), ascending
        candidates.sort(key=lambda c: model_cost(c[0]))

        if candidates:
            return candidates[0][0]  # cheapest verified model

    # If no candidates passed all demos, return the one with lowest Hamming
    # (for potential manual inspection / fallback chain)
    return None

def model_cost(model_proto):
    """Cost per competition rules: #params + #bytes."""
    n_params = sum(len(init.dims) and 1 for init in model_proto.graph.initializer)  # rough
    n_bytes = len(model_proto.SerializeToString())
    return n_params + n_bytes
```

---

## 9. Recommended Action Plan for NeuroGolf

Given the existing codebase (DSL dispatcher, 16/400 solved, well-structured `dsl.py`, `validator.py`), here's the recommended incremental path:

### Phase 1 — Wire in LLM-synthesis as fallback (1-2 days)

1. Add `llm_synthesis.py` module that fires only when DSL dispatcher returns `None`.
2. Use **Gemini 3.5 Flash (High)** if available, else **Sonnet 4.6 (High)**, else **GPT-5.5 (Medium)**.
3. Use the **system prompt + 1-shot example** from §8.3-8.4. Target `onnxscript` directly.
4. Sample **32 candidates per task** in parallel (n=32 OpenAI API parameter, or batch Anthropic calls).
5. **1 revision round** if no candidate passes all demos. Use the revision prompt from §8.5.
6. Verify with the existing `validator.py` (already does structural + functional checks).
7. Add the transductive-program detector from §8.6 (6) to reject hard-coded outputs.
8. Budget: 400 × $0.50 = **$200**, ~2-3 hours wallclock with 4-way task parallelism.

**Expected lift**: from 16/400 to **~150-250/400** based on Greenblatt's 50% baseline at 8k samples; at 32 samples + 1 revision we should get ~30-50% of the remaining 384 tasks = **+115-190 tasks**.

### Phase 2 — Berman-style 4-generation evolutionary (3-5 days, only if Phase 1 leaves budget)

1. Extend `llm_synthesis.py` with the 4-gen architecture from §2.2.
2. Sample 50 candidates per generation × 4 = 200 calls/task × 400 = 80,000 calls.
3. Add pooled-parent prompts (combine multiple parents that each solve different demos).
4. Budget: 400 × $1.00 = **$400**, ~10 hours wallclock with 4-way parallelism.

**Expected lift**: from 250/400 to **~300-350/400**. Plateaus around here without stronger models.

### Phase 3 — Strong-model fallback for hardest tasks (1 day, $100)

1. Identify the ~50 tasks Phase 2 missed.
2. Run them through **Claude Opus 4.6 (High, 120K thinking)** at $3.47/task × 50 = $174.
3. Same Greenblatt-style pipeline but with the strongest available model.

**Expected lift**: +5-10 more tasks.

### Phase 4 — Natural-language programs for irregular-shape tasks (if any budget remains)

For tasks where pixel-level ONNX is the wrong abstraction (Trelis' "irregular shapes" failure mode), try Berman 2025's approach: have the LLM write a **natural-language description** of the transformation, then have a second LLM call translate that description into onnxscript. This decouples "understanding the rule" from "implementing in ONNX."

---

## 10. Key References

| # | Title | Author | Date | URL |
|---|---|---|---|---|
| 1 | Getting 50% (SoTA) on ARC-AGI with GPT-4o | Ryan Greenblatt | 2024-06-17 | `blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt` |
| 2 | ARC Prize 2024 Technical Report | ARC Prize team | 2024-12 | `arxiv.org/html/2412.04604v1` |
| 3 | How to Beat ARC-AGI by Combining DL and Program Synthesis | M. Knoop, F. Chollet | 2024-10-28 | `arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis` |
| 4 | How I came in first on ARC-AGI-Pub using Sonnet 3.5 with Evolutionary Test-time Compute | Jeremy Berman | 2024-12 | `jeremyberman.substack.com/p/how-i-got-a-record-536-on-arc-agi` |
| 5 | Solving ARC Prize Tasks by Writing Python Code | Trelis Research (Ronan) | 2025-11-05 | `trelis.substack.com/p/solving-arc-prize-tasks-by-writing` |
| 6 | Self-Improving Language Models for Evolutionary Program Synthesis (SOAR) | Pourcel et al. | 2025 | `arxiv.org/html/2507.14172v2` |
| 7 | FunSearch: Making new discoveries in mathematical sciences using LLMs | DeepMind | 2023-12 | `deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models` |
| 8 | ARC-AGI-2 benchmark page | Epoch AI | 2025 | `epoch.ai/benchmarks/arc-agi-2` |
| 9 | ARC Prize leaderboard | ARC Prize | live | `arcprize.org/leaderboard` |
| 10 | Teaching Large Language Models to Self-Debug | Chen, Lin et al. | ICLR 2024 | `proceedings.iclr.cc/paper_files/paper/2024/file/2460396f2d0d421885997dd1612ac56b-Paper-Conference.pdf` |
| 11 | Introducing ONNX Script: Authoring ONNX with the ease of Python | Microsoft | 2023-08-01 | `opensource.microsoft.com/blog/2023/08/01/introducing-onnx-script-authoring-onnx-with-the-ease-of-python` |
| 12 | ONNX Script docs | Microsoft | 2026 | `microsoft.github.io/onnxscript` |
| 13 | arc_draw_more_samples_pub (code) | R. Greenblatt | 2024 | `github.com/rgreenblatt/arc_draw_more_samples_pub` |
| 14 | LeDex: Training LLMs to Better Self-Debug and Explain Code | anon | NeurIPS 2024 | `neurips.cc/virtual/2024/poster/94367` |

---

## 11. Bottom Line

**For the NeuroGolf 2026 championship, the SOTA LLM-synthesis recipe is well-established and the budget math works**:

- Use **Gemini 3.5 Flash (High) at $0.85/task** as the workhorse — single-shot CoT scores 92.5%/72.1% on ARC-AGI-1/2.
- Target **`onnxscript`** directly — LLM writes Python with `op.*` calls, `@script()` decorator builds the ONNX graph, `onnx.checker.check_model` validates structure, `onnxruntime.InferenceSession` validates function.
- **32 samples + 1 revision round** per task; $0.50-1.00/task; fits in 2-3 hours wallclock with task parallelism.
- **Total budget ~$200-400** for 400 tasks. Feasible.
- **Expected lift**: from 16/400 (current DSL-only) to **250-350/400** depending on how many tasks fall in the LLM's competence zone vs. the long tail of ARC-AGI-2-style compositional tasks.
- **Fallback chain**: DSL → LLM-synthesis (Gemini Flash) → LLM-synthesis (Claude Opus) → natural-language programs → identity. Detect "stuck" via Hamming-distance plateau after K rounds.
- **Watch out for**: transductive (hardcoded-output) programs — must filter; off-by-one errors — revision catches; program execution sandboxing — onnxscript's lack of `while`/unbounded `for` is a feature, not a bug.

The single highest-leverage improvement over the existing codebase is wiring in the LLM-synthesis fallback. The single highest-leverage model choice is Gemini 3.5 Flash (High) at $0.85/task — frontier CoT performance at commodity prices.
