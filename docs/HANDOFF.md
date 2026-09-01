# Marlowe — handoff

Read cold in ten minutes. Assumes you have the repo and can run the tests; assumes no
knowledge of the conversation that produced it.

**Goal.** Compress `Qwen/Qwen3.8-27B` into two shippable models via structure-aware depth
pruning plus distillation. **Marlowe-18B is the headline artifact**; Marlowe-22B is the
ladder's intermediate rung, not a lesser goal. The thesis is that the open-weight landscape
has a hole between Qwen3.5-9B (~22 AAII) and Qwen3.8-27B (~52), and an 18B that stays
intelligent at Q4_K_M lands in a tier nobody occupies.

The deliverable is **a base that quantises well**, not a checkpoint that fits one card.

---

## 1. Current state

| stage | status | notes |
|---|---|---|
| `stage1-smoke` | **complete** | converter blocker closed against real weights |
| `stage0-bitwidth` | ready to run | ~8 h; control curve, not a gate (§2) |
| `stage2-baselines` | **blocked** | needs a hosted bf16 endpoint (§4) |
| `stage3-score` | ready | 8 h, needs `data/calib.jsonl` (still a placeholder) |
| `stage4`–`stage8` | not started | |

**Running right now:** a memory-configuration search (`marlowe fitcheck`) against a
sizing-only 22B. Results were not in when this was written. See §5.

### Where things live

```
runs/marlowe-22b/models/parent      55.6 GB, the real Qwen3.8-27B (downloaded, verified)
runs/marlowe-22b/gguf/              smoke-2cut-iq3_m.gguf (12.2 GB) — the Stage 1 output
runs/marlowe-22b/manifests/         stage1-smoke.json (status: ok)
.tools/llama.cpp/                   checkout + CPU build, with the vendored patch applied
.tools/llama-cuda/                  prebuilt CUDA b10738 binaries — this is what gets used
vendor/0001-qwen35-*.patch          REQUIRED converter patch, pinned to upstream 8887a48
<scratch>/sizing-22b                throwaway 22B for memory sizing only; safe to delete
```

`marlowe doctor` prints the live environment, the llama.cpp backend, converter layout
support, and an itemised disk budget. Run it first.

### Environment facts worth not rediscovering

- **transformers must be ≥5.16.** 5.1 cannot parse `qwen3_5` at all; Stages 3/5/6 would
  fail in their first minute. Pinned in `pyproject.toml`.
- llama.cpp binaries are the **prebuilt CUDA** ones. No CUDA toolkit is installed and the
  machine is not elevated, so a source CUDA build is not available. This is fine: the
  vendored patch is Python-only and the C++ loader support was already upstream.
- The CPU-only llama.cpp build in `.tools/llama.cpp/build` still exists. `find_binary`
  prefers `.tools/llama-cuda`. Stages 0/2/7 **refuse** a CPU-only backend (28 h/variant vs
  2.8 h).

---

## 2. Decisions, and why

These do not survive in code. They are the expensive part of this document.

### The fp16 ladder, and why NF4-head candidates are gated

`heal.MEMORY_CANDIDATES`, best-quality-first:

```
fp16-head-2048         fp16 head, seq 2048, rank 32
fp16-head-1536         fp16 head, seq 1536, rank 32
fp16-head-1024         fp16 head, seq 1024, rank 32
fp16-head-1024-rank16  fp16 head, seq 1024, rank 16
--- approval required below ---
nf4-head-adamw32       NF4 head, fp32 Adam
nf4-head-adamw8        NF4 head, 8-bit Adam
```

The loss is top-K KL against **teacher logits**. Quantising `lm_head` injects noise into
exactly the quantity being matched — and `merge_adapters` loads the base in **bf16**, so the
adapter learns to cancel an error that never reaches inference. That is strictly worse than
a wash. bitsandbytes skips `lm_head` by default for this reason; the pipeline only overrides
that under explicit approval (`--allow-quantized-head`).

Ordering rationale: shortening the sequence is a real cost (less long-range structure per
step) but leaves the loss target intact. Halving the LoRA rank costs adapter capacity but
also leaves the target intact. Both are cheaper than corrupting the target.

If the fp16 ladder exhausts, the search **stops and reports** rather than stepping over the
cliff. Gated candidates are not even probed without approval.

Two savings are near-free and are in *every* candidate: `embed_tokens` on CPU (not a Linear,
so NF4 can't touch it; 2.5 GB; a lookup on CPU is cheap) and a chunked loss (`2048 × 248320`
is 1.02 GB in bf16 before any softmax intermediate).

### Stage 0 is a control curve, not a go/no-go

The original question — "is there a ~10 GB quant of the 27B that fixes circling?" — was
answered **no**, outside this pipeline. That is *why* the pruning is happening. No Stage 0
result changes that decision.

Its remaining job: give the parent's own bpw-vs-repetition response, so that when the healed
22B/18B are quantised down, repetition caused by **quantisation** can be separated from
repetition caused by **insufficient healing**. Without the parent curve those are confounded
and a bad IQ3_M number is uninterpretable.

### Mixed-length caching was rejected

Not for cost — it's cheap. `ShardMeta.seq_len` is already per-shard and the `.npz` payload
doesn't change shape, so it needs ~4 small code changes and no format migration.

It was rejected because it **doesn't solve the memory problem**. Peak VRAM is
`weights + optimizer + activations(seq)`; the first two are constant, so the peak of a mixed
run equals the peak of its longest sequence. Mixing changes how *often* you sit at the peak,
not what the peak is — and **OOM is a max statistic, not an average**.

The same error killed an earlier proposal to use gradient accumulation for long sequences:
accumulation reduces optimizer-step frequency, not per-forward activation memory.

### The memory plan is decided in Stage 5, confirmed in Stage 6

The teacher cache is built at a fixed sequence length and its distributions are
**position-aligned** — `iter_batches` refuses a cache whose length doesn't match training.
If the search ran inside Stage 6 and selected a shorter sequence, it would reject a cache
built six hours earlier.

So Stage 5 runs `search_memory_plan`, persists to `metrics/memory_plan.json`, and builds the
cache at the selected length. Stage 6 **loads** that plan and re-probes to confirm the
measurement still holds, but does not re-choose — a different answer there would invalidate
the cache it is about to consume.

### The prompt set is synthetic, and what that costs

`data/circling_prompts.jsonl` has 9 prompts and **zero validated circling triggers**. They
are plausible reasoning shapes, not prompts anyone watched the IQ3_XXS build loop on.

Consequence: the **relative** bpw-vs-repetition curve is real; **absolute** rates are not
workload-representative, and `trigger_subset` is empty by construction. Every
`RepetitionReport` from an unvalidated set carries a `caveat` string saying so, and
`headline()` appends `[synthetic prompts]`.

Prompt files are fingerprinted (sha256, count, trigger count) and the hash travels in every
report. Stage 0's curve and Stage 7's gate are comparable only if the hashes match —
`report_a.comparable_with(report_b)` checks contents, not paths.

### 18B is the headline; 22B is the intermediate

Marlowe-18B cannot be reached by a direct 27B→18B cut: that removes 36% of depth in one
step, outside the band where healing reliably recovers. Two sequential ~19% cuts from an
already-healed parent stay inside it. `configs/marlowe-18b.yaml` sets
`requires_healed_parent: true` and `stage8-ladder` refuses without it.

The 18B's healing budget (100M tokens) is sized for it as the target, not as leftover
capacity. The 22B's 50M is sized to make the ladder reachable.

**Stage 8 re-probes its own memory plan** against the actual 18B shape — the child runs at
`run_dir/<child name>`, so plan files don't collide, and the child config starts at
`lora_rank: 32`. This matters: at 41 layers the weights drop ~2.4 GB, likely reopening rank
32 or a longer sequence. Inheriting rung 1's plan would give the *harder* healing job less
adapter capacity than the easier one.

### Ship gate: three bit-widths, all required

`SHIP_BIT_WIDTHS = ("q4_K_M", "iq4_xs", "iq3_m")`. Under-healed weights carry larger
activation outliers and degrade unevenly across quantisation schemes, so passing at one
width proves nothing about the others. A width that was **never built counts as a failure**,
not an absence — otherwise a missing measurement reads as a pass.

Both gate criteria, per width: KL to the parent strictly below the user's IQ3_XXS build, and
repetition at or below the bf16 baseline.

### KL reference is Q8_0, not bf16

bf16 is 56 GB against 32 GB of RAM — llama.cpp would mmap and page from disk for the whole
pass. Q8_0 is ~28.6 GB and near-lossless, and every number in the project is a *relative*
comparison against the same reference file, so the substitution is free.
`--reference-outtype bf16` overrides. This does **not** affect the bf16 *repetition*
baseline, which comes from a hosted endpoint.

---

## 3. Bugs found, and the pattern

| bug | where | why it mattered |
|---|---|---|
| Converter regenerated layout from `full_attention_interval` | `conversion/qwen.py` upstream | mis-typed **15 of 52 layers** on a pruned stack, including attention layers marked recurrent (silently dropping their KV cache) |
| `range(64)` hardcoded in the surgery plan | `surgery.py` (original script) | invisible until the ladder's second rung, where the parent is 52 layers |
| transformers 5.1 lacks `qwen3_5` | environment | Stages 3/5/6 would die in minute one of an 8-hour job |
| `getattr(out, "x", None) or out[0]` | `teacher.py` | `or` calls `bool()` on a multi-element tensor → raises 6 h into the cache run |
| torch accounting instead of device memory | `probe_training`, then `estimate_peak_gb` | overstated headroom by ~1 GB; would have selected a config that OOMs at hour 30 **with its own record certifying it fit** |
| `lora_rank` not persisted in the memory plan | `save_memory_plan` | Stage 6 would reload the plan and silently revert the rank that made it fit |
| quants list narrower than the gate | `configs/marlowe-18b.yaml` | gate would fail after a week on *missing data*, looking like a quality failure |
| surgery not idempotent | `write_checkpoint` | collided with its own prior output after a full streaming pass |

**The pattern: every one of these loads, runs, and produces plausible output.** None crashes.
None OOMs. The model generates fluent text, the search reports a number, the config parses.
That is the failure shape this codebase produces, and it is what to look for.

Corollaries that earned their place:

- **A check that fails closed is only useful if it fails for the right reason.** The Stage 1
  probe initially reported the converter as broken when it was correct — llama.cpp names
  DeltaNet's fused projection `attn_qkv`, which the classifier read as attention. It now
  prefers the explicit `recurrent_layers` metadata (authoritative) over name inference.
- **The same wrong assumption tends to appear more than once.** The torch-vs-device confusion
  appeared in the probe and then in the estimator ten minutes later. An audit found no third
  instance; tests now pin all four call sites, including `preflight.probe` which was correct
  by accident.
- **A refusal that recommends a broken alternative is barely better than no refusal.** The
  long-context refusal suggested a fallback length that didn't itself fit, until the
  estimator was calibrated. A test now asserts the suggested ceiling clears the margin.

---

## 4. Open items

1. **Stage 2 needs a hosted bf16 endpoint.** It fails closed without one, by design — that
   baseline defines the repetition ship criterion, and "better than a badly quantised model"
   is a bar the *unhealed* checkpoint might clear. The operator is supplying an OpenRouter
   URL, model string, and key:
   ```bash
   marlowe run stage2-baselines --config configs/marlowe-22b.yaml \
     --hosted-base-url ... --hosted-model ... --hosted-api-key ... \
     --iq3-xxs-gguf <their current build>
   ```
   `--allow-missing-bf16-baseline` proceeds for the Stage 3–6 artifacts; Stage 7 then cannot
   pass.

2. **Real circling triggers may land before Stage 7.** They can be applied to already-built
   quants without redoing any sweep:
   ```bash
   marlowe repetition --gguf <quant> --prompts data/my_triggers.jsonl --out metrics/rerun.json
   ```
   Minutes, not hours.

3. **`data/calib.jsonl` and `data/heal_corpus.jsonl` are placeholders.** Stage 3 refuses to
   start unless the calibration set yields 4 sequences of 32768 tokens. See `data/README.md`.

4. **The memory search result is not in.** See §5.

---

## 5. The measurement in flight

`marlowe fitcheck --model <sizing-22b> --config configs/marlowe-22b.yaml`

Known so far: `fp16-head-2048` peaks at **16.64 GB device-level of 17.17 GB** (~0.53 GB
free), which is below the 1.0 GB `FIT_MARGIN_GB` and is rejected. The search moved on.

The estimator, calibrated against that measurement, predicts:

```
seq 1024   est peak 16.03 GB   est headroom 1.14 GB   fits
seq 1536   est peak 16.30 GB   est headroom 0.87 GB   marginal — measurement decides
seq 2048   est peak 16.58 GB   est headroom 0.59 GB   REJECT (measured 16.64)
```

Pre-committed calibration bands, agreed before the number arrived:

- measured 1024 ≤ **16.15 GB** → calibration holds, probe 1536 next
- measured 1024 > **16.3 GB** → context term still under-calibrated; skip 1536, probe
  `fp16-head-1024-rank16`, and re-derive `_EST_CUDA_CONTEXT_GB` from two points
- measured 1024 near **8 GB** → memory model wrong by 7 GB; stop and investigate before
  Stage 5

If the fp16 ladder exhausts entirely, **stop and ask** rather than taking an NF4 head.

---

## 6. Do not re-litigate

| settled | one-line reason |
|---|---|
| Whitelist removability (`linear_attention` only) | unknown layer types must fail closed; Flash-Next's QSA would be silently removable under a blacklist |
| Periods derived from `layer_types` | no constant 4 anywhere; the same code must work on 48- and 64-layer stacks |
| Ablation KL, not angular distance | angular distance nominates exactly the DeltaNet layers that must not be cut |
| Score at ≥8192, late positions | recurrent-state damage accumulates and is invisible near position zero |
| Both metrics at every checkpoint | KL is teacher-forced and cannot see circling |
| `full_attention_interval` removed + asserted | some runtimes regenerate `layer_types` from it |
| bf16 never resident | 55.6 GB against 16 GB VRAM / 32 GB RAM |
| Q8_0 KL reference | bf16 pages from disk; every number is relative to the same file anyway |
| Delete-temps by default | 519 GB retained doesn't fit; `--keep-intermediates` opts in |
| Plain peft, not Unsloth | Unsloth pins transformers/torch aggressively and would risk the validated stack; `fitcheck` measures throughput directly |
| Prebuilt CUDA llama.cpp | no toolkit, no elevation; the patch is Python-only |
| `FIT_MARGIN_GB = 1.0`, device-level | 0.5 GB is not a margin for a 3–5 day unattended run |
| Stage 1 before Stage 0 | if the converter can't express the layout, Stage 0's results are irrelevant |
| `model.norm` as an exact path | peft matches `modules_to_save` by suffix; bare `"norm"` also selects 3 layernorms per layer |

---

## 7. Working notes

- `pytest` — 287 tests, no GPU or weights needed. `pytest -m slow` adds the live QLoRA stack
  check (needs CUDA + bitsandbytes).
- `ruff check .` and `mypy marlowe` are clean and expected to stay that way.
- `tests/test_regressions.py` pins every bug in §3. They share the property that none of them
  raise on their own.
- Stages are resumable and manifest-gated. A manifest is a claim; declared outputs existing
  on disk is the evidence. `--force` re-runs.
- Large heredocs of Python through the shell repeatedly mangled `\n` inside f-strings during
  this build. Prefer writing files directly.
