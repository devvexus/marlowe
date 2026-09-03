# Marlowe — handoff

Read cold in ten minutes. Assumes you have the repo and can run the tests; assumes no
knowledge of the conversation that produced it.

**Goal.** Compress `Qwen/Qwen3.8-27B` into two shippable models via structure-aware depth
pruning plus distillation. **Marlowe-18B is the headline artifact**; Marlowe-22B is the
ladder's intermediate rung, not a lesser goal. The thesis is that the open-weight landscape
has a hole between Qwen3.5-9B (~22 AAII) and Qwen3.8-27B (~52), and an 18B that stays
intelligent at 4-bit lands in a tier nobody occupies. It ships at **IQ4_XS**; the 22B rung
ships at **IQ3_M**. See "Deployment width is not gate width" in §2 — the ship gate measures
three widths, only two of which anyone deploys.

The deliverable is **a base that quantises well**, not a checkpoint that fits one card.

---

## 0. What to do first

**Read `docs/TRANSFER_PLAN.md` first.** It is required, not background. This document
describes *how the pipeline runs*; that one describes *what it is for*, and the two-phase
structure it defines changes how several results here should be read:

* **Phase A** is healing -- KL to the 27B, everything the stages below implement. The KL ship
  gate applies here **and only here**.
* **Phase B** is uplift past the parent: after Phase A ships, a fresh rank-32 LoRA on the
  merged model, SFT on reasoning traces from stronger teachers. **KL to the 27B goes up by
  design.** A rising KL after Phase B is the objective succeeding. `ship_gate_multi` raises if
  handed a non-A phase rather than reporting a failure, because a gate that does not know
  this rejects the better model for working.
* The phases **cannot run together** -- "be the 27B" and "reason like Flash-Next" fight when
  combined, and compose when sequenced.
* **Both ship**, under different names and different claims. The 18B rung cuts from the
  **Phase A** 22B, because the ladder wants a parent-faithful intermediate.

Ordered. Do not infer priority from the rest of this document.

1. **Run Stage 4. It is the next thing, and everything it needs is on disk.**

   Stage 3 is complete: the cut list is `[4, 5, 8, 9, 13, 14, 16, 17, 37, 38, 40, 41]`, and it
   beats the positional control by 42% (§1a). Stage 4 applies it and writes the 22.3 B
   unhealed checkpoint, then measures it -- the unhealed floor is the number healing has to
   climb from, so it is recorded rather than assumed.

   ```bash
   marlowe run stage4-surgery --config configs/marlowe-22b.yaml
   ```

   ~1 h. Nothing blocks it.

2. **Stage 6 does not fit yet, and this is the open engineering problem.**

   §3a is the full ladder. It started 8.3 GB over the card and paging silently; it is now
   ~96 MiB short at rank 16, seq 1024, with the allocator capped at 0.96. Throughput went
   81 → 204 tok/s along the way, which halved the healing schedule.

   What is left is genuinely hard: the card is *physically* exhausted ("0 bytes free"), and
   the remaining levers are the `rank16-seq768` rung (unmeasured, the operator's fallback) or
   rank 8. Moving desktop applications to the iGPU does **not** help -- that was measured
   (§5c).

   Note the order of operations if you retry: the cap is derived from the measured context, so
   measure context bare first, then probe.

3. **The fit gate is the driver's Shared Usage counter. Not throughput, not torch.**

   A probe measured **74.5 tok/s while 6.8 GB over the card**. Under WDDM the driver pages
   instead of refusing, so an over-committed run can look fast for a few steps.
   `marlowe.gpumem` samples the adapter during the probe: idle is ~90 MB, and anything above
   250 MB over that floor is paging. `ProbeResult.fits()` uses it, torch accounting is the
   fallback.

4. **Before any multi-day run, soak it.** Every fit so far is an 8-step probe, and the failure
   the margin exists to prevent is fragmentation accumulating into an OOM at hour 30.
   `marlowe fitcheck --no-search --steps 500 --log-every 10` reports the slope of trapped
   fragmentation per 1000 steps. A flat slope after warm-up means the allocator reached a
   steady block pattern; a positive one sets a restart interval rather than failing the
   configuration.

5. **Stage 2 is blocked on the operator's OpenRouter endpoint** (§4). Do not work around it
   with `--allow-missing-bf16-baseline` unless asked: that baseline defines the repetition
   ship criterion.

6. **Stage 0 is partial and resumable at zero re-work** -- completed variants are read from
   disk. Budget it by variant *size*, not count (§5a): the wide ones are ~4x slower.

7. **The Unsloth evaluation is lower value than it was.** It was sized against a 15-day
   schedule at 73.5 tok/s; the stack now runs at 204 without it.

The GPU is the scarce resource and only one thing can use it at a time. `llama-quantize` from
the CUDA build initialises a CUDA context even for CPU-side work, so it cannot run alongside a
training probe that has taken the whole card. And `llama-server` does **not** die with its
parent -- kill it explicitly, or the next job silently starves.

---

## 1. Current state

| stage | status | notes |
|---|---|---|
| `stage1-smoke` | **complete** | converter blocker closed against real weights |
| `stage3-score` | **complete** | 3.84 h. Cut list selected; beats positional control by 42% (§1a) |
| `stage0-bitwidth` | **partial** | 4-5 of 8 variants measured. Resumable at zero re-work |
| `stage2-baselines` | **blocked** | needs a hosted bf16 endpoint (§4) |
| `stage4-surgery` | **ready** | Stage 3's cut list is on disk; this is the next thing to run |
| `stage5`–`stage8` | not started | Stage 6 is blocked on the fit (§3a) |

Both corpora are built (`data/calib.jsonl` 1.71M tokens / 34 sequences,
`data/heal_corpus.jsonl` 35.01M tokens), and the parent importance matrix is complete
(415 chunks, the full corpus). Those three were the blockers on everything downstream of
quantisation.

## 1a. Stage 3 result: the cut list

```
selected   [4, 5, 8, 9, 13, 14, 16, 17, 37, 38, 40, 41]
child      52 layers (16 full_attention, 36 linear_attention), 22.2968 B  (matches expected_params)
profile    runs/marlowe-22b/metrics/layer_profile.json  (+ .png)
```

**Measured selection beats the positional control by 42%**, which is the empirical
justification for the stage costing four hours rather than cutting evenly:

```
positional control  [4,9,13,18,24,29,33,38,44,49,53,58]   summed KL 0.217355
measured selection  [4,5,8,9,13,14,16,17,37,38,40,41]     summed KL 0.125685
overlap: 4 of 12
```

The selection clusters in two bands (4-17, 37-41) rather than spreading out, which is
structure even spacing cannot find. Ranked least-damaging first:
`16, 17, 5, 38, 37, 9, 13, 14, 41, 4, 40, 8, 45, 29, ...`

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

## 1b. Stage 0 (token-level only): the parent's bpw curve is flat

Five of eight variants measured. `runs/` is gitignored, so these numbers live here or
nowhere.

**Token-level only.** These `rep8`/`rep32` columns are n-gram repetition at `pp=0.0`, and
nothing more is planned: the semantic repetition harness is **dropped**. The objective is
transfer from the parent, measured by KL, and Stage 0 is a control curve on the *parent* --
it says where quantisation alone starts to hurt, which is context for reading Stage 7, not a
gate.

**The three custom mixes were never run, and as written they would not have been mixes.**
Their patterns used HF safetensors names (`q_proj`, `linear_attn.*`) against GGUFs that call
those `attn_q` / `attn_qkv` / `ssm_*`; llama-quantize ignores an unmatched pattern, so all
three would have produced plain `iq3_xxs` or plain `iq3_s` under three different names.
Fixed, with `assert_patterns_match` refusing a mix whose patterns match nothing -- see §3.

**Do not resume Stage 0 until `repetition.max_tokens` is raised above the observed circling
length.** Every completion so far hit the 2048 cap (`cap_hit` ~1.0), so the remaining
variants would add three more points measured at a ceiling rather than at a collapse.

```
recipe         GB   bpw     rep8    rep32   loop  cap_hit
iq3_xxs     11.19  3.27   0.0423   0.0029  0.000    0.995
iq3_s       12.42  3.63   0.0434   0.0042  0.000    1.000
iq3_m       12.58  3.67   0.0410   0.0037  0.000    1.000
iq4_xs      15.08  4.41   0.0386   0.0029  0.000    1.000
q4_k_s      15.59  4.55   0.0390   0.0033  0.000    0.995
```

**Read this cautiously.** Across 3.27-4.55 bpw, rep8 moves 0.0386-0.0434 and is **not
monotonic** -- `iq3_s` is the worst despite being wider than `iq3_xxs`. Zero hard loops
anywhere. The spread is plausibly inside the noise of 200 completions (95% CI at p=0.04 is
about +/- 2.7pp, far wider than the 0.5pp of observed spread).

Two reasons not to conclude "quantisation does not cause circling":

- **The prompt set is synthetic** and contains no confirmed trigger (§2). This measures that
  these nine prompts do not provoke circling at any width.
- **`cap_hit_rate` is ~1.000.** Effectively every completion hit the 2048-token ceiling
  rather than finishing. Circling develops over length, so anything emerging past 2048 tokens
  is invisible at this `repetition.max_tokens`. That is a plausible explanation for both the
  flat curve and the zero loop rate, and it is a harness limitation rather than a model
  result.

The curve can still do its declared job -- a baseline to subtract, so repetition in a healed
child can be attributed to healing debt rather than quantisation -- it is simply far less
informative than a curve with structure. If real triggers arrive,
`marlowe repetition --gguf <quant> --prompts <file>` re-measures without re-quantising, but
note Stage 0 deletes its candidates on *successful* completion: the five GGUFs are on disk
only because the run was stopped early.

## 1c. Stage 4 is done. The floor is measured, and the gate is tighter than it looks

**Complete.** Surgery -> GGUF convert -> child imatrix -> IQ3_M -> Q4_K_M, all verified.
Surgery streamed safetensors on CPU (`safe_open` per tensor, no `from_pretrained`), peak RSS
4.9 GB; nothing in this path ever made the 27B resident in bf16.

| artefact | size | detail |
| --- | --- | --- |
| `models/marlowe-22b-unhealed` | 44.6 GB | 52 layers, 36 linear + 16 full, 22.2968 B params, 683 tensors |
| `gguf/marlowe-22b-unhealed-bf16.gguf` | 44.61 GB | `block_count` 52, `recurrent_layers` length 52, matches `child_layout` element-for-element |
| `gguf/imatrix/child-unhealed-full.dat` | 11.0 MB | 415 chunks at 4096, 800 tensors, 1,699,840 tokens |
| `gguf/marlowe-22b-unhealed-iq3_m.gguf` | 10.495 GB | bpw 3.7656 |
| `gguf/marlowe-22b-unhealed-q4_k_m.gguf` | 13.739 GB | bpw 4.9296 |
| `metrics/reference.kld` | **77.28 GB** | 38 chunks at `-c 8192`, PPL 4.885, certified |

Note `stage4_surgery`'s docstring says the unhealed checkpoint "gets measured". **It does
not** -- the code runs surgery and asserts the parameter count, nothing else. The floor KL is
separate work, done through llama.cpp only.

### The floor, measured

Both against `metrics/reference.kld` (Q8_0 parent, held-out corpus, `-c 8192`):

| | kl_mean | kl_median | kl_p99 | kl_max | top-p agree |
| --- | --- | --- | --- | --- | --- |
| 27B iq3_xxs (gate baseline) | 0.144219 | 0.022859 | 1.647745 | 33.36 | 89.96% |
| **22B child IQ3_M (floor)** | **0.451331** | 0.153132 | 4.879992 | 36.84 | 77.98% |

0.45 nats unhealed is inside the expected band for 19% layer removal. The median moved 6.7x
against the mean's 3.1x, which says the damage is **diffuse rather than concentrated in a
tail** -- the case LoRA recovers well. Record this as the floor; it is the number healing has
to close.

### The collapse mode is a lock, not a decay

Measured on the unhealed child at IQ3_M, thinking preset, prompt `core-equations-ai`
("What are the core equations in AI"), 3072 max tokens:

| | rep8 | rep32 | longest repeated run | stop |
| --- | --- | --- | --- | --- |
| control, `presence_penalty=0.0` | 0.3424 | 0.2935 | **435 tokens** | length |
| `presence_penalty=1.5` | **0.0000** | **0.0000** | **0** | length |

The output does **not** degrade into noise. It stays coherent, reaches the ambiguity in the
prompt's own phrasing, begins quoting the term back to itself, and then locks:
`"AI" in "AI" (AI) "AI" in "AI" (AI) ...` for 435 tokens, never terminating. The quoted token
becomes its own highest-probability continuation and the distribution has nothing sharp
enough to escape it.

**This distinction sets what healing has to do.** A decay would mean the output distribution
was destroyed and has to be rebuilt. A lock means the distribution is still broadly right --
the model knows what it is doing until it reaches a fork it can no longer resolve -- and is
merely too flat to break a self-reinforcing cycle. **Healing has to sharpen a blurry
distribution, not reconstruct one.** That is the cheaper of the two problems, and it is
consistent with the diffuse damage signature (median moved 6.7x against the mean's 3.1x).

### The lock is not inherited. Pruning makes the model quantisation-fragile

All at `pp=0.0`, thinking preset, prompt `core-equations-ai`, seed 0.

| model | bpw | tokens | rep8 | longest run | locks? |
| --- | --- | --- | --- | --- | --- |
| parent Q8_0 (bf16 proxy) | ~8.5 | 3072 | 0.0000 | 0 | no |
| **parent IQ3_XXS** | **3.27** | 3072 | 0.0000 | 0 | **no** |
| child unhealed bf16 (NF4) | ~4.5 | 2048 | 0.0000 | 0 | no |
| **child unhealed IQ3_M** | **3.77** | 2048 | 0.0784 | 36 | **yes** |
| child unhealed IQ3_M | 3.77 | 3072 | 0.3424 | 435 | yes |

Read the two bold rows together. **The parent survives 3.27 bpw. The child fails at 3.77
bpw** -- half a bit *more* precision, and it locks anyway.

So neither cause is sufficient alone:

* pruning alone does not lock -- the child in bf16 is clean;
* quantisation alone does not lock -- the parent at a *lower* bit-width is clean;
* the two together do. **Pruning does not cause the repetition; it removes the margin that
  made the model tolerant of quantisation.**

An earlier reading of the bf16-vs-IQ3_M pair concluded "pruning does not reach the attractor,
quantisation tips it". That was right as far as it went and incomplete: it did not explain
why the *parent* tolerates even fewer bits. The parent controls are what distinguish
"quantisation is the cause" from "pruning made quantisation lethal", and only the second
survives the data.

**This is the strongest evidence so far that healing addresses the problem.** Healing's whole
effect is to restore the margin pruning removed -- that is what taking KL from 0.451 toward
~0.05 means -- and margin is precisely the property the parent has and the unhealed child
lacks. The mechanism is no longer a plausible story; it is the difference between two
measured rows.

It also means the collapse should **not** be treated as inherited from the 27B, and `pp=1.5`
should not be planned for as the shipped answer. The parent does not need it.

### Measured decode rates, and what they cost Stage 7

```
parent Q8_0,   -ngl 30 (partial offload, CPU-bound)     3.2 tok/s
parent IQ3_XXS, -ngl 64 (fully resident)               41.1 tok/s
```

**The Q8_0 repetition baseline is ~36 h, not ~14 h.** 200 completions x 2048 tokens =
409,600 tokens at 3.2 tok/s = 35.6 h single-slot. Partial offload is CPU-bound, so extra
llama-server slots will not scale it the way they do for a resident model -- assume the
parallel speedup is small until measured. Schedule it accordingly, after healing, when the
GPU is free.

### The ship gate stays at `presence_penalty=0.0`

`THINKING` is `presence_penalty=0.0, repeat_penalty=1.0`, and that is how the model actually
runs. The gate is measured there. Do not quietly move the gate to a penalised preset because
the numbers look better: that would be measuring a configuration the deployment does not use.

`presence_penalty=1.5` is the **documented fallback**, and it **works but is not free**. A
presence penalty suppresses tokens because they have already appeared, with no notion of
whether repeating them is correct. In reasoning traces -- which restate the problem, re-derive
intermediate results, and name the same variables repeatedly -- that is exactly the behaviour
being penalised. It buys termination at the cost of reasoning quality, and the cost is not
measured anywhere in this project.

**The target is a model that does not need it, like bf16.** Treat `pp=1.5` as an operational
escape hatch for a shipped build that still circles, not as a fix and not as a gate setting.

### The gate is tight at IQ3_M, and that is a width problem, not a healing problem

The comparison is not symmetric, and it is easy to read it as though it were:

* the **baseline** (27B iq3_xxs) carries **quantisation error only**
* the **child at IQ3_M** carries **pruning residual + its own quantisation error**

IQ3_M's advantage over IQ3_XXS is only ~0.04-0.06 nats. So for the child to clear the
baseline at IQ3_M, the *healing residual* has to land under roughly **0.05 nats** -- near
complete recovery of the pruning damage, with the width contributing almost nothing.

**Expect IQ4_XS to pass before IQ3_M does.** IQ3_M is the demanding width in
`SHIP_BIT_WIDTHS`, not the representative one. A run that passes at IQ4_XS and Q4_K_M and
fails at IQ3_M is the *expected* intermediate state, not a failure of the approach. Do not
retune the recipe on the strength of an IQ3_M miss alone.

### The mixer-protected mix is larger and worse. The FFN is not the tolerant part

Measured on the **unhealed** child, against `reference.kld`:

| | size | bpw | kl_mean | top-p agree |
| --- | --- | --- | --- | --- |
| child IQ3_M | 10.495 GB | 3.7656 | **0.451331** | 77.98% |
| child mixer-q5 | 10.951 GB | 3.9293 | 0.467476 | 77.51% |

More bits, higher KL, lower agreement. That comparison does not depend on matching bases --
the mix is a strictly larger file and it lost -- so it stands on its own.

The mix is **not** "IQ3_M with protected mixers". Its base is `iq3_xxs`, so relative to IQ3_M
it trades the FFN *down* (156 tensors, ~62% of parameters) to buy the mixers *up* (244
tensors to q5_K). The FFN loss outweighed the mixer gain.

`stage0_recipes` described the FFN as "62% of parameters and the most tolerant of them".
**That claim is now contradicted by measurement** on this architecture. Whatever the FFN is
doing in a gated-DeltaNet hybrid, it does not absorb quantisation error the way the comment
assumed.

**Missing control:** child `iq3_xxs`, which would separate "mixer protection helps but not
enough" from "mixer protection buys nothing". Not built. Do not carry the mix into Stage 7
without it -- the three outcomes lead to different decisions, and only one of them keeps the
recipe.

### No hosted endpoint: what is substituted, and what that costs

Stage 2 step 3 is **no longer endpoint-blocked**. Everything that needed OpenRouter has a
local stand-in, and each one is recorded as a proxy rather than as the thing it replaces.

* **bf16 controls -> parent Q8_0.** It is already the KL reference, its own KL to bf16 is
  under 0.001, and bf16 is 56 GB against 32 GB of RAM. Partial offload at `-ngl 30`.
* **bf16 repetition baseline -> the same Q8_0, locally.** ~14 h for 200 completions under
  partial offload. One-time, and only Stage 7 needs it, so it runs after healing when the GPU
  is free. **The manifest must record `Q8_0-as-proxy`**, not bf16: a baseline whose
  provenance is wrong is worse than a missing one, because it will be quoted.
* **Phase A traces -> the IQ3_M parent via llama-server**, 4 slots, pp=0.0, thinking preset,
  ~137 tok/s aggregate. 14M raw tokens targeted to net ~10M after filtering. **Record that
  traces come from IQ3_M, not bf16.** The teacher is itself quantised, so the student's
  ceiling is the quantised parent's behaviour, not the bf16 parent's -- that is a real cost
  and it must not be discovered later from a puzzling KL.
* **Phase B stays blocked** on the endpoint, which costs nothing: it runs after Stage 7.

Trace filter, applied at generation: drop when `rep8` exceeds the parent's own baseline, drop
on cap-hit without a stop token, drop under 1,500 tokens. **Log the yield** -- the filter
rate is the honest measure of how much the quantised teacher is costing.

### Stage 3 was one-shot, so cut interaction was never measured

Ablation KL was scored one layer at a time against the full model. The cut list removes
twelve layers *together*, and **eight of the twelve fall in periods 1-4**. Nothing in Stage 3
measured what happens when neighbours in the same period go at once.

If the floor comes in worse than the summed single-layer ablation KL predicts, this is the
first place to look, and it is the argument for running the **greedy-rescoring mode on the
18B rung** rather than one-shot again: greedy scores each candidate on top of the cuts
already taken, which is the only way interaction shows up.

### Stage 6: decide at the first checkpoint, not at the end

Stage 6 checkpoints every 10M tokens and runs both metrics at each. At the **first**
checkpoint (10M tokens), measure KL against `reference.kld` and branch:

* **below ~0.25 and still falling** -> continue, the budget is working
* **flattening above ~0.30** -> **stop and report**. The token budget is insufficient;
  whether to extend it is the operator's call, not the run's.

Do not spend the full 35M to discover this. The whole point of checkpointing at 10M is that
the answer is legible there, and a flat curve at 10M does not become a falling one at 35M.

### The KL reference must be held out from healing, and nothing else must be

Established here, because it is easy to over- or under-apply:

- **`data/calib.jsonl` overlaps `data/heal_corpus.jsonl`, and that is fine.** Both are built
  by `scripts/build_corpora.py` streaming the same proof-pile-2 arXiv shards from index 0, so
  the calibration set's arXiv content is a subset of the healing corpus *by construction* (6
  rows are byte-identical). It does not matter: Stage 3 measures parent behaviour under
  ablation and the imatrix measures weight importance. Neither is a generalisation claim.
  **Stage 3's result stands.**
- **The KL reference is different.** Every healing checkpoint is scored against it, so any
  shared document means the ship gate is partly measuring memorisation.

`data/kl_reference.jsonl` is therefore a third corpus:

- same composition as healing (~65% proof-pile-2 / ~35% fineweb-edu) so the KL is measured on
  the deployment distribution;
- drawn from shards healing never read, via an explicit `HELDOUT_SHARD_OFFSET` recorded in the
  manifest -- "which shards" is the only durable statement of what was excluded;
- ~300K tokens, enough for a stable per-token KL;
- **`assert_disjoint` fails the build** if the document-hash intersection with the healing
  corpus is non-empty. A reference that is 99% held out is not held out;
- sha256 recorded. **Once the reference `.kld` is built from it, the file must never change** --
  every child measurement is relative to those exact bytes.

### The reference pass, and the parameter that must never drift

```
model    runs/marlowe-22b/gguf/parent-q8_0.gguf   (28.6 GB, partial offload, slow, correct)
corpus   data/kl_reference.jsonl
context  -c 8192
```

Q8_0 with partial offload is the right reference rather than a smaller quant that fits: as
close to lossless as the machine allows, and hours once is the correct trade.

**`-c 8192` is load-bearing.** Every later `--kl-divergence` run must use the same context or
the chunking differs and the numbers are not comparable. It is recorded in
`runs/marlowe-22b/manifests/kl_reference.json` next to the corpus hash for exactly that
reason.

There is currently **no `.kld` on disk** -- it is a Stage 2 artefact and Stage 2 has never
run. It does not need the OpenRouter endpoint, which gates only the *repetition* baseline.

## 2. Decisions, and why

These do not survive in code. They are the expensive part of this document.

### The fp16 ladder, and why NF4-head candidates are gated

`heal.MEMORY_CANDIDATES`, best-quality-first:

**Rewritten. It used to spend sequence length; it now spends adapter capacity** -- seq_len
moved the measured peak by 0.04 GB across a 2x change, so the old ladder was tuning the one
parameter that did not matter (§3a).

```
rank32-chunk256   full adapter capacity, nothing given up
rank16-chunk256   half the adapter, loss target intact
rank8-chunk256    last rung before the loss target is touched
rank16-seq768     shorter sequence: the operator's fallback, unmeasured
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

**The NF4 gate is now nearly moot on throughput alone.** Measured: 9.8 tok/s for
`nf4-head + AdamW fp32` against 44.3 for the fp16 equivalent at the same sequence length —
**4.5x slower**. They lose on schedule before the quality argument is even reached. If the
fp16 ladder ever exhausts, the answer is almost certainly a smaller rank or a shorter
sequence, not a quantised head.

**Sequence length is a schedule decision, not only a memory one.** 1024 measured 73.5 tok/s
against 2048's 44.3 — 66% faster. Neither length is long enough to train DeltaNet state
eviction, so they are not meaningfully different for long-range exercise. **2048 winning the
memory search does not automatically make it the choice**; prefer 1024 unless there is a
specific reason not to.

Two savings are near-free and are in *every* candidate: `embed_tokens` on CPU (not a Linear,
so NF4 can't touch it; 2.5 GB) and a chunked loss (`2048 × 248320` is 1.02 GB in bf16 before
any softmax intermediate).

**"A lookup on CPU is cheap" was true of the lookup and false of accelerate's implementation
of it, and the difference cost this project days.** Naming `embed_tokens` in `device_map`
attaches an `AlignDevicesHook`, whose `pre_forward` copies the **entire 2.54 GB table onto
the device before every forward** and frees it after. The memory is therefore paid anyway —
as a transient on top of an already-full card, plus the fragmentation of allocating and
releasing 2.54 GB every step. The saving was zero and the cost was real.

Measured at seq 768, rank 16: after `empty_cache` the card had **2.49 GB free and the hook
needed 2.54 GB**. Fifty megabytes. That is the "~96 MiB short" the ladder chased, and *every
rung died on that same line* — before anything the ladder varies had been allocated, which is
why four rungs "OOM'd" in fifteen seconds and produced four identical non-measurements.

`CpuGatherEmbedding` replaces the module: `index_select` on the host weight, and only the
`[batch, seq, hidden]` result crosses the bus — 7.9 MB at seq 768 against 2540 MB, ~320×.
Forward-start free went 0.00 → 2.49 GB and every rung became measurable.

Three traps in implementing it, all of which reinstate the bug silently:

* **`register_buffer` is wrong.** Buffers are walked by `Module.to()`, so peft's device
  placement moves the table straight back onto the card — `allocated` jumped 12.78 → 15.33 GB,
  exactly the table. It must be a plain attribute: `nn.Module.__setattr__` intercepts
  Parameters and Modules, and a bare Tensor lands in `__dict__` where no device walk reaches it.
* **`remove_hook_from_module` materialises the offloaded weight onto the execution device on
  its way out**, and the old module keeps that GPU copy alive. Drop it by hand; GC is not
  fast enough when the next allocation is 50 MB from failing.
* **Do not pin the weight.** `pin_memory()` on 2.54 GB raised a raw `CUDA error: out of
  memory` from `cudaHostAlloc` on this 32 GB host and left the context unusable — the next
  `randint` failed. At 7.9 MB per forward, pinning buys microseconds against a 2.5 GB failure.

`requires_grad_(True)` on the gathered output is not optional: this path bypasses accelerate's
hook and therefore whatever `enable_input_require_grads()` attached to it, and gradient
checkpointing needs an input that requires grad or the LoRA layers below receive nothing.

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

**Fallback if the 22B cannot train on this card: cut 18B directly from the 27B.** This is a
decision the operator has already reserved, not a parameter to tune. A direct 23-cut removes
36% of depth in one step, outside the band where healing reliably recovers, so the result is
a *worse* 18B than the two-rung ladder would produce. But the 18B's NF4 body is ~7.77 GB
against the 22B's 9.88 GB, so it fits on this card where the 22B is marginal — and it trains
today. The trade is **a worse 18B, not no 18B**, and the 18B is the headline artifact.

Taking it means `configs/marlowe-18b.yaml` must drop `requires_healed_parent: true` and
`stage8-ladder`'s refusal must be overridden deliberately — both exist to stop this happening
by accident, and neither should be relaxed except as this decision.

### Stage 9 — MTP head retrain, per child (deferred, after Stage 7 ships)

Surgery drops the MTP draft head (`drop_mtp: true`, −0.38 B) because it is invalid once
layers are removed: it predicts from hidden states produced by a 64-layer stack, and after a
12- or 23-layer cut that relationship no longer holds. Every GGUF this pipeline builds is
converted with `--no-mtp` (see §3, the `block_count` mismatch).

It can be given back, and the cheap way is a separate stage rather than a healing-budget
line item. **Retrain the head standalone against the healed base, with the base frozen:**
forward through the base for hidden states, backward only through the head.

Why this is easy where Stage 6 is hard: the resident set is base weights + head + the head's
optimizer state. No LoRA, no adapter gradients, and **no 52-layer backward** — and the
52-layer backward transient is the entire reason Stage 6 is tight (13.1 GB steady against an
18.2 GB peak, §3a). It fits comfortably on the same card, converges fast once the base is
stable, and costs nothing from the healing budget.

Do the converter fix with the stage, not before: write `block_count` flags rather than
`num_hidden_layers`, and **confirm the MTP slot's flag against the loader** rather than
inferring it from `qwen35.cpp`'s interval fallback — that line guards on `i < n_layer()` and
it is not obvious whether `n_layer()` there is 64 or 65. Guessing the recurrent flag for a
block is precisely how this codebase produces a model that loads, runs, and is wrong.

`convert_to_gguf(..., no_mtp=False)` is preserved and tested for exactly this.

### Deployment width is not gate width

These are different questions and conflating them misreads the whole ship gate.

| model | deployment target | measured size | why |
|---|---|---|---|
| Marlowe-22B | **IQ3_M** (~3.66 bpw) | ~10.2 GB projected | the rung that runs on a 16 GB card |
| Marlowe-18B | **IQ4_XS** (~4.25 bpw) | ~9.6 GB projected | the headline: more bits at fewer layers |

**Q4_K_M is a gate width, not a deployment target for the 22B.** Measured on the sizing-22B:
Q4_K_M is **13.75 GB**, and llama.cpp needs roughly another 6 GB for KV cache, compute buffer
and CUDA context, so the 22B at Q4_K_M does not fit a 16 GB card and was never meant to. It is
in the gate because under-healed weights degrade *unevenly* across schemes, so a width nobody
ships still tells you whether the healing generalised.

That is also the quantified case for the 18B being the headline: at 41 layers the same recipe
lands near 11 GB at Q4_K_M and ~9.6 GB at IQ4_XS, which is the first configuration in the
ladder that is comfortable on the target card with real bits.

### Ship gate: three bit-widths, all required

`SHIP_BIT_WIDTHS = ("q4_K_M", "iq4_xs", "iq3_m")`. Under-healed weights carry larger
activation outliers and degrade unevenly across quantisation schemes, so passing at one
width proves nothing about the others. A width that was **never built counts as a failure**,
not an absence — otherwise a missing measurement reads as a pass.

**The gate is KL-first.** The objective is maximum knowledge transfer from the 27B into the
22B, and KL to the reference is the direct measure of it. Repetition is a sanity check, not a
second fidelity criterion.

* **Primary — KL to `reference.kld` at each width, strictly below the iq3_xxs parent
  baseline** (0.144219). This is the gate. It is a Phase A criterion only; see
  `docs/TRANSFER_PLAN.md` and `report.GATE_PHASE`.
* **Sanity — token-level `rep8` / `rep32` at 2K tokens, `presence_penalty=0.0`.** A
  degenerate model fails; a healthy one passes. It is a floor, not a gradient: do not read a
  small rep improvement as better transfer, and do not trade KL for it.

Two things this deliberately is **not**:

* Not a fix for the 27B's own circling. The parent circles; that is not this project's
  problem to solve, and a child that circles exactly as much as its parent has lost nothing.
* Not a semantic repetition measure. **The semantic harness is dropped from the plan.**
  Token-level n-gram repetition at `pp=0.0` is the whole of it. Building a semantic
  circling detector would be measuring a property the objective does not target.

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
| torch accounting instead of device memory | `probe_training`, then `estimate_peak_gb` | understated the footprint by ~1 GB; would have selected a config that OOMs at hour 30 **with its own record certifying it fit** |
| the *fix* for that then over-reported | `probe_training` | `total - free` includes the allocator's reserved pool, which under `expandable_segments` grows opportunistically and is never returned. Every candidate measured **17.17 GB, headroom 0.00** — identical, saturated, not a measurement |
| the *third* version double-counted the model | `probe_training` + base caching, same commit | measuring the baseline per-probe was correct until the search began caching the base across candidates. With a model already resident, `total - free` includes it and `max_memory_reserved()` includes it too — summed, the model is counted twice. Reported **41.16 GB on a 17.17 GB card**, headroom −23.99. Two individually-correct changes, made together. Fixed by measuring the CUDA context **once, globally**, while torch holds nothing |
| `lora_rank` not persisted in the memory plan | `save_memory_plan` | a candidate override the plan didn't carry. Had `fp16-head-1024-rank16` been selected, Stage 6 would have restored `seq_len` and head precision correctly and **silently reverted the rank that made it fit** — OOMing on a plan whose own record said it had been measured as fitting. The cleanest instance of the pattern: self-certifying wrong answer. Field list is now derived from the dataclass |
| quants list narrower than the gate | `configs/marlowe-18b.yaml` | gate would fail after a week on *missing data*, looking like a quality failure |
| surgery not idempotent | `write_checkpoint` | collided with its own prior output after a full streaming pass |
| `lm_head` resident at fp32, not bf16 | peft's `prepare_model_for_kbit_training` | it upcasts every non-quantised parameter. 5.09 GB instead of 2.54. The candidate was *named* `fp16-head` and `ProbeResult.lm_head_precision` reported `"fp16"` — derived from `cfg.quantize_lm_head`, never from the tensor |
| **parameter dtype is not stream dtype** | `assert_no_bf16_resident` | the same upcast made the norms fp32. That is **0.01 GB of parameters** — which the byte-level check waved through — but an fp32 norm emits fp32, the residual add promotes, and the stream ran fp32 for all 52 layers: 1.14 GB of saved checkpoint inputs instead of 0.55, plus a doubled backward recompute. The check measured the right property at the wrong layer, and no threshold would have caught it |
| memory metric saturated across candidates | `search_memory_plan` base caching | the allocator's high-water reservation is not returned between candidates, so every candidate after the first inherits the first one's peak. Reported 23.99 / 23.97 / 23.95 GB across a 2× change in sequence length — identical, saturated, not a measurement |
| `cuda_context_bytes()` returned 0 | `probe_training` via the search | it is measured lazily, and in the search path the first call happens after the cached base is resident. Its "subtract what torch reserved back out" fallback then collapses to zero, silently dropping the ~1.4 GB context term and overstating headroom by the same amount |

| `imatrix` parameter never passed | `quantize()` + every call site | the argument existed from the start and nothing supplied one. Stage 0 died on its first recipe -- after a 54 GB conversion, seven minutes in -- with "this quantization requires an importance matrix!" |
| an interrupted imatrix is indistinguishable from a finished one | `llama-imatrix` output | it writes a complete, valid GGUF at every periodic save. A kernel panic stopped one at 120 of ~420 chunks and the file was structurally perfect: all 64 blocks, 992 tensors. Only `imatrix.chunk_count` says otherwise, and nothing read it |
| Stage 0 re-measured variants it had already measured | `stage0_bitwidth` loop | quantisation was skipped when the GGUF existed; the ~50-minute harness was not. Stopping after three recipes and resuming would have re-generated 2.5 h before reaching the fourth |
| `llm_int8_enable_fp32_cpu_offload` missing | `score.load_4bit` | present in `heal.load_student` since it was written. Stage 3's load was refused outright: "Some modules are dispatched on the CPU or the disk" |
| `device_map="auto"` | `score.load_4bit` | `heal.load_student` documents that accelerate spills Linear4bit modules and dies reading `quant_state.offset.item()` on a meta tensor. Here it was a **segmentation fault** during load -- log ending mid-sentence, no traceback |
| a probe that could never succeed, defaulting to the wrong answer | `score.detect_return_style` | it called the decoder layer without `position_embeddings`, a required positional argument, so it **always** raised -- and the handler read that as "returns a tuple". This architecture returns a bare tensor, so every ablated layer handed back `(hidden,)` and the next real layer called `input_layernorm` on a tuple. Stage 3 ran the entire reference pass and died on its first candidate, eight minutes in |
| `estimated_hours` believed | every `@register` | Stage 0 declares 4.0 h and takes ~8.0; Stage 3 declares 8.0 h and takes ~3.5. Neither had ever been measured (§5) |

**The pattern: every one of these loads, runs, and produces plausible output.** None crashes.
None OOMs. The model generates fluent text, the search reports a number, the config parses.
That is the failure shape this codebase produces, and it is what to look for.

The memory metric is worth studying on its own: it was wrong **three times**, in three
different ways, and each fix was a correct response to the previous failure.

1. torch accounting — under-reported by ~1 GB (missing context and fragmentation)
2. `total - free` — saturated at exactly `total`, so every candidate looked identical
3. context + `max_memory_reserved()`, measured per probe — correct until the base model
   started being cached across candidates, at which point it double-counted the model and
   reported 41 GB on a 17 GB card

**Each failure was loud only by luck.** 0.00 headroom on every candidate and −23.99 GB are
both unmistakable; a subtler error in either direction would have been believed. Two lessons:
**check a corrected measurement against a case whose answer you already know** (a 17 GB card
cannot hold 41 GB), and **a change that is correct alone can break another change made in the
same commit** — the base cache and the per-probe baseline were each right in isolation.

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
- **Adding a field is not the same as plumbing it.** `lora_rank` became a candidate override
  and was not added to persistence; two commits later it would have caused an OOM. Anything
  a search can vary must be carried by whatever records the search's decision.

---

## 3a. Fitting the 22B on a 17.17 GB card

The 22B did not fit at any sequence length. It was ~8 GB over and paging silently, which is
why every throughput number before this was unstable (2048 measured 10.4, 44.3 and 74.5 tok/s
in three sessions). Sequence length was never the lever: peak moved **0.04 GB across a 2x
change** in seq_len, because almost none of the footprint is activations. What follows is
what actually moved it, measured at seq 1024 on the sizing-22B.

| change | effect | note |
|---|---|---|
| `lm_head` back to bf16 after peft's upcast | −2.54 GB | peft upcasts every non-quantised param to fp32 |
| skip the fp32 upcast entirely (`prepare_for_kbit_training`) | −2.55 GB, **+37% tok/s** | 0.01 GB of fp32 norms was promoting the whole residual stream |
| bf16 LoRA adapters (`autocast_adapter_dtype=False`) | fp32 residue 0.30 → 0.01 GB | collapsed the rank32/rank16 gap to 0.14 GB |
| post-load `empty_cache()` + peak reset | fragmentation 2.30 → 0.19 GB | the load transient was being held for the whole run |
| `max_split_size_mb:512,garbage_collection_threshold:0.8` | trapped 1.42 GiB → 327 MiB | supported on Windows; `expandable_segments` is **not** |
| allocator cap at `(total − context)/total` | paging 2.96 → 0.21 GB | turns silent 10x-slow paging into a real OOM |
| sub-layer checkpointing (mixer and FFN separately) | **−0.4 GB** | predicted 0.75; it is 0.4. Plan against the measured number |
| `PagedAdamW8bit` | ~0.35 GB | optimizer state is resident at the peak even though the step comes later |

Two things that did **not** work, and why, so they are not retried:

- **Smaller `loss_chunk`.** Byte-identical memory at 128 and 256, for −21% throughput. It
  bounds a transient the peak does not fall on.
- **Recomputing the loss chunks in backward.** Removes 1.02 GB of retained `log_softmax`
  output and buys nothing: loss backward completes *before* the decoder backward that sets
  the peak, so those tensors are already freed. Cost +5.1 GB reserved and −23% tok/s.

**The peak is one decoder layer's backward.** Steady state is 13.1 GB against an 18.2 GB
peak, with no phase boundary above 15.2 GB. Any lever that does not shrink that transient
does not shrink the peak.

**~1 GB of VRAM is held by the desktop.** Sixteen processes (browsers, Spotify, VS Code,
Steam) render on the discrete GPU, costing 1.38-1.52 GB and *drifting* by ~0.15 GB between
runs. That is larger than the remaining deficit, and it is invisible to torch — one more
reason the driver's Shared Usage counter is the gate.

### The fit rule changed: torch headroom said 0.11 GB on a config paging 7.51 GB

The old rule was `FIT_MARGIN_GB = 1.0` — a configuration fit if torch reported at least a
gigabyte of device headroom. Two runs of the same four rungs, one with the paging counter
broken and one with it working, show why that had to go:

```
                  BLIND (torch headroom)        WITH THE COUNTER
rank32-seq1024    0.11 GB, 201 tok/s            PAGES  7.51 GB, 216 tok/s
rank32-seq768     0.43 GB, 167 tok/s            FITS   0.09 GB, 183 tok/s
rank16-seq1024    0.12 GB, 198 tok/s            PAGES  0.66 GB, 210 tok/s
rank16-seq768     0.64 GB, 169 tok/s            FITS   0.22 GB, 108 tok/s
```

`rank32-seq1024` was the **fastest** rung measured and it was spilling **7.51 GB** to host.
Torch called that 0.11 GB of headroom, which under the old rule read as "tight" rather than
"disqualified". Headroom and residency are not the same measurement and here they disagreed
by three orders of magnitude.

**The rule now:**

* **Primary, and the only gate: driver Shared Usage over the idle floor ≤ 250 MB.**
* **Secondary, advisory, logged and never gated: torch device headroom ≥ 0.3 GB.**
* **Soak: 500 steps on the selected rung — paged bytes at the floor throughout, and trapped
  fragmentation growth < 50 MB per 1000 steps after warm-up.**
* **A restart supervisor built and exercised before Stage 6 begins.**

### An unvalidated counter is no verdict, not a pass

`ProbeResult.fits()` now raises `SamplerNotValidated` when the counter returned nothing, or
when nothing has proved *in this session* that it can see a spill. The old behaviour — fall
back to torch headroom when the counter is unavailable — is exactly the measure that missed
7.51 GB, so the fallback was worse than the failure.

This matters because the counter has been believed working, twice, while returning nothing.
`typeperf` buffers into its `-o` file: measured, the file is **0 bytes while typeperf is
still running** and 0 bytes after `terminate()`. Reading its stdout on a drainer thread is
what fixed it.

`validate_sampler()` supplies the proof in seconds and without a model: it over-commits VRAM
by 2 GB, which on WDDM the driver backs with host memory rather than refusing, and confirms
the counter reports the spill. Measured 1.34–1.42 GB detected. The heavyweight
known-positive remains `MARLOWE_CPU_GATHER_EMBED=0`, which restages the 2.54 GB embedding
table every forward: 11.585 GB across 103 samples.

### The allocator cap is opt-out now, because opt-in meant off

`cap_process_memory` returned `None` unless `MARLOWE_CUDA_MEMORY_FRACTION` was set, and it
was never set. **Every probe in this project ran with the cap inactive.** The safety net
existed, was documented, was tested, and was not deployed — the same defect class as a guard
that fails open, applied to the thing meant to catch the guard failing.

Cap and counter are belt and braces and they see different things. The cap makes torch
over-commitment *raise* instead of silently paging. The counter catches host-backed memory
the driver hands out *below* torch's own ceiling, which the cap cannot see at all.

## 3b. Schedule — the brief's estimate was 3-4x optimistic

The brief assumed ~200 tok/s, which came from an Unsloth-based estimate. This stack is plain
peft plus bitsandbytes. Measured on the 22.3B student (RTX 4080 Super, fp16 head, AdamW8bit,
micro_batch 1, gradient checkpointing, loss chunked at 256):

```
seq 2048, rank 32, AdamW8bit                44.3 tok/s
seq 1024, rank 32, AdamW8bit                73.5 tok/s
seq 2048, rank 32, NF4 head, AdamW fp32      9.8 tok/s
```

`marlowe schedule` regenerates this from whatever rate is current:

```
healing wall-clock at 73.5 tok/s (93 tok/s estimated for the 18B, scaled by depth)

  tokens         22B         18B    both rungs
     20M       3.1 d       2.5 d         5.6 d
     35M       5.5 d       4.3 d         9.9 d
     50M       7.9 d       6.2 d        14.1 d
    100M      15.7 d      12.4 d        28.2 d
```

**Decided: 35M tokens on both rungs** — 9.9 days healing plus ~4.5 days caching, ~15 days
end to end. The reasoning was explicitly *finish both rungs* rather than spend eight days on
the intermediate; 18B is the headline and a 22B healed to exhaustion is worth less than both
rungs completed. The configs previously read 50M/100M, which was 20.3 days of healing alone
and was sized against the 3-day-per-rung assumption.

`teacher.tokens` now matches `heal.tokens` on both rungs. They were left at 50M/100M when the
heal budgets came down, which would have spent Stage 5 caching tokens training never reads —
roughly a day per surplus 10M. `RunConfig.validate_token_budgets()` now refuses a cache
smaller than the heal budget (silent second epoch) and warns on a surplus above 10%.

Stage 5's teacher cache is a further unmeasured cost. The brief budgeted 6 hours from an
assumed 2500 tok/s — the same source as the 200 tok/s. Scaling the measured training rate by
a forward-only factor of ~3 and by depth (64 vs 52 layers) suggests **~3 days per rung at
50M tokens**, not 6 hours. `teacher_cache_days()` computes it; **it is an estimate and Stage 5
should replace it with its own logged rate.**

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

5. **Unsloth evaluation — not yet run.** Worth ~an hour against a 15-day schedule: 73 → 150
   tok/s would roughly halve it. `scripts/eval_unsloth.py` is written and its comparison
   logic is verified against synthetic inputs.

   **Isolated venv only — never the working environment.** The current stack is validated end
   to end (`tests/test_heal_stack.py`, 12 tests) and Unsloth pins transformers/torch
   aggressively. The benchmark claims are measured on Llama-family dense models.

   ```bash
   python -m venv .venv-unsloth
   .venv-unsloth/Scripts/pip install unsloth
   .venv-unsloth/Scripts/python scripts/eval_unsloth.py --model <student> --out unsloth.json
   python scripts/eval_unsloth.py --model <student> --stack peft --out peft.json
   python scripts/eval_unsloth.py --compare unsloth.json peft.json
   ```

   Three criteria, same data and seed, few hundred steps:

   - **tok/s** — is ~2x real on *this* architecture?
   - **peak VRAM, device-level** — savings might reopen seq 1536 or rank 32; a fallback might
     cost more. Either changes sizing, and the entire memory ladder was measured on plain
     peft. **If Unsloth is adopted, re-run `marlowe fitcheck` before Stage 5.**
   - **loss curve** — *the deciding one*. Divergence beyond 0.02 means its kernels compute
     something different on `qwen3_5`, and it is unusable at any speed.

   Unsloth patches model internals and must recognise the architecture to do so correctly.
   Gated DeltaNet, the fused `q_proj` output gate and the float32 recurrent state are all new
   and unusual. **Silent miscompute is this project's signature failure (§3), and a
   throughput win on wrong gradients is worse than no win.** A speedup below 1.25x is treated
   as a rejection in its own right: that is the signature of an unrecognised architecture
   quietly falling back to a generic path. The harness also captures any Unsloth warning
   mentioning an unsupported architecture.

---

## 5. Runtimes: measure them, the declared ones are guesses

Every stage carries `estimated_hours=` in its `@register`. **Both numbers that have now been
tested were wrong, in opposite directions**, and nothing in the codebase had ever checked
them. Treat the declaration as a placeholder until a stage has run once.

| stage | declared | measured | how |
|---|---|---|---|
| imatrix (parent) | -- | **86 min** | 6 segments, 415 chunks, full 1.7M-token corpus |
| `stage0-bitwidth` | 4.0 h | **~8.0 h** | 50 min harness + 11 min quantise, per variant, x8 |
| `stage3-score` | 8.0 h | **~3.5 h** | 301 s per candidate (75 s per forward) x 42 |

**Stage 0.** The harness is the cost, not the quantisation: 200 completions x 2048 tokens per
variant at ~137 tok/s aggregate across 4 llama-server slots. The 4.0 h declaration appears to
describe one variant, not the sweep of eight. Note `eval/kl.py` assumes `GPU_DECODE_TOK_S =
40`, which is single-stream; `repetition.parallel: 4` buys ~3.4x, so the code's own estimator
says 23 h and is also wrong.

**Stage 3.** Work is exactly `(42 eligible candidates + 1 reference) x n_seqs` forwards over
`score.seq_len` tokens -- 172 forwards at the shipped config. That count is solid, so the only
unknown is seconds per forward, and everything follows by multiplication. The reference pass
runs slower per forward (134 s) than the candidates (75 s) because of first-pass warm-up;
project from candidate intervals, not from the reference.

To measure a stage in flight rather than waiting for it: Stage 0 exposes decode progress via
llama-server's `/slots` (`next_token[0].n_decoded`, which resets per completion -- count the
resets), and Stage 3 logs `candidate scored` per candidate.

## 5a. A quantised variant above ~12 GB falls off a throughput cliff

Stage 0's harness runs at **137 tok/s aggregate** (4 llama-server slots) for variants up to
~12.6 GB, and at **31 tok/s** for `q4_k_s` at 15.59 GB -- a 4.4x collapse, turning a 50-minute
variant into 3.7 hours.

The cause is not the model size alone: llama-server is started with `-c ctx * parallel`, so
4 slots x 8192 is 32768 tokens of KV cache on top of the weights. At 15.59 GB of weights the
two no longer fit in 16 GB, layers spill to CPU, and decode drops by the ratio you would
expect from running part of a model on system RAM.

Consequences worth planning around:

- **Budget Stage 0 by variant size, not variant count.** The eight recipes are not eight
  equal units of work.
- `SHIP_BIT_WIDTHS` includes `q4_K_M`. On the 22B child that measured **13.75 GB**, which is
  in the same danger zone -- Stage 7's gate will be slow at that width for the same reason,
  and on the parent it would be slower still.
- If a wider variant is needed cheaply, lower `repetition.parallel` (fewer slots means less
  KV cache and possibly a fit) before touching `n_completions`, which costs statistics.

## 5b. Measuring a run in flight

Neither harness prints throughput, and both write results only when a unit of work finishes,
so a healthy run and a hung one look identical from the log. These are the two live signals:

- **Stage 0**: `curl http://127.0.0.1:8080/slots` -- each slot's `next_token[0].n_decoded`
  counts tokens for the *current* completion and resets when it finishes, so sample
  repeatedly and treat a decrease as `(2048 - prev) + cur`. Sampling naively across a reset
  reports ~0.2 tok/s and looks like a hang; it is not.
- **Stage 3**: `candidate scored` per candidate. Time the interval between two, multiply by
  42. Do not project from the reference pass -- it runs at 134 s/forward against the
  candidates' 75 s because of first-pass warm-up.

Corroborating signals that separate "expensive" from "hung": GPU power draw *fluctuating*
(compute- and memory-bound phases alternating) rather than pinned, and the process's CPU
time rising.

## 5c. Claims this document made that turned out to be wrong

### The paging sampler was correct in every reading. The interpretation was wrong

Not a broken instrument. Every number it returned was the true value of
`\GPU Adapter Memory\Shared Usage` at the moment it was sampled. What was wrong was the
assumption wrapped around it: that the sampling window covered a **steady state**, when it
in fact covered a **transient**.

After loading the 22B, driver host backing sits at 9.07 GB and falls monotonically at about
0.6 GB/s to 0.09 GB over roughly fifteen seconds, then stays flat for the rest of the run.
Measured at one-second resolution across a 60-step probe:

```
pre-training   9.07 9.07 9.07 9.07 9.07 9.07 9.07 8.98
training 1-10  8.28 7.66 7.02 6.44 5.78 5.18 4.54 3.89 3.24 2.58
then           0.09 ... 0.09      (~250 samples, ~4 minutes, flat, 191 tok/s)
```

`paged = max(shared) - min(shared)` is a sound definition -- the counter is a genuine
instantaneous gauge, confirmed by allocating 18.79 GB against 15.79 GB free and watching it
rise to 2.43 GB and fall back to 0.15 GB on release. But `max - min` over a window that
*contains a decay* returns the height of the decay, not the height of any spill during
training.

**One artefact produced every confusing paging number in this project:**

| reading | why |
| --- | --- |
| rung 1 `rank32-seq1024` 7.51 GB "PAGES" | probed first, immediately after the base load |
| rungs 2-4 0.09-0.66 GB "FITS" | probed later, after the drain had finished |
| standalone soaks 7.5-11.5 GB | each started a fresh process, so each caught its own drain |
| 100-step and 500-step soaks **identical to 11.547 GB** | the tell: a fixed event, not accumulation |

**`rank32-seq1024`'s PAGES verdict was this drain, not a spill.** The rung was never paging,
and neither was `rank32-seq768`, which reported 0.09 GB when probed second and 7.59 GB when
probed first -- the same rung, differing only in running order.

Two things nearly hid it. Byte-identical results at two run lengths is not a physical
measurement, and that is what forced the question. And an idle control -- three minutes with
no GPU work, showing 33 MB of desktop noise -- correctly cleared *other tenants* as the
cause, which made the number look confirmed when it had only been narrowed. Ruling out one
explanation is not establishing another.

The fix is `wait_for_drain()`, called before the measurement window opens, with the settling
time recorded on `ProbeResult.drain_seconds`. It is `None` when the model was loaded inside
the window, which marks that paging figure as not a training number rather than leaving it to
be read as one.

**A superseded intermediate is worth recording too.** `PagingReport.saturation()` was added
between these findings, on the theory that the platform provisions host backing once and
holds it. It returns the right verdict on the real series, but only because its warm-up
fraction happens to skip the decay -- right answer, wrong reason. It was deleted rather than
kept: two overlapping notions of the same thing, one of which works by coincidence, is worse
than one that works because the transient is waited out.

### "Pinning barely matters" was about bandwidth, and the failure was not about bandwidth

The judgement that pinning the 2.54 GB embedding table "barely matters at 7.9 MB per forward"
was correct *as a bandwidth claim* and was then used to dismiss a mechanism nobody had
measured. What actually happened is worse than immaterial: `pin_memory()` on 2.54 GB raised a
raw `CUDA error: out of memory` from `cudaHostAlloc` on this 32 GB host and left the context
unusable, so the next tiny `randint` failed. Not pinning is right here, for a reason that has
nothing to do with the reason given.

The related trap, when a small pinned staging buffer is eventually wanted: it must be sized
`seq x hidden`, not `vocab x hidden`. Pinning the table is what fails; pinning 7.9 MB does not.


Recorded because a handoff that only accumulates conclusions teaches the next reader to
trust it more than it deserves.

- **"Throughput is the fit signal; memory accounting is a diagnostic."** Falsified. A probe
  measured 74.5 tok/s while 6.8 GB over the card. Under WDDM the driver pages rather than
  refusing, so an over-committed run can look fast. The gate is the driver's Shared Usage
  counter (§0 item 3).
- **`score.n_seqs` should be raised to use the whole corpus.** Pushed repeatedly during one
  session, framed as "a one-line change now, or an 8-hour redo later". It would have made
  Stage 3 a **68-hour** job: `(42 + 1) x n_seqs` forwards. 4 is the compute budget and is
  correct. The framing was wrong in both directions -- it implied the change was cheap and
  that not making it was risky.
- **"Moving the desktop apps to the iGPU frees the 1.38 GB context."** It does not. That
  figure is this process's own WDDM reservation: with the card at 0 MiB and nvidia-smi
  reporting 254 MiB after CUDA init, `mem_get_info` still reports 1.42 GB used. Re-measuring
  after the apps moved gave a byte-identical deficit.
- **"Our IQ3_XXS is 12% larger than Unsloth's."** It is 2.3% larger (11.186 GB against
  10.935 GB). The "10 GB" was a rounded display figure, and the comparison tag was wrong --
  `marlowe-dusk:27b` is Q4_K_S with a vision tower, not IQ3_XXS.
- **"Our quant contains 64 vision tensors."** It contains zero. The detection pattern `v.`
  matched `attn_qkv.weight`.
- **`expandable_segments` reduces fragmentation here.** It is rejected outright on Windows
  (torch 2.5.1, `get_allocator_backend()` reports `native`). The 2.95 GB of fragmentation
  measured during training accrued with that flag set. What does work: `max_split_size_mb`
  and `garbage_collection_threshold`, which are supported (§3a).

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
