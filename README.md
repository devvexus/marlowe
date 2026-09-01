# Marlowe

Structure-aware depth pruning and knowledge distillation for hybrid linear/attention decoder
stacks. Built for `Qwen/Qwen3.8-27B`; the layout logic is architecture-agnostic.

Two shippable models, produced as a ladder:

| model | route | params | layers | ships at |
|---|---|---|---|---|
| **Marlowe-22B** | 27B, 12 cuts | 22.2968 B | 52 (36 linear + 16 attention) | IQ3_M, ~10.2 GB |
| **Marlowe-18B** | healed 22B, 11 more cuts | 18.0807 B | 41 (25 linear + 16 attention) | IQ4_XS, ~10.2 GB |

`marlowe plan` reproduces both figures from `config.json` with zero drift.

**Marlowe-18B is produced via the ladder, not a direct 27→18 cut.** A direct cut removes 36%
of depth in one step, outside the band where healing reliably recovers; two sequential ~19%
cuts from an already-healed parent stay inside it. `configs/marlowe-18b.yaml` sets
`requires_healed_parent: true` and `stage8-ladder` refuses to run without it.

**The AAII figures in the project brief are priors, not measurements.** They extrapolate from
depth-pruning literature on uniform transformers, which has never been validated on a hybrid
stack. This pipeline measures whether they hold. `report.py` deliberately does not compute an
AAII number — see [On AAII](#on-aaii).

---

## Quick start

```bash
pip install -e ".[dev,score,heal,plot]"
marlowe doctor                                    # what is installed, what will block what
marlowe plan   --config configs/marlowe-22b.yaml  # layout, budget, sizes, build order
marlowe run    stage1-smoke --config configs/marlowe-22b.yaml
```

Stages are individually resumable and skip when their manifest is current. A three-day
training run survives a crash.

---

## The five rules

These are correctness requirements. Violating any of them produces a model that loads, runs,
and is silently broken. Each is enforced in code and covered by tests.

### 1. Never remove an attention-class layer

Only 16 of 64 layers carry a KV cache and do exact token-to-token retrieval. DeltaNet's
fixed-size recurrent state cannot do precise long-range recall, so those 16 carry the 262K
context behaviour essentially alone.

This makes **contiguous block removal (Gromov et al., ShortGPT) the wrong algorithm** here: a
24-layer contiguous span takes six attention layers with it, short-context evals look fine,
and long-context recall is destroyed.

Enforced as a **whitelist**, not a blacklist. `arch.REMOVABLE_TYPES` contains exactly
`linear_attention`; every other type — `full_attention`, `sparse_attention`, anything this
module has never heard of — is attention-class and never removable, and an unrecognised type
fails closed with an explanation. Qwen3.8-Flash-Next (arch `qwen4_exp`) replaces Gated
Attention with Qwen Sparse Attention; under a blacklist those layers would have been silently
eligible.

### 2. Score on long sequences

Angular-distance metrics were derived for softmax-attention residual streams. A DeltaNet
layer's per-token output delta can be small while its contribution to state maintenance
across 100K tokens is large, so short calibration systematically nominates exactly the layers
you must not cut.

`score.py` uses ablation KL, calibrates at ≥8192 (32768 default, and it refuses below 8192),
and samples scored positions from the last three quarters of each sequence.

### 3. Renumber every `layer_idx`

Two caches are keyed by `layer_idx`: the KV cache for attention layers and the
conv/recurrent state cache for DeltaNet layers. Miss one and the model generates fluent,
confident nonsense with no error raised. On disk this is the tensor-name index
(`surgery.plan_surgery`); in a live module tree it is the attribute
(`surgery.renumber_live_modules`).

### 4. Null out `full_attention_interval`, then assert

After pruning the layout is not uniform and no interval describes it. Some runtimes
*regenerate* `layer_types` from that field instead of reading the explicit list, producing a
config that loads cleanly and builds the wrong stack. It is removed from both config levels,
and `verify_checkpoint` refuses a checkpoint where it survived.

Then, after every surgery and in CI, `assert_layer_types_match_tensors` proves each entry of
the rewritten `layer_types` matches the tensors actually present at that index. Three checks,
increasingly specific:

1. **Coverage** — every declared index has tensors; nothing sits past the declared end.
2. **Partition** — each declared type maps to exactly one tensor signature, and no two types
   share one. This is the generic check: it needs no knowledge of what the mixer submodules
   are called, so it works unchanged on a stack whose attention layers are QSA rather than
   Gated Attention. An off-by-one splits some type across two signatures and is caught here.
3. **Family** — where the mixer submodule name *is* recognised, a `linear_attention` layer
   must look linear and an attention-class layer must look like attention.

### 5. KL cannot detect the artefact being fixed

The motivating problem is **circling** — reasoning that loops and fails to terminate — seen at
IQ3_XXS (2.97 bpw) and absent at bf16, which is what says the cause is quantisation rather
than the model.

KL is measured teacher-forced on fixed text. Circling is an autoregressive failure that only
appears when the model samples its own continuations. **A checkpoint can have excellent KL and
still loop**, and under-healed pruned models loop too — different cause, identical symptom.
Every checkpoint is scored on both, and `ship_gate` fails closed if either is missing.

---

## Period structure is derived, never assumed

A period is a run of layers ending at an attention-class layer, read off `layer_types`.
Period count, per-period caps, and the protect-first/protect-last window all follow. No
constant `4` appears in `arch.py`.

| stack | layers | periods | budget |
|---|---|---|---|
| Qwen3.8-27B | 64 | 16 × len 4 | 28 |
| Flash-Next shape (3 GDN + 1 QSA) | 48 | 12 × len 4 | 20 |
| irregular | any | whatever it actually is | derived |

Constraints: at most `max_per_period` (2) removals per period, every period retains ≥1
removable layer, first and last period protected. On the 27B that gives a budget of 28, and
after 12 cuts the child still has budget 16 — so the 11-cut second rung fits.

---

## Layout

```
marlowe/
  arch.py        layer types, derived periods, param accounting, the rule-3.4 assertion
  score.py       Stage 3: ablation-KL scoring on a 4-bit model
  surgery.py     Stage 4/8: streaming safetensors layer removal
  teacher.py     Stage 5: top-K logprob cache
  heal.py        Stage 6: QLoRA distillation against the cache
  quantize.py    Stage 0/7: GGUF conversion, tensor-type mixes, the converter probe
  eval/kl.py     llama-perplexity --kl-divergence wrapper
  eval/repetition.py   generation-based loop detection
  eval/bench.py  GPQA Diamond, needle @128K
  report.py      unified metrics table, ship gate
  pipeline.py    stage registry, manifests, resumability
  stages.py      the wiring
  cli.py         doctor / plan / run / surgery / verify / probe / report
configs/         marlowe-22b.yaml, marlowe-18b.yaml
docs/original/   the two scripts this started from, and what changed
```

## Parameter accounting

Computed from config fields, not transcribed, so it stays correct for a pruned child. It
reconstructs the published figures exactly (`tests/test_arch.py`):

| component | count | each | total |
|---|---|---|---|
| FFN (every layer) | 64 | 267.4 M | 17.11 B |
| DeltaNet mixer | 48 | 115.9 M | 5.56 B |
| Gated attention mixer | 16 | 104.9 M | 1.68 B |
| embed_tokens + lm_head (untied) | 2 | 1.271 B | 2.54 B |

`linear_attention` block = 383.279 M · attention block = 372.245 M · base after
`--drop-vision --drop-mtp` = 26.8961 B, so `params(n) = 26.8961 − 0.383279 n`.

The `attn_output_gate` fuses a multiplicative gate into `q_proj`, making it **[12288, 5120]**
rather than [6144, 5120]. Omitting that under-counts by 31.5 M/layer, and it is also the
component Stage 0 protects at higher precision.

---

## Stages

Build order is not stage order. **Stage 1 runs first**: if the converter cannot handle a
non-uniform layout, Stage 0's results are irrelevant until that is fixed.

| stage | ~time | what |
|---|---|---|
| `stage1-smoke` | 4 h | 2-cut surgery → GGUF → quantize → generate. **Tests the converter blocker.** |
| `stage0-bitwidth` | 4 h | bpw-vs-repetition curve on the *unpruned* 27B. Where does circling stop? |
| `stage2-baselines` | 4 h | bf16 KL reference; IQ3_XXS KL (the number to beat); repetition on both |
| `stage3-score` | 8 h | ablation-KL damage profile, ~172 forward passes, plus a positional control |
| `stage4-surgery` | 1 h | 12 cuts; measure the unhealed floor |
| `stage5-teacher` | 6 h | 50M-token top-16 logprob cache (~5 GB) |
| `stage6-heal` | 72 h | QLoRA against the cache; both metrics at every checkpoint |
| `stage7-ship` | 4 h | merge, needle @128K, quantize candidates, ship gate |
| `stage8-ladder` | 96 h | re-score against the **healed** parent, repeat for 18B |

### The Stage 1 blocker

`convert_hf_to_gguf.py` may regenerate the layout from `full_attention_interval` rather than
reading a non-uniform `layer_types`. If it does, everything downstream is dead.

`quantize.probe_converter` reads the produced GGUF back and checks which blocks actually carry
attention tensors. It uses a small built-in GGUF metadata reader, so it needs no llama.cpp
binaries and runs anywhere:

```bash
marlowe probe --model ./Marlowe-22B --gguf ./marlowe-22b-bf16.gguf
```

A converter that regenerated the layout produces a clean 3:1 alternation over the *pruned*
layer count — which looks plausible and is wrong. That is exactly what the probe catches. If
it fails, patch the converter, put the patched copy at `vendor/convert_hf_to_gguf.py` (it is
picked up automatically), and upstream it.

### Stage 3 outputs a standalone artefact

No one has published layer-redundancy measurements on a hybrid linear/attention stack.
Whether DeltaNet layers are more or less redundant than attention layers, and whether the 3:1
ratio is load-bearing or arbitrary, are open questions the profile answers — independent of
whether the pruning run succeeds. Written as `layer_profile.json` plus a plot, alongside a
positional-selection control. **If measured selection does not beat positional selection, that
is a finding worth reporting, not a bug to tune away.**

### Ship gate

Both required, neither negotiable:

- KL to the bf16 parent **strictly below** the user's current IQ3_XXS build.
- repetition rate **at or below** the bf16 baseline.

A missing measurement fails the gate rather than passing it. Needle @128K below 0.9 and a
build over 10.0 GB warn but do not block.

---

## Hardware

RTX 4080 Super (16 GB), 32 GB system RAM. Every design decision follows from those.

**No stage may hold bf16 weights resident** — the full model is 55.6 GB. Scoring, teacher
caching and healing all load NF4, and `preflight.assert_no_bf16_resident` asserts it rather
than assuming it, raising with the largest offenders if the quantisation config was ignored.

Two consequences worth knowing about:

- **Logits are never materialised in full.** `seq_len × 248320 × 4` bytes is 8 GB at 8K and
  33 GB at 32K. `score.py` runs the decoder for hidden states and applies `lm_head` only at
  the sampled positions; long sequences prefill in chunks so activation memory tracks the
  chunk, not the sequence. `heal.py` does the same for the training loss.
- **The deployment budget is ~10.0 GB of weights**, empirically, not theoretically: a 27B
  IQ3_XXS at 10.0 GB with a Q8 KV cache at 32K context is at the limit today. The ~6 GB of
  overhead does not shrink when you prune — it scales with hidden size (5120, unchanged) and
  the 16 preserved attention layers.

`marlowe doctor` reports the disk budget against actual free space. Peak is ~187 GB; the
brief's 150 GB assumption is already tight.

---

## Sampling

```
thinking:     temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
non-thinking: presence_penalty=1.5
```

Running thinking mode with the non-thinking preset produces repetition that looks exactly like
quantisation damage. Every eval pins the preset explicitly and records which it used; the
repetition harness **refuses** the non-thinking preset outright, because it would manufacture
the artefact being measured. `write_modelfile` bakes the thinking preset into the Modelfile
for the same reason.

## Metrics

| metric | what it catches | tool |
|---|---|---|
| KL to bf16 parent | general fidelity loss | `llama-perplexity --kl-divergence` |
| top-1 agreement | same, interpretable | same |
| **repetition rate (n=8, 32)** | **circling** | `eval/repetition.py` |
| cap-hit rate | non-termination | same |
| loop rate | cap hit **and** a periodic tail | same |
| GPQA Diamond | reasoning, milestones only | `eval/bench.py` |
| needle @128K | long-context / DeltaNet state | same |
| tok/s decode | throughput claim | `llama-bench` |

Repetition harness: 200 completions × 2048 tokens, thinking preset pinned, on a fixed held-out
prompt set. Prompts tagged `known_trigger` are reported as a separate subset — an aggregate
over easy prompts hides a regression on the hard ones.

`loop_rate` requires both a cap hit and a periodic tail. Either alone is too loose: long
answers hit the cap legitimately, and a cycle inside an otherwise-terminating answer is not
the failure being chased.

### On AAII

True AAII v4.1.1 is not reproducible here. It is nine evaluations (GDPval-AA v2, τ³-Banking,
Terminal-Bench v2.1, SciCode, HLE, GPQA Diamond, CritPt, AA-Omniscience, AA-LCR) weighted
34% Agents / 24% Coding / 24% Scientific Reasoning / 18% General, and AA-Omniscience is a
private dataset. This pipeline reports GPQA Diamond and the KL/repetition suite and claims no
AAII number.

58% of AAII weight is agentic plus coding — long generative traces where errors compound.
Expect measured degradation to exceed what the MCQA-style GPQA column suggests.

---

## Development

```bash
pytest            # 150 tests, no GPU or weights needed
ruff check .
mypy marlowe      # strict
```

Tests run against a synthetic mini-checkpoint with realistic tensor naming and **three**
competing `.layers.` namespaces — text (12 layers), a vision-tower decoy (27), and an MTP
decoy (1) — so prefix detection is proven to select by depth rather than by name or ordering.

## Known risks

| risk | detection | fallback |
|---|---|---|
| GGUF converter can't read non-uniform `layer_types` | Stage 1, day one | patch and vendor the converter |
| circling threshold above 3.66 bpw | Stage 0 | skip 22B, go direct to 18B at IQ4_XS |
| 50M tokens insufficient for 12 cuts | KL plateau, Stage 6 day 2 | fall back to 8 cuts, re-heal on cached data |
| measured selection no better than positional | Stage 3 control | report as finding; use positional |
| 22B at IQ3_M exceeds 10 GB in practice | Stage 7 | Q4 K-cache, 24K context, or partial offload |
| long-context regression from 2K healing | needle @128K, Stage 7 | `heal.long_context_tokens: 5000000` at 16K |
