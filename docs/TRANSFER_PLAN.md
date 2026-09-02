# Marlowe — Maximum Transfer Plan

**Goal:** the healed 22B should be as close to the 27B's reasoning capability as the
architecture allows. Not "a smaller model," not "fix the 27B's circling" — maximum
knowledge and reasoning transfer from parent to child.

**The metric:** KL divergence to the parent, measured against the certified 77 GB
reference at 8192 context. Everything in this document is a lever on that number.
Repetition metrics are a sanity check that the model is not degenerate; they are not a
fidelity measure.

**Why this supersedes the original plan:** the original healing corpus was documents —
arXiv, math, web text. Distilling on documents transfers document modelling. Distilling
on the parent's own reasoning traces transfers reasoning behaviour. Those are different
things, and the second is what a PI model needs.

---

## Baseline, measured

```
                          kl_mean   kl_median   top-1 agree
27B iq3_xxs (gate)        0.144     0.023       89.96%
22B unhealed IQ3_M        0.451     0.153       77.98%
```

The unhealed child sits 0.31 nats from the gate. The damage is diffuse (median moved
6.7×) and manifests as a sampling lock, not decay — presence penalty 1.5 eliminates it
completely. That is the healable case: a blurry distribution to sharpen, not a broken one
to rebuild.

The gate is asymmetric: the child pays pruning residual plus its own quantisation; the
baseline pays quantisation only. IQ3_M's advantage over IQ3_XXS is ~0.04–0.06 nats, so the
healing residual must land under ~0.05 to pass at that width. Expect IQ4_XS to pass before
IQ3_M. An IQ3_M-only miss is the expected intermediate state, not grounds to retune.

---

## Levers, ranked by expected effect on transfer

### 1. Reasoning traces — the largest lever, and it comes in two phases

Traces are text, not logits. That distinction opens two distinct uses with two distinct
teachers, and they must run in sequence, not together.

**Phase A — healing traces, from the parent.** Generate ~10M tokens of the 27B's own
reasoning on in-domain prompts and mix them into the healing corpus at ~30%. Trained via
the top-K KL against the local 4-bit 27B like every other corpus text. Objective: be the
27B. Measured by KL to the reference.

**Phase B — uplift traces, from stronger teachers.** After Phase A has healed the student
to the parent, generate a second trace set from models that outrank the 27B, and SFT on
them with plain cross-entropy. Objective: reason better than the 27B. Measured by
benchmarks, *not* by KL to the 27B — see "Gate implications" below.

Candidates for Phase B, with the operator's numbers:

```
Qwen3.8-27B          ~$0.37/task   52 AAII   42 tok/s   (the parent)
Qwen3.8-Flash-Next   ~$0.10/task   56 AAII   88 tok/s   (same family, preferred)
GLM-5.3-Flash        ~$0.09/task   57 AAII   45 tok/s   (different family, diversity)
```

Flash-Next is the primary Phase B teacher: same family, closest thinking style, cheaper
than the parent. GLM as a second source for diversity if budget allows. Their traces are
re-wrapped in Qwen3.8's chat template with `<think>` tags — the reasoning content
transfers, the wrapper is Qwen's. No vocab alignment is needed because SFT uses hard
labels tokenised with the student's tokeniser.

This is the DeepSeek-R1-Distill playbook: R1-Distill-Qwen-32B beat Qwen-32B by SFT on a
stronger model's traces. It works, it is standard, and it means the 22B is not capped at
the parent's 52.

**Why Phase A and B cannot run together:** the healing loss says "be the 27B"; the uplift
loss says "reason like Flash-Next." Combined, they fight. Sequenced, they compose. Phase B
is the second LoRA round (lever 6) with a real objective.

**Honest ceiling:** a 22B cannot become a 57. Capacity bounds it, not teacher quality.
Expect 2–5 points of uplift on reasoning-heavy benchmarks, which puts the 22B at or
slightly above the parent — a real possibility, not a ceiling.

**Phase A source:** hosted bf16 Qwen3.8-27B via OpenRouter, `reasoning_effort=xhigh`,
thinking preset (`temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`). Not
the local 4-bit parent — it is the wrong precision for the *target behaviour* and at ~40
tok/s it would take ~70 hours. Hosted is ~$25 for 10M output tokens.

**Prompts:** ~1,000, in-domain for a research/PI model. Sources, in priority order:
- arXiv abstracts from proof-pile-2 → "explain the method and its limitations" /
  "propose a follow-up experiment" / "identify the weakest assumption"
- Math and logic problems from MATH-train, proof-pile-2's OpenWebMath problems
- Synthetic PI tasks: "design an experiment to test X," "critique this study design,"
  "plan a research programme for Y"

**Contamination rule:** no prompt may come from any evaluation set — GPQA (any split),
HLE, AIME 2024+, MMLU-Pro, LiveCodeBench, SciCode. Hash every prompt and assert
disjointness against those sets before generation. A contaminated healing corpus makes
every downstream benchmark number worthless.

**Format:** the full chat-templated sequence — system/user/assistant with the `<think>`
block intact — not the raw trace text. The student must learn the assistant behaviour in
the format it will be used, and the thinking block *is* the reasoning behaviour being
transferred.

**Logprobs:** the local 4-bit teacher caches logprobs over these traces like any other
corpus text. Optional upgrade: if the hosted endpoint returns `top_logprobs`, use the bf16
parent's distribution for the trace subset — higher-quality signal on exactly the tokens
that matter most. Check whether the provider exposes it; do not assume.

**Provenance:** prompt source, model string, sampling params, generation date, and the
prompt-hash manifest all recorded alongside the traces.

### 2. Top-K from 16 to 64 — free

The top-16 captures most probability mass on confident tokens but as little as 60% on
uncertain ones, and uncertain positions are where reasoning signal concentrates. K=64
costs 4× cache disk (~14 GB at 35M tokens; affordable against ~110 GB free after the
reference) and zero training compute. More distribution shape per token for the same
token budget.

### 3. Rank 32 over rank 16 — capacity beats context

If Stage 5's memory search lands on `rank16-seq768`, measure `rank32-seq768`. The 32→16
gap was 0.14 GB and seq 768 saves ~0.25 GB, so rank 32 at 768 probably fits. Neither 768
nor 1024 trains long-range state eviction, so sequence length is not buying transfer;
rank is what determines how much the adapters can absorb.

### 4. Confirm LoRA targets every projection

q, k, v, o, gate, up, down, and the DeltaNet-specific projections. If the target list is
attention-only (a common default), the FFN cannot adapt and transfer is capped. One-line
check before Stage 6.

### 5. Extend tokens if the curve says so

35M was sized to finish both rungs on schedule. If the 10M checkpoint shows KL still
falling at 35M, keep going on the 22B. The 18B is downstream of the healed 22B, so a
better-healed intermediate produces a better headline. Decide from the slope, not the
budget.

### 6. Second LoRA round on the merged model — this is Phase B

Merge Phase A's adapters into the bf16 base. Evaluate the ship gate on that checkpoint.
Then a fresh rank-32 LoRA round on the merged model, SFT on the Phase B traces. Merge
again. Two stacked rounds also get around the rank ceiling the card imposes.

If Phase A's KL is still falling at 35M, extend Phase A before starting B — uplift on an
under-healed base transfers less.

---

## Gate implications — read this before evaluating anything after Phase B

The ship gate is **KL to the 27B, strictly below the `iq3_xxs` baseline.** That gate
measures healing. It is evaluated on the **Phase A merged checkpoint**, before any uplift.

After Phase B, KL to the 27B goes **up** — by design. The student now reasons in ways the
parent does not, so it diverges from the parent's distribution on exactly the tokens where
it improved. **A rising KL after Phase B is the objective succeeding, not the gate
failing.** If nobody writes this down, the gate rejects the better model.

Phase B is measured by:
- GPQA Diamond, HLE subset, HumanEval+ — the anchor-interpolation set
- Repetition sanity at pp=0.0 (must not regress from the Phase A checkpoint)
- Needle at 128K (must not regress)
- KL to the 27B is recorded but **not gated**

Both checkpoints ship: the Phase A model as "Marlowe-22B" (healed, parent-faithful), the
Phase B model as "Marlowe-22B-Reasoning" or similar. They are different products with
different claims and the naming should say so.

---

## Corpus composition, revised

```
~30%  parent reasoning traces (lever 1)       ~10M tokens
~45%  proof-pile-2 (arXiv / OpenWebMath / AlgebraicStack)
~25%  fineweb-edu
```

`data/heal_corpus.jsonl` is rebuilt with this mix. `teacher.tokens` and `heal.tokens` stay
coupled. `data/kl_reference.jsonl` must remain disjoint — assert against the new corpus
before anything else runs.

---

## Quantisation for shipping

The mixer-protected mix, as a ship width for the child: `linear_attn.*`, `q_proj` (with its
fused output gate), `k/v/o_proj` at Q5_K; FFN at IQ3_XXS; embeddings at Q4_K. The
DeltaNet projections and attention gate are where the parent's knowledge is encoded and
where quantisation error compounds along the sequence. Protecting them serves the
transfer goal directly. Measure on the unhealed child now and the healed child later.

---

## Roadmap

1. **Now:** diagnostic 1 (bf16 unhealed at pp=0.0) → onset → Stage 5 search-alone.
   Push for rank 32.
2. **Before the teacher cache builds:** K=64; Phase A traces generated from the 27B and
   mixed in; corpus rebuilt; disjointness re-asserted. About one day, ~$25.
3. **Stage 6 = Phase A** with the decision point at 10M tokens: KL below ~0.25 and
   falling → continue; flattening above ~0.30 → stop and report.
4. **Stage 7 on the Phase A checkpoint:** three widths plus the mixer-protected mix.
   Expect IQ4_XS to pass first. **This is the KL gate, and it is the only place the KL
   gate applies.**
5. **Phase B:** generate uplift traces from Flash-Next (and GLM if budget allows),
   ~$100–150 for ~1,000 tasks. Fresh LoRA on the merged Phase A model, SFT, merge.
   Evaluate by benchmarks. Ship as a separate artifact.
6. **18B rung** with greedy re-scoring, not one-shot — Stage 3's one-shot mode never
   measured cut interaction, and 8 of 12 cuts landed in periods 1–4. Cut from the
   **Phase A** 22B, not the Phase B one: the ladder wants a parent-faithful intermediate,
   and Phase B can be re-run on the 18B separately.

---

## Expected outcome

With levers 1–3 applied to Phase A: IQ3_M pass moves from a coin-flip-against to roughly
even; IQ4_XS moves to likely. The Phase A 22B at IQ4_XS (~12.6 GB) is the most probable
first shipped artifact and should be the best parent-faithful model in that footprint.

Phase B can push the 22B to or slightly above the parent on reasoning-heavy work — 2–5
benchmark points is the honest expectation, capacity-bounded. That would make it the
best *reasoning* model in the footprint, which is a different and stronger claim than
"best compression of the 27B."

The thesis — that compression from a strong parent beats native training at the child
size — is untested until the first healed checkpoint. Everything above is about giving
that test the best chance of a clean answer.

---

## Operator inputs

- OpenRouter key: serves the bf16 repetition baseline, Phase A trace generation (~$25),
  and Phase B trace generation (~$100–150). The trace spend is the highest-leverage money
  in the project.
- Nothing else.