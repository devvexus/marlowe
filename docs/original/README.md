# Original scripts

The two scripts this pipeline started from, kept verbatim for provenance. Neither is on the
import path; both have been superseded.

## What was carried forward

* **`surgery.py` -> `marlowe/surgery.py`.** The streaming design (one tensor at a time, no
  GPU, no full model load) and `detect_stack`'s auto-detection of the text decoder prefix by
  matching the max layer index against `num_hidden_layers`, so the 27-layer vision tower is
  never selected. That heuristic is now `arch.detect_text_stack_prefix` and has a test with
  three competing `.layers.` namespaces.
* **`prune_qwen38.py` -> `marlowe/score.py`.** The ablation-KL formulation, the `Identity`
  passthrough that mirrors the decoder's return style, the late-position sampling, and the
  greedy-with-rescoring selector (now opt-in via `score.mode: greedy`).

## What was changed, and why

* **`plan()` hardcoded `range(64)`** when building the kept-index list. On any parent that is
  not 64 layers deep, the remap silently included indices that do not exist -- which is
  exactly the 22B -> 18B ladder step. Now derived from the source config.
  (`tests/test_surgery.py::TestLadder`)
* **The scorer loaded bf16 with `device_map="auto"`.** 55.6 GB against 16 GB of VRAM and
  32 GB of RAM. Now NF4, with `preflight.assert_no_bf16_resident` asserting it rather than
  assuming it.
* **Full logits were materialised.** `seq_len x 248320 x 4` bytes is 8 GB per 8K sequence and
  33 GB at 32K. `score.py` now runs the decoder for hidden states and applies `lm_head` only
  at the sampled positions, prefilling long sequences in chunks.
* **No layer_types/tensor consistency assertion.** Rule 3.4 is now enforced after every
  surgery and in CI, and generalised to arbitrary layer types.
* **Removability was a blacklist** (`!= "full_attention"`). Now a whitelist: only
  `linear_attention` is removable, and anything unrecognised fails closed.
* **The period length 4 was assumed** in `--auto` and in the "at least one DeltaNet per
  period" check. Now derived from `layer_types`.
* The docstring claimed peak memory was one tensor (~250 MB). The read side is; the write
  side buffers an output shard, because `safetensors.save_file` takes a complete dict.
  `--shard-size` bounds it, and the docstring now says so.
