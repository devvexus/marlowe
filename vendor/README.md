# vendor

## `0001-qwen35-explicit-recurrent-layers.patch` — REQUIRED

Without this patch, every GGUF this project produces from a pruned checkpoint is silently
mis-typed. It is not optional and it is tracked in git deliberately: a run is not
reproducible if the converter is not pinned.

### What is wrong upstream

`conversion/qwen.py` writes only `<arch>.full_attention_interval`, defaulting to 4 when the
key is absent:

```python
self.gguf_writer.add_full_attention_interval(self.hparams.get("full_attention_interval", 4))
```

`src/models/qwen35.cpp` already *prefers* an explicit per-layer array and falls back to the
interval only when it is missing:

```cpp
if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, ...)) {
    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
    }
}
```

The loader is fine. The gap is that `gguf-py` had no constant and no writer for
`<arch>.attention.recurrent_layers`, so nothing could ever emit it.

### Why it only breaks pruned models

For a uniform 64-layer stack, interval 4 is correct and nobody notices. For a depth-pruned
stack it is not: removing linear-attention layers leaves periods of unequal length that no
interval describes. Marlowe additionally *removes* `full_attention_interval` from the child
config (rule 3.4), so `.get(..., 4)` falls through to the default and the mismatch is
guaranteed rather than incidental.

Measured on a 52-layer stack (Qwen3.8-27B minus 12 evenly spaced linear layers), the stock
converter causes the loader to mis-type **15 of 52 layers**, including `full_attention`
layers marked recurrent — which silently drops their KV cache. The model loads, runs at full
speed, and is wrong.

That is the entire Stage 1 blocking risk, confirmed.

### The fix

Three small additions, all upstreamable:

1. `gguf-py/gguf/constants.py` — the `Keys.Attention.RECURRENT_LAYERS` constant.
2. `gguf-py/gguf/gguf_writer.py` — `add_recurrent_layers()`, modelled on the existing
   `add_sliding_window_pattern()`.
3. `conversion/qwen.py` — emit the explicit array whenever `config.json` carries
   `layer_types`, and stop inventing an interval that was never in the config.

### Applying it

```bash
git clone https://github.com/ggml-org/llama.cpp .tools/llama.cpp
git -C .tools/llama.cpp apply vendor/0001-qwen35-explicit-recurrent-layers.patch
marlowe doctor        # "layout support: OK"
```

The pinned upstream commit is recorded at the top of the patch. `find_converter()` resolves
`$LLAMA_CPP_ROOT`, then `.tools/llama.cpp`, then `PATH`.

### Verifying it

`marlowe doctor` reports layout support statically — no weights, no binaries, no conversion.
`marlowe.quantize.converter_writes_explicit_layout()` is the same check as an API, and Stage 1
calls it before doing any work. Once a GGUF exists, `marlowe probe --model <hf-dir> --gguf
<out.gguf>` checks the produced file itself.

Upstream has split the model classes out of `convert_hf_to_gguf.py` into a `conversion/`
package, so the converter is no longer a single vendorable file — hence a patch plus a pinned
commit rather than a copied script.
