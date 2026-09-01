# vendor

`convert_hf_to_gguf.py` goes here if Stage 1 shows the stock converter mishandles a
non-uniform `layer_types`. `quantize.find_converter` checks this directory first, then
`$LLAMA_CPP_ROOT`, then `PATH`.

A patched copy is tracked in git deliberately: every GGUF this project produces depends on it,
and a run is not reproducible if the converter is not pinned. Record the upstream commit it
was forked from at the top of the file, and open the upstream PR.

`marlowe probe --model <hf-dir> --gguf <out.gguf>` is the check. It needs no llama.cpp
binaries.
