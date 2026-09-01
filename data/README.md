# Data

None of these are shipped with real content. Each explains what it needs and why.

## `calib.jsonl` -- Stage 3 scoring, Stage 2/7 KL corpus

JSONL, one object per line with a `text` field. **Long documents.** The scorer packs text into
fixed-length sequences and refuses to start if it cannot fill `score.n_seqs` sequences of
`score.seq_len` tokens, which is 4 x 32768 = 131072 tokens minimum, and realistically several
times that so the sequences are not all from one document.

Long is not a preference. Angular-distance and short-snippet scoring systematically nominate
exactly the DeltaNet layers that must not be cut: a linear-attention layer's per-token output
delta can be small while its contribution to state maintenance across 100K tokens is large.
Short calibration cannot see the second thing.

## `heal_corpus.jsonl` -- Stage 5 teacher cache, Stage 6 healing

Same format. In-domain data where available. At this compression ratio there is no obligation
to recover the parent's general capability, only the part actually used, so weight this
toward the work the model will really do.

Sizing: `teacher.tokens` (default 50M) is what gets cached, so the corpus needs at least that
many tokens or the cache loop will run short and healing will start reusing data.

## `circling_prompts.jsonl` -- the repetition harness

The scaffold in this repo is a starting point, not the real input. Replace it.

The most valuable rows are prompts you have personally watched the IQ3_XXS build circle on.
Tag those `"known_trigger": true`: they are reported as a separate subset, because an
aggregate over easy prompts can hide a regression on exactly the hard ones this project
exists to fix.

Fields: `id`, `text`, optional `known_trigger` (bool), optional `tags` (list).

Held out means held out. Do not draw these from `heal_corpus.jsonl`.

## `gpqa_diamond.csv` -- Stage 7 milestone eval

Gated on HuggingFace. Accept the terms at huggingface.co/datasets/Idavidrein/gpqa and export
the Diamond split here. 198 questions. Official column names are read directly; a simplified
JSONL with `question` / `correct` / `incorrect` also works.
