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

**`calib.jsonl` OVERLAPS `heal_corpus.jsonl` and is NOT a held-out set.** Both are built by
`scripts/build_corpora.py` streaming the same proof-pile-2 arXiv shards from index 0, so the
calibration set's arXiv content is a subset of the healing corpus by construction (6 rows are
byte-identical). That is harmless for its actual jobs -- Stage 3 measures parent behaviour
under ablation, and the importance matrix measures weight importance; neither is a
generalisation claim. **Do not use it to evaluate a healed model.** For that,
`kl_reference.jsonl` exists.

## `kl_reference.jsonl` -- the KL yardstick, held out from healing

Same composition as `heal_corpus.jsonl` (~65% proof-pile-2, ~35% fineweb-edu) so KL is
measured on the deployment distribution, but drawn from shards the healing corpus never read.
312,147 tokens = 38 chunks at `-c 8192`.

Two holdout mechanisms, because the sources resolve their shards differently: proof-pile-2
sources get explicit URLs built from `HELDOUT_SHARD_OFFSET`, while fineweb-edu lists its
shards from the repo at stream time and is offset inside `_stream`. The manifest records
`shards_actually_read`, not the declared table -- they were once different.

`assert_disjoint` **fails the build** if any document hash collides with the healing corpus.
A reference that is 99% held out is not held out: every healing checkpoint is scored against
this file, so a shared document means the ship gate is partly measuring memorisation.

This is not theoretical. On its first real run the check failed with **112 collisions**:
`_offset` was set on the fineweb-edu spec and `_stream` never read it, so a third of the
"held-out" reference was the exact shards healing trains on. Nothing else would have caught
it -- the file looked correct and the manifest claimed a holdout that had not been applied.
Do not weaken this check into a warning.

**Once a reference `.kld` has been built from it, this file must never change.** Every child
measurement is relative to those exact bytes. Its sha256, token count, shard offsets and the
`-c 8192` context are recorded in `runs/<name>/manifests/kl_reference.json`.

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
