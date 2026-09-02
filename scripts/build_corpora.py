"""Build data/calib.jsonl and data/heal_corpus.jsonl for a reasoning-domain model.

Marlowe is meant to act as a PI: logic, scientific reasoning, planning. Both corpora are
built to that domain, from the same sources, for different jobs.

**calib.jsonl** -- Stage 3 layer scoring, and the importance matrix for quantisation. In
domain, because layer damage rankings depend on the text they are measured on, and long,
because a linear-attention layer's per-token output delta can be small while its contribution
to state maintenance across 100K tokens is large. Short calibration systematically nominates
exactly the DeltaNet layers that must not be cut. arXiv papers are concatenated into
sequences above the 32768-token floor; a few pg19 books are included as a length control so
the calibration is not purely technical.

**heal_corpus.jsonl** -- Stage 5 teacher cache and Stage 6 healing. ~65% proof-pile-2 (35%
arXiv, 15% OpenWebMath, 15% AlgebraicStack) and ~35% fineweb-edu, because a PI writes and
plans in plain language as well as equations.

Licensing: proof-pile-2 and fineweb-edu are both ODC-By. Marlowe is a commercial model, so
that attribution requirement travels with it. AlgebraicStack derives from the-stack; if this
is extended with more code, keep the permissive-license filter on.

A note on arXiv categories: proof-pile-2's arXiv split carries only timestamp, yymm,
arxiv_id, language and url -- there is no subject category to group by. Sequences are grouped
by submission month, which is the available locality proxy. True subject grouping would need
an arXiv API lookup per id. The requirement that actually matters here is length.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PP2 = "https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/main"
FWE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main"

#: Shards are streamed in order and stopped as soon as the budget is met, so listing a few
#: is enough; more are here only so a larger budget does not run dry mid-build.
SOURCES: dict[str, dict[str, Any]] = {
    "arxiv": {
        "urls": [f"{PP2}/arxiv/train/arXiv_{i:03d}.jsonl.zst" for i in range(8)],
        "license": "ODC-By",
        "note": "proof-pile-2 arXiv (cleaned LaTeX)",
    },
    "open-web-math": {
        "urls": [f"{PP2}/open-web-math/train/shard-{i:04d}.jsonl.zst" for i in range(8)],
        "license": "ODC-By",
        "note": "proof-pile-2 OpenWebMath",
    },
    "algebraic-stack": {
        "urls": [
            f"{PP2}/algebraic-stack/train/{name}.jsonl.zst"
            for name in ("python0000", "c0000", "haskell0000", "lean0000", "matlab0000")
        ],
        "license": "ODC-By (the-stack, permissive-licensed subset)",
        "note": "proof-pile-2 AlgebraicStack",
    },
    "fineweb-edu": {
        # Resolved from the repo, not constructed. Shard naming is not uniform across
        # crawls -- the oldest use `train-00000-of-00014.parquet` and newer ones
        # `000_00000.parquet` -- so a guessed URL 404s, and _stream's per-shard skip turned
        # that into a corpus silently missing its entire general-language third.
        "repo": "HuggingFaceFW/fineweb-edu",
        "prefix": "data/CC-MAIN-2025-18/",
        "license": "ODC-By",
        "note": "fineweb-edu (general capability)",
        "format": "parquet",
    },
    "pg19": {
        "hf": "emozilla/pg19",
        "license": "public domain (Project Gutenberg, pre-1919)",
        "note": "length control for calibration",
    },
}


def _tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _stream(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield records from a source, shard by shard, without downloading everything."""
    from datasets import load_dataset

    if "hf" in spec:
        yield from load_dataset(spec["hf"], streaming=True, split="train")
        return
    builder = spec.get("format", "json")
    urls = spec.get("urls")
    if urls is None:
        from huggingface_hub import HfApi

        repo, prefix = spec["repo"], spec["prefix"]
        names = [
            s.rfilename
            for s in HfApi().dataset_info(repo).siblings
            if s.rfilename.startswith(prefix)
        ]
        if not names:
            raise RuntimeError(
                f"no files under {prefix!r} in {repo}. The shard layout changed; listing the "
                f"repo is the fix, guessing the names is what broke."
            )
        urls = [f"https://huggingface.co/datasets/{repo}/resolve/main/{n}" for n in sorted(names)]
    # Held-out sources start reading past whatever the healing corpus consumed. For sources
    # with an explicit `urls` list the caller does this by constructing different URLs; for
    # repo-listed sources the names are resolved here, so the offset has to be applied here
    # too. It was set on the spec and never read, which made the fineweb-edu third of the
    # "held-out" reference the same shards healing had already trained on -- caught by
    # assert_disjoint with 112 collisions, which is exactly what that check is for.
    offset = int(spec.get("_offset", 0))
    if offset:
        if offset >= len(urls):
            raise RuntimeError(
                f"_offset={offset} skips past all {len(urls)} shards; nothing would be read."
            )
        urls = urls[offset:]
    for url in urls:
        try:
            # The generic builder, pointed at the file: proof-pile-2 still ships a loading
            # script, and datasets 4.x refuses those outright.
            ds = load_dataset(builder, data_files=url, streaming=True, split="train")
            yield from ds
        except Exception as exc:  # noqa: BLE001 - one bad shard must not end the build
            print(f"    shard failed ({type(exc).__name__}: {str(exc)[:80]}), skipping: {url}",
                  file=sys.stderr, flush=True)


def take_tokens(
    name: str, budget: int, tok: Any, *, min_chars: int = 400, seed: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """Stream ``name`` until ``budget`` tokens have been collected."""
    spec = SOURCES[name]
    out: list[dict[str, Any]] = []
    total = 0
    t0 = time.time()
    for rec in _stream(spec):
        text = rec.get("text") or rec.get("content") or ""
        if len(text) < min_chars:
            continue
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        # Carry the grouping key. Dropping it here made build_calib's month grouping a
        # silent no-op: every document fell into "unknown" and concatenation degenerated to
        # sequential order, which looks identical in the output.
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        out.append({
            "text": text, "source": name, "tokens": n,
            "yymm": str(meta.get("yymm", "")) or None,
        })
        total += n
        if total >= budget:
            break
        if len(out) % 2000 == 0:
            print(f"    {name}: {total / 1e6:.2f}M / {budget / 1e6:.2f}M tokens "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"  {name}: {total / 1e6:.2f}M tokens in {len(out)} docs ({time.time() - t0:.0f}s)",
          flush=True)
    # Fail loudly on a short source.
    #
    # _stream skips a shard it cannot open so one bad URL does not end a long build. That is
    # right for one shard and wrong for all of them: the first run of this script produced a
    # corpus with 0.00M of its 12.25M fineweb-edu tokens -- a third of the mix missing -- and
    # reported success. A corpus quietly missing its general-language component would have
    # trained a model that only ever saw equations, and nothing downstream would have said so.
    if total < budget * 0.9:
        raise RuntimeError(
            f"{name} yielded {total / 1e6:.2f}M tokens against a {budget / 1e6:.2f}M budget. "
            f"Shards were skipped or the source is exhausted -- check the 'shard failed' "
            f"lines above. Refusing to write a corpus whose composition does not match its "
            f"manifest."
        )
    return out, total


def build_calib(out_path: Path, tok: Any, *, target_tokens: int, seq_len: int,
                n_books: int, seed: int, max_book_tokens: int = 0) -> dict[str, Any]:
    """arXiv concatenated to >= seq_len tokens, plus pg19 books as a length control."""
    print(f"calibration: target {target_tokens / 1e6:.1f}M tokens, "
          f"sequences >= {seq_len} tokens", flush=True)
    # Over-collect: concatenation discards whatever is left in a short group.
    docs, _ = take_tokens("arxiv", int(target_tokens * 1.3), tok, seed=seed)

    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        by_month[d.get("yymm") or "unknown"].append(d)

    rows: list[dict[str, Any]] = []
    packed = 0
    buf: list[str] = []
    buf_tokens = 0
    for month in sorted(by_month):
        for d in by_month[month]:
            buf.append(d["text"])
            buf_tokens += d["tokens"]
            if buf_tokens >= seq_len:
                rows.append({"text": "\n\n".join(buf), "source": "proof-pile-2/arxiv",
                             "tokens": buf_tokens, "group": month})
                packed += buf_tokens
                buf, buf_tokens = [], 0
                if packed >= target_tokens:
                    break
        if packed >= target_tokens:
            break
    print(f"  packed {len(rows)} sequences, {packed / 1e6:.2f}M tokens", flush=True)

    # Truncate each book to a bounded length.
    #
    # These are a *control*, and an uncapped one swamps the thing it is controlling: the
    # first book pg19 yields is the King James Bible at 1.22M tokens, which alone is 3.5x
    # the entire arXiv portion at the default budget. Damage rankings measured on a corpus
    # that is mostly scripture would not describe a model doing scientific reasoning. Two
    # sequence-lengths each is long enough to exercise recurrent state and small enough to
    # stay a minority of the corpus.
    cap = max_book_tokens or seq_len * 2
    books = 0
    book_tokens = 0
    for rec in _stream(SOURCES["pg19"]):
        text = rec.get("text", "")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < seq_len:
            continue
        if len(ids) > cap:
            text = tok.decode(ids[:cap])
            n = cap
        else:
            n = len(ids)
        rows.append({"text": text, "source": "pg19", "tokens": n,
                     "group": rec.get("short_book_title", ""),
                     "truncated_from": len(ids) if len(ids) > cap else None})
        books += 1
        book_tokens += n
        if books >= n_books:
            break
    print(f"  + {books} pg19 books, {book_tokens / 1e6:.2f}M tokens", flush=True)

    rng = random.Random(seed)
    rng.shuffle(rows)
    _write(out_path, rows)
    return {
        "target_tokens": target_tokens,
        "seq_len_floor": seq_len,
        "sequences": len(rows),
        "tokens": packed + book_tokens,
        "composition": {"proof-pile-2/arxiv": packed, "pg19": book_tokens},
        "grouping": "arxiv concatenated by submission month (yymm); the corpus carries no "
                    "subject category",
    }


def build_heal(out_path: Path, tok: Any, *, target_tokens: int, seed: int) -> dict[str, Any]:
    """~65% proof-pile-2 (35/15/15) and ~35% fineweb-edu."""
    mix = {
        "arxiv": 0.35,
        "open-web-math": 0.15,
        "algebraic-stack": 0.15,
        "fineweb-edu": 0.35,
    }
    print(f"healing corpus: target {target_tokens / 1e6:.1f}M tokens, mix {mix}", flush=True)
    rows: list[dict[str, Any]] = []
    actual: dict[str, int] = {}
    for name, frac in mix.items():
        docs, got = take_tokens(name, int(target_tokens * frac), tok, seed=seed)
        rows.extend(docs)
        actual[name] = got

    rng = random.Random(seed)
    rng.shuffle(rows)
    _write(out_path, rows)
    total = sum(actual.values())
    return {
        "target_tokens": target_tokens,
        "tokens": total,
        "documents": len(rows),
        "requested_mix": mix,
        "actual_mix": {k: round(v / total, 4) for k, v in actual.items()},
        "composition": actual,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "meta": {
                k: v for k, v in r.items() if k != "text"}}, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(rows)} rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/marlowe-22b/models/parent",
                    help="tokenizer source; budgets are in ITS tokens")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--heal", action="store_true")
    ap.add_argument("--kl-reference", action="store_true",
                    help="build the held-out KL yardstick (disjoint from --heal's output)")
    ap.add_argument("--calib-tokens", type=int, default=1_500_000)
    ap.add_argument("--heal-tokens", type=int, default=35_000_000)
    # 300K corpus tokens is 36 chunks at -c 8192, and the reference .kld it produces is
    # ~73 GB: the file stores a distribution over all 248,320 vocabulary entries for every
    # scored token. Raising this costs 0.24 GB per thousand tokens, permanently.
    ap.add_argument("--kl-tokens", type=int, default=300_000)
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--books", type=int, default=3)
    ap.add_argument("--max-book-tokens", type=int, default=0,
                    help="cap per book; 0 means 2x --seq-len")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--run-dir", default="runs/marlowe-22b",
                    help="also write the manifest here, so each artifact carries "
                         "its provenance rather than pointing at a shared file")
    args = ap.parse_args()

    tok = _tokenizer(args.model)
    out_dir = Path(args.out_dir)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "tokenizer": args.model,
        "licenses": {k: v.get("license") for k, v in SOURCES.items()},
        "commercial_use": "proof-pile-2 and fineweb-edu are ODC-By; attribution travels with "
                          "the shipped model. AlgebraicStack derives from the-stack -- keep "
                          "the permissive-license filter on if extending with more code.",
    }
    if args.calib:
        manifest["calib"] = build_calib(
            out_dir / "calib.jsonl", tok, target_tokens=args.calib_tokens,
            seq_len=args.seq_len, n_books=args.books, seed=args.seed,
            max_book_tokens=args.max_book_tokens,
        )
    if args.heal:
        manifest["heal_corpus"] = build_heal(
            out_dir / "heal_corpus.jsonl", tok,
            target_tokens=args.heal_tokens, seed=args.seed,
        )
    if args.kl_reference:
        manifest["kl_reference"] = build_kl_reference(
            out_dir / "kl_reference.jsonl", tok,
            target_tokens=args.kl_tokens,
            heal_path=out_dir / "heal_corpus.jsonl", seed=args.seed,
        )
        # Its own manifest, separate from the shared one: this file is permanent and every
        # KL number in the project is relative to it, so its provenance must not be
        # overwritten the next time an unrelated corpus is rebuilt.
        kpath = Path(args.run_dir) / "manifests" / "kl_reference.json" if args.run_dir else None
        if kpath:
            kpath.parent.mkdir(parents=True, exist_ok=True)
            kpath.write_text(json.dumps(manifest["kl_reference"], indent=2), encoding="utf-8")
            print(f"manifest: {kpath}", flush=True)

    payload = json.dumps(manifest, indent=2)
    mpath = out_dir / "corpus_manifest.json"
    mpath.write_text(payload, encoding="utf-8")
    print(f"manifest: {mpath}", flush=True)
    # A copy travels with the run. The corpus in data/ is mutable and shared; a model built
    # six weeks from now must be able to say which corpus produced it without trusting that
    # data/ still holds the same bytes.
    if args.run_dir:
        rpath = Path(args.run_dir) / "manifests" / "corpus_manifest.json"
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(payload, encoding="utf-8")
        print(f"manifest: {rpath}", flush=True)
    return 0



# ---------------------------------------------------------------------------
# held-out KL reference
# ---------------------------------------------------------------------------

#: Where the held-out reference starts reading, per source.
#:
#: The healing corpus streams each source from index 0 and stops when its budget is met, so
#: anything it touched lives at the start. Offsetting past that is what makes the reference
#: held out by construction rather than by luck -- and the offsets are recorded in the
#: manifest because "which shards" is the only durable statement of what was excluded.
HELDOUT_SHARD_OFFSET = {
    "arxiv": 20,
    "open-web-math": 20,
    "algebraic-stack": 0,  # named files, not indexed; a different set is used below
    "fineweb-edu": None,   # resolved from the repo listing, offset applied there
}

#: AlgebraicStack files the healing corpus did not read (it used *0000 of each language).
HELDOUT_ALGEBRAIC = ("python0001", "c0001", "haskell0001", "lean0001", "matlab0001")


def _doc_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def heal_document_hashes(path: str | Path) -> set[str]:
    """Every document in the healing corpus, by content hash."""
    out: set[str] = set()
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(_doc_hash(json.loads(line).get("text", "")))
    return out


def assert_disjoint(rows: list[dict[str, Any]], heal_hashes: set[str]) -> None:
    """Refuse to write a reference that shares any document with the training corpus.

    This is the whole point of the file. Every healing checkpoint is scored against it, so a
    single shared document means the ship gate is partly measuring memorisation. Checked
    before anything is written, and fatal -- a reference that is 99% held out is not held out.
    """
    collisions = [r for r in rows if _doc_hash(r["text"]) in heal_hashes]
    if collisions:
        raise RuntimeError(
            f"KL reference shares {len(collisions)} document(s) with the healing corpus. "
            f"The reference must be disjoint or the ship gate scores memorisation. "
            f"Increase HELDOUT_SHARD_OFFSET so the reference reads shards healing never "
            f"touched."
        )


def build_kl_reference(
    out_path: Path, tok: Any, *, target_tokens: int, heal_path: Path, seed: int
) -> dict[str, Any]:
    """A held-out yardstick with the healing corpus's composition.

    Same mix as healing (~65% proof-pile-2, ~35% fineweb-edu) so the KL is measured on the
    distribution the model is meant to serve, but drawn from shards healing never read.
    """
    import hashlib

    mix = {"arxiv": 0.35, "open-web-math": 0.15, "algebraic-stack": 0.15, "fineweb-edu": 0.35}
    print(f"KL reference: target {target_tokens / 1e6:.2f}M tokens, held out from {heal_path}",
          flush=True)
    # An absent healing corpus yields an empty hash set, and assert_disjoint would then pass
    # on anything. "No collisions" and "nothing to collide with" are indistinguishable
    # downstream, so the distinction has to be made here.
    if not Path(heal_path).exists():
        raise FileNotFoundError(
            f"{heal_path} does not exist, so disjointness cannot be checked. Building the "
            f"reference now would produce a file that only appears held out."
        )
    heal_hashes = heal_document_hashes(heal_path)
    if not heal_hashes:
        raise RuntimeError(
            f"{heal_path} yielded no document hashes. Refusing to certify disjointness "
            f"against an empty set."
        )
    print(f"  healing corpus documents to exclude: {len(heal_hashes)}", flush=True)

    rows: list[dict[str, Any]] = []
    actual: dict[str, int] = {}
    # What each source was ACTUALLY read from, recorded per source as the build applies it.
    #
    # Transcribing HELDOUT_SHARD_OFFSET into the manifest instead described the intent rather
    # than the act: fineweb-edu's offset is set on the spec here, not in that table, so the
    # manifest reported `fineweb-edu: None` for the one source that had 112 collisions. The
    # manifest is the only durable statement of what was excluded, so it has to be written
    # from the values that were used.
    effective: dict[str, Any] = {}
    for name, frac in mix.items():
        spec = dict(SOURCES[name])
        off = HELDOUT_SHARD_OFFSET.get(name)
        if name == "algebraic-stack":
            spec["urls"] = [f"{PP2}/algebraic-stack/train/{n}.jsonl.zst"
                            for n in HELDOUT_ALGEBRAIC]
            effective[name] = {"mode": "explicit files", "files": list(HELDOUT_ALGEBRAIC)}
        elif name == "fineweb-edu":
            spec = dict(spec)
            spec["_offset"] = 20
            effective[name] = {"mode": "shard index offset into repo listing",
                               "offset": 20, "prefix": spec.get("prefix")}
        elif off:
            spec["urls"] = [f"{PP2}/{name}/train/"
                            + (f"arXiv_{i:03d}.jsonl.zst" if name == "arxiv"
                               else f"shard-{i:04d}.jsonl.zst")
                            for i in range(off, off + 6)]
            effective[name] = {"mode": "explicit shard URLs", "offset": off,
                               "shards": [u.rsplit("/", 1)[-1] for u in spec["urls"]]}
        else:
            # No holdout applied. Recorded rather than omitted: a source with no offset is
            # reading the same shards healing did, and that must be visible in the manifest
            # instead of inferred from a missing key.
            effective[name] = {"mode": "NO OFFSET APPLIED", "offset": 0}
        SOURCES[f"_heldout_{name}"] = spec
        docs, got = take_tokens(f"_heldout_{name}", int(target_tokens * frac), tok, seed=seed)
        for d in docs:
            d["source"] = name
        rows.extend(docs)
        actual[name] = got

    assert_disjoint(rows, heal_hashes)
    print("  disjointness verified: 0 shared documents with the healing corpus", flush=True)

    rng = random.Random(seed)
    rng.shuffle(rows)
    _write(out_path, rows)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    total = sum(actual.values())
    print(f"  sha256: {digest}", flush=True)
    return {
        "path": str(out_path),
        "sha256": digest,
        "tokens": total,
        "documents": len(rows),
        "composition": actual,
        "requested_mix": mix,
        "shard_offsets_declared": {**HELDOUT_SHARD_OFFSET,
                                   "algebraic_files": list(HELDOUT_ALGEBRAIC)},
        "shards_actually_read": effective,
        "held_out_from": str(heal_path),
        "heal_documents_excluded": len(heal_hashes),
        "warning": "This file is the project's KL yardstick. Once a reference .kld is built "
                   "from it, it must never change -- every child measurement is relative to "
                   "these exact bytes.",
    }
if __name__ == "__main__":
    sys.exit(main())
