# Dataset

## Provenance

| | |
|---|---|
| **Corpus** | LiveCareer resume corpus |
| **Obtained from** | <https://huggingface.co/datasets/Darshan-04/Resume-classification> |
| **Declared licence** | MIT (on the hub repository) |
| **Upstream origin** | Mirror of the Kaggle dataset `snehaanbhawal/resume-dataset`, scraped from publicly published resume examples on livecareer.com |
| **Records** | 2,484 |
| **Format** | CSV: `ID`, `Resume_str`, `Resume_html`, `Category` |
| **Retrieval** | Direct HTTPS, no credentials — `scripts/download_corpus.py` |
| **Integrity** | SHA-256 verified at download; recorded in `MANIFEST.json` |

The `Resume_html` column is deliberately ignored: it duplicates the text column
plus markup that would pollute the feature space.

### Things to be aware of

- **This is real personal data.** Scraped from real people's published resumes,
  containing names, e-mail addresses, telephone numbers and employment history.
  `data/raw/` is git-ignored and must stay that way. PII is redacted before any
  feature is computed.
- **Verify the licence yourself** before any commercial use. The hub repository
  declares MIT; the ultimate upstream terms are livecareer.com's.
- **The labels are embedded in the documents.** LiveCareer resumes print the job
  title as a header, so a document partly announces its own category. See
  *Header ablation* below.

### Why not something bigger?

No larger legitimate public resume corpus exists. The widely circulated
alternative (`UpdatedResumeDataSet.csv`, 962 rows) is heavily duplicated and has
poorly documented provenance. Datasets advertising tens of thousands of resumes
are predominantly LLM-generated, which this project excludes.

**~2,500 real documents is the honest ceiling from public sources.**

---

## Schema

One row per resume, stored as JSON Lines (`corpus.jsonl`). JSONL, not CSV:
resume text is full of embedded newlines and quotes, which is exactly what broke
the predecessor project's CSV.

| Field | Type | Purpose |
|---|---|---|
| `record_id` | str | `sha256(source + "::" + source_id)[:16]` — stable across rebuilds |
| `source` / `source_version` / `source_id` | str | Provenance |
| `resume_text` | str | Ingestion-normalised text |
| `label` | str | Canonical taxonomy class id |
| `label_raw` | str | Original label, **never modified** — keeps mapping auditable |
| `text_sha256` | str | Hash of aggressively normalised text — the dedup key |
| `text_length_chars` / `text_length_words` | int | Profiling and gating |
| `duplicate_group_id` | str \| null | Cluster membership, when non-trivial |
| `ingested_at` | str | ISO 8601 UTC |

`record_id` is derived from content, not row order, so rebuilding never
renumbers anything and split assignments stay valid.

---

## Build pipeline

```
rc-build-dataset
```

```
raw rows → validate → deduplicate → split → corpus.jsonl + splits.json
```

Outputs land in `data/processed/v1/`:

| File | Contents |
|---|---|
| `corpus.jsonl` | Validated records |
| `splits.json` | `record_id` → train / validation / test |
| `quarantine.jsonl` | Every rejected row, with a reason code |
| `report.json` / `report.md` | Profiling and quality report |
| `MANIFEST.json` | Raw hashes, git commit, seed, thresholds, output hashes |

### Validation gate

Nothing is dropped silently. Every rejection carries a code, and the build
asserts `read == kept + quarantined` before finishing.

| Code | Meaning |
|---|---|
| `LABEL_MISSING` | No category on the source row |
| `LABEL_UNMAPPED` | Label not in the taxonomy — a **build failure signal**, not a silent drop |
| `TEXT_EMPTY` | No text after normalisation |
| `TEXT_TOO_SHORT` | Below `RC_MIN_RESUME_CHARS` / `RC_MIN_RESUME_WORDS` |
| `ID_DUPLICATE` | Colliding record id |
| `EXACT_DUPLICATE` | Identical normalised content |
| `NEAR_DUPLICATE` | Cosine similarity ≥ `RC_NEAR_DUPLICATE_THRESHOLD` |

### Deduplication

Two tiers, both **before** splitting:

1. **Exact** — SHA-256 over aggressively normalised text.
2. **Near** — character 5-gram TF-IDF cosine ≥ 0.95, clustered with union-find,
   computed in row blocks so memory stays bounded.

Each cluster collapses to one representative (lowest `record_id`); the rest are
quarantined with their cluster id. Auditable, not blind deletion.

On this corpus: **2 exact** duplicates and **13 near** duplicates. The near
duplicates are the interesting ones — hashing alone would have left all 13 free
to straddle the train/test boundary.

### Splitting

70 / 15 / 15, stratified by label, seed 42, persisted by `record_id` in
`splits.json` and never regenerated on the fly. Every one of the 24 classes
appears in all three splits.

---

## Current build

| | |
|---|---|
| Rows read | 2,484 |
| Kept | 2,468 |
| Quarantined | 16 (1 empty, 2 exact dup, 13 near dup) |
| Classes | 24 |
| Train / Validation / Test | 1,727 / 370 / 371 |
| Median length | 5,538 chars · 751 words |
| Class balance | 22 (BPO) to 119 (Information Technology) — 5.4× |

Full detail in `data/processed/v1/report.md`.

---

## Taxonomies

Both are TOML, committed to git, and read with the standard library's
`tomllib` — human-curatable at zero dependency cost.

### Categories — `data/taxonomy/categories_v1.toml`

24 industry-level classes. Every source label maps to exactly one; an unmapped
label fails the build. There is deliberately **no `other` class** — a
heterogeneous bucket has no coherent decision boundary and drags a linear model
down.

### Skills — `data/taxonomy/skills_v1.toml`

170 skills across 21 categories. Curated reproducibly:

1. Mine discriminative unigrams and bigrams **from the TRAIN split only** —
   mining the full corpus would leak test vocabulary into a visible feature.
2. Keep terms naming a transferable, checkable capability.
3. Discard job titles, employer names, generic filler.
4. Group surface forms under one canonical id via `aliases`.

Multi-domain by necessity: the corpus spans 24 industries, so an IT-only skill
list would return nothing for two thirds of it.

Coverage on the training split: **99.2%** of documents match at least one skill,
median 8 per document.

---

## Header ablation

LiveCareer resumes print the job title at the top, so the top discriminative
term for nearly every class is that class's own name. That is legitimate signal
— a real uploaded resume has a title too — but it makes the task easier than it
looks, and leaving it unmeasured would be dishonest.

```bash
rc-train --strip-header-chars 300
```

This re-runs the entire protocol with the first 300 characters removed from
every document, and writes the result to `models/ablations/` rather than the
serving version namespace -- an ablation is a measurement, not a release.

**Result on this corpus:**

| Run | Test macro-F1 | Test accuracy |
|---|---:|---:|
| Full document | 0.5731 | 0.6199 |
| First 300 chars removed | 0.5590 | 0.6011 |
| **Difference** | **-0.0141** | **-0.0188** |

The drop is small. The model is therefore *not* simply reading the job title off
the top of the page: nearly all of its performance comes from the body of the
document. That was worth measuring rather than assuming, in either direction.

---

## Reproducing

```bash
python scripts/download_corpus.py   # SHA-256 verified
rc-build-dataset                    # deterministic given the same raw bytes
```

Rebuilding from identical raw input produces byte-identical `corpus.jsonl` and
`splits.json`: record ids are content-derived, iteration is sorted, JSON keys
are sorted and newlines are forced to `\n`.

The manifest ties everything together — raw file hashes, git commit, taxonomy
version, seed, thresholds, and output hashes — so any model can be traced back
to the exact bytes it was trained on.
