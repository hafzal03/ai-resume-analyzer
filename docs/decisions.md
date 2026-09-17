# Design decisions

Each entry records what was decided, why, and what it cost. Several exist
because the predecessor system made the opposite choice and broke.

---

## 1. Two levels of text normalisation, one shared function

**Decision.** `ingestion_normalize` (conservative, stored) and
`feature_normalize` (aggressive, never stored). The fitted vectorizer carries
`preprocessor=feature_normalize` *inside* the pickled artifact.

**Why.** Training and serving cannot drift apart, because they are the same
function object travelling inside the same file. The previous system shipped a
vectorizer with 37 features while its dataset produced 46, and nothing noticed.

**Cost.** The artifact depends on the package being importable at load time.
Acceptable: it is our own package.

---

## 2. Punctuation is preserved selectively

**Decision.** `+ # . -` survive normalisation; leading punctuation and trailing
`. _ -` are stripped.

**Why.** `c++`, `c#`, `f#`, `node.js` and `back-end` are all skills. A naive
stripper turns `c++` into `c` and loses them.

**Subtlety found during implementation.** A one-character lookbehind is not
enough. Inside `c++` it succeeds at the *second* `+` (whose neighbour is
punctuation, not a letter) and silently eats it. Both rules are anchored to token
boundaries instead. There is a regression test for exactly this.

---

## 3. PII redaction sits at the feature level

**Decision.** E-mail addresses, URLs, phone numbers and long digit runs become
placeholder tokens before any feature is computed.

**Why.** Two reasons, and the second is the less obvious one. Privacy is
expected. But a candidate's name or e-mail is also a *fingerprint*: unredacted,
it lets the model recognise a person who appears in more than one split rather
than learn the category. Redaction is a leakage control as much as a privacy
control.

---

## 4. Deduplicate before splitting, collapse rather than delete

**Decision.** Exact (hash) then near (char 5-gram cosine ≥ 0.95), both before
the split. Clusters collapse to one representative; the rest are quarantined
with their cluster id.

**Why.** A resume in both train and test inflates every metric. On this corpus
hashing alone would have missed 13 of the 15 duplicates.

**Cost.** O(n²) similarity. Fine at 2,500 documents, computed in blocks to bound
memory; would need MinHash/LSH past ~50,000.

---

## 5. JSONL, not CSV, for the corpus

**Decision.** Newline-delimited JSON.

**Why.** Resume text is full of embedded newlines and quotes. CSV ambiguity is
precisely what made the predecessor's dataset unreliable. JSONL is streamable,
line-diffable and unambiguous, and needs no `pyarrow`.

---

## 6. Data is git-ignored; manifests are committed

**Decision.** `data/raw/`, `data/processed/` and `models/` are excluded from
git. Their `MANIFEST.json` files are not.

**Why.** Size, third-party licensing, and personal data all argue against
committing the bytes. But provenance must be under version control, so the
manifests — raw file hashes, git commit, taxonomy version, seed, thresholds,
output hashes — are tracked instead. Any model traces back to the exact bytes it
was trained on.

---

## 7. Taxonomies in TOML, read with `tomllib`

**Decision.** Category and skill vocabularies are TOML files, not Python
literals and not YAML.

**Why.** They are curated by humans and reviewed in diffs, so they belong in
data, not code. TOML is readable, and `tomllib` is in the standard library —
YAML would have cost a dependency for no gain.

---

## 8. No `other` category

**Decision.** Unmapped labels fail the build rather than landing in a catch-all.

**Why.** A heterogeneous bucket has no coherent decision boundary; it becomes a
dumping ground that drags the whole model down. A corpus that grows a new
category should force a taxonomy decision, loudly.

---

## 9. Model selection compares tuned models, not default ones

**Decision.** Cross-validate all families at defaults, then hyperparameter-tune
the top two and re-compare before choosing.

**Why.** Comparing families at arbitrary fixed hyperparameters can crown the
wrong winner — a family that responds well to tuning can start behind and finish
ahead.

**Cost.** Training takes 15–25 minutes rather than 12.

---

## 10. The production model is fitted on TRAIN only

**Decision.** Deliberately *not* refitted on TRAIN+VALIDATION after selection.

**Why.** The abstention threshold was tuned on validation against a TRAIN-fitted
model. Refitting would make that threshold an in-sample quantity and the one
honest uncertainty estimate the system has would become decorative.

**Cost.** 370 documents of training signal. With 1,727 already in hand, not
worth the trade.

---

## 11. Confidence uses probability *and* margin

**Decision.** A prediction is committed to only if top probability ≥ threshold
**and** (top − runner-up) ≥ margin. Both tuned on validation, subject to an 80%
coverage floor.

**Why.** The two fail differently. A model can be 0.45 confident and decisive
(runner-up 0.05) or 0.45 confident and genuinely torn (runner-up 0.43).
Probability alone cannot distinguish them. The coverage floor exists because
abstaining on everything would be perfectly accurate and perfectly useless.

---

## 12. Persistence is optional; database failure degrades, never blocks

**Decision.** No `RC_DATABASE_URL` means no persistence and a normal service. A
database that is configured but unreachable logs a warning and the service still
starts. A mid-request failure adds a warning to the result.

**Why.** Classification does not depend on recording it. Coupling them would
turn a database outage into an outage.

---

## 13. Resume text is not stored by default

**Decision.** The schema stores a SHA-256, the prediction, and document
statistics. Text requires `RC_STORE_RESUME_TEXT=true`. There is no filename
column.

**Why.** Data minimisation. The questions the table needs to answer — what was
predicted, by which model version, can it be reproduced — need no document.
Filenames routinely contain candidate names.

---

## 14. No `age` column, or any protected characteristic

**Decision.** Removed entirely.

**Why.** The predecessor schema had one. It was always `NULL` because no form
ever collected it — but an automated hiring-adjacent decision stored beside a
protected characteristic is a discrimination liability with no offsetting
benefit.

---

## 15. Uploads are processed in memory and never written to disk

**Decision.** No upload directory.

**Why.** The predecessor wrote every upload to `uploads/` using the raw
client-supplied filename — a path-traversal exposure and an unbounded store of
personal data. Neither is needed: PDF parsing works from a byte buffer.

---

## 16. Models are mounted into the container, not baked in

**Decision.** `models/` is a read-only bind mount; the Dockerfile ships no
artifact.

**Why.** An image with an embedded model can silently disagree with the dataset
in the repository. Mounting keeps one obvious source of truth, and keeps the
image small.

---

## 17. waitress instead of gunicorn

**Decision.** waitress as the production WSGI server.

**Why.** gunicorn does not run on Windows, and this project is developed on
Windows and deployed to Linux. waitress runs on both, so development and
production use the same server.

---

## 18. "No external AI" is a test, not a promise

**Decision.** `tests/test_no_external_ai.py` asserts that no provider SDK is
importable, that none appears in declared dependencies, and that no source file
references one — including the serving path making no network calls at all.

**Why.** A constraint that only lives in a README erodes. One that fails CI does
not.

---

## 19. `.env` is shared with Compose, so foreign keys are tolerated

**Decision.** `Settings` uses `extra="ignore"` rather than `extra="forbid"`, and
strictness for the `RC_` namespace is enforced by two explicit checks instead:
one over `os.environ`, one over the dotenv file (`unknown_dotenv_keys`).
Compose-only variables use their own prefix (`POSTGRES_*`, `COMPOSE_*`) and
never `RC_`.

**Why.** A project's `.env` is read by *both* the application and Docker
Compose, and Compose requires its substitution variables in that same file.
With `extra="forbid"`, `cp .env.example .env` made the application refuse to
start -- `POSTGRES_DB` was an "extra input". Worse, the template itself had put
`RC_BIND_HOST` in the application's namespace, where it was correctly rejected
as an unknown setting.

Found by running the documented setup path end to end; the regression was caught
by the existing test suite the moment a real `.env` existed.

**Cost.** A genuinely unknown non-`RC_` key is now silently ignored. Accepted:
that namespace is not ours to police. What matters is that a mistyped *setting*
still fails loudly, which both explicit checks preserve.

`tests/test_config.py::test_shipped_env_example_is_loadable` now asserts the
template stays loadable, so this cannot regress unnoticed.
