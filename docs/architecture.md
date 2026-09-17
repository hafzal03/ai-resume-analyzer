# Architecture

## The shape of the system

Two execution contexts share one code library. This separation is the single
most important structural decision in the project.

```
OFFLINE (batch, reproducible)              ONLINE (stateless service)
──────────────────────────────             ──────────────────────────────
data/raw/<source>/<version>/               HTTP request (PDF or text)
  immutable, checksummed                          │
        │                                         ▼
        ▼  ingest + validate               extraction/  PDF → text
        ▼  deduplicate  ← before split            │
        ▼  stratified split                       ▼
        ▼  ┌──────────────────────┐        preprocessing/  ← SHARED
        │  │ preprocessing/       │               │
        │  │ skills/              │  shared       ▼
        │  │ features/            │  ────────►  features/  ← SHARED
        │  └──────────────────────┘               │
        ▼  model selection + tuning               ▼
        ▼  evaluate once on TEST           classifier → probabilities
        ▼                                         │
 models/v<N>/                                     ▼
   pipeline.joblib                         inference/confidence
   metadata.json  ───────── loaded ──────►   label | "Uncertain"
                                                  │
                                                  ▼
                                            db/  (optional)
                                                  │
                                                  ▼
                                            PostgreSQL
```

## The load-bearing invariant

`preprocessing/text.py` is imported by **both** contexts, and the fitted
vectorizer carries `preprocessor=feature_normalize` *inside* the pickled
artifact. Training and serving therefore cannot disagree about how text is
normalised — not by convention, but by construction.

This is the defect that broke the predecessor system: its shipped vectorizer had
37 features while its own dataset produced 46, and nothing detected the drift.

## Module responsibilities

| Module | Owns | Must not |
|---|---|---|
| `config.py` | Env-driven settings, validated at startup | Contain literals or secrets |
| `preprocessing/` | The two normalisation levels | Know about PDFs, models or the DB |
| `extraction/` | PDF → text, with typed failures | Know about ML or the DB |
| `skills/` | Taxonomy + deterministic matcher | Call a model |
| `features/` | Vectoriser construction | Fit on anything but TRAIN |
| `dataset/` | Ingest, validate, dedup, split, report | Know about HTTP |
| `models/` | Train, evaluate, version, load | Import Flask |
| `inference/` | Runtime pipeline + confidence policy | Import training code or pandas |
| `db/` | ORM, sessions, repositories | Contain business logic |
| `service.py` | Orchestration | Know about HTTP status codes |
| `api/` | Routes, error envelope, templates | Contain business logic |

## Two levels of text normalisation

| Level | Function | Stored? | Does |
|---|---|---|---|
| Ingestion | `ingestion_normalize` | ✅ as `resume_text` | NFKC, control/zero-width removal, whitespace collapse. **Keeps case and punctuation.** |
| Feature | `feature_normalize` | ❌ never | casefold, PII redaction, digit tokens, punctuation cleanup |

Storing close to the source means feature preprocessing can change without
re-ingesting the corpus.

PII redaction sits at the feature level for two reasons. Privacy is the obvious
one. The subtler one is leakage: a candidate's name or e-mail is a fingerprint,
and an unredacted fingerprint lets a model recognise a *person* who appears in
more than one split instead of learning the category.

## Confidence and abstention

Two signals gate every prediction, because they fail differently:

- **Top probability** — how sure the model is.
- **Margin** — how far ahead the leader is. A 0.45/0.05 split is decisive; a
  0.45/0.43 split is a coin flip. Probability alone cannot tell them apart.

Both are tuned on the **validation** split, never on test, subject to a coverage
floor — abstaining on everything would be perfectly accurate and perfectly
useless.

## Leakage controls

| Vector | Control | Enforced by |
|---|---|---|
| Same resume in train and test | Exact + near dedup **before** split | `dataset/dedup.py` |
| Duplicate cluster straddling splits | Clusters collapsed to one representative | `dataset/build.py` |
| Vectoriser fitted on all data | Fitted inside the pipeline on TRAIN only | `models/train.py` |
| Skill vocabulary mined from all data | Mined from TRAIN only | documented in `skills_v1.toml` |
| Person identifiable across splits | PII redaction | `preprocessing/text.py` |
| Test used for tuning | Sealed until stage 5 | `models/train.py` |
| Reshuffling between runs | Splits persisted by `record_id` | `splits.json` |

Three of these are assertions in `tests/test_dataset.py`, so a refactor cannot
quietly reintroduce contamination.

## Why the production model is fitted on TRAIN only

It would be conventional to refit on TRAIN+VALIDATION for the extra data. That
is deliberately not done here: the abstention threshold was tuned on validation
against a TRAIN-fitted model, and refitting would make the threshold an
in-sample quantity. With 1,727 training documents the extra 370 are not worth
compromising the one honest estimate the system has.

## Degradation behaviour

| Failure | Behaviour |
|---|---|
| No `RC_DATABASE_URL` | Classifies normally, records nothing |
| Database configured but unreachable | Starts anyway, logs a warning, persistence off |
| Database fails mid-request | Result returned with a warning; never lost |
| No trained model | `503 MODEL_UNAVAILABLE`, with instructions |
| Scanned PDF | `422 PDF_NO_TEXT` — refused, never guessed from empty text |
| Oversized upload | `413`, rejected by Werkzeug before application code |

## No external AI

No request path touches an external model provider. The container classifies
correctly with all outbound network access blocked — `tests/test_no_external_ai.py`
asserts the dependency tree and source contain no provider SDK.
