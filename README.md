# Resume Classifier

Classifies a resume into one of **24 professional categories** and extracts the
skills it mentions — using classical NLP and machine learning, running entirely
on your own machine.

Upload a PDF (or paste text), get a category, a confidence score, the ranked
alternatives, and a list of recognised skills. When the evidence is weak the
answer is **Uncertain** rather than a forced guess.

---

## No AI services, no paid dependencies

This project does **not** call OpenAI, Anthropic, Gemini, Mistral, Groq, Cohere,
or any hosted inference API, and uses no LangChain, LangGraph, RAG, MCP, agent
framework or prompt-based classification.

The intelligence is a scikit-learn model trained on a labelled corpus, plus a
deterministic skill taxonomy. **The application classifies correctly with all
outbound network access blocked.** That is enforced by
`tests/test_no_external_ai.py`, not just promised.

Every dependency is free and open source.

---

## Quick start

You need **Python 3.11+** and about 500 MB of disk space.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # macOS / Linux
pip install -e ".[train,dev]"
```

Check what is present:

```bash
rc-info
```

### 1. Get the corpus

The training data is **not** in this repository (size, licensing, and it
contains real people's resumes). Download it once:

```bash
python scripts/download_corpus.py
```

This fetches the LiveCareer resume corpus (2,484 resumes, MIT-licensed mirror),
writes it to `data/raw/livecareer_resumes/v1/`, and records a SHA-256 manifest.
See [docs/dataset.md](docs/dataset.md) for provenance and licensing.

### 2. Build the dataset

```bash
rc-build-dataset
```

Validates, deduplicates and splits the corpus, then writes a quality report to
`data/processed/v1/report.md`.

### 3. Train a model

```bash
rc-train
```

Runs baselines, cross-validated model selection, hyperparameter tuning,
confidence-policy tuning, and one final held-out evaluation. Takes 15–25
minutes. The model lands in `models/v1/`.

### 4. Run it

```bash
rc-serve
```

Open <http://127.0.0.1:5000>.

---

## Command reference

| Command | Does |
|---|---|
| `rc-info` | Show what is installed, built and trained |
| `rc-build-taxonomy` | Validate the category and skill taxonomies |
| `rc-build-dataset` | Build the processed corpus from raw data |
| `rc-train` | Train, select and evaluate a model |
| `rc-train --strip-header-chars 300` | Header ablation (see below) |
| `rc-serve` | Run the web application |
| `rc-serve --dev` | Run with the Flask dev server instead of waitress |
| `rc-classify --file cv.pdf` | Classify one PDF from the shell |
| `rc-classify --text "..." --json` | Classify text, emit JSON |

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/api/v1/health` | Liveness (checks nothing else) |
| `GET` | `/api/v1/ready` | Readiness: model loaded, DB reachable |
| `GET` | `/api/v1/model` | Active model version, metrics, provenance |
| `GET` | `/api/v1/categories` | The 24 classes |
| `GET` | `/api/v1/skills` | The skill vocabulary |
| `POST` | `/api/v1/classify` | multipart PDF, field `resume` |
| `POST` | `/api/v1/classify/text` | JSON `{"text": "..."}` |

```bash
curl -X POST http://127.0.0.1:5000/api/v1/classify \
     -F "resume=@my_resume.pdf"
```

Errors always use one envelope, and never leak a traceback:

```json
{
  "error": { "code": "PDF_NO_TEXT", "message": "no extractable text found..." },
  "request_id": "a1b2c3d4"
}
```

| Status | When |
|---|---|
| `400` | Malformed request or no input |
| `413` | File or page count over the limit |
| `415` | Not a PDF (checked by magic bytes, not by filename) |
| `422` | Encrypted PDF, scanned PDF, or text too short |
| `503` | No trained model available |

---

## Docker

```bash
cp .env.example .env
# edit .env and set POSTGRES_PASSWORD
docker compose up --build
```

The stack is PostgreSQL 16 plus the web service. Startup ordering is enforced:
`db` becomes healthy, then `migrate` runs Alembic and exits, and only then does
`web` start.

Train a model *before* `docker compose up` — `models/` is mounted read-only
rather than baked into the image, so an image can never ship a stale model.

**Verified behaviour** (Docker 28.3.2, linux containers):

| Property | Result |
|---|---|
| Image size | 471 MB |
| Runs as | `appuser` (uid 1001), not root |
| Database port | **not published** — reachable only on the internal network |
| Credentials | no defaults; the stack refuses to start without them |
| `models/` mount | read-only; writes are refused |
| Corpus / secrets in image | none — `data/raw`, `data/processed` empty, no `.env` |
| Training deps in runtime image | none — `pandas`, `matplotlib`, `pytest` all absent |
| Data survives `docker compose down` | yes, via the named volume |
| Migrations on restart | idempotent, no re-application |
| Errors in production mode | typed codes (`415`, `422`, `413`), no traceback leaked |

**Classifies with no network at all.** The headline guarantee, tested directly:

```bash
docker run --rm --network none   -v "$PWD/models:/app/models:ro"   -v "$PWD/data/taxonomy:/app/data/taxonomy:ro"   ai-resume-analyzer-main-web   rc-classify --text "EXECUTIVE CHEF. Fifteen years leading kitchens..."
# prediction : Chef / Culinary
# confidence : 0.828
```

Stop the stack with `docker compose down` (add `-v` to also delete the database
volume).

---

## Configuration

Everything is read from `RC_*` environment variables, optionally via `.env`.
Nothing is hard-coded. Unknown `RC_*` variables are **rejected at startup**, so
a typo fails loudly instead of silently using a default.

See [.env.example](.env.example) for the full annotated list. The ones that
matter most:

| Variable | Default | Notes |
|---|---|---|
| `RC_ENVIRONMENT` | `development` | `development` / `testing` / `production` |
| `RC_DEBUG` | `false` | **Refuses to be true in production** |
| `RC_DATABASE_URL` | unset | Persistence is off unless set |
| `RC_STORE_RESUME_TEXT` | `false` | Leave false; see Privacy |
| `RC_MAX_UPLOAD_BYTES` | 10 MB | |
| `RC_MODEL_VERSION` | newest | Pin a specific model |

---

## Privacy

Resumes are personal data. The defaults reflect that:

- **Resume text is not stored.** Only a SHA-256 hash, the prediction, and
  document statistics. Storing text requires explicitly setting
  `RC_STORE_RESUME_TEXT=true`.
- **Uploaded files are never written to disk.** They are processed in memory.
- **No filename column** — filenames routinely contain candidate names.
- **No age column, or any other protected characteristic.** An automated
  hiring-adjacent decision beside a protected attribute is a liability with no
  offsetting benefit.
- **PII is redacted before features are computed** — e-mail addresses, phone
  numbers and URLs become placeholder tokens.
- **Logs record decisions, never resume content.**

---

## How well does it actually work?

See `models/v1/metadata.json` for the numbers from your own training run, or
`GET /api/v1/model`. Honest framing:

- **Macro-F1 is the headline metric**, not accuracy. The corpus has a 5.4×
  class imbalance, so accuracy alone would hide a class failing completely.
- **Baselines are reported alongside.** A stratified-random baseline scores
  about 0.05 macro-F1 on 24 classes; that is the number real performance should
  be read against.
- **The classes genuinely overlap.** "Finance" versus "Banking" versus
  "Accountant" are not cleanly separable from resume text, and the model
  confuses them the way a person would.
- **The corpus embeds its own labels** -- and it was measured, not assumed.
  LiveCareer resumes print the job title as a header, so a document partly
  announces its own category. Re-running the whole protocol with the first 300
  characters stripped (`rc-train --strip-header-chars 300`) costs only
  **0.0141 macro-F1** (0.5731 -> 0.5590), so nearly all the model's performance
  comes from the body of the resume rather than its header.

The test split is opened exactly once, at the end of training. It is never used
for model selection or threshold tuning.

---

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest
pytest --cov              # with coverage
pytest -m "not slow"      # skip the slow ones
```

Tests that need a built corpus or a trained model skip themselves cleanly when
those are absent, so a fresh clone can run the suite immediately.

---

## Project layout

```
src/resume_classifier/
├── config.py              env-driven settings, validated at startup
├── logging_config.py      stdout logging
├── cli.py                 console entry points
├── preprocessing/         text normalisation  ← SHARED by training and serving
├── extraction/            PDF → text, with typed failures
├── skills/                taxonomy + deterministic matcher
├── features/              vectoriser construction
├── dataset/               ingest, validate, dedup, split, report
├── models/                train, evaluate, version, load
├── inference/             runtime pipeline + confidence policy
├── db/                    ORM, sessions, repositories
├── service.py             orchestration
└── api/                   routes, error envelope, templates

data/taxonomy/             version-controlled, committed
data/raw/                  immutable corpora      (git-ignored)
data/processed/            built dataset          (git-ignored)
models/                    trained artifacts      (git-ignored)
alembic/                   database migrations
docs/                      architecture, dataset, decisions
```

Data and models are git-ignored; their **manifests are committed**, so
provenance lives in version control even though the bytes do not.

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the pieces fit, and why
- [docs/dataset.md](docs/dataset.md) — corpus provenance, licensing, schema
- [docs/decisions.md](docs/decisions.md) — design decisions and their reasons

---

## Known limitations

- **Industry-level, not role-level.** The corpus cannot distinguish "Java
  Developer" from "DevOps Engineer" — both are Information Technology.
- **~2,500 training documents.** No larger legitimate public resume corpus
  exists; the bigger ones in circulation are LLM-generated.
- **English only.**
- **No OCR.** Scanned PDFs are refused, not guessed at.
- **Thin minority classes.** BPO has 22 documents; its per-class metrics are
  correspondingly noisy.
