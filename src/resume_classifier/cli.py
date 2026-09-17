"""Command-line entry points.

Installed as console scripts by ``pyproject.toml``:

    rc-build-dataset   build the corpus from the raw source
    rc-train           run model selection and persist the winner
    rc-serve           run the web application
    rc-info            show what is installed, built and trained
    rc-classify        classify a single file or text from the shell

Each command prints something useful on failure rather than a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from resume_classifier.config import Settings, get_settings
from resume_classifier.logging_config import configure_logging


def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings)
    return settings


def build_dataset_main(argv: list[str] | None = None) -> int:
    """Entry point for ``rc-build-dataset``."""
    parser = argparse.ArgumentParser(description="Build the processed corpus from raw data.")
    parser.add_argument(
        "--source",
        default="livecareer_resumes",
        help="registered source adapter name (default: livecareer_resumes)",
    )
    args = parser.parse_args(argv)

    from resume_classifier.dataset import build_dataset
    from resume_classifier.dataset.ingest import IngestError

    settings = _settings()
    try:
        result = build_dataset(settings, source=args.source)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.summary())
    print(f"report: {result.output_dir / 'report.md'}")
    return 0


def train_main(argv: list[str] | None = None) -> int:
    """Entry point for ``rc-train``."""
    parser = argparse.ArgumentParser(description="Train and select a classification model.")
    parser.add_argument(
        "--strip-header-chars",
        type=int,
        default=0,
        metavar="N",
        help=(
            "ablation: drop the first N characters of every document, to measure "
            "how much the model relies on the job title in the header"
        ),
    )
    args = parser.parse_args(argv)

    from resume_classifier.models.train import train

    settings = _settings()
    try:
        metadata = train(settings, strip_header_chars=args.strip_header_chars)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    test = metadata["metrics"]["test"]
    print()
    print(f"model {metadata['version']} saved")
    print(f"  estimator   : {metadata['selected']['estimator']}")
    print(f"  features    : {metadata['selected']['features']['name']}")
    print(f"  test macro-F1: {test['macro_f1']}   accuracy: {test['accuracy']}")
    return 0


def serve_main(argv: list[str] | None = None) -> int:
    """Entry point for ``rc-serve``."""
    parser = argparse.ArgumentParser(description="Run the web application.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="use the Flask development server instead of waitress",
    )
    args = parser.parse_args(argv)

    from resume_classifier.api.app import create_app
    from resume_classifier.models.artifacts import ModelNotFoundError

    settings = _settings()
    host = args.host or settings.host
    port = args.port or settings.port

    try:
        app = create_app(settings)
    except ModelNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dev:
        app.run(host=host, port=port, debug=settings.debug)
        return 0

    from waitress import serve

    print(f"serving on http://{host}:{port}")
    serve(app, host=host, port=port, threads=8)
    return 0


def info_main(argv: list[str] | None = None) -> int:
    """Entry point for ``rc-info``: report what exists and what is missing."""
    argparse.ArgumentParser(description="Show project status.").parse_args(argv)

    import resume_classifier
    from resume_classifier.models.artifacts import list_versions

    settings = _settings()
    corpus = settings.dataset_dir / "corpus.jsonl"
    versions = list_versions(settings.models_dir)

    print(f"resume-classifier {resume_classifier.__version__}")
    print(f"  environment : {settings.environment.value}")
    print(f"  data dir    : {settings.data_dir}")
    print(f"  models dir  : {settings.models_dir}")
    print()
    print(f"  raw corpus  : {'present' if settings.raw_data_dir.is_dir() else 'MISSING'}")
    if corpus.is_file():
        records = sum(1 for _ in corpus.open(encoding="utf-8"))
        print(f"  dataset     : {settings.dataset_version} ({records:,} records)")
    else:
        print("  dataset     : NOT BUILT -- run 'rc-build-dataset'")
    print(f"  models      : {', '.join(versions) if versions else 'NONE -- run rc-train'}")
    print(f"  persistence : {'enabled' if settings.database_enabled else 'disabled'}")
    return 0


def classify_main(argv: list[str] | None = None) -> int:
    """Entry point for ``rc-classify``: classify one document from the shell."""
    parser = argparse.ArgumentParser(description="Classify a resume.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="path to a PDF")
    source.add_argument("--text", help="raw resume text")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    from resume_classifier.extraction.pdf import PdfExtractionError
    from resume_classifier.inference.engine import TextTooShortError
    from resume_classifier.models.artifacts import ModelNotFoundError
    from resume_classifier.service import build_service

    settings = _settings()
    try:
        service = build_service(settings)
        outcome = (
            service.analyse_pdf(args.file.read_bytes())
            if args.file
            else service.analyse_text(args.text or "")
        )
    except (ModelNotFoundError, PdfExtractionError, TextTooShortError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(outcome.to_dict(), indent=2))
        return 0

    result = outcome.result
    print(f"prediction : {result.display_label}")
    print(f"confidence : {result.confidence:.3f} (margin {result.margin:.3f})")
    print(f"skills     : {', '.join(s.display for s in result.skills) or '(none)'}")
    for warning in result.warnings:
        print(f"warning    : {warning}")
    return 0


def build_taxonomy_main(argv: list[str] | None = None) -> int:
    """Entry point for ``rc-build-taxonomy``: validate the taxonomy files."""
    argparse.ArgumentParser(description="Validate taxonomy files.").parse_args(argv)

    from resume_classifier.dataset.taxonomy import TaxonomyError, load_taxonomy
    from resume_classifier.skills.taxonomy import SkillTaxonomyError, load_skill_taxonomy

    settings = _settings()
    try:
        categories = load_taxonomy(settings.taxonomy_dir / "categories_v1.toml")
        skills = load_skill_taxonomy(settings.taxonomy_dir / "skills_v1.toml")
    except (TaxonomyError, SkillTaxonomyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"categories : v{categories.version}, {len(categories.classes)} classes -- valid")
    print(
        f"skills     : v{skills.version}, {len(skills.skills)} skills across "
        f"{len(skills.categories)} categories -- valid"
    )
    return 0
