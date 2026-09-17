"""Flask application factory.

The factory pattern keeps construction explicit and testable: a test builds an
app with its own settings and its own service, with no global state and no
import-time side effects.

Debug mode is never enabled from here. It follows configuration, and
configuration refuses to enable it in production.
"""

from __future__ import annotations

import logging

from flask import Flask

from resume_classifier.api.errors import register_error_handlers
from resume_classifier.api.routes import api, web
from resume_classifier.config import Settings, get_settings
from resume_classifier.logging_config import configure_logging
from resume_classifier.service import ResumeAnalysisService, build_service

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    service: ResumeAnalysisService | None = None,
) -> Flask:
    """Build the application.

    Args:
        settings: Configuration. Defaults to the process-wide settings.
        service: Pre-built service, injected by tests to avoid loading a model
            from disk.
    """
    resolved = settings or get_settings()
    configure_logging(resolved)

    app = Flask(__name__, template_folder="templates", static_folder=None)

    # Werkzeug rejects oversized uploads before they reach application code.
    app.config["MAX_CONTENT_LENGTH"] = resolved.max_upload_bytes
    app.config["JSON_SORT_KEYS"] = False
    app.config["SETTINGS"] = resolved

    app.extensions["resume_service"] = service or build_service(resolved)

    app.register_blueprint(api)
    app.register_blueprint(web)
    register_error_handlers(app)

    logger.info(
        "application ready environment=%s debug=%s model=%s persistence=%s",
        resolved.environment.value,
        resolved.debug,
        app.extensions["resume_service"].engine.model_version,
        app.extensions["resume_service"].persistence_enabled,
    )
    return app
