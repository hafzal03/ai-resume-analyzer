"""Initial schema: analyses and analysis_skills.

Revision ID: 0001
Revises:
Create Date: 2026-09-16

Deliberate omissions, for privacy and legal reasons:
  * no uploaded filename column (filenames routinely contain candidate names)
  * no age or any other protected characteristic
  * ``resume_text`` is nullable and only populated when RC_STORE_RESUME_TEXT is
    explicitly enabled; the default records a hash instead of the document.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("predicted_label", sa.String(length=64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("margin", sa.Float(), nullable=False),
        sa.Column("is_uncertain", sa.Boolean(), nullable=False),
        sa.Column("text_sha256", sa.String(length=64), nullable=False),
        sa.Column("text_char_count", sa.Integer(), nullable=False),
        sa.Column("text_word_count", sa.Integer(), nullable=False),
        sa.Column("input_kind", sa.String(length=16), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("resume_text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])
    op.create_index("ix_analyses_model_version", "analyses", ["model_version"])
    op.create_index("ix_analyses_predicted_label", "analyses", ["predicted_label"])
    op.create_index("ix_analyses_text_sha256", "analyses", ["text_sha256"])

    op.create_table(
        "analysis_skills",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("skill_id", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("occurrences", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_skills_analysis_id", "analysis_skills", ["analysis_id"])
    op.create_index("ix_analysis_skills_skill_id", "analysis_skills", ["skill_id"])


def downgrade() -> None:
    op.drop_index("ix_analysis_skills_skill_id", table_name="analysis_skills")
    op.drop_index("ix_analysis_skills_analysis_id", table_name="analysis_skills")
    op.drop_table("analysis_skills")

    op.drop_index("ix_analyses_text_sha256", table_name="analyses")
    op.drop_index("ix_analyses_predicted_label", table_name="analyses")
    op.drop_index("ix_analyses_model_version", table_name="analyses")
    op.drop_index("ix_analyses_created_at", table_name="analyses")
    op.drop_table("analyses")
