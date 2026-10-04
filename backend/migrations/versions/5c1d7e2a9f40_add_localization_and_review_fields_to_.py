"""add localization and review fields to inspections

Revision ID: 5c1d7e2a9f40
Revises: 39bd1b8da129
Create Date: 2026-10-04 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '5c1d7e2a9f40'
down_revision: Union[str, Sequence[str], None] = '39bd1b8da129'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add-only, every column nullable with no server default and no backfill: every existing
    # inspection keeps NULL ("not computed"), never a made-up confidence or localization.
    # Populated for new inspections by app.inspections.service (app.ai.inference.localization).
    op.add_column('inspections', sa.Column('ai_confidence', sa.Float(), nullable=True))
    op.add_column('inspections', sa.Column('review_required', sa.Boolean(), nullable=True))
    op.add_column('inspections', sa.Column('review_reason', sa.String(length=255), nullable=True))
    op.add_column('inspections', sa.Column('localization', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    # Storage-relative path of the heatmap PNG under storage/heatmaps; internal, never serialized.
    op.add_column('inspections', sa.Column('heatmap_path', sa.String(length=500), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('inspections', 'heatmap_path')
    op.drop_column('inspections', 'localization')
    op.drop_column('inspections', 'review_reason')
    op.drop_column('inspections', 'review_required')
    op.drop_column('inspections', 'ai_confidence')
