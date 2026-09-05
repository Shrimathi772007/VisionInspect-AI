"""add ai prediction fields to inspections

Revision ID: b33cf1824856
Revises: 3ba354d22dca
Create Date: 2026-09-05 14:40:49.166274

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b33cf1824856'
down_revision: Union[str, Sequence[str], None] = '3ba354d22dca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('inspections', sa.Column('ai_prediction', sa.String(length=20), nullable=True))
    op.add_column('inspections', sa.Column('ai_reconstruction_error', sa.Float(), nullable=True))
    op.add_column('inspections', sa.Column('ai_threshold', sa.Float(), nullable=True))
    op.add_column('inspections', sa.Column('ai_model_name', sa.String(length=100), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('inspections', 'ai_model_name')
    op.drop_column('inspections', 'ai_threshold')
    op.drop_column('inspections', 'ai_reconstruction_error')
    op.drop_column('inspections', 'ai_prediction')
