"""add source field to inspections

Revision ID: 3ba354d22dca
Revises: 3dbc453c0101
Create Date: 2026-08-29 16:01:04.508450

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ba354d22dca'
down_revision: Union[str, Sequence[str], None] = '3dbc453c0101'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


inspection_source_enum = sa.Enum('upload', 'mvtec_ad', name='inspection_source')


def upgrade() -> None:
    """Upgrade schema."""
    inspection_source_enum.create(op.get_bind(), checkfirst=True)
    op.add_column(
        'inspections',
        sa.Column('source', inspection_source_enum, server_default='upload', nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('inspections', 'source')
    inspection_source_enum.drop(op.get_bind(), checkfirst=True)
