"""add category to products

Revision ID: 39bd1b8da129
Revises: e94d6730c867
Create Date: 2026-10-03 13:54:34.364051

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '39bd1b8da129'
down_revision: Union[str, Sequence[str], None] = 'e94d6730c867'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # The product's MVTec AD category (e.g. "tile"), used to pick the AI model for uploaded
    # images. Nullable with no server default and no data backfill: every existing product
    # keeps NULL ("no category"), so none of them starts receiving AI predictions until a
    # quality engineer deliberately sets one. Values are validated in the API layer
    # (app.dataset.categories), not by a database constraint.
    op.add_column('products', sa.Column('category', sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('products', 'category')
