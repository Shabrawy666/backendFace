"""Remove dummy_column from Student model

Revision ID: a524cd0be58e
Revises: ede5ee9ec3a2
Create Date: 2025-04-11 16:46:19.010492

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a524cd0be58e'
down_revision = 'ede5ee9ec3a2'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('student', schema=None) as batch_op:
        batch_op.drop_column('dummy_column')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('student', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dummy_column', sa.VARCHAR(length=255), autoincrement=False, nullable=True))

    # ### end Alembic commands ###
