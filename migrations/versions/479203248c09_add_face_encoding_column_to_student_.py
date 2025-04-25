"""Add face_encoding column to student table

Revision ID: 479203248c09
Revises: a1968fb9a866
Create Date: 2024-12-31 20:10:44.277743

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '479203248c09'
down_revision = 'a1968fb9a866'
branch_labels = None
depends_on = None


def upgrade():
    # Add face_encoding column to the student table
    op.add_column('student', sa.Column('face_encoding', sa.Text(), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # Drop the face_encoding column if the migration is rolled back
    op.drop_column('student', 'face_encoding')

    # ### end Alembic commands ###
