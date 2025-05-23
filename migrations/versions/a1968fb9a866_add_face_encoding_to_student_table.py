"""Add face_encoding to student table

Revision ID: a1968fb9a866
Revises: 
Create Date: 2024-12-31 20:04:03.123269

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1968fb9a866'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('course',
    sa.Column('course_id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('course_name', sa.String(length=255), nullable=False),
    sa.Column('sessions', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('course_id')
    )
    op.create_table('student',
    sa.Column('student_id', sa.String(length=11), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('password', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('face_encoding', sa.LargeBinary(), nullable=False),  # Add this line
    sa.PrimaryKeyConstraint('student_id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('student_id')
    )
    op.create_table('teacher',
    sa.Column('teacher_id', sa.String(length=11), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('password', sa.String(length=255), nullable=False),
    sa.PrimaryKeyConstraint('teacher_id'),
    sa.UniqueConstraint('teacher_id')
    )
    op.create_table('attendancelog',
    sa.Column('attendance_id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('course_id', sa.Integer(), nullable=False),
    sa.Column('session', sa.Integer(), nullable=True),
    sa.Column('teacher_id', sa.String(length=11), nullable=False),
    sa.Column('student_id', sa.String(length=11), nullable=False),
    sa.Column('date', sa.Date(), nullable=True),
    sa.Column('time', sa.Time(), nullable=True),
    sa.Column('status', sa.String(length=10), nullable=True),
    sa.ForeignKeyConstraint(['course_id'], ['course.course_id'], ),
    sa.ForeignKeyConstraint(['student_id'], ['student.student_id'], ),
    sa.ForeignKeyConstraint(['teacher_id'], ['teacher.teacher_id'], ),
    sa.PrimaryKeyConstraint('attendance_id')
    )
    # ### end Alembic commands ###



def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('attendancelog')
    op.drop_table('teacher')
    op.drop_table('student')
    op.drop_table('course')
    # ### end Alembic commands ###
