"""Initial schema - create all tables

Revision ID: 001
Revises:
Create Date: 2026-01-01 00:00:00
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── users ──────────────────────────────────────────────────────────────
    op.create_table(
        'users',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=True),
        sa.Column('password_hash', sa.String(255), nullable=True),
        sa.Column('name', sa.String(100), nullable=True),
        sa.Column('role', sa.Enum('viewer', 'analyst', 'admin', name='userrole'),
                  nullable=False, server_default='analyst'),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('NOW()')),
        sa.Column('last_login', sa.DateTime, nullable=True),
    )
    op.create_index('ix_users_email', 'users', ['email'])

    # ── sessions ───────────────────────────────────────────────────────────
    op.create_table(
        'sessions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('NOW()')),
        sa.Column('expires_at', sa.DateTime, nullable=False),
    )

    # ── jobs ───────────────────────────────────────────────────────────────
    op.create_table(
        'jobs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('session_id', UUID(as_uuid=True),
                  sa.ForeignKey('sessions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('file_size_mb', sa.Float, nullable=True),
        sa.Column('status', sa.Enum(
            'queued', 'processing', 'completed', 'failed', 'cancelled',
            name='jobstatus'
        ), nullable=False, server_default='queued'),
        sa.Column('progress', sa.Integer, nullable=False, server_default='0'),
        sa.Column('progress_message', sa.String(255), server_default='Queued for processing'),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('error_type', sa.String(100), nullable=True),
        sa.Column('processing_time_seconds', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('NOW()')),
        sa.Column('completed_at', sa.DateTime, nullable=True),
    )
    op.create_index('ix_jobs_user_id', 'jobs', ['user_id'])
    op.create_index('ix_jobs_session_id', 'jobs', ['session_id'])
    op.create_index('ix_jobs_status', 'jobs', ['status'])
    op.create_index('ix_jobs_created_at', 'jobs', ['created_at'])

    # ── results ────────────────────────────────────────────────────────────
    op.create_table(
        'results',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', UUID(as_uuid=True),
                  sa.ForeignKey('jobs.id', ondelete='CASCADE'),
                  unique=True, nullable=False),
        sa.Column('structural_analysis', JSONB, nullable=True),
        sa.Column('statistical_analysis', JSONB, nullable=True),
        sa.Column('model_recommendations', JSONB, nullable=True),
        sa.Column('insights', JSONB, nullable=True),
        sa.Column('dataset_info', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('NOW()')),
    )

    # ── dataset_metadata ───────────────────────────────────────────────────
    op.create_table(
        'dataset_metadata',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', UUID(as_uuid=True),
                  sa.ForeignKey('jobs.id', ondelete='CASCADE'),
                  unique=True, nullable=False),
        sa.Column('row_count', sa.Integer, nullable=True),
        sa.Column('column_count', sa.Integer, nullable=True),
        sa.Column('missing_percentage', sa.Float, nullable=True),
        sa.Column('duplicate_rows', sa.Integer, nullable=True),
        sa.Column('data_structure_type', sa.String(50), nullable=True),
        sa.Column('critical_issues_count', sa.Integer, server_default='0'),
        sa.Column('high_issues_count', sa.Integer, server_default='0'),
        sa.Column('medium_issues_count', sa.Integer, server_default='0'),
        sa.Column('primary_model_recommendation', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('NOW()')),
    )


def downgrade() -> None:
    op.drop_table('dataset_metadata')
    op.drop_table('results')
    op.drop_table('jobs')
    op.drop_table('sessions')
    op.drop_table('users')
    op.execute("DROP TYPE IF EXISTS jobstatus")
    op.execute("DROP TYPE IF EXISTS userrole")