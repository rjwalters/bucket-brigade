"""
SQLAlchemy models for Bucket Brigade Agent Registry.

This module defines the database schema for:
- agents: Core agent information (id, name, author, code path)
- submissions: Submission history and validation results
- agent_metadata: Extended metadata (tags, version, description)
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Agent(Base):
    """
    Core agent table storing basic information and code location.

    Attributes:
        id: Unique agent identifier
        name: Agent display name
        author: Agent creator
        code_path: Filesystem path to agent Python module
        created_at: When agent was first submitted
        updated_at: When agent was last modified
        active: Whether agent is currently active/valid
    """

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    code_path = Column(String(512), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    active = Column(Boolean, default=True, nullable=False, index=True)

    # Relationships
    submissions = relationship(
        "Submission", back_populates="agent", cascade="all, delete-orphan"
    )
    agent_metadata = relationship(
        "AgentMetadata",
        back_populates="agent",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Agent(id={self.id}, name='{self.name}', author='{self.author}')>"


class Submission(Base):
    """
    Submission history tracking all agent submission attempts.

    Attributes:
        id: Unique submission identifier
        agent_id: Foreign key to agents table
        validation_passed: Whether validation succeeded
        validation_errors: JSON array of validation error messages
        validation_warnings: JSON array of validation warnings
        test_stats: JSON object with test run statistics
        submitted_at: When submission occurred
    """

    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(
        Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    validation_passed = Column(Boolean, nullable=False)
    validation_errors = Column(JSON, nullable=True)
    validation_warnings = Column(JSON, nullable=True)
    test_stats = Column(JSON, nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    agent = relationship("Agent", back_populates="submissions")

    def __repr__(self):
        return f"<Submission(id={self.id}, agent_id={self.agent_id}, passed={self.validation_passed})>"


class AgentMetadata(Base):
    """
    Extended metadata for agents including description, version, tags.

    Attributes:
        id: Unique metadata identifier
        agent_id: Foreign key to agents table (one-to-one)
        description: Agent description
        version: Semantic version string
        tags: JSON array of tags/keywords
        license: License identifier
        repository_url: Optional URL to source repository
    """

    __tablename__ = "agent_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    description = Column(Text, nullable=True)
    version = Column(String(50), default="1.0.0", nullable=False)
    tags = Column(JSON, nullable=True)
    license = Column(String(100), nullable=True)
    repository_url = Column(String(512), nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="agent_metadata")

    def __repr__(self):
        return f"<AgentMetadata(agent_id={self.agent_id}, version='{self.version}')>"
