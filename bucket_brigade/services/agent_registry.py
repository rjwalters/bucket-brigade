"""
Agent Registry Service for Bucket Brigade.

This service handles agent submissions, validation, storage, and retrieval
with full database persistence to PostgreSQL.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from ..db.models import Agent, Submission, AgentMetadata
from ..db.connection import get_session
from ..agents import (
    load_agent_from_file,
    load_agent_from_string,
    get_agent_metadata,
    validate_agent_behavior,
    create_agent_instance,
    AgentValidationError,
    AgentSecurityError,
)


class AgentRegistryService:
    """
    Service for managing agent submissions and validation.

    This service provides:
    - Agent submission with validation
    - Agent storage in filesystem
    - Database persistence with PostgreSQL
    - Agent retrieval and listing
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the Agent Registry Service.

        Args:
            storage_dir: Directory for storing agent code files (default: agents/submitted)
        """
        if storage_dir is None:
            # Default to agents/submitted relative to project root
            self.storage_dir = Path("agents/submitted")
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def submit_agent(
        self,
        agent_code: str,
        name: str,
        author: str,
        description: Optional[str] = None,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        license: Optional[str] = None,
        repository_url: Optional[str] = None,
        test_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit a new agent for validation and storage.

        Args:
            agent_code: Python source code as string
            name: Agent display name
            author: Agent creator
            description: Optional agent description
            version: Semantic version string
            tags: Optional list of tags/keywords
            license: Optional license identifier
            repository_url: Optional repository URL
            test_run: Whether to run behavioral validation tests

        Returns:
            Dictionary with submission results:
                - success: bool
                - agent_id: int (if successful)
                - errors: List[str]
                - warnings: List[str]
                - stats: Dict (if test_run=True)
        """
        result = {
            "success": False,
            "agent_id": None,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        session_gen = get_session()
        session = next(session_gen)

        try:
            # Validate agent code
            try:
                agent_class = load_agent_from_string(agent_code, validate=True)
            except AgentSecurityError as e:
                result["errors"].append(f"Security violation: {e}")
                self._record_submission(session, None, False, result["errors"], result["warnings"], None)
                return result
            except AgentValidationError as e:
                result["errors"].append(f"Validation error: {e}")
                self._record_submission(session, None, False, result["errors"], result["warnings"], None)
                return result
            except Exception as e:
                result["errors"].append(f"Failed to load agent: {e}")
                self._record_submission(session, None, False, result["errors"], result["warnings"], None)
                return result

            # Run behavioral tests if requested
            if test_run:
                try:
                    agent_instance = create_agent_instance(agent_class, 0, name="TestAgent")
                    validation_results = validate_agent_behavior(agent_instance, max_steps=20)

                    if not validation_results["valid"]:
                        result["errors"].extend(validation_results["errors"])
                        result["warnings"].extend(validation_results["warnings"])
                        self._record_submission(
                            session, None, False, result["errors"], result["warnings"], validation_results["stats"]
                        )
                        return result

                    result["stats"] = validation_results["stats"]
                    result["warnings"].extend(validation_results.get("warnings", []))

                except Exception as e:
                    result["errors"].append(f"Behavioral validation failed: {e}")
                    self._record_submission(session, None, False, result["errors"], result["warnings"], None)
                    return result

            # Create agent record
            agent = Agent(
                name=name,
                author=author,
                code_path="",  # Will be updated after saving file
                active=True,
            )
            session.add(agent)
            session.flush()  # Get agent ID

            # Save agent code to filesystem
            agent_file_path = self.storage_dir / f"agent_{agent.id}.py"
            with open(agent_file_path, "w") as f:
                f.write(agent_code)

            # Update agent with code path
            agent.code_path = str(agent_file_path)
            session.add(agent)

            # Create metadata record
            metadata = AgentMetadata(
                agent_id=agent.id,
                description=description or "No description provided",
                version=version,
                tags=tags or [],
                license=license,
                repository_url=repository_url,
            )
            session.add(metadata)

            # Record successful submission
            self._record_submission(
                session, agent.id, True, result["errors"], result["warnings"], result.get("stats")
            )

            session.commit()

            result["success"] = True
            result["agent_id"] = agent.id

        except Exception as e:
            session.rollback()
            result["errors"].append(f"Database error: {e}")

        finally:
            session.close()

        return result

    def submit_agent_from_file(
        self,
        file_path: str,
        name: Optional[str] = None,
        author: Optional[str] = None,
        test_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit an agent from a file.

        Args:
            file_path: Path to agent Python file
            name: Optional agent name (extracted from metadata if not provided)
            author: Optional author name (extracted from metadata if not provided)
            test_run: Whether to run behavioral validation tests

        Returns:
            Dictionary with submission results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {
                "success": False,
                "agent_id": None,
                "errors": [f"File not found: {file_path}"],
                "warnings": [],
                "stats": {},
            }

        # Read agent code
        with open(file_path, "r") as f:
            agent_code = f.read()

        # Try to extract metadata from code
        try:
            agent_class = load_agent_from_file(str(file_path), validate=True)
            metadata = get_agent_metadata(agent_class)

            if name is None:
                name = metadata.get("name", file_path.stem)
            if author is None:
                author = metadata.get("author", "Unknown")

            return self.submit_agent(
                agent_code=agent_code,
                name=name,
                author=author,
                description=metadata.get("description"),
                version=metadata.get("version", "1.0.0"),
                tags=metadata.get("tags", []),
                test_run=test_run,
            )

        except Exception as e:
            return {
                "success": False,
                "agent_id": None,
                "errors": [f"Failed to process file: {e}"],
                "warnings": [],
                "stats": {},
            }

    def get_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """
        Get agent information by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with agent information or None if not found
        """
        session_gen = get_session()
        session = next(session_gen)

        try:
            agent = session.query(Agent).filter(Agent.id == agent_id).first()

            if agent is None:
                return None

            return {
                "id": agent.id,
                "name": agent.name,
                "author": agent.author,
                "code_path": agent.code_path,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
                "active": agent.active,
                "metadata": {
                    "description": agent.agent_metadata.description if agent.agent_metadata else None,
                    "version": agent.agent_metadata.version if agent.agent_metadata else "1.0.0",
                    "tags": agent.agent_metadata.tags if agent.agent_metadata else [],
                    "license": agent.agent_metadata.license if agent.agent_metadata else None,
                    "repository_url": agent.agent_metadata.repository_url if agent.agent_metadata else None,
                },
            }

        finally:
            session.close()

    def list_agents(
        self,
        active_only: bool = True,
        author: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List agents with optional filtering.

        Args:
            active_only: Only return active agents
            author: Filter by author name
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of agent information dictionaries
        """
        session_gen = get_session()
        session = next(session_gen)

        try:
            query = session.query(Agent)

            if active_only:
                query = query.filter(Agent.active == True)

            if author:
                query = query.filter(Agent.author == author)

            query = query.order_by(Agent.created_at.desc())
            query = query.limit(limit).offset(offset)

            agents = query.all()

            return [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "author": agent.author,
                    "created_at": agent.created_at.isoformat(),
                    "updated_at": agent.updated_at.isoformat(),
                    "active": agent.active,
                    "description": agent.agent_metadata.description if agent.agent_metadata else None,
                    "version": agent.agent_metadata.version if agent.agent_metadata else "1.0.0",
                    "tags": agent.agent_metadata.tags if agent.agent_metadata else [],
                }
                for agent in agents
            ]

        finally:
            session.close()

    def get_agent_submissions(self, agent_id: int) -> List[Dict[str, Any]]:
        """
        Get submission history for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of submission records
        """
        session_gen = get_session()
        session = next(session_gen)

        try:
            submissions = (
                session.query(Submission)
                .filter(Submission.agent_id == agent_id)
                .order_by(Submission.submitted_at.desc())
                .all()
            )

            return [
                {
                    "id": sub.id,
                    "agent_id": sub.agent_id,
                    "validation_passed": sub.validation_passed,
                    "validation_errors": sub.validation_errors,
                    "validation_warnings": sub.validation_warnings,
                    "test_stats": sub.test_stats,
                    "submitted_at": sub.submitted_at.isoformat(),
                }
                for sub in submissions
            ]

        finally:
            session.close()

    def load_agent_code(self, agent_id: int) -> Optional[str]:
        """
        Load agent source code from filesystem.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent source code as string or None if not found
        """
        agent_info = self.get_agent(agent_id)

        if agent_info is None:
            return None

        code_path = Path(agent_info["code_path"])

        if not code_path.exists():
            return None

        with open(code_path, "r") as f:
            return f.read()

    def _record_submission(
        self,
        session: Session,
        agent_id: Optional[int],
        passed: bool,
        errors: List[str],
        warnings: List[str],
        stats: Optional[Dict[str, Any]],
    ) -> None:
        """
        Record a submission attempt in the database.

        Args:
            session: Database session
            agent_id: Agent ID (None if agent creation failed)
            passed: Whether validation passed
            errors: List of error messages
            warnings: List of warning messages
            stats: Test statistics dictionary
        """
        if agent_id is not None:
            submission = Submission(
                agent_id=agent_id,
                validation_passed=passed,
                validation_errors=errors if errors else None,
                validation_warnings=warnings if warnings else None,
                test_stats=stats,
            )
            session.add(submission)
