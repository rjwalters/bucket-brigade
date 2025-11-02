#!/usr/bin/env python3
"""
Migrate data from SQLite to PostgreSQL.

This script reads data from an existing SQLite database and imports it
into the PostgreSQL schema.
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bucket_brigade.db import init_db, get_session
from bucket_brigade.db.models import Agent, Submission, AgentMetadata


def read_sqlite_data(sqlite_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read all data from SQLite database.

    Args:
        sqlite_path: Path to SQLite database file

    Returns:
        Dictionary with table data
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    data = {
        "agents": [],
        "submissions": [],
        "agent_metadata": [],
    }

    # Read agents table if exists
    try:
        cursor.execute("SELECT * FROM agents")
        data["agents"] = [dict(row) for row in cursor.fetchall()]
        print(f"   Found {len(data['agents'])} agents")
    except sqlite3.OperationalError:
        print("   No agents table found in SQLite")

    # Read submissions table if exists
    try:
        cursor.execute("SELECT * FROM submissions")
        data["submissions"] = [dict(row) for row in cursor.fetchall()]
        print(f"   Found {len(data['submissions'])} submissions")
    except sqlite3.OperationalError:
        print("   No submissions table found in SQLite")

    # Read agent_metadata table if exists
    try:
        cursor.execute("SELECT * FROM agent_metadata")
        data["agent_metadata"] = [dict(row) for row in cursor.fetchall()]
        print(f"   Found {len(data['agent_metadata'])} metadata records")
    except sqlite3.OperationalError:
        print("   No agent_metadata table found in SQLite")

    conn.close()
    return data


def import_to_postgresql(data: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Import data into PostgreSQL.

    Args:
        data: Dictionary with table data from SQLite
    """
    session_gen = get_session()
    session = next(session_gen)

    try:
        # Import agents
        agent_id_map = {}
        for agent_data in data["agents"]:
            agent = Agent(
                name=agent_data.get("name", "Unknown"),
                author=agent_data.get("author", "Unknown"),
                code_path=agent_data["code_path"],
                created_at=(
                    datetime.fromisoformat(agent_data["created_at"])
                    if "created_at" in agent_data
                    else datetime.utcnow()
                ),
                updated_at=(
                    datetime.fromisoformat(agent_data["updated_at"])
                    if "updated_at" in agent_data
                    else datetime.utcnow()
                ),
                active=agent_data.get("active", True),
            )
            session.add(agent)
            session.flush()  # Get the new ID
            agent_id_map[agent_data["id"]] = agent.id

        print(f"   Imported {len(data['agents'])} agents")

        # Import submissions
        for submission_data in data["submissions"]:
            old_agent_id = submission_data["agent_id"]
            new_agent_id = agent_id_map.get(old_agent_id)

            if new_agent_id:
                submission = Submission(
                    agent_id=new_agent_id,
                    validation_passed=submission_data["validation_passed"],
                    validation_errors=submission_data.get("validation_errors"),
                    validation_warnings=submission_data.get("validation_warnings"),
                    test_stats=submission_data.get("test_stats"),
                    submitted_at=(
                        datetime.fromisoformat(submission_data["submitted_at"])
                        if "submitted_at" in submission_data
                        else datetime.utcnow()
                    ),
                )
                session.add(submission)

        print(f"   Imported {len(data['submissions'])} submissions")

        # Import metadata
        for metadata_data in data["agent_metadata"]:
            old_agent_id = metadata_data["agent_id"]
            new_agent_id = agent_id_map.get(old_agent_id)

            if new_agent_id:
                metadata = AgentMetadata(
                    agent_id=new_agent_id,
                    description=metadata_data.get("description"),
                    version=metadata_data.get("version", "1.0.0"),
                    tags=metadata_data.get("tags"),
                    license=metadata_data.get("license"),
                    repository_url=metadata_data.get("repository_url"),
                )
                session.add(metadata)

        print(f"   Imported {len(data['agent_metadata'])} metadata records")

        session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description="Migrate from SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        required=True,
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing PostgreSQL tables before migration (DANGEROUS!)",
    )

    args = parser.parse_args()

    print("🔄 SQLite → PostgreSQL Migration")
    print("=" * 50)

    # Check SQLite file exists
    if not Path(args.sqlite_path).exists():
        print(f"❌ SQLite file not found: {args.sqlite_path}")
        sys.exit(1)

    try:
        # Initialize PostgreSQL
        print("\n1️⃣ Initializing PostgreSQL schema...")
        init_db(drop_existing=args.drop_existing)
        print("✅ PostgreSQL schema ready")

        # Read SQLite data
        print("\n2️⃣ Reading data from SQLite...")
        data = read_sqlite_data(args.sqlite_path)

        # Import to PostgreSQL
        print("\n3️⃣ Importing data to PostgreSQL...")
        import_to_postgresql(data)

        print("\n✅ Migration completed successfully!")
        print("\n🎉 All data has been migrated to PostgreSQL!")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
