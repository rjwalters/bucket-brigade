#!/usr/bin/env python3
"""
Initialize the Bucket Brigade PostgreSQL database.

This script creates all tables and initializes the schema.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bucket_brigade.db import init_db


def main():
    """Initialize the database."""
    print("🔧 Initializing Bucket Brigade database...")
    print("=" * 50)

    try:
        init_db(drop_existing=False)
        print("✅ Database initialized successfully!")
        print("\n📋 Created tables:")
        print("   - agents")
        print("   - submissions")
        print("   - agent_metadata")
        print("\n🎉 Ready to accept agent submissions!")

    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
