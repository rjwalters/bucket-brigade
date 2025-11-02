#!/usr/bin/env python3
"""
Agent submission and validation script for Bucket Brigade.

This script allows users to submit their custom agents for validation
and testing against the platform.
"""

import sys
from pathlib import Path
from typing import Optional
import typer

# Add the bucket_brigade package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.agents import (
    load_agent_from_file,
    create_agent_instance,
    get_agent_metadata,
    validate_agent_behavior,
    AgentValidationError,
    AgentSecurityError
)


def submit_agent(agent_file: str, test_run: bool = True, verbose: bool = False) -> None:
    """
    Submit and validate an agent file.

    Args:
        agent_file: Path to the agent Python file
        test_run: Whether to run behavioral tests
        verbose: Enable verbose output
    """
    print("🔍 Bucket Brigade Agent Submission")
    print("=" * 50)
    print(f"📁 Agent file: {agent_file}")

    try:
        # Load and validate agent
        print("⏳ Loading agent...")
        agent_class = load_agent_from_file(agent_file, validate=True)

        print("✅ Agent loaded successfully!")

        # Get metadata
        metadata = get_agent_metadata(agent_class)
        print("\n📋 Agent Metadata:")
        print(f"   Name: {metadata.get('name', 'Unknown')}")
        print(f"   Author: {metadata.get('author', 'Unknown')}")
        print(f"   Description: {metadata.get('description', 'No description')}")
        print(f"   Version: {metadata.get('version', '1.0.0')}")
        print(f"   Tags: {', '.join(metadata.get('tags', []))}")

        if test_run:
            print("\n🧪 Running behavioral validation...")
            # Create test instance
            agent = create_agent_instance(agent_class, 0, name="TestAgent")

            # Run validation
            results = validate_agent_behavior(agent, max_steps=20)

            if results['valid']:
                print("✅ Behavioral validation passed!")
                print("\n📊 Test Results:")
                stats = results['stats']
                print(f"   Steps run: {stats['steps_run']}")
                print(f"   Avg reward: {stats.get('avg_reward', 0):.3f}")
                print(f"   Valid actions: {stats.get('valid_actions_pct', 0):.3f}")
                print(f"   Game completed: {stats['game_completed']}")
            else:
                print("❌ Behavioral validation failed!")
                print("\n❌ Errors:")
                for error in results['errors']:
                    print(f"   • {error}")
                if results['warnings']:
                    print("\n⚠️ Warnings:")
                    for warning in results['warnings']:
                        print(f"   • {warning}")
                return

        print("\n🎉 Agent submission successful!")
        print("Your agent is ready for evaluation in the Bucket Brigade tournament!")

        if verbose:
            print("\n🔧 Technical Details:")
            print(f"   Agent class: {agent_class.__name__}")
            print(f"   Module: {agent_class.__module__}")
            print(f"   Methods: {[m for m in dir(agent_class) if not m.startswith('_')]}")

    except AgentSecurityError as e:
        print(f"🚨 SECURITY VIOLATION: {e}")
        print("Your agent contains forbidden code patterns.")
        print("Please review the submission guidelines and remove any")
        print("system calls, file access, or dangerous imports.")

    except AgentValidationError as e:
        print(f"❌ VALIDATION ERROR: {e}")
        print("Your agent doesn't meet the required interface.")
        print("Please check the agent template and ensure all")
        print("required methods and functions are implemented.")

    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {e}")
        print("An unexpected error occurred during submission.")
        print("Please check your code and try again.")


def create_agent_template(output_file: str = "my_agent.py") -> None:
    """
    Create a new agent template file.

    Args:
        output_file: Where to save the template
    """
    template_path = Path(__file__).parent.parent / "bucket_brigade" / "agents" / "agent_template.py"

    if not template_path.exists():
        print("❌ Agent template not found!")
        return

    output_path = Path(output_file)

    # Copy template
    import shutil
    shutil.copy2(template_path, output_path)

    print(f"✅ Agent template created: {output_path}")
    print("\n📝 Next steps:")
    print("1. Edit the template with your agent logic")
    print("2. Test locally: python scripts/submit_agent.py my_agent.py")
    print("3. Submit when ready!")


def main(
    agent_file: Optional[str] = typer.Argument(None, help="Path to agent file to submit"),
    create_template: bool = typer.Option(False, help="Create a new agent template"),
    template_output: str = typer.Option("my_agent.py", help="Output file for template"),
    test_run: bool = typer.Option(True, help="Run behavioral tests"),
    verbose: bool = typer.Option(False, help="Enable verbose output")
):
    """Submit and validate Bucket Brigade agents."""

    if create_template:
        create_agent_template(template_output)
    elif agent_file:
        submit_agent(agent_file, test_run, verbose)
    else:
        print("❌ Please specify an agent file to submit or use --create-template")
        print("\nUsage examples:")
        print("  python scripts/submit_agent.py my_agent.py")
        print("  python scripts/submit_agent.py --create-template")
        print("  python scripts/submit_agent.py --help")


if __name__ == "__main__":
    typer.run(main)
