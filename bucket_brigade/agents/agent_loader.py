"""
Secure agent loading and validation system for Bucket Brigade.

This module provides functionality to safely load, validate, and execute
user-submitted agents while preventing security issues.
"""

import importlib.util
import sys
import os
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path
import hashlib
import logging

from .agent_base import AgentBase

logger = logging.getLogger(__name__)

# Allowed imports for agent submissions
ALLOWED_IMPORTS = {
    'numpy', 'np',  # numpy as np
    'typing',  # Type hints
    'math',  # Basic math
    'random',  # Random number generation
    'collections',  # Data structures
    'itertools',  # Iteration utilities
    'functools',  # Function utilities
}

# Forbidden patterns that indicate potentially malicious code
FORBIDDEN_PATTERNS = [
    'import os', 'import subprocess', 'import sys',
    'open(', 'eval(', 'exec(', '__import__(',
    'globals(', 'locals(', 'dir(__builtins__)',
    'getattr(', 'setattr(', 'hasattr(', 'delattr(',
    'input(', 'print(',  # Debug prints not allowed in production
]

class AgentValidationError(Exception):
    """Raised when agent validation fails."""
    pass

class AgentSecurityError(Exception):
    """Raised when agent contains security violations."""
    pass

def validate_agent_code(source_code: str) -> None:
    """
    Validate agent source code for security and correctness.

    Args:
        source_code: The Python source code as a string

    Raises:
        AgentSecurityError: If code contains forbidden patterns
        AgentValidationError: If code doesn't meet requirements
    """
    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in source_code:
            raise AgentSecurityError(f"Forbidden pattern detected: {pattern}")

    # Check for required functions
    if 'def create_agent(' not in source_code:
        raise AgentValidationError("Missing required function: create_agent()")

    if 'class ' not in source_code and 'AgentBase' not in source_code:
        raise AgentValidationError("Agent must inherit from AgentBase or implement same interface")

    # Check imports (basic validation)
    lines = source_code.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            # Extract module name
            if line.startswith('import '):
                modules = line[7:].split(',')
                for module in modules:
                    module = module.strip().split()[0]
                    if module and module not in ALLOWED_IMPORTS:
                        raise AgentSecurityError(f"Forbidden import: {module}")
            elif line.startswith('from '):
                module = line[5:].split()[0]
                if module and module not in ALLOWED_IMPORTS:
                    raise AgentSecurityError(f"Forbidden import: {module}")

def load_agent_from_file(file_path: str, validate: bool = True) -> Type[AgentBase]:
    """
    Load an agent class from a Python file.

    Args:
        file_path: Path to the Python file
        validate: Whether to validate the code for security

    Returns:
        The agent class

    Raises:
        AgentValidationError: If validation fails
        AgentSecurityError: If security check fails
        ImportError: If file cannot be loaded
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Agent file not found: {file_path}")

    # Read and validate source code
    with open(file_path, 'r') as f:
        source_code = f.read()

    if validate:
        validate_agent_code(source_code)

    # Create a unique module name
    module_name = f"agent_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"

    try:
        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check for required create_agent function
        if not hasattr(module, 'create_agent'):
            raise AgentValidationError("Module missing create_agent function")

        # Test agent creation
        create_func = getattr(module, 'create_agent')
        test_agent = create_func(0, name="TestAgent")

        if not isinstance(test_agent, AgentBase):
            # Allow agents that don't inherit but implement the interface
            required_methods = ['reset', 'act', 'agent_id']
            for method in required_methods:
                if not hasattr(test_agent, method):
                    raise AgentValidationError(f"Agent missing required method: {method}")

        # Check metadata if present
        metadata = getattr(module, 'AGENT_METADATA', None)
        if metadata:
            required_fields = ['name', 'author', 'description']
            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"Agent metadata missing field: {field}")

        return type(test_agent)

    except Exception as e:
        raise AgentValidationError(f"Failed to load agent: {e}")

def load_agent_from_string(source_code: str, module_name: str = "user_agent",
                          validate: bool = True) -> Type[AgentBase]:
    """
    Load an agent from source code string.

    Args:
        source_code: Python source code as string
        module_name: Name for the loaded module
        validate: Whether to validate code

    Returns:
        The agent class
    """
    if validate:
        validate_agent_code(source_code)

    try:
        # Load module from string
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            raise ImportError("Could not create module spec")

        module = importlib.util.module_from_spec(spec)

        # Execute code in module
        exec(source_code, module.__dict__)

        # Check for required create_agent function
        if not hasattr(module, 'create_agent'):
            raise AgentValidationError("Module missing create_agent function")

        # Test agent creation
        create_func = getattr(module, 'create_agent')
        test_agent = create_func(0, name="TestAgent")

        # Validate agent interface
        required_methods = ['reset', 'act', 'agent_id']
        for method in required_methods:
            if not hasattr(test_agent, method):
                raise AgentValidationError(f"Agent missing required method: {method}")

        return type(test_agent)

    except Exception as e:
        raise AgentValidationError(f"Failed to load agent from string: {e}")

def create_agent_instance(agent_class: Type[AgentBase], agent_id: int,
                         **kwargs) -> AgentBase:
    """
    Create an instance of an agent class.

    Args:
        agent_class: The agent class to instantiate
        agent_id: Agent ID
        **kwargs: Additional arguments for agent creation

    Returns:
        Agent instance
    """
    # Try to use create_agent function if available
    module = sys.modules.get(agent_class.__module__)
    if module and hasattr(module, 'create_agent'):
        create_func = getattr(module, 'create_agent')
        return create_func(agent_id, **kwargs)
    else:
        # Fallback to direct instantiation
        return agent_class(agent_id, **kwargs)

def get_agent_metadata(agent_class: Type[AgentBase]) -> Dict[str, Any]:
    """
    Get metadata for an agent class.

    Args:
        agent_class: The agent class

    Returns:
        Metadata dictionary
    """
    module = sys.modules.get(agent_class.__module__)
    if module and hasattr(module, 'AGENT_METADATA'):
        return getattr(module, 'AGENT_METADATA')
    else:
        return {
            'name': agent_class.__name__,
            'author': 'Unknown',
            'description': 'No description provided',
            'version': '1.0.0',
            'tags': []
        }

def validate_agent_behavior(agent: AgentBase, max_steps: int = 100) -> Dict[str, Any]:
    """
    Validate agent behavior by running it in a test environment.

    Args:
        agent: Agent instance to test
        max_steps: Maximum steps to run

    Returns:
        Validation results
    """
    from ..envs import BucketBrigadeEnv, default_scenario

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    try:
        # Create test environment
        env = BucketBrigadeEnv(default_scenario(4))
        obs = env.reset()

        actions_taken = []
        rewards_received = []

        # Run for a few steps
        for step in range(min(max_steps, 50)):
            action = agent.act(obs)
            actions_taken.append(action.tolist())

            # Validate action format
            if not isinstance(action, np.ndarray) or action.shape != (2,):
                results['valid'] = False
                results['errors'].append(f"Invalid action format at step {step}: {action}")
                break

            if not (0 <= action[0] <= 9):
                results['warnings'].append(f"House index out of range at step {step}: {action[0]}")

            if action[1] not in [0, 1]:
                results['errors'].append(f"Invalid mode at step {step}: {action[1]}")

            obs, rewards, dones, info = env.step([action])  # Single agent
            rewards_received.append(float(rewards[0]))

            if env.done:
                break

        results['stats'] = {
            'steps_run': len(actions_taken),
            'total_reward': sum(rewards_received),
            'avg_reward': sum(rewards_received) / len(rewards_received) if rewards_received else 0,
            'game_completed': env.done
        }

    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Runtime error: {e}")

    return results
