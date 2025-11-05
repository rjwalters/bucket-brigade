# Contributing to Bucket Brigade

Thank you for your interest in contributing to Bucket Brigade! This document provides guidelines and information for contributors.

## 🏗️ Development Workflow

We use **Loom** for AI-powered development orchestration. See [CLAUDE.md](CLAUDE.md) and [LOOM_AGENTS.md](LOOM_AGENTS.md) for details on our automated development workflow.

### For Contributors

1. **Fork the repository** and create a feature branch
2. **Follow the Loom workflow** for coordinated development
3. **Test thoroughly** before submitting PRs
4. **Use conventional commits** for clear change tracking

## 📋 Contribution Types

### 🔧 Code Contributions

- **Bug fixes**: Fix issues in existing functionality
- **Features**: Implement new capabilities following existing patterns
- **Refactoring**: Improve code quality without changing behavior
- **Documentation**: Improve or add documentation

### 🎮 Agent Development

- **Custom agents**: Submit new agent strategies for evaluation
- **Agent improvements**: Enhance existing heuristic agents
- **RL policies**: Train and submit learned policies

### 📊 Research Contributions

- **Scenario design**: Propose new test scenarios
- **Analysis**: Provide statistical analysis of agent performance
- **Benchmarking**: Compare against existing methods

## 🚀 Getting Started

### Prerequisites

- **Python 3.9 - 3.13** (3.14+ not yet supported due to PyO3 0.22.6 limitations)
  - We recommend Python 3.12 (see `.python-version`)
  - Use `uv` package manager for best experience
- Node.js 18+ with pnpm
- Git
- Rust toolchain (for building `bucket-brigade-core`)

### Local Development Setup

```bash
# Clone and setup
git clone https://github.com/your-username/bucket-brigade.git
cd bucket-brigade

# Install all dependencies
pnpm run install:all

# Run tests
pnpm run test

# Start development servers
pnpm run dev
```

### Development Commands

```bash
# Python development
uv run pytest                    # Run tests
uv run ruff format .             # Format code
uv run ruff check . --fix        # Lint code
uv run mypy .                    # Type check

# Web development
pnpm run format                  # Format web code
pnpm run lint:biome              # Lint web code
pnpm run typecheck               # Type check web code

# Full project
pnpm run test                     # Run all tests
pnpm run format                   # Format all code
pnpm run lint:fix                 # Fix all linting issues
pnpm run typecheck                # Type check everything
```

## 🧪 Testing

### Python Tests
```bash
uv run pytest tests/              # Run all tests
uv run pytest tests/test_environment.py  # Run specific test
uv run pytest --cov=bucket_brigade  # With coverage
```

### Web Tests
```bash
pnpm run test                    # Run Playwright tests
pnpm run test:headed            # Run in visible browser
pnpm run test:ui                # Run with test UI
```

### Integration Tests
```bash
# Test agent submission pipeline
uv run python scripts/submit_agent.py my_agent.py

# Test scenario validation
uv run python scripts/test_scenarios.py trivial_cooperation
```

## 🤖 Agent Submission

### Requirements

Your agent must:
- Inherit from `AgentBase`
- Implement `act(obs: dict) -> np.ndarray` method
- Return actions as `[house_index, mode_flag]`
- Handle observation format correctly

### Validation Process

```bash
# 1. Create agent template
uv run python scripts/submit_agent.py --create-template

# 2. Implement your strategy in my_agent.py
# 3. Validate and test
uv run python scripts/submit_agent.py my_agent.py
```

### Agent Interface

```python
class MyAgent(AgentBase):
    def __init__(self, agent_id: int, name: str = "MyAgent"):
        super().__init__(agent_id, name)

    def reset(self):
        # Reset between games
        pass

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        # obs contains: signals, locations, houses, last_actions, scenario_info
        return np.array([0, 1])  # [house_index, mode_flag]
```

## 📝 Pull Request Process

1. **Create a branch** from `main` following the pattern:
   - `feature/description-of-feature`
   - `fix/issue-description`
   - `docs/update-documentation`

2. **Make changes** following our coding standards

3. **Test thoroughly**:
   - All existing tests pass
   - New functionality has tests
   - Integration tests work

4. **Update documentation** if needed

5. **Create PR** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/videos for UI changes

6. **Wait for review** - our Judge agent will review automatically

## 🎯 Coding Standards

### Python

- **Formatting**: Ruff format with Black-compatible settings
- **Linting**: Ruff with our configuration
- **Types**: Full type annotations, checked with mypy
- **Imports**: Absolute imports, grouped by standard library → third-party → local

### TypeScript/React

- **Formatting**: Prettier with our config
- **Linting**: Biome for fast, comprehensive checking
- **Types**: Strict TypeScript configuration
- **Components**: Functional components with hooks

### General

- **Commits**: Use conventional commit format
- **Documentation**: Update docs for any API changes
- **Security**: No hardcoded secrets or sensitive data

## 🔍 Code Review Process

### Automated Review (Loom)

Our **Judge** agent automatically reviews PRs for:
- Code quality and style
- Test coverage
- Documentation updates
- Breaking changes

### Human Review

For complex changes, human review focuses on:
- Architectural decisions
- Performance implications
- Security considerations
- Research methodology

## 🐛 Reporting Issues

### Bug Reports

Please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python/Node versions)
- Relevant logs or error messages

### Feature Requests

Please include:
- Clear description of the proposed feature
- Use case and motivation
- Potential implementation approach
- Impact on existing functionality

## 📊 Performance Contributions

When contributing performance improvements:

1. **Benchmark** before and after changes
2. **Document** the improvement metrics
3. **Consider** memory usage and scalability
4. **Test** across different scenarios

## 🎓 Research Contributions

For academic or research contributions:

1. **Methodology**: Clearly document experimental setup
2. **Reproducibility**: Provide scripts and configurations
3. **Validation**: Compare against established baselines
4. **Documentation**: Include in research papers or technical reports

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and features
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check our docs in the `docs/` directory

## 🙏 Recognition

Contributors are recognized through:
- GitHub contributor statistics
- Attribution in release notes
- Research paper acknowledgments
- Agent leaderboard credits

Thank you for contributing to Bucket Brigade! 🚀
