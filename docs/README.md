# Bucket Brigade Documentation

## Overview

This documentation directory contains detailed guides, specifications, and research materials for the Bucket Brigade platform.

## Quick Start

- **[Game Mechanics](game_mechanics.md)** - Complete rules and game mechanics
- **[Agent Roster](AGENT_ROSTER.md)** - Available AI agents (Firefighter, Hero, Free Rider, etc.)
- **[API Reference](../API.md)** - Technical API and data structures

## Documentation Map

### 🎮 Game & Design
- **[game_mechanics.md](game_mechanics.md)** - Canonical game rules and mechanics
- **[AGENT_ROSTER.md](AGENT_ROSTER.md)** - AI agent specifications and behaviors
- **[HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md)** - Parameter optimization guides

### 🧠 Research & Analysis
- **[../RANKING_METHODOLOGY.md](../RANKING_METHODOLOGY.md)** - Policy ranking and evaluation
- **[../SCENARIO_RESEARCH.md](../SCENARIO_RESEARCH.md)** - Scenario-based research framework
- **[curriculum_learning.md](curriculum_learning.md)** - Learning curriculum design

### 🔭 Research Vision & Philosophy
- **[background_closed_vs_open_world.md](background_closed_vs_open_world.md)** - Closed vs. open world learning foundations
- **[technical_marl_review.md](technical_marl_review.md)** - Comprehensive technical MARL review
- **[vision_future_directions.md](vision_future_directions.md)** - Long-term research questions and goals
- **[roadmap_phased_plan.md](roadmap_phased_plan.md)** - Phased roadmap to meta-game exploration
- **[library/](library/)** - Research paper summaries and references

### 💻 Development & Implementation
- **[../API.md](../API.md)** - API reference and data structures
- **[IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)** - Development checklist
- **[PERFORMANCE.md](PERFORMANCE.md)** - Performance analysis and benchmarks

### 🏗️ Architecture & Planning
- **[development/CLASS_DESIGN.md](development/CLASS_DESIGN.md)** - Class structure and API design
- **[SIMPLIFIED_ARCHITECTURE.md](SIMPLIFIED_ARCHITECTURE.md)** - Architecture overview
- **[WEB_UI_MOCKUP.md](WEB_UI_MOCKUP.md)** - Web interface design

### 📋 Project Management
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- **[../LOOM_AGENTS.md](../LOOM_AGENTS.md)** - AI-powered development workflow

## Key Concepts

### Game Elements
- **Houses**: 10 circular positions, each Safe, Burning, or Ruined
- **Agents**: 4-10 players, each owning one house
- **Signals**: Work/Rest intentions (may be deceptive)
- **Actions**: Choose house and work/rest mode
- **Scenarios**: 12 named configurations testing different cooperation aspects

### Agent Types
- **Heuristic Agents**: Parameterized behavioral models (Firefighter, Coordinator, etc.)
- **RL Agents**: Neural network policies trained with PPO
- **Evolutionary Agents**: Optimized through genetic algorithms

### Research Methods
- **Tournament Play**: Large-scale mixed-team competitions
- **Nash Equilibrium**: Strategic analysis of optimal play
- **Ranking**: Statistical evaluation of agent performance

## Contributing to Documentation

1. **Canonical Sources**: Reference [game_mechanics.md](game_mechanics.md) for game rules
2. **Cross-References**: Link to related documents for context
3. **Consistent Terminology**: Use standardized terms (Safe/Burning/Ruined, not SAFE/BURNING/RUINED)
4. **Clear Structure**: Follow established document patterns

## Directory Structure

```
docs/
├── game_mechanics.md                   # Canonical game rules
├── AGENT_ROSTER.md                     # AI agent specifications
├── HYPERPARAMETER_TUNING.md            # Parameter optimization
├── curriculum_learning.md              # Learning curriculum design
├── background_closed_vs_open_world.md  # Closed vs. open world foundations
├── technical_marl_review.md            # Technical MARL methods review
├── vision_future_directions.md         # Long-term research vision
├── roadmap_phased_plan.md              # Phased implementation roadmap
├── IMPLEMENTATION_CHECKLIST.md         # Development checklist
├── PERFORMANCE.md                      # Performance analysis
├── SIMPLIFIED_ARCHITECTURE.md          # Architecture overview
├── WEB_UI_MOCKUP.md                    # Web interface design
├── library/                            # Research paper summaries
│   ├── README.md                       # Library index and search guide
│   ├── marl/                           # Multi-agent RL papers
│   ├── safety/                         # AI safety papers
│   ├── evolution/                      # Evolutionary algorithms
│   ├── communication/                  # Emergent communication
│   ├── open-world/                     # Open-world environments
│   ├── game-theory/                    # Nash equilibria, norms
│   └── templates/                      # Paper summary template
├── archive/                            # Deprecated documents
├── development/                        # Development planning
├── features/                           # Feature specifications
├── game-design/                        # Game design documents
└── implementation/                     # Implementation details
```

---

*For the main project README, see [../README.md](../README.md).*
*For development workflow, see [../LOOM_AGENTS.md](../LOOM_AGENTS.md).*
