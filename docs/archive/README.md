# Archived Documentation

This directory contains documentation for features that have been removed or significantly changed during the project simplification.

## Archived Documents

### AGENT_SUBMISSION_GUIDE.md
Documentation for the community agent submission system, which allowed users to submit custom agents for validation and inclusion in tournaments. This feature was removed to simplify the project and focus on core research capabilities.

### AGENT_REGISTRY_API.md
API documentation for the agent registry backend service. The backend infrastructure (PostgreSQL, FastAPI, job queue) was removed in favor of a static site architecture with local computation.

### DATABASE_SETUP.md
Setup guide for PostgreSQL database used by the agent registry. No longer needed as the project now uses JSON files for storing statistical summaries instead of a database.

## Why Were These Removed?

In November 2024, the project underwent a simplification to focus on core research goals:

1. **Running large-scale tournaments** (1000+ games) with statistical validation
2. **Extracting individual policy performance** through rigorous analysis
3. **Evolving optimal heuristic agents** via evolutionary algorithms
4. **Training neural network policies** using PufferLib/PPO

The agent submission infrastructure, while valuable for community engagement, added significant complexity that distracted from these research goals. The simplification removed:

- Backend services (PostgreSQL, FastAPI, job queue)
- Agent validation and sandboxing system
- Community agent registry
- Tournament scheduling infrastructure

## Current Focus

The project now focuses on:

- **Statistical analysis tools** for experiment validation
- **Simple web demo** showing single-game visualization
- **Research workflows** for comparing teams and scenarios
- **RL training** for learning optimal policies

## Accessing Old Features

If you need to reference the old agent submission system:

1. Check out commit `6809999` or earlier (before simplification)
2. See PRs #70, #71, #72 for details on what was removed
3. Review these archived docs for implementation details

The core game engine, scenarios, agents, and RL training remain unchanged and fully functional.
