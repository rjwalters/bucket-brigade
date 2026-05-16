# 📖 Glossary of Terms

This document defines canonical terminology used throughout the Bucket Brigade project to ensure consistency across documentation and code.

---

## Core Game Concepts

### Policy / Strategy
**Canonical Term:** Policy or Strategy (interchangeable)

The decision-making logic used by a player in the bucket brigade game. Policies determine what actions to take based on observations.

**Types:**
- **Heuristic Policy**: Rule-based logic with configurable parameters
- **Neural Policy**: Deep learning model trained via RL
- **Evolved Policy**: Heuristic policy optimized through genetic algorithms

**Avoid:** "Agent" (ambiguous with Loom roles), "Individual" (evolution-specific), "Genome" (refers to parameters, not the policy itself)

**Examples:**
- ✅ "The firefighter policy prioritizes burning houses"
- ✅ "Evolved strategies outperformed handcrafted policies"
- ❌ "The agent cooperated with other agents"

---

### Scenario
**Canonical Term:** Scenario

A named configuration of game parameters that creates a specific strategic environment.

**Components:**
- Fire spread probability (β)
- Solo extinguish probability (κ)
- Spark probability (p)
- Rewards/penalties (A, L, c)
- Number of nights (N)

**Related Terms:**
- **Scenario Instance / Game**: A single playthrough of a scenario
- **Episode**: Same as game (RL terminology)

**Avoid:** "Environment" (too general), "Configuration" (too generic)

**Examples:**
- ✅ "The trivial_cooperation scenario has κ=0.95"
- ✅ "We ran 100 games of the chain_reaction scenario"
- ❌ "The environment parameters were adjusted"

---

### Genome / Parameters
**Canonical Term:** Genome (evolution context), Parameters (general context)

The numerical values that configure a heuristic policy's behavior.

**Context:**
- **Evolution**: genome (array of 10 floats in [0,1])
- **General**: policy parameters

**Avoid:** Mixing "genome" and "parameters" in the same context

**Examples:**
- ✅ "The evolved genome encodes work tendency and risk tolerance"
- ✅ "Policy parameters are optimized through evolution"
- ❌ "The agent's parameters define its genome"

---

### Free-Riding
**Canonical Term:** Free-riding (hyphenated)

The strategy of benefiting from collective action without contributing. In Bucket Brigade, this means resting while others extinguish fires.

**Variants to avoid:** Free rider, freeriding, free riding

**Examples:**
- ✅ "Free-riding is the dominant Nash equilibrium"
- ❌ "Free riders benefit from cooperation"

---

## Development & Tooling

### Loom Role
**Canonical Term:** Loom role (when referring to development automation)

An automated development worker that performs specific tasks in the Loom orchestration system.

**Types:**
- Builder, Judge, Curator, Architect, Hermit, Doctor, Guide, Champion, Driver

**Avoid:** "Agent" (conflicts with game policies), "Worker" (too generic)

**Examples:**
- ✅ "The Builder role implements new features"
- ✅ "Loom roles coordinate through GitHub labels"
- ❌ "The agent will fix the tests"

---

### Individual
**Canonical Term:** Individual (evolution-specific only)

A member of an evolutionary population, containing a genome and fitness score.

**Context:** Use only in evolution algorithm discussions, not for general policy references.

**Examples:**
- ✅ "The genetic algorithm evaluates each individual's fitness"
- ✅ "Top individuals are selected for reproduction"
- ❌ "Each individual plays 10 games" (use "each policy" instead)

---

## Research Terminology

### Nash Equilibrium
**Canonical Term:** Nash equilibrium

A strategic profile where no player can improve their payoff by unilaterally changing strategy.

**Types:**
- **Pure Nash**: Single deterministic strategy per player
- **Mixed Nash**: Probability distribution over strategies

**Related:**
- **Symmetric Nash**: All players use the same strategy/distribution
- **Support**: Set of strategies with non-zero probability in mixed Nash

**Examples:**
- ✅ "Free-riding is a pure Nash equilibrium"
- ✅ "The mixed Nash equilibrium assigns 70% to cooperation"

---

### Evolution
**Canonical Term:** Evolution or Evolutionary Algorithm

The process of optimizing policy parameters through genetic algorithms.

**Key Terms:**
- **Population**: Set of genomes being evolved
- **Generation**: One iteration of evolution (evaluate, select, breed, mutate)
- **Fitness**: Performance metric for a genome (mean scenario payoff)
- **Selection**: Choosing parent genomes based on fitness
- **Crossover**: Combining parent genomes to create offspring
- **Mutation**: Random perturbation of genome values

**Phases:**
- **Phase 1.0**: Specialist evolution (one scenario)
- **Phase 1.5**: Generalist evolution (multiple scenarios)

**Examples:**
- ✅ "Evolution discovered a cooperation strategy"
- ✅ "The population converged after 50 generations"

---

### Training
**Canonical Term:** Training (reinforcement learning context)

The process of optimizing a neural network policy through RL algorithms (PPO).

**Related:**
- **Timestep**: One environment interaction
- **Batch**: Set of experiences used for one policy update
- **Episode**: One complete game playthrough
- **Reward**: Feedback signal for the RL algorithm

**Avoid:** Confusing with evolution

**Examples:**
- ✅ "Neural policies are trained with PPO"
- ✅ "Training achieved 1M timesteps"
- ❌ "We evolved a neural network"

---

## Architecture Terms

### Rust Core
**Canonical Term:** bucket-brigade-core or Rust core

The high-performance Rust implementation of the game engine.

**Features:**
- Native Python bindings (PyO3)
- WebAssembly compilation
- 100x speedup over Python

**Related:**
- **WASM**: Browser-compiled Rust for web interface
- **PyO3**: Python bindings for the Rust implementation

**Examples:**
- ✅ "The Rust core provides a 100x speedup"
- ✅ "WASM enables browser-based simulations"

---

### Frontend vs Backend
**Current Status:** Browser-only static site (no backend)

**Terms:**
- **Web Interface**: Browser-based visualization and interaction
- **WASM**: Client-side game engine
- **Backend API**: Planned future server component (not implemented)

**Examples:**
- ✅ "The web interface uses WASM for game simulation"
- ⚠️ "The backend API endpoints are proposed but not implemented"

---

## Quick Reference Table

| Concept | Use | Don't Use | Context |
|---------|-----|-----------|---------|
| Game decision-maker | Policy, Strategy | Agent | General |
| Evolution member | Individual | Agent, Genome | Evolution only |
| Decision parameters | Genome | Parameters | Evolution only |
| Decision parameters | Parameters | Genome | General |
| Named config | Scenario | Environment | Game setup |
| Single playthrough | Game, Episode | Simulation | Gameplay |
| Selfish strategy | Free-riding | Free rider | Game theory |
| Dev automation | Loom role | Agent | Development |
| RL optimization | Training | Evolution | Neural nets |
| GA optimization | Evolution | Training | Heuristics |
| Rust implementation | Rust core, bucket-brigade-core | Backend | Architecture |

---

## Usage Guidelines

1. **Context Matters**: Use the term that's clearest in context
2. **Consistency Within Documents**: Don't mix synonyms in the same document
3. **Favor Specificity**: "Evolved policy" > "strategy" > "agent"
4. **Cross-Reference**: Link to this glossary when introducing technical terms
5. **Update As Needed**: Propose glossary changes via pull request

---

## Contributing

If you find terminology inconsistencies or want to propose new canonical terms:

1. Check this glossary first
2. Use the canonical term in your contribution
3. If you need a new term, open an issue or PR to update this glossary

---

*Last Updated: 2025-11-05*
