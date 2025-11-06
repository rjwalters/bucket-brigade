# Scenario-Based Research Framework

> ⚠️ **PROPOSED FRAMEWORK - NOT YET IMPLEMENTED**
>
> This document describes a comprehensive research organization framework that **has not been built yet**. The directory structure, automation scripts, and workflows described below are proposals for future implementation.
>
> **For current research structure**, see:
> - [experiments/README.md](experiments/README.md) - Current experiment organization
> - [experiments/RESEARCH_SUMMARY.md](experiments/RESEARCH_SUMMARY.md) - Actual research findings
> - [experiments/nash/README.md](experiments/nash/README.md) - Nash equilibrium research
> - [experiments/evolution/README.md](experiments/evolution/README.md) - Evolution research

## Vision & Motivation

The Bucket Brigade game features carefully designed **named scenarios** - each with hand-tuned parameters that tell a specific story about coordination, cooperation, and strategic behavior. These scenarios serve as controlled experimental environments for studying multi-agent dynamics.

This document outlines a comprehensive research framework for:

1. **Understanding** - What makes each scenario unique?
2. **Prediction** - What strategies should theoretically work best?
3. **Discovery** - Can we find optimal strategies through evolution?
4. **Validation** - Do empirical results match theoretical predictions?

## Current State

### Named Scenarios (12 total)

From `bucket_brigade/envs/scenarios.py`:

1. **trivial_cooperation** - Easy cooperation, fires extinguish readily
2. **early_containment** - Aggressive start, requires early coordination
3. **greedy_neighbor** - Social dilemma, high work cost creates free-riding incentive
4. **sparse_heroics** - Few workers needed, efficiency matters
5. **rest_trap** - Fires usually self-extinguish, but not always
6. **chain_reaction** - High spread requires distributed teams
7. **deceptive_calm** - Occasional flare-ups reward honest signaling
8. **overcrowding** - Too many workers reduce efficiency
9. **mixed_motivation** - Self-interest conflicts with team optimum
10. **default** - Balanced baseline scenario
11. **easy** - Easier cooperation task
12. **hard** - More challenging cooperation task

### Existing Analysis Tools

- `scripts/test_scenarios.py` - Tests with hand-tuned heuristic agents
- `scripts/evolve_agents.py` - Genetic algorithm for strategy discovery
- `scripts/analyze_nash_equilibrium.py` - Nash equilibrium via Double Oracle

### The Gap

These tools exist but work in isolation. We need:
- **Unified framework** for running all three analyses per scenario
- **Structured outputs** (JSON + Markdown) for comparison and visualization
- **Research scripts** that orchestrate multi-method analysis
- **Web integration** for interactive exploration of results

## Proposed Directory Structure

```
experiments/
├── scenarios/
│   ├── trivial_cooperation/
│   │   ├── config.json              # Scenario parameters & research config
│   │   ├── heuristics/              # Hand-tuned baseline agents
│   │   │   ├── agents.json          # Defined heuristic agent parameters
│   │   │   ├── results.json         # Performance data (payoffs, win rates)
│   │   │   └── report.md            # Human-readable analysis
│   │   ├── evolved/                 # Evolutionary algorithm results
│   │   │   ├── generation_0000/
│   │   │   │   ├── population.json  # Population snapshot
│   │   │   │   └── stats.json       # Diversity, fitness stats
│   │   │   ├── generation_0050/
│   │   │   ├── best_agent.json      # Champion from evolution
│   │   │   ├── evolution_trace.json # Full evolutionary trajectory
│   │   │   └── report.md            # Analysis of evolutionary dynamics
│   │   ├── nash/                    # Nash equilibrium analysis
│   │   │   ├── equilibrium.json     # Nash strategy distribution
│   │   │   ├── payoff_matrix.json   # Strategy payoffs
│   │   │   ├── convergence.json     # Double Oracle iterations
│   │   │   └── report.md            # Game-theoretic interpretation
│   │   ├── comparison/              # Cross-method analysis
│   │   │   ├── tournament.json      # Head-to-head matchups
│   │   │   ├── strategy_distance.json # How similar are strategies?
│   │   │   └── report.md            # Synthesis & insights
│   │   └── README.md                # Scenario story & research questions
│   │
│   ├── greedy_neighbor/             # Same structure for each scenario
│   │   ├── config.json
│   │   ├── heuristics/
│   │   ├── evolved/
│   │   ├── nash/
│   │   ├── comparison/
│   │   └── README.md
│   │
│   └── [... 7 more scenarios ...]
│
├── scripts/                         # Research orchestration scripts
│   ├── run_scenario_research.py     # Master script for full analysis
│   ├── analyze_heuristics.py        # Evaluate hand-tuned agents
│   ├── run_evolution.py             # Run evolutionary optimization
│   ├── compute_nash.py              # Nash equilibrium computation
│   ├── cross_scenario_analysis.py   # Compare across scenarios
│   └── generate_web_bundle.py       # Package for web visualization
│
└── web_data/                        # Aggregated data for web UI
    ├── scenario_index.json          # All scenarios metadata
    ├── scenario_summaries.json      # Key findings per scenario
    └── full_dataset.json            # Complete research data
```

## Research Dimensions

### 1. Hand-Tuned Heuristics (Baseline)

**Purpose**: Establish ground truth with interpretable strategies

**Agents** (from `bucket_brigade/agents/archetypes.py`):
- **Firefighter** - High cooperation, honest signaling
- **Free Rider** - Selfish, minimal work
- **Hero** - Always works, altruistic
- **Coordinator** - High trust in signals, balances cooperation
- **Liar** - Deceptive signaling

**Analysis**:
- All-vs-all tournaments (every pairing)
- Individual payoffs in mixed teams
- Stability of team compositions

**Output** (`heuristics/results.json`):
```json
{
  "scenario": "greedy_neighbor",
  "agents": [
    {"name": "firefighter", "params": [1.0, 0.9, ...]},
    {"name": "free_rider", "params": [0.7, 0.2, ...]}
  ],
  "tournament_results": [
    {
      "team": [0, 0, 0, 0],  // All firefighters
      "payoffs": [45.2, 44.8, 45.5, 44.9],
      "mean_payoff": 45.1,
      "saved_houses": 8.2,
      "ruined_houses": 1.8
    },
    {
      "team": [0, 0, 0, 1],  // 3 firefighters + 1 free rider
      "payoffs": [42.1, 41.9, 42.3, 51.2],  // Free rider exploits
      "mean_payoff": 44.4
    }
  ],
  "rankings": [
    {"agent": "free_rider", "mean_payoff": 48.5},
    {"agent": "firefighter", "mean_payoff": 43.2}
  ]
}
```

### 2. Evolutionary Algorithms (Discovery)

**Purpose**: Discover optimal strategies through natural selection

**Algorithm**: Genetic algorithm with:
- Tournament selection
- Uniform/arithmetic crossover
- Gaussian mutation
- Diversity maintenance

**Generational Snapshots**: Capture population every N generations
- Population distribution in parameter space
- Diversity metrics (genetic variance)
- Fitness statistics (mean, max, min, std)

**Output** (`evolved/evolution_trace.json`):
```json
{
  "scenario": "greedy_neighbor",
  "config": {
    "population_size": 100,
    "num_generations": 200,
    "mutation_rate": 0.1,
    "crossover_rate": 0.7,
    "fitness_type": "mean_reward"
  },
  "generations": [
    {
      "generation": 0,
      "best_fitness": 38.2,
      "mean_fitness": 22.1,
      "diversity": 0.85
    },
    {
      "generation": 50,
      "best_fitness": 51.3,
      "mean_fitness": 44.7,
      "diversity": 0.42
    }
  ],
  "best_agent": {
    "generation": 187,
    "fitness": 52.8,
    "params": [0.62, 0.31, 0.15, 0.88, 0.42, 0.19, 0.08, 0.05, 0.71, 0.23],
    "classification": "Balanced Free Rider"
  },
  "convergence": {
    "converged": true,
    "generation": 187,
    "plateau_length": 15
  }
}
```

### 3. Nash Equilibrium Analysis (Theory)

**Purpose**: Predict stable strategic configurations

**Algorithm**: Double Oracle
- Start with archetypal strategies
- Iteratively add best responses
- Solve restricted matrix game via linear programming
- Converge to Nash equilibrium

**Output** (`nash/equilibrium.json`):
```json
{
  "scenario": "greedy_neighbor",
  "equilibrium": {
    "type": "mixed",  // "pure" or "mixed"
    "support_size": 2,
    "distribution": {
      "0": 0.62,  // Index to strategy pool
      "1": 0.38
    },
    "expected_payoff": 46.3
  },
  "strategy_pool": [
    {
      "index": 0,
      "params": [0.88, 0.71, 0.52, 0.63, 0.48, 0.71, 0.11, 0.02, 0.22, 0.68],
      "classification": "Cooperative",
      "archetype_distance": {"firefighter": 0.21, "coordinator": 0.18}
    },
    {
      "index": 1,
      "params": [0.31, 0.19, 0.08, 0.91, 0.21, 0.12, 0.09, 0.01, 0.88, 0.11],
      "classification": "Free Rider",
      "archetype_distance": {"free_rider": 0.15}
    }
  ],
  "convergence": {
    "iterations": 12,
    "converged": true,
    "final_improvement": 0.003
  },
  "interpretation": {
    "equilibrium_type": "social_dilemma",
    "cooperation_rate": 0.62,
    "free_riding_rate": 0.38,
    "insight": "High work cost (c=1.0) creates mixed equilibrium with 62% cooperation"
  }
}
```

### 4. Cross-Method Comparison

**Purpose**: Synthesize findings across all approaches

**Analysis**:
- **Tournament**: Evolved agent vs Nash strategies vs Heuristics
- **Strategy Distance**: How similar are the discovered strategies?
- **Performance Gap**: Does evolution find Nash? Better than Nash?
- **Robustness**: Which strategies are most stable?

**Output** (`comparison/tournament.json`):
```json
{
  "scenario": "greedy_neighbor",
  "contestants": [
    {"id": "heuristic_firefighter", "source": "heuristic"},
    {"id": "heuristic_free_rider", "source": "heuristic"},
    {"id": "evolved_champion", "source": "evolution"},
    {"id": "nash_mixed_0", "source": "nash", "probability": 0.62},
    {"id": "nash_mixed_1", "source": "nash", "probability": 0.38}
  ],
  "matchups": [
    {
      "team": ["evolved_champion", "evolved_champion", "evolved_champion", "evolved_champion"],
      "mean_payoff": 52.8,
      "individual_payoffs": [52.9, 52.7, 52.8, 52.8]
    },
    {
      "team": ["nash_mixed_0", "nash_mixed_0", "nash_mixed_0", "nash_mixed_0"],
      "mean_payoff": 46.3,
      "individual_payoffs": [46.1, 46.4, 46.3, 46.2]
    }
  ],
  "strategy_distances": {
    "evolved_vs_nash_cooperative": 0.18,
    "evolved_vs_nash_free_rider": 0.42,
    "evolved_vs_heuristic_free_rider": 0.12
  },
  "findings": {
    "evolution_finds_nash": false,
    "evolution_payoff_advantage": 6.5,
    "closest_to_evolved": "heuristic_free_rider",
    "insight": "Evolution discovered strategy similar to free rider but with higher coordination"
  }
}
```

## Research Scripts

### 1. Master Orchestration Script

**`experiments/scripts/run_scenario_research.py`**

```python
"""
Run complete research pipeline for a scenario.

Usage:
    python experiments/scripts/run_scenario_research.py trivial_cooperation
    python experiments/scripts/run_scenario_research.py --all
    python experiments/scripts/run_scenario_research.py greedy_neighbor --skip-evolution
"""

def run_scenario_research(
    scenario_name: str,
    skip_heuristics: bool = False,
    skip_evolution: bool = False,
    skip_nash: bool = False,
    skip_comparison: bool = False,
):
    """Run full research pipeline for a scenario."""

    scenario_dir = Path(f"experiments/scenarios/{scenario_name}")
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # 1. Heuristic baseline
    if not skip_heuristics:
        print(f"[1/4] Analyzing hand-tuned heuristics...")
        run_heuristic_analysis(scenario_name, scenario_dir / "heuristics")

    # 2. Evolutionary optimization
    if not skip_evolution:
        print(f"[2/4] Running evolutionary algorithm...")
        run_evolution(scenario_name, scenario_dir / "evolved")

    # 3. Nash equilibrium
    if not skip_nash:
        print(f"[3/4] Computing Nash equilibrium...")
        compute_nash_equilibrium(scenario_name, scenario_dir / "nash")

    # 4. Cross-method comparison
    if not skip_comparison:
        print(f"[4/4] Cross-method comparison...")
        run_comparison(scenario_name, scenario_dir / "comparison")

    # Generate reports
    generate_scenario_reports(scenario_name, scenario_dir)

    print(f"\n✅ Research complete for {scenario_name}")
    print(f"📂 Results: {scenario_dir}")
```

### 2. Heuristic Analysis

**`experiments/scripts/analyze_heuristics.py`**

```python
"""
Evaluate hand-tuned heuristic agents in a scenario.

Output:
  - heuristics/agents.json
  - heuristics/results.json
  - heuristics/report.md
"""

def analyze_heuristics(scenario_name: str, output_dir: Path):
    """Run all-vs-all tournament with heuristic archetypes."""

    scenario = get_scenario_by_name(scenario_name, num_agents=4)

    # Define agent pool
    agents = [
        ("firefighter", FIREFIGHTER_PARAMS),
        ("free_rider", FREE_RIDER_PARAMS),
        ("hero", HERO_PARAMS),
        ("coordinator", COORDINATOR_PARAMS),
        ("liar", LIAR_PARAMS),
    ]

    # All possible team compositions (with replacement, 4 agents)
    results = []
    for team_composition in generate_team_compositions(agents, team_size=4):
        payoffs = run_tournament_match(team_composition, scenario, num_games=100)
        results.append({
            "team": team_composition,
            "payoffs": payoffs
        })

    # Save structured results
    save_json(output_dir / "results.json", results)

    # Generate markdown report
    generate_heuristic_report(results, scenario_name, output_dir / "report.md")
```

### 3. Evolution Wrapper

**`experiments/scripts/run_evolution.py`**

```python
"""
Run genetic algorithm to discover optimal strategies.

Output:
  - evolved/generation_XXXX/population.json (snapshots)
  - evolved/best_agent.json
  - evolved/evolution_trace.json
  - evolved/report.md
"""

def run_evolution(scenario_name: str, output_dir: Path):
    """Run evolutionary algorithm with generational snapshots."""

    scenario = get_scenario_by_name(scenario_name, num_agents=4)

    config = EvolutionConfig(
        population_size=100,
        num_generations=200,
        snapshot_interval=10,  # Save every 10 generations
        ...
    )

    ga = GeneticAlgorithm(config)

    def snapshot_callback(generation: int, population: Population):
        """Save population snapshot."""
        snapshot_dir = output_dir / f"generation_{generation:04d}"
        snapshot_dir.mkdir(exist_ok=True)
        save_population_snapshot(population, snapshot_dir)

    result = ga.evolve(progress_callback=snapshot_callback)

    # Save final results
    save_json(output_dir / "best_agent.json", result.best_individual.to_dict())
    save_json(output_dir / "evolution_trace.json", {
        "generations": result.fitness_history,
        "diversity": result.diversity_history,
        ...
    })

    generate_evolution_report(result, scenario_name, output_dir / "report.md")
```

### 4. Nash Computation

**`experiments/scripts/compute_nash.py`**

```python
"""
Compute Nash equilibrium using Double Oracle algorithm.

Output:
  - nash/equilibrium.json
  - nash/payoff_matrix.json
  - nash/convergence.json
  - nash/report.md
"""

def compute_nash_equilibrium(scenario_name: str, output_dir: Path):
    """Compute Nash equilibrium via Double Oracle."""

    scenario = get_scenario_by_name(scenario_name, num_agents=4)

    solver = DoubleOracle(
        scenario=scenario,
        num_simulations=1000,
        max_iterations=50,
        verbose=True
    )

    equilibrium = solver.solve()

    # Save results
    save_json(output_dir / "equilibrium.json", {
        "type": "mixed" if len(equilibrium.distribution) > 1 else "pure",
        "distribution": equilibrium.distribution,
        "strategy_pool": [s.tolist() for s in equilibrium.strategy_pool],
        "expected_payoff": equilibrium.payoff,
        ...
    })

    save_json(output_dir / "convergence.json", {
        "iterations": equilibrium.iterations,
        "converged": equilibrium.converged,
        ...
    })

    generate_nash_report(equilibrium, scenario_name, output_dir / "report.md")
```

### 5. Cross-Scenario Analysis

**`experiments/scripts/cross_scenario_analysis.py`**

```python
"""
Analyze trends and patterns across all scenarios.

Output:
  - experiments/cross_scenario_summary.json
  - experiments/cross_scenario_report.md
"""

def cross_scenario_analysis(scenario_names: list[str]):
    """Compare results across multiple scenarios."""

    summary = {
        "scenarios": [],
        "findings": {}
    }

    for scenario_name in scenario_names:
        scenario_dir = Path(f"experiments/scenarios/{scenario_name}")

        # Load results
        heuristic_results = load_json(scenario_dir / "heuristics" / "results.json")
        evolved_results = load_json(scenario_dir / "evolved" / "best_agent.json")
        nash_results = load_json(scenario_dir / "nash" / "equilibrium.json")

        # Aggregate
        summary["scenarios"].append({
            "name": scenario_name,
            "nash_type": nash_results["type"],
            "evolved_fitness": evolved_results["fitness"],
            "best_heuristic": find_best_heuristic(heuristic_results),
            ...
        })

    # Identify patterns
    summary["findings"] = {
        "pure_nash_scenarios": [s["name"] for s in summary["scenarios"] if s["nash_type"] == "pure"],
        "mixed_nash_scenarios": [s["name"] for s in summary["scenarios"] if s["nash_type"] == "mixed"],
        "evolution_beats_nash": count_evolution_advantage(summary["scenarios"]),
        ...
    }

    save_json("experiments/cross_scenario_summary.json", summary)
    generate_cross_scenario_report(summary, "experiments/cross_scenario_report.md")
```

## Output Formats

### JSON Schema

**Structured data for programmatic access and web visualization**

```typescript
// experiments/schemas/scenario_config.json
interface ScenarioConfig {
  name: string;
  parameters: {
    beta: number;
    kappa: number;
    c: number;
    // ... full scenario params
  };
  research_config: {
    num_heuristic_games: number;
    evolution_generations: number;
    nash_simulations: number;
  };
  story: string;
  research_questions: string[];
}

// experiments/schemas/heuristic_results.json
interface HeuristicResults {
  scenario: string;
  agents: Agent[];
  tournament_results: TournamentMatch[];
  rankings: AgentRanking[];
}

// experiments/schemas/evolution_trace.json
interface EvolutionTrace {
  scenario: string;
  config: EvolutionConfig;
  generations: GenerationSnapshot[];
  best_agent: Individual;
  convergence: ConvergenceInfo;
}

// experiments/schemas/nash_equilibrium.json
interface NashEquilibrium {
  scenario: string;
  equilibrium: {
    type: "pure" | "mixed";
    support_size: number;
    distribution: Record<string, number>;
    expected_payoff: number;
  };
  strategy_pool: Strategy[];
  convergence: ConvergenceInfo;
  interpretation: GameTheoreticInterpretation;
}
```

### Markdown Reports

**Human-readable analysis for documentation and papers**

**Example: `experiments/scenarios/greedy_neighbor/nash/report.md`**

```markdown
# Nash Equilibrium Analysis: Greedy Neighbor

## Scenario Overview

The **Greedy Neighbor** scenario creates a social dilemma through high work cost (c=1.0) combined with moderate fire spread (β=0.15) and extinguish rate (κ=0.4).

**Key Parameters**:
- Fire spread (β): 0.15 (low)
- Extinguish efficiency (κ): 0.4 (moderate)
- Work cost (c): 1.0 (high) ← Creates free-riding incentive
- Number of agents: 4

## Equilibrium Results

**Type**: Mixed Strategy Nash Equilibrium

**Support**: 2 strategies

**Distribution**:
- Strategy 1 (Cooperative): 62%
- Strategy 2 (Free Rider): 38%

**Expected Payoff**: 46.3

### Strategy 1: Cooperative (62% probability)

**Parameters**:
```
honesty:          ████████████████████ 0.88
work_tendency:    ██████████████░░░░░░ 0.71
neighbor_help:    ██████████░░░░░░░░░░ 0.52
own_priority:     ████████████░░░░░░░░ 0.63
...
```

**Classification**: Cooperative Helper

**Closest Archetype**: Firefighter (distance: 0.21)

### Strategy 2: Free Rider (38% probability)

**Parameters**:
```
honesty:          ██████░░░░░░░░░░░░░░ 0.31
work_tendency:    ███░░░░░░░░░░░░░░░░░ 0.19
neighbor_help:    █░░░░░░░░░░░░░░░░░░░ 0.08
own_priority:     ██████████████████░░ 0.91
rest_bias:        █████████████████░░░ 0.88
...
```

**Classification**: Selfish Free Rider

**Closest Archetype**: Free Rider (distance: 0.15)

## Game-Theoretic Interpretation

### Why Mixed Equilibrium?

The high work cost (c=1.0) creates a **classic social dilemma**:

1. **If everyone cooperates**: Total payoff is high, but individuals can improve by free-riding
2. **If everyone free-rides**: Fires spread, total payoff collapses
3. **Equilibrium balances**: 62% cooperation prevents collapse while 38% free-riding is tolerated

### Mixing Indifference Condition

At equilibrium, both strategies earn **equal expected payoff**:

```
U(Cooperate | 62% cooperate) = U(Free-ride | 62% cooperate) = 46.3
```

This indifference is what allows mixing - if one strategy had higher payoff, all agents would switch to it.

### Stability Analysis

**Nash Property**: Neither strategy can improve payoff by deviating

- Cooperative agents: Can't gain by switching to free-riding (already balanced)
- Free-riding agents: Can't gain by switching to cooperation (already balanced)

**Evolutionarily Stable**: No mutant strategy can invade

## Convergence

**Algorithm**: Double Oracle

**Iterations**: 12

**Converged**: Yes

**Final Improvement**: 0.003 (below ε=0.01 threshold)

### Convergence Trace

| Iteration | Pool Size | Expected Payoff | Improvement |
|-----------|-----------|-----------------|-------------|
| 1         | 4         | 42.1            | -           |
| 5         | 6         | 45.2            | 0.62        |
| 10        | 8         | 46.1            | 0.18        |
| 12        | 9         | 46.3            | 0.003       |

## Predictions vs Reality

**Theoretical Prediction**: Mixed equilibrium with moderate cooperation

**Actual Result**: ✅ Confirmed

- **Cooperation rate**: 62% (theory predicted 55-70%)
- **Free-riding rate**: 38% (theory predicted 30-45%)
- **Social dilemma**: Clearly visible in payoff structure

## Comparison to Other Scenarios

| Scenario           | Equilibrium Type | Cooperation Rate |
|--------------------|------------------|------------------|
| Trivial Cooperation| Pure             | 100%             |
| **Greedy Neighbor**| **Mixed**        | **62%**          |
| Sparse Heroics     | Mixed            | 25%              |

The Greedy Neighbor sits in the middle - cooperation is valuable but costly.

---

*Analysis generated: 2025-11-03*
*Computation time: 87.3 seconds*
```

## Web Integration

### Data Bundle for Web UI

**`experiments/scripts/generate_web_bundle.py`**

```python
"""
Package scenario research data for web visualization.

Output:
  - experiments/web_data/scenario_index.json
  - experiments/web_data/full_dataset.json
"""

def generate_web_bundle():
    """Create JSON bundle for web UI consumption."""

    scenarios = []
    for scenario_dir in Path("experiments/scenarios").iterdir():
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name

        # Load all data
        config = load_json(scenario_dir / "config.json")
        heuristic = load_json(scenario_dir / "heuristics" / "results.json")
        evolved = load_json(scenario_dir / "evolved" / "evolution_trace.json")
        nash = load_json(scenario_dir / "nash" / "equilibrium.json")
        comparison = load_json(scenario_dir / "comparison" / "tournament.json")

        scenarios.append({
            "name": scenario_name,
            "config": config,
            "data": {
                "heuristics": heuristic,
                "evolution": evolved,
                "nash": nash,
                "comparison": comparison
            }
        })

    # Save aggregated data
    save_json("experiments/web_data/full_dataset.json", {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "scenarios": scenarios
    })

    # Create lightweight index
    save_json("experiments/web_data/scenario_index.json", [
        {
            "name": s["name"],
            "story": s["config"]["story"],
            "nash_type": s["data"]["nash"]["equilibrium"]["type"],
            "evolved_fitness": s["data"]["evolution"]["best_agent"]["fitness"]
        }
        for s in scenarios
    ])
```

### Web UI Features

The web application (already exists at `web/`) can consume this data to provide:

1. **Scenario Explorer**
   - Grid view of all scenarios
   - Filter by Nash type (pure/mixed), difficulty, cooperation level
   - Click to drill down into scenario details

2. **Scenario Detail View**
   - Interactive parameter visualization
   - Tabbed views: Heuristics | Evolution | Nash | Comparison
   - Strategy visualization (radar charts, parameter heatmaps)

3. **Evolution Visualization**
   - Animated fitness progression over generations
   - Population diversity scatter plot
   - Strategy parameter evolution (line charts)

4. **Nash Equilibrium Viz**
   - Strategy distribution pie chart
   - Payoff matrix heatmap
   - Best response graph (which strategies beat which)

5. **Cross-Scenario Comparison**
   - Scatter plot: work cost vs cooperation rate
   - Identify patterns (when does pure vs mixed equilibrium occur?)
   - Export findings as publication-ready figures

## Research Workflow

### Standard Workflow (Per Scenario)

```bash
# 1. Run complete analysis for a scenario
python experiments/scripts/run_scenario_research.py greedy_neighbor

# Output structure created:
# experiments/scenarios/greedy_neighbor/
#   ├── heuristics/results.json + report.md
#   ├── evolved/evolution_trace.json + report.md
#   ├── nash/equilibrium.json + report.md
#   └── comparison/tournament.json + report.md

# 2. View markdown reports
cat experiments/scenarios/greedy_neighbor/nash/report.md

# 3. Load JSON for programmatic analysis
python -c "
import json
with open('experiments/scenarios/greedy_neighbor/nash/equilibrium.json') as f:
    eq = json.load(f)
    print(f'Equilibrium type: {eq[\"equilibrium\"][\"type\"]}')
    print(f'Expected payoff: {eq[\"equilibrium\"][\"expected_payoff\"]}')
"
```

### Batch Workflow (All Scenarios)

```bash
# Run analysis for all scenarios
python experiments/scripts/run_scenario_research.py --all

# Cross-scenario analysis
python experiments/scripts/cross_scenario_analysis.py

# Generate web bundle
python experiments/scripts/generate_web_bundle.py

# View in web UI
cd web
npm run dev
# Navigate to http://localhost:5173/scenarios
```

### Incremental Workflow (Skip Steps)

```bash
# Already have heuristics and evolution, just compute Nash
python experiments/scripts/run_scenario_research.py greedy_neighbor \
  --skip-heuristics \
  --skip-evolution

# Only run evolution (for tweaking GA parameters)
python experiments/scripts/run_scenario_research.py greedy_neighbor \
  --skip-heuristics \
  --skip-nash \
  --skip-comparison
```

## Research Questions by Scenario

### Trivial Cooperation
- **Q**: Is pure cooperation the Nash equilibrium?
- **Q**: Can evolution converge to full cooperation?
- **Q**: How robust is cooperation to free-riders?

### Greedy Neighbor
- **Q**: What cooperation rate emerges in equilibrium?
- **Q**: Can evolution find the Nash mixed strategy?
- **Q**: At what work cost does cooperation collapse?

### Sparse Heroics
- **Q**: How many workers are optimal per fire?
- **Q**: Does Nash predict "minimal effort" strategies?
- **Q**: Can evolution discover efficient coordination?

### Deceptive Calm
- **Q**: Is honesty incentive-compatible in equilibrium?
- **Q**: Do evolved agents learn to trust signals?
- **Q**: What happens with heterogeneous honesty levels?

### Chain Reaction
- **Q**: Does Nash equilibrium involve spatial strategies?
- **Q**: Can evolution discover distributed coverage?
- **Q**: How important is early coordination?

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create directory structure (`experiments/scenarios/`)
- [ ] Implement master orchestration script
- [ ] Refactor existing scripts to fit new structure
- [ ] Add JSON schema validation

### Phase 2: Core Scripts (Week 2)
- [ ] `analyze_heuristics.py` - Tournament evaluation
- [ ] `run_evolution.py` - GA with snapshots
- [ ] `compute_nash.py` - Double Oracle wrapper
- [ ] Report generation (Markdown)

### Phase 3: Comparison & Analysis (Week 3)
- [ ] Cross-method tournament
- [ ] Strategy distance metrics
- [ ] `cross_scenario_analysis.py`
- [ ] Web data bundle generation

### Phase 4: Documentation & Web (Week 4)
- [ ] Per-scenario README.md files
- [ ] Cross-scenario report
- [ ] Web UI integration
- [ ] Publication-ready visualizations

## Future Extensions

### Advanced Analysis
- **Sensitivity Analysis**: How do results change with parameter perturbations?
- **Robustness Testing**: Test strategies against adversarial opponents
- **Transfer Learning**: Do strategies from one scenario work in others?

### Additional Methods
- **Reinforcement Learning**: Compare PPO/DQN to heuristics/evolution/Nash
- **Bayesian Nash**: Model incomplete information about opponent types
- **Correlated Equilibrium**: Use signal mechanism as correlation device

### Visualization Enhancements
- **3D Parameter Space**: Visualize strategy distributions in reduced dimensions
- **Evolutionary Phylogeny**: Track lineage of successful strategies
- **Interactive Nash Explorer**: Adjust parameters, see equilibrium shift in real-time

---

## Summary

This framework transforms isolated analysis scripts into a **comprehensive research platform**:

1. **Unified Structure**: All scenarios analyzed with same methodology
2. **Multi-Method**: Heuristics (baseline), Evolution (discovery), Nash (theory)
3. **Reproducible**: JSON configs + deterministic seeds
4. **Accessible**: Markdown reports for humans, JSON for programs
5. **Scalable**: Easy to add new scenarios or methods
6. **Web-Ready**: Direct integration with visualization layer

The goal is to answer: **What strategies work best, and why?**

By combining empirical optimization (evolution) with theoretical prediction (Nash), we can validate our understanding of multi-agent dynamics and discover surprising emergent behaviors.

---

*Document version: 1.0*
*Last updated: 2025-11-03*
