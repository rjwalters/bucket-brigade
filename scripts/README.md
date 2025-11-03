# Policy Team Testing CLI Tools

This directory contains command-line tools for testing policy teams against game scenarios, providing a parallel UX to the web demo for controlled testing and development.

## Overview

The CLI tools allow you to:
- Test specific team compositions against various scenarios
- Compare multiple teams side-by-side with statistical analysis
- Validate that preferred strategies succeed on appropriate scenarios
- Run automated integration tests via pytest

## Tools

### 1. `test_team.py` - Main Testing Tool

Test a team of agents against specific or random scenarios.

**Usage:**
```bash
# Test a team against 100 random scenarios
uv run python scripts/test_team.py test firefighter,coordinator,hero

# Test against specific scenario types
uv run python scripts/test_team.py test firefighter,hero --scenarios early_containment --count 20

# Test with multiple scenario types
uv run python scripts/test_team.py test "free_rider,liar,opportunist" \
  --scenarios "greedy_neighbor,rest_trap" --count 50

# Save results to file
uv run python scripts/test_team.py test strategist,coordinator,firefighter,hero \
  --output results/test1.json --verbose

# Quick test with fewer scenarios
uv run python scripts/test_team.py test maverick,random,hero --count 10
```

**Commands:**
- `test` - Run tournament with a team
- `list-archetypes` - Show all available agent archetypes
- `list-scenarios` - Show all available scenario types

**Output:**
- Team performance statistics (mean, std, median, range)
- Houses saved metrics
- Game length statistics
- Individual agent contributions ranked by performance

### 2. `compare_teams.py` - Team Comparison Tool

Compare multiple team configurations side-by-side with statistical significance testing.

**Usage:**
```bash
# Compare two teams
uv run python scripts/compare_teams.py compare \
  --team1 firefighter,hero,coordinator \
  --team2 free_rider,liar,opportunist

# Compare on specific scenarios
uv run python scripts/compare_teams.py compare \
  --team1 hero,hero,hero \
  --team2 free_rider,free_rider,free_rider \
  --scenarios greedy_neighbor \
  --count 100

# Compare up to 4 teams
uv run python scripts/compare_teams.py compare \
  --team1 firefighter,coordinator,hero \
  --team2 strategist,strategist,strategist \
  --team3 opportunist,liar,free_rider \
  --team4 cautious,cautious,cautious \
  --scenarios early_containment \
  --count 50

# Save results
uv run python scripts/compare_teams.py compare \
  --team1 hero,hero \
  --team2 free_rider,free_rider \
  --scenarios rest_trap \
  --output results/comparison.json
```

**Features:**
- Side-by-side performance comparison
- Statistical significance testing (t-test) for 2-team comparisons
- Rich formatted table output
- JSON export for further analysis

## Agent Archetypes

Available agent behavioral profiles:

- **firefighter** - Reliable team player, always signals honestly, helps neighbors
- **coordinator** - Excellent at reading signals and organizing team response
- **hero** - Maximum effort and altruism, never rests while fires burn
- **strategist** - Analyzes situation carefully, maximizes efficiency
- **free_rider** - Prefers to rest and let teammates handle fires
- **opportunist** - Laser-focused on protecting own property
- **cautious** - Avoids risky situations, conservative approach
- **liar** - Sends false signals to mislead teammates
- **maverick** - High variance, unpredictable by design
- **random** - All parameters randomized each game

## Scenario Types

Available scenario configurations:

- **trivial_cooperation** - Fires are rare and extinguish easily
- **early_containment** - Fires start aggressive but can be stopped early
- **greedy_neighbor** - Social dilemma between self-interest and cooperation
- **sparse_heroics** - Few workers can make the difference
- **rest_trap** - Fires usually extinguish themselves, but not always
- **chain_reaction** - High spread requires distributed teams
- **deceptive_calm** - Occasional flare-ups reward honest signaling
- **overcrowding** - Too many workers reduce efficiency
- **mixed_motivation** - Ownership creates self-interest conflicts

## Integration Tests

Automated pytest integration tests validate strategy performance hypotheses.

**Run all validation tests:**
```bash
uv run pytest tests/test_strategy_validation.py -v
```

**Run specific test class:**
```bash
uv run pytest tests/test_strategy_validation.py::TestCooperativeStrategies -v
```

**Run specific test:**
```bash
uv run pytest tests/test_strategy_validation.py::TestCooperativeStrategies::test_early_containment_cooperative_wins -v
```

**Test categories:**
- `TestCooperativeStrategies` - Verify cooperative teams outperform on cooperative scenarios
- `TestSelfishStrategiesExcel` - Verify selfish strategies can perform well in competitive scenarios
- `TestArchetypeSpecializations` - Verify specific archetypes excel in their designed scenarios

## Example Workflows

### Validate Cooperative Strategies

Test that cooperative teams significantly outperform non-cooperative teams on cooperative scenarios:

```bash
# Via comparison tool
uv run python scripts/compare_teams.py compare \
  --team1 firefighter,coordinator,hero \
  --team2 free_rider,liar,opportunist \
  --scenarios early_containment \
  --count 100

# Via pytest
uv run pytest tests/test_strategy_validation.py::TestCooperativeStrategies -v
```

### Validate Selfish Strategies

Test that selfish strategies are competitive in scenarios where cooperation isn't required:

```bash
# Via comparison tool
uv run python scripts/compare_teams.py compare \
  --team1 hero,hero,hero \
  --team2 free_rider,free_rider,free_rider \
  --scenarios greedy_neighbor \
  --count 100

# Via pytest
uv run pytest tests/test_strategy_validation.py::TestSelfishStrategiesExcel -v
```

### Test Specific Scenario

Test preferred strategies on a specific scenario:

```bash
# Test that early containment rewards cooperation
uv run python scripts/test_team.py test firefighter,coordinator,hero \
  --scenarios early_containment \
  --count 50 \
  --verbose

# Compare to non-cooperative team
uv run python scripts/test_team.py test free_rider,liar,opportunist \
  --scenarios early_containment \
  --count 50 \
  --verbose
```

### Analyze Team Diversity

Test whether team composition diversity matters:

```bash
# Homogeneous team
uv run python scripts/test_team.py test firefighter,firefighter,firefighter \
  --scenarios early_containment \
  --count 50

# Diverse team
uv run python scripts/test_team.py test firefighter,coordinator,hero \
  --scenarios early_containment \
  --count 50

# Statistical comparison
uv run python scripts/compare_teams.py compare \
  --team1 firefighter,firefighter,firefighter \
  --team2 firefighter,coordinator,hero \
  --scenarios early_containment \
  --count 50
```

## Output Format

### test_team.py Output

```
============================================================
📊 TOURNAMENT RESULTS
============================================================

🏆 Team Performance:
   Mean Team Reward: 245.67 ± 45.23
   Median: 250.00
   Range: [150.00, 320.00]

🏠 Houses Saved:
   Mean: 7.50 ± 1.20
   Median: 8.00
   Success Rate (≥5): 92.0%

⏱️  Game Length:
   Mean Nights: 12.5 ± 2.3

👥 Agent Contributions (by mean reward):
   1. firefighter  (0):  85.23 ± 12.45
   2. hero         (2):  82.11 ± 15.67
   3. coordinator  (1):  78.33 ± 11.22
```

### compare_teams.py Output

```
                        Team Comparison Results
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Team                   ┃ Mean Reward┃ Std Dev ┃ Houses Saved┃ Success Rate┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ firefighter,coordinator│     278.45│  ±35.67│         8.20│       95.0% │
│ free_rider,liar,oppor  │     189.23│  ±52.34│         5.40│       65.0% │
└────────────────────────┴────────────┴─────────┴─────────────┴─────────────┘

📊 Statistical Significance Test (t-test):
   t-statistic: 12.3456
   p-value: 0.0001
   ✅ Significant difference detected (p < 0.05)
   🏆 firefighter,coordinator,hero performs significantly better
```

## Dependencies

All dependencies are managed via `pyproject.toml`:
- numpy - Numerical operations
- scipy - Statistical tests
- typer - CLI framework
- rich - Rich terminal formatting
- pytest - Testing framework (dev dependency)

Install with:
```bash
uv sync
```

## Related Files

- `bucket_brigade/envs/scenarios.py` - Scenario definitions
- `bucket_brigade/agents/heuristic_agent.py` - Agent implementation
- `web/src/utils/agentArchetypes.ts` - Web archetype definitions (reference)
- `tests/test_strategy_validation.py` - Integration tests
