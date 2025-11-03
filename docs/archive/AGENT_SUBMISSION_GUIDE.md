# Agent Submission Guide

Welcome to the Bucket Brigade distributed agent discovery system! This guide explains how to submit high-performing agent configurations discovered through browser-based tournaments.

## Overview

After running tournaments in your browser and discovering agent configurations that perform well, you can submit them to the community registry. This creates a crowdsourced collection of proven strategies that helps everyone discover effective agent behaviors.

## Why Submit Agents?

- **Share discoveries** - Help the community benefit from your findings
- **Build the knowledge base** - Contribute to the collective understanding of effective strategies
- **Get recognition** - Your submitted agents will be attributed to you in the registry
- **Advance research** - Support the development of evolutionary algorithms and meta-strategies

## Submission Requirements

### Minimum Thresholds

Your agent must meet these criteria:

1. **Sample Size**: Tested in at least **20 tournament games**
2. **Performance**: Either:
   - Win rate > 55%, OR
   - Score in top 25% of all agents tested
3. **Parameter Validity**: All parameters must be in range 0.0-1.0
4. **Scenario Diversity**: Tested in at least one scenario type
5. **No Duplicates**: Not an exact copy of an existing agent

### Data Integrity

- Performance data must be from actual tournament runs
- Parameters must match the tested configuration exactly
- Scenario results should be accurate and reproducible

## How to Submit

### Step 1: Run Tournaments

1. Open the Bucket Brigade web app
2. Navigate to the **Team Builder**
3. Create or select a team composition
4. Run a tournament with at least 20 games
5. Review the results to identify high-performing agents

### Step 2: Gather Performance Data

From the tournament results, collect:

- **Agent parameters** (all 10 values)
- **Win rate** or **average score**
- **Games played**
- **Scenarios tested** and performance per scenario
- **Individual contribution** metrics

### Step 3: Submit via GitHub Issue

1. Go to the [GitHub Issues](https://github.com/rjwalters/bucket-brigade/issues/new/choose) page
2. Select **"Agent Submission"** template
3. Fill in the form with your agent's data
4. Review the checklist to ensure requirements are met
5. Submit the issue

### Step 4: Wait for Review

- Maintainers will validate your submission
- Automated checks verify parameter validity and performance thresholds
- If approved, your agent will be added to the registry
- You'll be notified when your submission is processed

## Submission Form Fields

### Agent Name
A descriptive, memorable name for your configuration.

**Good examples**:
- "Heroic Coordinator"
- "Tactical Free Rider"
- "Balanced Strategist"

**Avoid**:
- Generic names like "Agent 1" or "Test"
- Names that duplicate existing agents

### Base Archetype
Which archetype your agent is based on or most similar to:
- Firefighter
- Coordinator
- Hero
- Strategist
- Free Rider
- Opportunist
- Cautious
- Liar
- Maverick
- Random
- Custom (if significantly different from all archetypes)

### Agent Parameters
All 10 behavioral parameters (copy from tournament results):

```yaml
honesty_bias: 0.85        # Probability of truthful signaling (0-1)
work_tendency: 0.90       # Base probability to choose WORK mode (0-1)
neighbor_help_bias: 0.70  # Preference for helping burning neighbors (0-1)
own_house_priority: 0.40  # Preference for defending own house (0-1)
risk_aversion: 0.50       # Sensitivity to number of burning houses (0-1)
coordination_weight: 0.70 # Trust in other agents' signals (0-1)
exploration_rate: 0.10    # Randomness in action selection (0-1)
fatigue_memory: 0.50      # Inertia to repeat last action (0-1)
rest_reward_bias: 0.10    # Preference for rest if fires are low (0-1)
altruism_factor: 0.80     # Willingness to work despite personal cost (0-1)
```

### Tournament Performance
Statistics from your tournament runs:

```
Win Rate: 72% (36/50 games)
Average Score: 45.2
Average Contribution: +8.3
Games Played: 50
```

### Tested Scenarios
List scenarios where you tested the agent with performance breakdown:

```
- Greedy Neighbor: 32-18 (64% win rate)
- Chain Reaction: 28-22 (56% win rate)
- Sparse Heroics: 15-10 (60% win rate)
```

### Tags (Optional)
Comma-separated descriptive tags:

```
cooperative, reliable, defensive, honest
```

### Strategy Notes (Optional)
Context about when your agent performs well:

```
This agent excels in scenarios with high cooperation requirements.
Works well with other honest signalers.
Struggles in sparse fire scenarios where rest is optimal.
```

## Validation Process

### Automated Checks

When you submit, automated validation checks:

1. **Parameter ranges** - All values between 0.0 and 1.0
2. **Performance threshold** - Meets minimum win rate or score
3. **Sample size** - At least 20 games tested
4. **Duplicate detection** - Not an exact match of existing agent

### Manual Review

Maintainers will:

1. Verify performance data is reasonable
2. Check for gaming/manipulation attempts
3. Ensure submission follows community guidelines
4. Add approved agents to the registry

### Rejection Reasons

Submissions may be rejected if:

- Parameters are invalid or out of range
- Performance data is clearly fabricated
- Agent is a duplicate of existing submission
- Insufficient testing (< 20 games)
- Below performance threshold

## After Approval

Once approved, your agent will:

1. Be added to `web/public/data/known-good-agents.json`
2. Appear in the community registry
3. Be available for other users to import and test
4. Contribute to the collective knowledge base

## Registry Structure

The known-good agents registry is a JSON file with this structure:

```json
{
  "version": 1,
  "schema_version": "1.0.0",
  "last_updated": "2025-01-15T10:30:00Z",
  "agents": [
    {
      "id": "heroic-coordinator-001",
      "name": "Heroic Coordinator",
      "archetype": "Firefighter",
      "parameters": { ... },
      "performance": {
        "win_rate": 0.72,
        "avg_score": 48.3,
        "games_played": 50,
        "scenarios_tested": ["greedy_neighbor", "chain_reaction"]
      },
      "metadata": {
        "submitted_at": "2025-01-15T10:30:00Z",
        "tags": ["cooperative", "reliable"],
        "notes": "...",
        "submitter": "username"
      }
    }
  ]
}
```

## Best Practices

### Testing Your Agent

- Test in **multiple scenarios** to understand strengths/weaknesses
- Run at least **30-50 games** for reliable statistics
- Test against **diverse team compositions**
- Record **specific performance metrics** per scenario

### Naming Conventions

- Use descriptive names that hint at strategy
- Avoid version numbers (e.g., "v2", "final")
- Keep names concise (2-4 words)

### Documentation

- Provide clear strategy notes
- Explain when the agent excels and struggles
- Include insights about team composition effects
- Share interesting observations

### Community Guidelines

- Submit only agents you've actually tested
- Don't spam with minor parameter variations
- Report performance data honestly
- Give credit if building on others' work

## FAQ

### How long does review take?
Typically 1-3 days, depending on submission volume.

### Can I update a submitted agent?
Yes, but it will be treated as a new submission with a new ID.

### What if my submission is rejected?
You'll receive feedback on why. You can resubmit after addressing issues.

### Can I submit multiple agents?
Yes, but please ensure each is meaningfully different and tested.

### How is parameter space explored?
Through distributed computation - each user explores different parameter regions and shares successful discoveries.

### What makes an agent "good"?
Consistent performance across scenarios, clear strategy, and meaningful contribution to the knowledge base.

## Next Steps

1. **Run tournaments** in the browser to discover effective agents
2. **Analyze results** to identify high performers
3. **Submit** via GitHub Issue template
4. **Share insights** about what makes your agents successful

## Support

- **Questions**: Open a [GitHub Discussion](https://github.com/rjwalters/bucket-brigade/discussions)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/rjwalters/bucket-brigade/issues)
- **Contributions**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

*This guide is part of the distributed agent discovery system - treating agent optimization as a crowdsourced parameter search problem.*
