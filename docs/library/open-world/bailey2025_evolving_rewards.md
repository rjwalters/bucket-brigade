# Continuously Evolving Rewards in an Open-Ended Environment

**Authors**: Bailey, J.
**Venue**: Journal of Machine Learning Research 2025
**Link**: TBD
**Tags**: #open-world #reward-evolution #phase2 #phase4

## TL;DR

Addresses the challenge of reward specification in open-ended environments where goals naturally emerge and shift over time, proposing frameworks for reward functions that evolve alongside agent capabilities and environmental changes.

## Key Contributions

- Framework for time-varying reward functions in open-ended settings
- Analysis of how fixed rewards become obsolete as environments evolve
- Proposals for adaptive reward mechanisms
- Connection between reward evolution and value alignment

## Relevance to Bucket Brigade

- **Research goals**: Directly motivates Phase 4 (reflective agents with mutable reward weights)
- **Methods**: Challenges fixed reward assumption even in controlled scenarios
- **Theoretical framework**: Open-world requires rethinking reward as dynamic, not static

## Key Findings

### Fixed Rewards Break in Open Worlds

When environments are non-stationary:
- Original reward function may become poor proxy for intended behavior
- New opportunities/threats emerge that weren't in training
- Agent needs to adapt objectives, not just strategies

### Reward Evolution Mechanisms

1. **Human-in-loop**: Continuous reward specification updates
2. **Learned reward models**: Models that update with new data
3. **Meta-objectives**: Higher-level goals that guide reward adaptation
4. **Population-level**: Different agents pursue different reward structures, selection favors successful ones

### Alignment Implications

Continuously evolving rewards raise questions:
- Who decides how rewards evolve?
- How to ensure evolution aligns with human values?
- Can agents participate in shaping their own objectives?

### Practical Challenges

- Stability: Frequent reward changes can prevent skill consolidation
- Consistency: Hard to compare agent performance across time
- Oversight: Difficult to verify reward changes are beneficial

## Methods

- **Approach**: Theoretical framework with illustrative examples
- **Analysis**: Formal treatment of time-varying reward functions
- **Evaluation**: Conceptual, not extensive empirical validation

## Related Work

- **Extends**: Reward modeling, inverse RL
- **Related to**: [lee2021_novelty_generator](lee2021_novelty_generator.md) - Open-world dynamics
- **Related to**: [langosco2022_goal_misgeneralization](../safety/langosco2022_goal_misgeneralization.md) - Goal specification challenges

## Implementation Notes

Practical implications for Bucket Brigade:

### Phase 2 Application
- Scenarios could have slowly drifting reward parameters
- Test if agents adapt to gradual vs. sudden reward changes
- Measure performance degradation under reward shift

### Phase 4 Application
- Core motivation: Let agents propose reward modifications
- Meta-objective: Rewards should encourage robustness across scenarios
- Evaluation: Do evolved rewards generalize better than fixed ones?

### Design Considerations
- **Smoothness**: Gradual vs. abrupt reward changes
- **Announcement**: Do agents know reward changed?
- **Reversibility**: Can old rewards be restored?
- **Meta-stability**: How often should rewards evolve?

## Notable Quotes

> "In open-ended environments, the notion of a fixed objective becomes untenable."

> "Reward functions must evolve alongside the environment and agent capabilities."

## Open Questions

- How to formalize "beneficial" reward evolution?
- Can agents learn to recognize when their rewards should change?
- What prevents runaway reward evolution (agents just making tasks easier)?

## Citation

```bibtex
@article{bailey2025evolving,
  author = {Bailey, J.},
  title = {Continuously Evolving Rewards in an Open-Ended Environment},
  journal = {Journal of Machine Learning Research},
  year = {2025}
}
```
