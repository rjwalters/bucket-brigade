# SILK: Regulating Open Normative Multiagent Systems

**Authors**: Mashayekhi, M., et al.
**Venue**: IJCAI 2016
**Link**: [IJCAI Proceedings](https://www.ijcai.org/proceedings/2016/)
**Tags**: #game-theory #norms #open-systems #phase4

## TL;DR

Framework for regulating behavior in open multi-agent systems where agents can enter/leave dynamically, using emergent norms rather than fixed rules to maintain coordination despite changing population.

## Key Contributions

- Distinction between closed (fixed membership) and open (dynamic membership) multi-agent systems
- Norm emergence mechanisms for open systems
- SILK framework for adaptive regulation
- Evidence that open systems require different governance than closed ones

## Relevance to Bucket Brigade

- **Research goals**: Phase 4 (norm formation) - how populations develop shared behavioral standards
- **Methods**: Inspiration for cultural transmission of strategies
- **Theoretical framework**: Open-world dynamics require emergent, not imposed, coordination

## Key Findings

### Open vs. Closed Systems

**Closed systems**:
- Fixed set of agents
- Can pre-coordinate strategies
- Central authority can enforce rules

**Open systems**:
- Agents join/leave unpredictably
- New agents lack coordination history
- Norms must emerge and propagate

### Norm Emergence Process

1. **Local interactions**: Agents interact in pairs or small groups
2. **Successful patterns spread**: Effective coordination strategies are copied
3. **Population-level convergence**: Norms become widespread
4. **Resilience to churn**: Norms persist even as individuals leave

### SILK Framework Components

- **Norm recognition**: How agents identify prevailing norms
- **Norm adoption**: When to conform vs. innovate
- **Norm propagation**: How norms spread through population
- **Norm enforcement**: Incentivizing conformity (reputation, sanctions)

### Robustness to Newcomers

Key challenge: New agents enter without knowledge of norms
- **Observation period**: New agents watch before acting
- **Mentorship**: Existing agents teach newcomers
- **Sanctions**: Punish norm violations to signal expectations
- **Redundancy**: Multiple agents demonstrate same norm

## Methods

- **Experimental setup**: Simulated multi-agent systems with dynamic entry/exit
- **Algorithms**: Norm learning via reinforcement and imitation
- **Evaluation**: Norm convergence speed, stability, efficiency of coordination

## Related Work

- **Extends**: Norm emergence literature (Shoham, Axelrod)
- **Related to**: [baker2019_hide_seek](../evolution/baker2019_hide_seek.md) - Emergent behaviors
- **Related to**: Cultural evolution, institutional economics

## Implementation Notes

Practical implications for Bucket Brigade:

### Phase 3 Application (Population Dynamics)
- Population turnover: Replace agents each generation
- Do successful strategies persist despite turnover?
- Measure "cultural transmission" of cooperation patterns

### Phase 4 Application (Norm Formation)
- Start with diverse reward weights
- Allow agents to observe and copy successful peers
- Track convergence to shared reward structures
- Test robustness: Introduce new agents with random weights

### Design Considerations

**Observation mechanisms**:
- Can agents see others' reward weights?
- Can they see others' actions and outcomes?

**Adoption rules**:
- Copy most successful peer?
- Weighted average of multiple peers?
- Probabilistic adoption based on performance difference?

**Evaluation**:
- Norm consensus: Variance in population reward weights
- Norm stability: Persistence over generations
- Norm efficiency: Do emerged norms improve group outcomes?

## Notable Quotes

> "Open systems require emergent norms rather than fixed rules to handle dynamic membership."

> "Successful coordination patterns spread through population even without central authority."

## Open Questions

- How many interactions needed for norm to stabilize?
- Can multiple norms coexist stably (norm polymorphism)?
- What happens when norms conflict with individual incentives?
- How to bootstrap norms in initially chaotic populations?

## Citation

```bibtex
@inproceedings{mashayekhi2016silk,
  author = {Mashayekhi, M. and others},
  title = {SILK: Regulating Open Normative Multiagent Systems},
  booktitle = {Proceedings of IJCAI 2016},
  year = {2016}
}
```
