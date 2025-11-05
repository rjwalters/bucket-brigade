# An Open-World Novelty Generator for Authoring RL Environments

**Authors**: Lee, S., et al.
**Venue**: MIWAI 2021
**Link**: [OpenReview](https://openreview.net/forum?id=rJgEKyrtPr)
**Tags**: #open-world #novelty #phase2 #continual-learning

## TL;DR

Proposes a novelty generator system for creating open-world RL environments where agents face unexpected changes and novel situations, arguing that closed-world assumptions are "rarely appropriate" for complex domains and agents need continuous adaptation capabilities.

## Key Contributions

- Distinction between closed-world (static, predictable) and open-world (dynamic, novel) environments
- Framework for generating novel situations during training/deployment
- Emphasis on continual adaptation over convergent optimization
- Authoring tools for environment designers to specify novelty types

## Relevance to Bucket Brigade

- **Research goals**: Direct motivation for Phase 2 (multi-scenario agents) and open-world progression
- **Methods**: Validates our scenario-switching approach as stepping stone to true open-world capability
- **Theoretical framework**: Articulates why closed-world mastery alone is insufficient

## Key Findings

### Closed-World Assumptions Break Down

Traditional RL assumes:
- Fixed set of states, actions, dynamics
- Stationary reward function
- Known environment model

These assumptions "rarely appropriate" in real-world domains where:
- New objects/agents appear
- Rules change
- Goals shift
- Unforeseen circumstances arise

### Novelty as Training Signal

Instead of treating novel situations as failures, use them as:
- **Exploration driver**: Novel states indicate under-explored regions
- **Capability test**: Agent robustness measured by handling novelty
- **Adaptation trigger**: Signal to update policy or world model

### Types of Novelty

1. **State novelty**: New objects, configurations, or features
2. **Action novelty**: New possible actions become available
3. **Transition novelty**: Environment dynamics change
4. **Reward novelty**: Objective shifts or new goals emerge
5. **Agent novelty**: New agents enter or leave the system

### Authoring Framework

Provide tools for designers to specify:
- Probability distributions over novelty types
- Timing of novelty introduction (gradual vs. sudden)
- Magnitude of change (minor variation vs. fundamental shift)
- Dependencies between novelty events

## Methods

- **Experimental setup**: Prototype system demonstrated on simple RL tasks
- **Evaluation metrics**: Agent adaptation speed, performance degradation, recovery time
- **Novelty injection**: Controlled perturbations to environment parameters

## Related Work

- **Extends**: Domain randomization, procedural content generation
- **Related to**: [bailey2025_evolving_rewards](bailey2025_evolving_rewards.md) - Dynamic objectives
- **Complements**: UED (unsupervised environment design), meta-learning

## Implementation Notes

Practical implications for Bucket Brigade:

### Phase 2 Implementation
- Start with **reward novelty**: Change scenario mid-episode
- Add **agent novelty**: Vary team composition
- Progress to **transition novelty**: Modify fire spread rules
- Eventually **state novelty**: New house configurations

### Novelty Schedule
- **Curriculum**: Start predictable, increase novelty gradually
- **Frequency**: How often to inject novelty (every N episodes? timesteps?)
- **Detection**: Can agent recognize when novelty occurred?

### Evaluation Framework
- **Baseline**: Performance on familiar scenarios (no novelty)
- **Immediate impact**: Performance drop right after novelty
- **Adaptation**: How quickly performance recovers
- **Transfer**: Does handling one novelty type help with others?

### Design Considerations
- Too much novelty → agent never consolidates skills
- Too little novelty → agent overfits to training distribution
- Need balanced exploration-exploitation in novelty schedule

## Notable Quotes

> "Closed-world assumptions are rarely appropriate for complex domains where agents must face unexpected changes and continue to adapt."

> "Novelty should be treated not as a failure mode but as a fundamental aspect of realistic environments."

## Open Questions

- How to distinguish beneficial novelty (drives learning) from harmful noise?
- Can agents learn to predict or anticipate certain types of novelty?
- What's the relationship between novelty handling and out-of-distribution generalization?
- How to evaluate whether an agent is truly adaptable vs. just memorizing many scenarios?

## Citation

```bibtex
@inproceedings{lee2021novelty,
  author = {Lee, S. and others},
  title = {An Open-World Novelty Generator for Authoring RL Environments},
  booktitle = {MIWAI 2021},
  year = {2021}
}
```
