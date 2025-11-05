# Deep Multiagent Reinforcement Learning: Challenges and Directions

**Authors**: Wong, A., et al.
**Venue**: Artificial Intelligence Review 2023
**Link**: [Springer](https://link.springer.com/article/10.1007/s10462-023-10409-1)
**Tags**: #marl #survey #phase1 #non-stationarity

## TL;DR

Comprehensive survey of deep MARL covering key challenges (non-stationarity, partial observability, credit assignment, scalability) and categorizing algorithmic approaches (value-based, policy-based, actor-critic, evolutionary methods).

## Key Contributions

- Systematic categorization of MARL challenges and solution approaches
- Analysis of non-stationarity as the central MARL problem
- Comparison of different algorithm families and their trade-offs
- Discussion of evolutionary algorithms as alternative to gradient-based methods
- Identification of open research directions

## Relevance to Bucket Brigade

- **Research goals**: Informs Phase 1 algorithm selection (PPO, GA, Nash analysis)
- **Methods**: Validates our multi-method approach (gradient-based + evolutionary)
- **Theoretical framework**: Emphasizes non-stationarity challenge even in closed worlds, motivating our progression to open-world scenarios

## Key Findings

### Non-Stationarity is Central Challenge

Multi-agent environments are inherently non-stationary because each agent's learning changes the environment from other agents' perspectives. This creates an "infinite loop of agents adapting to other agents" that can prevent convergence.

### Algorithm Categories

- **Value-based** (Q-learning variants): Simple but struggle with large action spaces
- **Policy-based** (policy gradients): Better for continuous actions, more sample-efficient
- **Actor-critic** (PPO, MADDPG): Combine benefits of both, currently dominant
- **Evolutionary** (ES, GA): Naturally handle non-stationarity through population diversity

### Evolutionary Advantages

"Evolutionary algorithms have been shown to work well with nonstationarity and partial observability" by maintaining diverse populations rather than single policies. This directly motivates our Phase 1 GA track and Phase 3 population-level work.

### Open Problems

- Scalability to many agents (>100)
- Transfer learning across tasks
- Sample efficiency in complex environments
- Theoretical convergence guarantees

## Methods

- **Experimental setup**: Survey and synthesis methodology, not empirical
- **Algorithms covered**: DQN variants, REINFORCE, A3C, PPO, MADDPG, QMIX, evolutionary strategies
- **Evaluation metrics**: Sample efficiency, final performance, scalability, robustness
- **Benchmark environments**: Particle environments, SMAC, Hanabi, multi-agent Atari

## Related Work

- **Extends**: Earlier MARL surveys (Busoniu 2008, Zhang 2019)
- **Related to**: [baker2019_hide_seek](../evolution/baker2019_hide_seek.md) - Population-based approaches
- **Complements**: [lee2021_novelty_generator](../open-world/lee2021_novelty_generator.md) - Open-world extensions

## Implementation Notes

Practical implications for Bucket Brigade:

- **PPO is reasonable choice** for Phase 1: Well-established, sample-efficient, works with continuous/discrete actions
- **GA should use tournament selection** to maintain diversity under non-stationarity
- **CTDE approach** (centralized training, decentralized execution) could help stabilize learning in Bucket Brigade
- **Opponent modeling** may be overkill for cooperative scenarios but worth exploring in competitive ones

## Notable Quotes

> "Evolutionary algorithms have been shown to work well with nonstationarity and partial observability by continually using an evolving population rather than a single static agent."

> "The fundamental challenge in MARL is that agents concurrently updating their policies make the environment dynamics change from another agent's perspective."

## Open Questions

- How to balance optimization (convergence) with adaptation (handling non-stationarity)?
- Can we develop metrics that predict which algorithm family will work best for a given task?
- What is the right granularity of opponent modeling in cooperative MARL?

## Citation

```bibtex
@article{wong2023marl,
  author = {Wong, A. and others},
  title = {Deep Multiagent Reinforcement Learning: Challenges and Directions},
  journal = {Artificial Intelligence Review},
  year = {2023},
  volume = {56},
  pages = {5023--5084},
  doi = {10.1007/s10462-023-10409-1}
}
```
