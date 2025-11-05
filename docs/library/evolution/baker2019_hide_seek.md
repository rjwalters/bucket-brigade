# Emergent Tool Use From Multi-Agent Autocurricula

**Authors**: Baker, B., Kanitscheider, I., Markov, T., Wu, Y., Powell, G., McGrew, B., Mordatch, I.
**Venue**: arXiv 2019 (OpenAI)
**Link**: [arXiv:1909.07528](https://arxiv.org/abs/1909.07528)
**Tags**: #evolution #autocurriculum #emergence #phase3 #open-ended

## TL;DR

Multi-agent competition in a simple hide-and-seek game with physics objects generates an emergent autocurriculum spanning six distinct strategy phases (including tool use and environment manipulation), demonstrating that open-ended complexity can arise from simple rules and self-play without explicit reward shaping.

## Key Contributions

- Demonstration of emergent autocurricula from pure self-play competition
- Six distinct emergent strategy phases discovered automatically
- Evidence that tool use and environment manipulation can emerge without explicit reward
- Proof that simple competitive objectives can drive open-ended skill discovery
- Transfer learning: skills acquired in hide-and-seek transfer to explicit object manipulation tasks

## Relevance to Bucket Brigade

- **Research goals**: Direct inspiration for Phase 3 (population resilience) and emergence of specialized strategies
- **Methods**: Validates population-based training as source of innovation beyond single-agent optimization
- **Theoretical framework**: Concrete example of open-world dynamics generating endless adaptation

## Key Findings

### Six Emergent Strategy Phases

1. **Hiders run away**: Basic evasion
2. **Seekers learn to chase**: Simple pursuit
3. **Hiders build shelters**: Tool use emerges (moving boxes to create barriers)
4. **Seekers use ramps**: Counter-strategy using environmental objects
5. **Hiders lock ramps**: More sophisticated defense (securing tools to prevent exploitation)
6. **Seekers exploit physics**: Surfing on boxes through unintended physics interactions

Each phase emerged spontaneously as one side developed a counter-strategy to the other's current best approach.

### No Reward Shaping Required

The only reward signal: hiders get +1 if hidden at end of round, seekers get +1 if they see a hider. No explicit rewards for tool use, building, or any specific strategy. All complex behaviors emerged from this simple competitive pressure.

### Population Diversity Crucial

Training used **population-based self-play** with:
- Multiple concurrent training runs
- Policies playing against diverse opponents (not just latest)
- No single "optimal" strategy due to rock-paper-scissors dynamics

This prevented premature convergence to local optima and maintained exploration pressure.

### Transfer to Explicit Tasks

After hide-and-seek training, agents could quickly fine-tune on explicit object manipulation tasks (like "move box to location") showing they had genuinely acquired reusable skills, not just task-specific behaviors.

### 481 Million Episodes Required

Emergent complexity takes time: billions of environment steps across population. This highlights computational cost of true open-ended learning.

## Methods

- **Experimental setup**: 3D physics environment, 4 agents (hiders vs. seekers), movable objects (boxes, ramps)
- **Algorithms used**: PPO with population-based training, opponent sampling from diverse skill pool
- **Evaluation metrics**: Hide success rate, strategy progression tracking, transfer task performance
- **Environment**: MuJoCo physics simulation with vision-based observations

## Related Work

- **Extends**: Self-play literature (AlphaGo, Dota) to emergent tool use
- **Related to**: POET (open-ended environment generation), PBT (population-based training)
- **Cited by**: Extensive follow-up on emergent complexity in games
- **Complements**: [wolff2023_emergent_language](../communication/wolff2023_emergent_language.md) - Emergence from interaction

## Implementation Notes

Practical implications for Bucket Brigade:

- **Population-based training essential** for Phase 3: Single-agent convergence would miss emergent strategies
- **Diverse opponent sampling** prevents exploitation collapse: Must maintain varied strategy pool
- **Long training horizons**: Emergence requires patience (100M+ episodes), budget accordingly
- **Transfer evaluation**: Test if learned behaviors generalize beyond training scenarios
- **Code available**: OpenAI released environments and some training code

Specific techniques:
- **Opponent sampling**: Mix of current best, historical, and diverse policies
- **Policy architecture**: Vision-based (no direct access to game state), recurrent for memory
- **Team composition**: Fixed team sizes (3 hiders, 1-2 seekers)

## Notable Quotes

> "We find that agents learn to use tools and modify their environment even though these behaviors are not explicitly rewarded."

> "Simple competitive objectives can drive the emergence of complex, intelligent behavior through multi-agent interaction."

> "Each new strategy by one team creates a new challenge for the other team, leading to a progression of increasingly complex behaviors."

## Open Questions

- Does autocurriculum plateau eventually, or is it truly endless?
- How to guide emergence toward safe/aligned behaviors rather than exploitation?
- Can similar emergence occur in cooperative settings, or does competition drive it?
- How to scale to even larger populations and more diverse environments?

## Citation

```bibtex
@article{baker2019emergent,
  author = {Baker, Bowen and Kanitscheider, Ingmar and Markov, Todor and Wu, Yi and Powell, Glenn and McGrew, Bob and Mordatch, Igor},
  title = {Emergent Tool Use From Multi-Agent Autocurricula},
  journal = {arXiv preprint arXiv:1909.07528},
  year = {2019}
}
```
