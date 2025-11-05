# Research Library

This directory contains summaries of key papers, articles, and resources relevant to the Bucket Brigade project. Each summary provides a concise overview, highlights relevance to our work, and captures actionable insights.

## Purpose

- **Quick reference**: Access key findings without re-reading full papers
- **Link rot protection**: Local summaries persist even if external links break
- **Contextualization**: Each summary explains relevance to Bucket Brigade
- **Knowledge base**: Build institutional memory for the project
- **Onboarding**: Help contributors understand the research landscape

## Organization

Papers are organized by primary topic:

### Multi-Agent Reinforcement Learning (`marl/`)
Core MARL algorithms, challenges, and surveys.

- [wong2023_marl_survey.md](marl/wong2023_marl_survey.md) - Comprehensive MARL challenges and methods

### AI Safety (`safety/`)
Specification gaming, reward misspecification, goal misgeneralization.

- [langosco2022_goal_misgeneralization.md](safety/langosco2022_goal_misgeneralization.md) - Goal misgeneralization in deep RL
- [deepmind2020_specification_gaming.md](safety/deepmind2020_specification_gaming.md) - Specification gaming examples
- [bronsdon2025_marl_security.md](safety/bronsdon2025_marl_security.md) - Security risks in MARL

### Evolution & Open-Ended Learning (`evolution/`)
Evolutionary algorithms, autocurricula, population-based training.

- [baker2019_hide_seek.md](evolution/baker2019_hide_seek.md) - Emergent tool use from autocurricula
- [openai2019_hide_seek_blog.md](evolution/openai2019_hide_seek_blog.md) - Accessible summary of hide-and-seek

### Open-World Environments (`open-world/`)
Non-stationarity, novelty, continual learning.

- [lee2021_novelty_generator.md](open-world/lee2021_novelty_generator.md) - Open-world novelty for RL environments
- [bailey2025_evolving_rewards.md](open-world/bailey2025_evolving_rewards.md) - Continuously evolving rewards

### Communication (`communication/`)
Emergent communication, coordination, language.

- [wolff2023_emergent_language.md](communication/wolff2023_emergent_language.md) - Emergent language in open-ended environments

### Game Theory (`game-theory/`)
Nash equilibria, norms, multi-agent coordination.

- [mashayekhi2016_silk.md](game-theory/mashayekhi2016_silk.md) - Regulating open normative systems

## Index by Tag

### #marl
- [wong2023_marl_survey](marl/wong2023_marl_survey.md)
- [baker2019_hide_seek](evolution/baker2019_hide_seek.md)

### #safety
- [langosco2022_goal_misgeneralization](safety/langosco2022_goal_misgeneralization.md)
- [deepmind2020_specification_gaming](safety/deepmind2020_specification_gaming.md)
- [bronsdon2025_marl_security](safety/bronsdon2025_marl_security.md)

### #evolution
- [baker2019_hide_seek](evolution/baker2019_hide_seek.md)
- [openai2019_hide_seek_blog](evolution/openai2019_hide_seek_blog.md)

### #open-world
- [lee2021_novelty_generator](open-world/lee2021_novelty_generator.md)
- [bailey2025_evolving_rewards](open-world/bailey2025_evolving_rewards.md)

### #communication
- [wolff2023_emergent_language](communication/wolff2023_emergent_language.md)

### #game-theory
- [mashayekhi2016_silk](game-theory/mashayekhi2016_silk.md)

### #phase1 (Closed-World Mastery)
- [wong2023_marl_survey](marl/wong2023_marl_survey.md)

### #phase2 (Adaptive Multi-Scenario)
- [lee2021_novelty_generator](open-world/lee2021_novelty_generator.md)
- [bailey2025_evolving_rewards](open-world/bailey2025_evolving_rewards.md)

### #phase3 (Population Resilience)
- [baker2019_hide_seek](evolution/baker2019_hide_seek.md)

### #phase4 (Reflective Agents)
- [langosco2022_goal_misgeneralization](safety/langosco2022_goal_misgeneralization.md)
- [mashayekhi2016_silk](game-theory/mashayekhi2016_silk.md)

## Contributing New Summaries

1. **Use the template**: Copy `templates/paper_summary.md` as a starting point
2. **Choose the right directory**: Place in the most relevant category
3. **Naming convention**: `lastname[s]YYYY_shortname.md` (e.g., `baker2019_hide_seek.md`)
4. **Tag appropriately**: Use tags from the index above
5. **Context matters**: Focus the "Relevance to Bucket Brigade" section
6. **Update this README**: Add entry to category and tag sections

## Search Tips

```bash
# Find all papers about safety
grep -l "#safety" docs/library/*/*.md

# Find papers relevant to Phase 2
grep -l "#phase2" docs/library/*/*.md

# Search for specific terms across summaries
grep -r "reward hacking" docs/library/

# List all papers by a specific author
grep -r "^**Authors**:.*Baker" docs/library/
```

## Citation Format

When referencing papers from this library in other docs:

```markdown
As demonstrated in hide-and-seek experiments [Baker et al. 2019](../library/evolution/baker2019_hide_seek.md),
autocurricula can emerge from simple competitive dynamics.
```

## Related Documentation

- [Technical MARL Review](../technical_marl_review.md) - References many papers in this library
- [Closed vs. Open World Background](../background_closed_vs_open_world.md) - Conceptual framework
- [Vision and Future Directions](../vision_future_directions.md) - Research questions these papers inform
