# Research Notebook 📔

A chronological record of our research journey - experiments, insights, discoveries, and lessons learned.

## What is This?

The Research Notebook is our **living history** of the Bucket Brigade project. Unlike traditional research papers that present polished final results, the notebook captures:

- **The process**: What we tried, why we tried it, what happened
- **The insights**: What we learned, what surprised us, what patterns emerged
- **The narrative**: The story of how our understanding evolved over time
- **The context**: Why decisions were made, what constraints existed, what alternatives we considered

## Philosophy

**Research is a journey, not just a destination.**

Traditional research dissemination focuses on final results. But understanding *how* we got there is often more valuable than just knowing *where* we ended up. The notebook captures:

1. **Failed experiments** - What didn't work and why
2. **Pivotal moments** - When our understanding shifted
3. **Technical decisions** - Why we chose one approach over another
4. **Surprises** - Unexpected results that led to new questions
5. **Lessons learned** - What we'd do differently next time

## Structure

Each entry follows a narrative format:

```markdown
# Research Notebook Entry: [Date]

**Date**: YYYY-MM-DD
**Author**: [Name]
**Status**: [Brief status/milestone]
**Tags**: `tag1`, `tag2`, `tag3`

## The Story

[Narrative describing what happened, what we learned, what's next]

## Key Findings

[Bullet points of important discoveries]

## What's Next

[Clear action items and open questions]
```

## Writing Guidelines

**Tell a story:**
- Use narrative structure (problem → exploration → discovery)
- Explain *why* decisions were made, not just *what* happened
- Include context that future readers (or future you) will need

**Be honest:**
- Document failures and dead ends
- Acknowledge uncertainty and open questions
- Explain what you don't understand

**Focus on insights:**
- What did you *learn*, not just what did you *do*
- What surprised you? What changed your thinking?
- What implications do the results have?

**Make it useful:**
- Include enough detail for reproducibility
- Link to relevant code, data, and documentation
- Clearly state next steps and open questions

## Entry Naming Convention

Format: `YYYY-MM-DD_descriptive_title.md`

Examples:
- `2025-11-07_game_mechanics_fix_and_v7_completion.md`
- `2025-11-15_nash_v7_equilibrium_analysis.md`
- `2025-11-20_ppo_vs_evolution_comparison.md`

## How to Add an Entry

1. Create a new file in `research_notebook/` with format `YYYY-MM-DD_title.md`
2. Use the template structure above
3. Write in narrative form (tell the story!)
4. Run `pnpm run build:research` to publish to website
5. Commit both the source entry and generated website content

## Building for Website

The research notebook is automatically built into the website:

```bash
# Build and update website content
pnpm run build:research

# This copies notebook entries to web/public/research/notebook/
# and generates content_index.json for the browser
```

## Website Integration

The notebook is browsable on the website at `/research/notebook`:

- **Chronological view**: All entries in reverse chronological order
- **Tag filtering**: Filter by tags (evolution, nash, ppo, milestone, etc.)
- **Search**: Full-text search across all entries
- **Rich rendering**: Markdown with syntax highlighting, charts, images

## Tags

Use consistent tags for easy filtering:

**Research Areas**:
- `evolution` - Evolutionary algorithm research
- `nash-equilibrium` - Nash equilibrium analysis
- `ppo-training` - PPO/MARL training
- `tournaments` - Tournament and ranking research

**Topics**:
- `game-mechanics` - Game implementation and physics
- `analysis` - Data analysis and insights
- `infrastructure` - Tools, testing, deployment
- `methodology` - Research methods and approaches

**Milestones**:
- `milestone` - Major achievements or completions
- `breakthrough` - Significant discoveries
- `pivot` - Change in direction or approach
- `retrospective` - Looking back and lessons learned

## Current Entries

1. **2025-11-07**: Game Mechanics Fix and V7 Completion
   - Tags: `game-mechanics`, `v7-evolution`, `ppo-training`, `nash-equilibrium`, `milestone`
   - Status: First correct-mechanics results complete
   - Key finding: Previous experiments used inconsistent physics, new V7/PPO results are first valid data

## Companion: Research Library

The **Research Library** complements the notebook with comprehensive documentation:

- **Notebook** (this): Chronological story of progress
- **Library** (`experiments/*.md`, `docs/*.md`): Comprehensive reference docs

Both are published to the website and cross-referenced for easy navigation.

## For Future Researchers

If you're reading old entries and notice:
- Broken links → Please update them
- Outdated information → Add a note at the top
- Missing context → Consider writing a "retrospective" entry

The notebook is a living document. Keep it accurate and useful!

---

**Browse the notebook**: [https://rjwalters.github.io/bucket-brigade/research/notebook](https://rjwalters.github.io/bucket-brigade/research/notebook)

**Last Updated**: 2025-11-07
