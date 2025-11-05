# Evolution Documentation Archive

This directory contains historical evolution experiment documentation that has been superseded by newer experiments and consolidated documentation.

**Date Archived**: 2025-11-05

## Why These Were Archived

These documents served their purpose during active experimentation but are no longer needed for current work:

- **V4_STATUS.md** - Historical status updates during V4 run (superseded by main README)
- **V4_POST_COMPLETION_GUIDE.md** - Post-completion steps for V4 (superseded by V5_NEXT_STEPS.md)
- **INTENSIVE_EVOLUTION.md** - Planning document for intensive evolution runs (completed)
- **INTENSIVE_EVOLUTION_RESULTS.md** - Results analysis (consolidated into main README)
- **ONGOING_SPECIALIST_V3.md** - V3 experiment notes (consolidated into main README)
- **EVOLUTION_TIMING_ANALYSIS.md** - Timing analysis for old runs (superseded by V3/V4/V5 actual results)

## Current Documentation

For current evolution research, see:

- **../README.md** - Main evolution research document (comprehensive, up-to-date)
- **../RUST_SINGLE_SOURCE_OF_TRUTH.md** - Critical infrastructure resolution
- **../V5_EVOLUTION_PLAN.md** - Current V5 experiment configuration
- **../V5_NEXT_STEPS.md** - V5 post-completion guide

Historical debugging documents are kept at the parent level for reference:
- **../V3_FITNESS_BUG_ANALYSIS.md** - Original train/test mismatch discovery
- **../V4_CRITICAL_FAILURE_ANALYSIS.md** - V4 diagnostics
- **../V4_EVOLUTION_PLAN.md** - V4 configuration (reference value)

## Data Cleanup

Along with archiving these documents, we also cleaned up experiment data:

**Deleted**:
- `evolved_v2/` directories (empty, abandoned)
- `generation_*` snapshot directories from `evolved/` and `evolved_v3/` (redundant with evolution_trace.json)

**Retained**:
- All `best_agent.json` files (final evolved genomes)
- All `evolution_trace.json` files (convergence data)
- All `comparison/` directories (tournament results)

**Space Saved**: ~9MB (39MB → 30MB)

## If You Need These Documents

These documents are kept in git history and can be restored if needed. They provide valuable context for understanding the evolution research process and debugging journey.
