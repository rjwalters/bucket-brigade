# Team Builder Tournament Feature

## Overview

A retro-inspired team builder that lets users compose a team of up to 10 agents and test them against 100 randomized scenarios. The UX draws inspiration from classic NES Ice Hockey's character selection, combining simple aesthetics with deep strategic gameplay.

## User Experience

### Core Flow

```
Team Selection → Agent Configuration → Tournament Execution → Results Analysis
```

### 1. Team Selection Screen (`/team-builder`)

**Layout**: Circular arrangement matching the game board (10 house positions)

```
┌─────────────────────────────────────────────────────────────┐
│                  🔥 BUILD YOUR BRIGADE 🔥                    │
│                                                              │
│                      Team Name: [___________]                │
│                                                              │
│                          [House 0]                           │
│                             🧑‍🚒                              │
│                         Firefighter                          │
│                                                              │
│        [House 9]                               [House 1]     │
│           💤                                      📋         │
│        Free Rider                             Coordinator    │
│                                                              │
│    [House 8]        [TOWN CENTER]          [House 2]        │
│       🤥            🔥 🏘️ 🔥               🦸               │
│      Liar                                    Hero            │
│                                                              │
│        [House 7]                               [House 3]     │
│           🎯                                      😰         │
│        Strategist                             Cautious       │
│                                                              │
│                          [House 6]                           │
│                             💰                               │
│                         Opportunist                          │
│                                                              │
│                     [House 5]   [House 4]                    │
│                        🎲          ❓                        │
│                     Maverick    Random                       │
│                                                              │
│   Team Size: 10/10    [+ Add Position] [- Remove Last]      │
│                                                              │
│   [Load Template ▼] [Save Team] [Start Tournament →]        │
└─────────────────────────────────────────────────────────────┘
```

**Interactions**:
- Click any house to open agent selector modal
- Drag-and-drop agents between positions (future)
- Keyboard shortcuts: 0-9 for houses, arrow keys to cycle agents
- Empty positions shown as gray "?" icons

### 2. Agent Selector Modal

**Inspired by**: RPG character selection screens, retro game aesthetics

```
┌─────────────────────────────────────────────────────────────┐
│              Select Agent for House 3                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│     [◄ Prev]        THE FIREFIGHTER        [Next ►]         │
│                                                              │
│                   ████████████████                           │
│                   ██    🧑‍🚒    ██                          │
│                   ████████████████                           │
│                                                              │
│              "First to respond, last to rest"               │
│                                                              │
│   ┌──────────────────────────────────────────────────┐      │
│   │ Stats:                                           │      │
│   │                                                  │      │
│   │   Honesty        ██████████░  10/10             │      │
│   │   Work Ethic     █████████░░   9/10             │      │
│   │   Altruism       ████████░░░   8/10             │      │
│   │   Coordination   ███████░░░░   7/10             │      │
│   │   Risk Taking    █████░░░░░░   5/10             │      │
│   │   House Priority ████░░░░░░░   4/10             │      │
│   │   Rest Bias      ██░░░░░░░░░   2/10             │      │
│   │                                                  │      │
│   │ Strategy Profile:                                │      │
│   │ • Always signals honestly                        │      │
│   │ • Prioritizes urgent fires over rest            │      │
│   │ • Helps neighbors proactively                    │      │
│   │ • Coordinates well with teammates                │      │
│   └──────────────────────────────────────────────────┘      │
│                                                              │
│            [Select This Agent] [Randomize]                   │
│                                                              │
│   Quick Select: [All Firefighters] [Balanced Mix]           │
└─────────────────────────────────────────────────────────────┘
```

**Features**:
- Scroll through all available archetypes
- Visual stat bars (0-10 scale)
- Natural language strategy description
- Quick-select options for common patterns
- Randomize button for experimentation

### 3. Team Templates

**Pre-defined Templates**:

| Template Name | Description | Composition |
|---------------|-------------|-------------|
| **Perfect Cooperation** | Maximum teamwork, honest signals | 4× Firefighter, 2× Coordinator, 2× Hero, 2× Strategist |
| **Social Dilemma** | Mix of cooperators and free-riders | 3× Firefighter, 2× Coordinator, 3× Free Rider, 2× Opportunist |
| **Chaos Squad** | Unpredictable and experimental | 2× Maverick, 2× Random, 2× Liar, 2× Opportunist, 2× Free Rider |
| **All-Stars** | Best performers from each category | 3× Firefighter, 2× Hero, 2× Coordinator, 2× Strategist, 1× Cautious |
| **Selfish Strategy** | Every agent for themselves | 6× Opportunist, 4× Free Rider |
| **Honest Workers** | No deception, pure effort | 5× Firefighter, 3× Coordinator, 2× Hero |
| **Minimal Team** | Small but mighty (4 agents) | 2× Firefighter, 1× Coordinator, 1× Hero |
| **Balanced Research** | Scientific control group | 1 of each archetype (10 total) |

### 4. Tournament Execution Screen

**Real-time Progress Display**:

```
┌─────────────────────────────────────────────────────────────┐
│                 TOURNAMENT IN PROGRESS                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Scenario: 47/100                                          │
│   ████████████████████████░░░░░░░░░░░░░░░░░░ 47%          │
│                                                              │
│   Performance Summary:                                       │
│   ┌────────────────────────────────────────────────┐        │
│   │  Current Score:  241.6                         │        │
│   │  Mean Score:     238.4                         │        │
│   │  Best Score:     289.2  (Scenario #23)        │        │
│   │  Worst Score:    187.3  (Scenario #41)        │        │
│   │                                                 │        │
│   │  Houses Saved:   72% average                   │        │
│   │  Work Efficiency: 84%                          │        │
│   └────────────────────────────────────────────────┘        │
│                                                              │
│   Live Preview:                                              │
│   [Mini game board showing current scenario - animated]     │
│                                                              │
│   Est. Time Remaining: 12s                                   │
│                                                              │
│   [Pause] [View Current Game Details] [Cancel Tournament]   │
└─────────────────────────────────────────────────────────────┘
```

**Technical Features**:
- Progress bar updates every 5-10 scenarios
- Running statistics calculation
- Live game preview (optional, can be disabled for performance)
- Pause/resume capability
- Background execution continues if user navigates away

### 5. Results Screen

**Comprehensive Analysis Dashboard**:

```
┌─────────────────────────────────────────────────────────────┐
│              🎉 TOURNAMENT COMPLETE! 🎉                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Team: "Dream Team Alpha"          Date: 2025-11-02        │
│                                                              │
│   ┌─── Overall Performance ──────────────────────────┐      │
│   │  Final Average Score:  241.6                     │      │
│   │  Median Score:         239.8                     │      │
│   │  Std. Deviation:       24.3                      │      │
│   │  Success Rate:         87% (scenarios won)       │      │
│   └──────────────────────────────────────────────────┘      │
│                                                              │
│   Score Distribution:                                        │
│   ┌──────────────────────────────────────────────────┐      │
│   │  30│                                              │      │
│   │  25│            ▁▃▅███████▅▃▁                   │      │
│   │  20│          ▁▃███████████████▃▁               │      │
│   │  15│        ▁▃███████████████████▃▁             │      │
│   │  10│      ▁▃█████████████████████████▃▁         │      │
│   │   5│    ▁▃███████████████████████████████▃▁     │      │
│   │   0└────┴────┴────┴────┴────┴────┴────┴────    │      │
│   │     150  200  250  300  350  400  450  500      │      │
│   └──────────────────────────────────────────────────┘      │
│                                                              │
│   🏆 MVP Agents (by marginal contribution):                 │
│   ┌──────────────────────────────────────────────────┐      │
│   │  🥇 House 1 (Firefighter):  +42.3 avg          │      │
│   │  🥈 House 3 (Coordinator):  +38.7 avg          │      │
│   │  🥉 House 6 (Hero):         +35.1 avg          │      │
│   │                                                  │      │
│   │  📊 House 4 (Strategist):   +28.4 avg          │      │
│   │  📊 House 7 (Firefighter):  +24.8 avg          │      │
│   └──────────────────────────────────────────────────┘      │
│                                                              │
│   🚫 Weakest Links:                                          │
│   ┌──────────────────────────────────────────────────┐      │
│   │  House 2 (Free Rider):      -12.4 avg          │      │
│   │  House 9 (Liar):            -8.7 avg           │      │
│   └──────────────────────────────────────────────────┘      │
│                                                              │
│   Scenario Type Performance:                                 │
│   • Trivial Cooperation:  ████████░░  85%                   │
│   • Early Containment:    ███████░░░  73%                   │
│   • Chain Reaction:       ██████░░░░  62%                   │
│   • Sparse Heroics:       ████████░░  81%                   │
│                                                              │
│   [💾 Save Team] [🔄 Run Again] [⚖️ Compare Teams]          │
│   [📊 Detailed Analytics] [🎬 View Best Replay]             │
│   [📤 Share Results] [🏠 Back to Builder]                   │
└─────────────────────────────────────────────────────────────┘
```

### 6. Team Comparison Mode

**Side-by-Side Analysis**:

```
┌─────────────────────────────────────────────────────────────┐
│                    TEAM COMPARISON                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Team A: "Dream Team"         vs.    Team B: "Chaos Squad" │
│                                                              │
│   ┌──────────────────────┐     ┌──────────────────────┐    │
│   │ Avg Score:  241.6    │     │ Avg Score:  198.3    │    │
│   │ Houses:     72%      │     │ Houses:     54%      │    │
│   │ Efficiency: 84%      │     │ Efficiency: 61%      │    │
│   └──────────────────────┘     └──────────────────────┘    │
│                                                              │
│   Winner: Team A by 43.3 points (p < 0.001)                │
│                                                              │
│   Head-to-Head by Scenario Type:                            │
│   • Trivial Cooperation:  A wins (85% vs 78%)              │
│   • Early Containment:    A wins (73% vs 41%)              │
│   • Chain Reaction:       A wins (62% vs 38%)              │
│   • Sparse Heroics:       Tie   (81% vs 79%)               │
│                                                              │
│   Key Differences:                                          │
│   • Team A has better coordination (3 coordinators)         │
│   • Team B more unpredictable (higher variance)            │
│   • Team A dominates high-pressure scenarios               │
│   • Team B performs better in sparse-work scenarios        │
│                                                              │
│   [Load Team A] [Load Team B] [Tournament Rematch]         │
└─────────────────────────────────────────────────────────────┘
```

## Agent Archetypes

### Complete Roster

| # | Name | Icon | Tagline | Primary Traits |
|---|------|------|---------|----------------|
| 1 | **Firefighter** | 🧑‍🚒 | "First to respond, last to rest" | High work ethic, honest, proactive |
| 2 | **Free Rider** | 💤 | "Let others do the heavy lifting" | Low work ethic, high rest bias, selfish |
| 3 | **Coordinator** | 📋 | "Teamwork makes the dream work" | High coordination, honest signals |
| 4 | **Liar** | 🤥 | "Trust me, I'm working hard" | Low honesty, deceptive signals |
| 5 | **Hero** | 🦸 | "I'll save everyone or die trying" | Max altruism, fearless, tireless |
| 6 | **Strategist** | 🎯 | "Calculated action over emotion" | Low exploration, high coordination |
| 7 | **Opportunist** | 💰 | "My house, my rules" | Max self-interest, low altruism |
| 8 | **Cautious** | 😰 | "Better safe than sorry" | High risk aversion, conservative |
| 9 | **Maverick** | 🎲 | "Unpredictable by design" | High exploration, inconsistent |
| 10 | **Random** | ❓ | "Chaos incarnate" | All parameters randomized |

### Parameter Profiles

```typescript
const AGENT_ARCHETYPES: Record<string, AgentArchetype> = {
  firefighter: {
    name: "Firefighter",
    icon: "🧑‍🚒",
    color: "#dc2626", // red
    tagline: "First to respond, last to rest",
    description: "Reliable team player who prioritizes putting out fires over personal rest.",
    parameters: {
      honesty_bias: 1.0,
      work_tendency: 0.9,
      neighbor_help_bias: 0.7,
      own_house_priority: 0.4,
      risk_aversion: 0.5,
      coordination_weight: 0.7,
      exploration_rate: 0.1,
      fatigue_memory: 0.5,
      rest_reward_bias: 0.1,
      altruism_factor: 0.8
    },
    strategyNotes: [
      "Always signals honestly",
      "Prioritizes urgent fires over rest",
      "Helps neighbors proactively",
      "Coordinates well with teammates"
    ]
  },

  free_rider: {
    name: "Free Rider",
    icon: "💤",
    color: "#7c3aed", // purple
    tagline: "Let others do the heavy lifting",
    description: "Prefers to rest and let teammates handle the fires.",
    parameters: {
      honesty_bias: 0.7,
      work_tendency: 0.2,
      neighbor_help_bias: 0.2,
      own_house_priority: 0.9,
      risk_aversion: 0.8,
      coordination_weight: 0.3,
      exploration_rate: 0.2,
      fatigue_memory: 0.8,
      rest_reward_bias: 0.9,
      altruism_factor: 0.1
    },
    strategyNotes: [
      "Signals work but often rests",
      "Only works when own house threatened",
      "Relies on teammates for firefighting",
      "Maximizes personal rest rewards"
    ]
  },

  coordinator: {
    name: "Coordinator",
    icon: "📋",
    color: "#2563eb", // blue
    tagline: "Teamwork makes the dream work",
    description: "Excellent at reading signals and organizing team response.",
    parameters: {
      honesty_bias: 0.9,
      work_tendency: 0.6,
      neighbor_help_bias: 0.6,
      own_house_priority: 0.5,
      risk_aversion: 0.5,
      coordination_weight: 1.0,
      exploration_rate: 0.05,
      fatigue_memory: 0.4,
      rest_reward_bias: 0.4,
      altruism_factor: 0.6
    },
    strategyNotes: [
      "Highly responsive to team signals",
      "Avoids redundant work on same fire",
      "Balances work and rest strategically",
      "Fills gaps in team coverage"
    ]
  },

  liar: {
    name: "Liar",
    icon: "🤥",
    color: "#16a34a", // green
    tagline: "Trust me, I'm working hard",
    description: "Sends false signals to mislead teammates.",
    parameters: {
      honesty_bias: 0.1,
      work_tendency: 0.5,
      neighbor_help_bias: 0.3,
      own_house_priority: 0.7,
      risk_aversion: 0.4,
      coordination_weight: 0.6,
      exploration_rate: 0.3,
      fatigue_memory: 0.5,
      rest_reward_bias: 0.6,
      altruism_factor: 0.2
    },
    strategyNotes: [
      "Signals opposite of actual intent",
      "Creates confusion in team coordination",
      "Prioritizes self-interest",
      "Exploits others' trust"
    ]
  },

  hero: {
    name: "Hero",
    icon: "🦸",
    color: "#eab308", // gold
    tagline: "I'll save everyone or die trying",
    description: "Maximum effort, maximum altruism. Never gives up.",
    parameters: {
      honesty_bias: 1.0,
      work_tendency: 1.0,
      neighbor_help_bias: 0.9,
      own_house_priority: 0.2,
      risk_aversion: 0.1,
      coordination_weight: 0.5,
      exploration_rate: 0.1,
      fatigue_memory: 0.9,
      rest_reward_bias: 0.0,
      altruism_factor: 1.0
    },
    strategyNotes: [
      "Never rests while fires burn",
      "Helps everyone equally",
      "Ignores personal costs",
      "Consistent and predictable"
    ]
  },

  strategist: {
    name: "Strategist",
    icon: "🎯",
    color: "#1e3a8a", // navy
    tagline: "Calculated action over emotion",
    description: "Analyzes situation carefully before acting.",
    parameters: {
      honesty_bias: 0.9,
      work_tendency: 0.6,
      neighbor_help_bias: 0.5,
      own_house_priority: 0.5,
      risk_aversion: 0.7,
      coordination_weight: 0.9,
      exploration_rate: 0.05,
      fatigue_memory: 0.3,
      rest_reward_bias: 0.5,
      altruism_factor: 0.6
    },
    strategyNotes: [
      "Calculates optimal response",
      "Avoids wasteful overwork",
      "Responds to risk level",
      "Minimal exploration, maximum efficiency"
    ]
  },

  opportunist: {
    name: "Opportunist",
    icon: "💰",
    color: "#ea580c", // orange
    tagline: "My house, my rules",
    description: "Laser-focused on protecting own property.",
    parameters: {
      honesty_bias: 0.6,
      work_tendency: 0.6,
      neighbor_help_bias: 0.1,
      own_house_priority: 1.0,
      risk_aversion: 0.6,
      coordination_weight: 0.2,
      exploration_rate: 0.2,
      fatigue_memory: 0.6,
      rest_reward_bias: 0.7,
      altruism_factor: 0.0
    },
    strategyNotes: [
      "Only defends own house",
      "Ignores team fires",
      "Maximizes personal reward",
      "Doesn't coordinate with others"
    ]
  },

  cautious: {
    name: "Cautious",
    icon: "😰",
    color: "#facc15", // yellow
    tagline: "Better safe than sorry",
    description: "Avoids risky situations, prefers conservative approach.",
    parameters: {
      honesty_bias: 0.9,
      work_tendency: 0.4,
      neighbor_help_bias: 0.4,
      own_house_priority: 0.7,
      risk_aversion: 0.9,
      coordination_weight: 0.8,
      exploration_rate: 0.05,
      fatigue_memory: 0.7,
      rest_reward_bias: 0.6,
      altruism_factor: 0.4
    },
    strategyNotes: [
      "Works less when many fires present",
      "Prioritizes self-preservation",
      "Honest but conservative",
      "Avoids overcommitment"
    ]
  },

  maverick: {
    name: "Maverick",
    icon: "🎲",
    color: "#ec4899", // pink
    tagline: "Unpredictable by design",
    description: "High variance strategy, keeps opponents guessing.",
    parameters: {
      honesty_bias: 0.5,
      work_tendency: 0.5,
      neighbor_help_bias: 0.5,
      own_house_priority: 0.5,
      risk_aversion: 0.5,
      coordination_weight: 0.5,
      exploration_rate: 1.0,
      fatigue_memory: 0.3,
      rest_reward_bias: 0.5,
      altruism_factor: 0.5
    },
    strategyNotes: [
      "High exploration rate",
      "Tries different strategies",
      "Unpredictable behavior",
      "Good for learning optimal patterns"
    ]
  },

  random: {
    name: "Random",
    icon: "❓",
    color: "#6b7280", // gray
    tagline: "Chaos incarnate",
    description: "All parameters randomized each game. Pure chaos.",
    parameters: {
      // Randomized on each instantiation
      honesty_bias: 0.5,
      work_tendency: 0.5,
      neighbor_help_bias: 0.5,
      own_house_priority: 0.5,
      risk_aversion: 0.5,
      coordination_weight: 0.5,
      exploration_rate: 0.5,
      fatigue_memory: 0.5,
      rest_reward_bias: 0.5,
      altruism_factor: 0.5
    },
    isRandomized: true,
    strategyNotes: [
      "Parameters change every game",
      "Completely unpredictable",
      "Useful for testing adaptability",
      "Baseline for comparison"
    ]
  }
};
```

## Technical Architecture

### Component Structure

```
web/src/
├── pages/
│   └── TeamBuilder.tsx           # Main page component
├── components/
│   ├── team-builder/
│   │   ├── TeamBuilderLayout.tsx      # Overall layout
│   │   ├── TeamSelector.tsx           # Circular team display
│   │   ├── AgentCard.tsx              # Individual agent display
│   │   ├── AgentSelectorModal.tsx     # Agent selection UI
│   │   ├── AgentStatsDisplay.tsx      # Stat bars visualization
│   │   ├── TournamentRunner.tsx       # Progress display
│   │   ├── TournamentResults.tsx      # Results dashboard
│   │   ├── TeamComparison.tsx         # Side-by-side comparison
│   │   ├── ScoreDistribution.tsx      # Histogram chart
│   │   └── TemplateSelector.tsx       # Pre-built templates
├── utils/
│   ├── teamBuilder.ts            # Team management logic
│   ├── tournamentEngine.ts       # Tournament orchestration
│   └── agentContributions.ts     # Contribution estimation
└── types/
    └── teamBuilder.ts            # TypeScript definitions
```

### Data Types

```typescript
// Core types
export interface AgentArchetype {
  name: string;
  icon: string;
  color: string;
  tagline: string;
  description: string;
  parameters: AgentParameters;
  strategyNotes: string[];
  isRandomized?: boolean;
}

export interface AgentParameters {
  honesty_bias: number;
  work_tendency: number;
  neighbor_help_bias: number;
  own_house_priority: number;
  risk_aversion: number;
  coordination_weight: number;
  exploration_rate: number;
  fatigue_memory: number;
  rest_reward_bias: number;
  altruism_factor: number;
}

export interface TeamComposition {
  id: string;
  name: string;
  positions: (AgentArchetype | null)[]; // Up to 10 slots
  createdAt: number;
  modifiedAt: number;
  tournamentHistory: string[]; // Tournament result IDs
}

export interface TournamentConfig {
  teamId: string;
  numScenarios: number; // Default 100
  scenarioTypes?: string[]; // Optional filter
  seed?: number; // For reproducibility
}

export interface TournamentProgress {
  current: number;
  total: number;
  results: ScenarioResult[];
  statistics: {
    mean: number;
    median: number;
    stdDev: number;
    min: number;
    max: number;
  };
  startTime: number;
  estimatedCompletion: number;
}

export interface TournamentResult {
  id: string;
  teamId: string;
  teamName: string;
  timestamp: number;
  duration: number;
  config: TournamentConfig;
  scenarios: ScenarioResult[];
  statistics: TournamentStatistics;
  agentContributions: AgentContribution[];
  scenarioTypePerformance: Record<string, number>;
}

export interface TournamentStatistics {
  mean: number;
  median: number;
  stdDev: number;
  min: number;
  max: number;
  q25: number;
  q75: number;
  successRate: number; // % of scenarios "won"
  housesSavedAvg: number;
  workEfficiency: number;
}

export interface AgentContribution {
  position: number;
  archetype: string;
  avgContribution: number;
  consistency: number; // 0-1 score
  mvpCount: number; // Times this agent was top performer
  rank: number;
}

export interface ScenarioResult {
  scenarioId: string;
  scenarioType: string;
  teamScore: number;
  agentScores: number[];
  housesSaved: number;
  nightsPlayed: number;
  replayData: any; // Full game replay
}

export interface TeamComparison {
  teamA: TournamentResult;
  teamB: TournamentResult;
  winner: 'A' | 'B' | 'tie';
  scoreDifference: number;
  significance: number; // Statistical significance
  scenarioWins: { A: number; B: number; ties: number };
  scenarioTypeComparison: Record<string, { A: number; B: number }>;
}
```

### Storage Schema

```typescript
// Session Storage Keys
const STORAGE_KEYS = {
  TEAMS: 'bucket_brigade_teams',
  TOURNAMENTS: 'bucket_brigade_tournaments',
  ACTIVE_TEAM: 'bucket_brigade_active_team',
  TEMPLATES: 'bucket_brigade_templates'
} as const;

// Storage structure
interface StorageSchema {
  teams: Record<string, TeamComposition>;
  tournaments: Record<string, TournamentResult>;
  activeTeamId: string | null;
  templates: Record<string, TeamComposition>;
}
```

### Tournament Engine

```typescript
class TournamentEngine {
  private workers: Worker[] = [];
  private numWorkers: number;
  private wasmReady = false;

  constructor(numWorkers = navigator.hardwareConcurrency || 4) {
    this.numWorkers = numWorkers;
  }

  async initialize(): Promise<void> {
    // Initialize Web Workers with WASM
    for (let i = 0; i < this.numWorkers; i++) {
      const worker = new Worker(
        new URL('../workers/gameWorker.ts', import.meta.url),
        { type: 'module' }
      );
      await this.initializeWorker(worker);
      this.workers.push(worker);
    }
    this.wasmReady = true;
  }

  async runTournament(
    team: TeamComposition,
    config: TournamentConfig,
    onProgress?: (progress: TournamentProgress) => void
  ): Promise<TournamentResult> {
    if (!this.wasmReady) {
      await this.initialize();
    }

    // Generate scenarios
    const scenarios = this.generateScenarios(config);

    // Distribute work across workers
    const scenariosPerWorker = Math.ceil(scenarios.length / this.numWorkers);
    const startTime = Date.now();

    const workerPromises = this.workers.map(async (worker, i) => {
      const start = i * scenariosPerWorker;
      const end = Math.min(start + scenariosPerWorker, scenarios.length);
      const batch = scenarios.slice(start, end);

      return this.runBatch(worker, team, batch, (partial) => {
        // Aggregate progress from all workers
        if (onProgress) {
          const progress = this.aggregateProgress(partial, start);
          onProgress(progress);
        }
      });
    });

    // Wait for all workers to complete
    const batchResults = await Promise.all(workerPromises);
    const allResults = batchResults.flat();

    // Calculate statistics and contributions
    return this.buildTournamentResult(
      team,
      config,
      allResults,
      Date.now() - startTime
    );
  }

  private async runBatch(
    worker: Worker,
    team: TeamComposition,
    scenarios: Scenario[],
    onProgress: (results: ScenarioResult[]) => void
  ): Promise<ScenarioResult[]> {
    return new Promise((resolve, reject) => {
      const results: ScenarioResult[] = [];

      worker.onmessage = (e) => {
        const { type, data } = e.data;

        if (type === 'progress') {
          results.push(data.result);
          onProgress(results);
        } else if (type === 'complete') {
          resolve(results);
        } else if (type === 'error') {
          reject(new Error(data.message));
        }
      };

      worker.postMessage({
        type: 'runBatch',
        team: this.serializeTeam(team),
        scenarios
      });
    });
  }

  private generateScenarios(config: TournamentConfig): Scenario[] {
    // Generate random scenarios or use predefined distributions
    const scenarios: Scenario[] = [];

    for (let i = 0; i < config.numScenarios; i++) {
      // Balanced distribution across scenario types
      const scenarioType = this.selectScenarioType(i, config);
      scenarios.push(generateScenario(scenarioType, config.seed));
    }

    return scenarios;
  }

  private buildTournamentResult(
    team: TeamComposition,
    config: TournamentConfig,
    results: ScenarioResult[],
    duration: number
  ): TournamentResult {
    const statistics = this.calculateStatistics(results);
    const agentContributions = this.estimateContributions(results, team);
    const scenarioTypePerformance = this.analyzeScenarioTypes(results);

    return {
      id: generateId(),
      teamId: team.id,
      teamName: team.name,
      timestamp: Date.now(),
      duration,
      config,
      scenarios: results,
      statistics,
      agentContributions,
      scenarioTypePerformance
    };
  }

  private estimateContributions(
    results: ScenarioResult[],
    team: TeamComposition
  ): AgentContribution[] {
    // Simplified Shapley value estimation
    // For each position, estimate marginal contribution
    const contributions = team.positions.map((agent, position) => {
      if (!agent) {
        return null;
      }

      // Average the agent's individual score across all scenarios
      const avgScore = results.reduce((sum, r) => {
        return sum + r.agentScores[position];
      }, 0) / results.length;

      // Calculate consistency (inverse of variance)
      const scores = results.map(r => r.agentScores[position]);
      const variance = this.calculateVariance(scores);
      const consistency = 1 / (1 + variance / 100);

      // Count MVP appearances (top score in scenario)
      const mvpCount = results.filter(r => {
        const maxScore = Math.max(...r.agentScores);
        return r.agentScores[position] === maxScore;
      }).length;

      return {
        position,
        archetype: agent.name,
        avgContribution: avgScore,
        consistency,
        mvpCount,
        rank: 0 // Will be set after sorting
      };
    }).filter(Boolean) as AgentContribution[];

    // Rank by contribution
    contributions.sort((a, b) => b.avgContribution - a.avgContribution);
    contributions.forEach((c, i) => c.rank = i + 1);

    return contributions;
  }

  private calculateStatistics(results: ScenarioResult[]): TournamentStatistics {
    const scores = results.map(r => r.teamScore);
    scores.sort((a, b) => a - b);

    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const median = scores[Math.floor(scores.length / 2)];
    const q25 = scores[Math.floor(scores.length * 0.25)];
    const q75 = scores[Math.floor(scores.length * 0.75)];

    const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);

    const successRate = results.filter(r => r.housesSaved >= 5).length / results.length;
    const housesSavedAvg = results.reduce((sum, r) => sum + r.housesSaved, 0) / results.length;

    // Calculate work efficiency (houses saved per total work actions)
    const totalWorkActions = results.reduce((sum, r) => {
      return sum + r.agentScores.length * r.nightsPlayed;
    }, 0);
    const workEfficiency = (housesSavedAvg * 10) / (totalWorkActions / results.length);

    return {
      mean,
      median,
      stdDev,
      min: scores[0],
      max: scores[scores.length - 1],
      q25,
      q75,
      successRate,
      housesSavedAvg,
      workEfficiency
    };
  }

  cleanup(): void {
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    this.wasmReady = false;
  }
}
```

### Web Worker Implementation

```typescript
// workers/gameWorker.ts
import init, { WasmBucketBrigade } from 'bucket-brigade-wasm';

let wasmReady = false;

self.onmessage = async (e) => {
  const { type, team, scenarios } = e.data;

  if (type === 'init') {
    await init();
    wasmReady = true;
    self.postMessage({ type: 'ready' });
    return;
  }

  if (type === 'runBatch') {
    if (!wasmReady) {
      await init();
      wasmReady = true;
    }

    try {
      for (const scenario of scenarios) {
        const result = await runSingleGame(team, scenario);

        // Send progress update
        self.postMessage({
          type: 'progress',
          data: { result }
        });
      }

      self.postMessage({ type: 'complete' });
    } catch (error) {
      self.postMessage({
        type: 'error',
        data: { message: error.message }
      });
    }
  }
};

async function runSingleGame(team, scenario): Promise<ScenarioResult> {
  const engine = new WasmBucketBrigade(scenario);

  const agents = team.positions.map((archetype, i) => {
    return createHeuristicAgent(i, archetype.parameters);
  });

  const agentScores: number[] = [];
  let teamScore = 0;
  let nightsPlayed = 0;
  let housesSaved = 0;

  while (!engine.is_done()) {
    // Get observations for all agents
    const observations = engine.get_observations();

    // Each agent decides action
    const actions = agents.map((agent, i) => {
      return agent.act(observations[i]);
    });

    // Execute step
    const stepResult = engine.step(actions);
    nightsPlayed++;
  }

  // Get final results
  teamScore = engine.get_team_score();
  agentScores.push(...engine.get_agent_scores());
  housesSaved = engine.get_houses_saved();

  return {
    scenarioId: scenario.id,
    scenarioType: scenario.type,
    teamScore,
    agentScores,
    housesSaved,
    nightsPlayed,
    replayData: engine.get_replay()
  };
}
```

## Performance Targets

### WASM Engine Performance

- **Single game (50 nights avg)**: ~5ms
- **100 scenarios**: ~500ms sequential
- **With 4 workers**: ~125ms parallel
- **Target total time**: < 2 seconds (including overhead)

### UI Responsiveness

- **Team selection**: Instant feedback (<16ms)
- **Agent modal**: Smooth transitions (60fps)
- **Progress updates**: Every 10 scenarios (10-20ms)
- **Results rendering**: < 100ms

### Memory Budget

- **Base app**: ~2MB
- **WASM module**: ~100KB
- **Worker overhead**: ~1MB per worker
- **Total**: < 10MB for 4 workers

## Implementation Phases

### Phase 1: Core UI (Week 1)
- [ ] Create TeamBuilder page route
- [ ] Implement TeamSelector circular layout
- [ ] Build AgentSelectorModal with carousel
- [ ] Create AgentCard component
- [ ] Add team size controls
- [ ] Implement local storage for teams

### Phase 2: Agent System (Week 1)
- [ ] Define all 10 agent archetypes
- [ ] Create AgentStatsDisplay component
- [ ] Implement parameter-to-stats visualization
- [ ] Add strategy notes rendering
- [ ] Create template system

### Phase 3: Tournament Engine (Week 2)
- [ ] Implement TournamentEngine class
- [ ] Create Web Worker setup
- [ ] Integrate WASM engine calls
- [ ] Add progress tracking
- [ ] Build scenario generator

### Phase 4: Results & Analysis (Week 2)
- [ ] Create TournamentResults dashboard
- [ ] Implement score distribution chart
- [ ] Build agent contribution estimation
- [ ] Add scenario type analysis
- [ ] Create results export/sharing

### Phase 5: Advanced Features (Week 3)
- [ ] Team comparison mode
- [ ] Challenge mode
- [ ] Leaderboards integration
- [ ] Achievement system
- [ ] Social sharing

### Phase 6: Polish & Testing (Week 3)
- [ ] Retro visual styling
- [ ] Animations and transitions
- [ ] Mobile responsiveness
- [ ] Performance optimization
- [ ] End-to-end tests

## Success Metrics

### User Engagement
- Average time on team builder page
- Number of teams created per session
- Tournament completion rate
- Template usage vs custom teams

### Performance
- Tournament completion time < 5s
- Zero UI freezing during execution
- Smooth 60fps animations
- Memory usage < 50MB

### Functionality
- All 10 archetypes working correctly
- Statistical analysis accuracy
- Agent contribution estimation correlation
- Replay system integration

## Future Enhancements

### Short-term
- Custom agent creator (parameter sliders)
- Historical performance tracking
- Team evolution recommendations
- Social leaderboards

### Long-term
- Multi-player tournaments
- Real-time team vs team
- Agent marketplace
- ML-powered team suggestions
- Integration with RL training

## Testing Strategy

### Unit Tests
- Agent parameter validation
- Tournament statistics calculation
- Contribution estimation logic
- Storage operations

### Integration Tests
- Full tournament execution
- Worker communication
- WASM integration
- Results aggregation

### E2E Tests
```typescript
test('complete team builder flow', async ({ page }) => {
  await page.goto('/team-builder');

  // Select agents
  await page.click('[data-position="0"]');
  await page.click('[data-archetype="firefighter"]');
  await page.click('text=Select This Agent');

  // ... select remaining agents

  // Start tournament
  await page.click('text=Start Tournament');

  // Wait for completion
  await page.waitForSelector('text=TOURNAMENT COMPLETE');

  // Verify results
  const score = await page.textContent('[data-testid="final-score"]');
  expect(parseFloat(score)).toBeGreaterThan(0);
});
```

## Accessibility

- Keyboard navigation for all interactions
- Screen reader support for agent stats
- High contrast mode support
- Reduced motion option
- Clear focus indicators
- Semantic HTML structure

## Documentation

- User guide with screenshots
- Strategy guide for each archetype
- Tips for team composition
- Understanding results metrics
- FAQ section

---

**Status**: 🟢 Ready for Implementation
**Priority**: High
**Estimated Effort**: 3 weeks
**Dependencies**: WASM engine, Web Workers support
**Target Release**: Q1 2025
