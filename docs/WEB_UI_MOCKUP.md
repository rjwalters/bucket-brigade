# 🎨 Simplified Web UI - Design Mockup

**Version**: 2.0 (Single-Game Focus)
**Last Updated**: 2025-11-03

---

## 🎯 Design Goals

1. **Single Game Focus** - One game at a time, fully visualized
2. **Educational** - Help users understand game dynamics
3. **Interactive** - Easy to explore different scenarios
4. **Analysis** - Show what happened and why
5. **Fast** - No backend, instant loading

---

## 📱 Page Structure

### Navigation (Simplified)

```
┌──────────────────────────────────────────────────────────┐
│  🔥 Bucket Brigade                  [Dashboard] [Settings]│
└──────────────────────────────────────────────────────────┘
```

Only 2 main pages:
1. **Dashboard** - Game launcher (team + scenario selection)
2. **Settings** - Basic preferences

*Game Replay is reached by clicking "Run Game" on Dashboard*

---

## 🏠 Dashboard Page (Main Entry)

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🔥 Bucket Brigade                       [Dashboard] [Settings]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                  Watch a Bucket Brigade Game                     │
│         Experience cooperation, deception, and firefighting      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐     ┌──────────────────────────────┐ │
│  │   🤖 Select Team     │     │   🌍 Select Scenario         │ │
│  ├──────────────────────┤     ├──────────────────────────────┤ │
│  │                      │     │                              │ │
│  │ ○ All Firefighters   │     │ ○ Trivial Cooperation       │ │
│  │   (4x Firefighter)   │     │   Easy fires, obvious win   │ │
│  │                      │     │                              │ │
│  │ ○ All Coordinators   │     │ ○ Early Containment         │ │
│  │   (4x Coordinator)   │     │   Time pressure, fast spread│ │
│  │                      │     │                              │ │
│  │ ○ All Heroes         │     │ ○ Greedy Neighbor           │ │
│  │   (4x Hero)          │     │   Self-interest dilemma     │ │
│  │                      │     │                              │ │
│  │ ○ Mixed Team         │     │ ○ Chain Reaction            │ │
│  │   (2F, 1C, 1H)       │     │   High fire spread          │ │
│  │                      │     │                              │ │
│  │ ○ Free Riders        │     │ ○ Sparse Heroics            │ │
│  │   (2F, 2FR)          │     │   Minimal workers needed    │ │
│  │                      │     │                              │ │
│  │ ○ Custom Team...     │     │ ○ Rest Trap                 │ │
│  │   [Edit]             │     │   Rare but dangerous fires  │ │
│  │                      │     │                              │ │
│  │  🎲 Randomize        │     │  🎲 Randomize               │ │
│  │                      │     │                              │ │
│  └──────────────────────┘     └──────────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    [▶  Run Game  (50ms)]                        │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📚 Learn More:                                                 │
│  • What is Bucket Brigade?                                      │
│  • How do agents decide?                                        │
│  • What are scenarios?                                          │
│  • Research paper                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Team Selection Details

**Preset Teams** (radio buttons with descriptions):

1. **All Firefighters** - "Aggressive firefighters who always work"
2. **All Coordinators** - "Balanced agents who signal honestly"
3. **All Heroes** - "Risk-takers who save distant houses"
4. **Mixed Team** - "2 Firefighters, 1 Coordinator, 1 Hero"
5. **Free Riders** - "2 Firefighters, 2 Free Riders (test cooperation)"
6. **Custom Team** - Opens modal to pick each agent individually

**Custom Team Modal**:
```
┌───────────────────────────────────────────────────────┐
│  Build Your Team (4 agents)                          │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Agent 1: [Firefighter ▼]  [?] Info                  │
│  Agent 2: [Coordinator ▼]  [?] Info                  │
│  Agent 3: [Hero        ▼]  [?] Info                  │
│  Agent 4: [Free Rider  ▼]  [?] Info                  │
│                                                       │
│  Available agents:                                    │
│  • Firefighter - Always works, goes to fires         │
│  • Coordinator - Balances work and rest              │
│  • Hero - Takes risks to save distant houses         │
│  • Free Rider - Minimizes work, relies on others     │
│  • Greedy Neighbor - Protects own house only         │
│  • Deceptive - Lies in signals                       │
│                                                       │
│                [Cancel]  [Save Team]                  │
└───────────────────────────────────────────────────────┘
```

### Scenario Selection Details

**Scenarios** (radio buttons with short descriptions):

1. **Trivial Cooperation** - "Easy fires, clear benefit to working together"
2. **Early Containment** - "Fast fire spread, requires immediate coordination"
3. **Greedy Neighbor** - "Social dilemma: protect own house or help others?"
4. **Chain Reaction** - "High spread rate, distributed teams needed"
5. **Sparse Heroics** - "Few workers needed, overworking is wasteful"
6. **Rest Trap** - "Usually safe to rest, but disasters require response"
7. **Deceptive Calm** - "Rare fire outbreaks reward honest signaling"
8. **Overcrowding** - "Too many workers reduce efficiency"
9. **Mixed Motivation** - "House ownership creates conflicting goals"

Each has a `[?]` info icon that shows full description on hover/click.

---

## 🎮 Game Replay Page (After "Run Game")

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🔥 Bucket Brigade                       [Dashboard] [Settings]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🎮 Mixed Team vs. Early Containment Scenario                   │
│  Night 12/30  •  Houses Safe: 7/10  •  Score: 241.6            │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │                    GAME VISUALIZATION                      │ │
│  │                                                            │ │
│  │                      🏠 🏠 🏠                             │ │
│  │                    🏠         🏠                         │ │
│  │                  🏠             🏠                       │ │
│  │                    🏠         🏠                         │ │
│  │                      🏠 🏠 🏠                             │ │
│  │                                                            │ │
│  │  🔥 = Burning    👤 = Agent    💧 = Working              │ │
│  │  ✅ = Safe       ❌ = Ruined    💤 = Resting              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ◀◀  ◀  ▶  ▶▶   [Night 12/30]   Speed: ━━━━○─────        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐   ┌─────────────────────────────────────┐│
│  │  Agent Status    │   │  Turn Details                       ││
│  ├──────────────────┤   ├─────────────────────────────────────┤│
│  │                  │   │                                     ││
│  │ Firefighter      │   │ Night 12 Events:                    ││
│  │ 🏠 House 0       │   │                                     ││
│  │ 📍 Location: 3   │   │ • Agent 1 → House 3 (WORK)         ││
│  │ 💪 Working       │   │ • Agent 2 → House 2 (REST)         ││
│  │ 📊 Score: 32.5   │   │ • Agent 3 → House 5 (WORK)         ││
│  │                  │   │ • Agent 4 → House 0 (WORK)         ││
│  │ Coordinator      │   │                                     ││
│  │ 🏠 House 2       │   │ Fires Extinguished:                 ││
│  │ 📍 Location: 2   │   │ • House 3 ✓ (1 worker)             ││
│  │ 💤 Resting       │   │ • House 5 ✓ (1 worker)             ││
│  │ 📊 Score: 28.0   │   │                                     ││
│  │                  │   │ New Fires:                          ││
│  │ Hero             │   │ • House 7 (spread from 6)           ││
│  │ 🏠 House 5       │   │                                     ││
│  │ 📍 Location: 5   │   │ Rewards:                            ││
│  │ 💪 Working       │   │ • Agent 1: +2.5                     ││
│  │ 📊 Score: 45.2   │   │ • Agent 2: -0.5 (rest)             ││
│  │                  │   │ • Agent 3: +3.2                     ││
│  │ Free Rider       │   │ • Agent 4: +1.8                     ││
│  │ 🏠 House 7       │   │                                     ││
│  │ 📍 Location: 0   │   └─────────────────────────────────────┘│
│  │ 💤 Resting       │                                          │
│  │ 📊 Score: -5.0   │                                          │
│  │                  │                                          │
│  └──────────────────┘                                          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📊 POST-GAME ANALYSIS (Shown when game ends)                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  🎉 Game Complete!                                         │ │
│  │                                                            │ │
│  │  Final Score: 241.6 (Excellent - top 20%)                 │ │
│  │  Nights Played: 18 / min 12                               │ │
│  │  Houses Saved: 7/10 (70%)                                 │ │
│  │                                                            │ │
│  │  ┌──────────────────────────────────────────────────────┐│ │
│  │  │  Individual Performance                              ││ │
│  │  ├──────────────────────────────────────────────────────┤│ │
│  │  │                                                      ││ │
│  │  │  Firefighter (Agent 1)       Score: 32.5            ││ │
│  │  │  Contribution: 28%           ━━━━━━━━━━━━━─────    ││ │
│  │  │  Strategy: Aggressive fire control                  ││ │
│  │  │  • Worked 14/18 nights                              ││ │
│  │  │  • Extinguished 8 fires                             ││ │
│  │  │  • Protected own house + neighbors                  ││ │
│  │  │                                                      ││ │
│  │  │  Coordinator (Agent 2)       Score: 28.0            ││ │
│  │  │  Contribution: 24%           ━━━━━━━━━━────────    ││ │
│  │  │  Strategy: Balanced work/rest                       ││ │
│  │  │  • Worked 10/18 nights                              ││ │
│  │  │  • Honest signaling (100%)                          ││ │
│  │  │  • Efficient resource use                           ││ │
│  │  │                                                      ││ │
│  │  │  Hero (Agent 3)              Score: 45.2            ││ │
│  │  │  Contribution: 38%           ━━━━━━━━━━━━━━━━───  ││ │
│  │  │  Strategy: Risk-taking saves                        ││ │
│  │  │  • Worked 12/18 nights                              ││ │
│  │  │  • Saved distant houses (3)                         ││ │
│  │  │  • High-value interventions                         ││ │
│  │  │                                                      ││ │
│  │  │  Free Rider (Agent 4)        Score: -5.0            ││ │
│  │  │  Contribution: 10%           ━━─────────────────    ││ │
│  │  │  Strategy: Minimal effort                           ││ │
│  │  │  • Worked 2/18 nights (11%)                         ││ │
│  │  │  • Benefited from team work                         ││ │
│  │  │  • Low individual contribution                      ││ │
│  │  │                                                      ││ │
│  │  └──────────────────────────────────────────────────────┘│ │
│  │                                                            │ │
│  │  ┌──────────────────────────────────────────────────────┐│ │
│  │  │  Key Insights                                        ││ │
│  │  ├──────────────────────────────────────────────────────┤│ │
│  │  │                                                      ││ │
│  │  │  ✅ Team coordinated well in early game (nights 1-5)││ │
│  │  │  ✅ Hero's risk-taking paid off (saved 3 houses)    ││ │
│  │  │  ⚠️  Free Rider underperformed (drag on team)       ││ │
│  │  │  📈 Could improve: More coordination nights 10-15   ││ │
│  │  │                                                      ││ │
│  │  └──────────────────────────────────────────────────────┘│ │
│  │                                                            │ │
│  │  [📥 Download Replay JSON]    [🔄 Try Again]              │ │
│  │  [🏠 Back to Dashboard]                                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Live Visualization**
   - Houses in a circle (like clock positions)
   - Color-coded states: Green (safe), Red (burning), Gray (ruined)
   - Agents shown as avatars with icons
   - Animations for movement and fire spread

2. **Replay Controls**
   - Previous/Next night buttons
   - Skip to start/end
   - Speed slider (0.5x to 4x)
   - Current night indicator

3. **Agent Status Panel**
   - Each agent's current state
   - House ownership
   - Current location
   - Action (working/resting)
   - Running score

4. **Turn Details Panel**
   - Events that happened this turn
   - Who went where
   - What got extinguished
   - New fires that spawned
   - Rewards earned

5. **Post-Game Analysis** (Appears when done=true)
   - Final scores and rankings
   - Individual contributions
   - Strategy summaries
   - Key moments/insights
   - Performance vs. optimal

---

## ⚙️ Settings Page

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🔥 Bucket Brigade                       [Dashboard] [Settings]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ⚙️ Settings                                                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  🎨 Appearance                                             │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  Theme:  ○ Light  ● Dark  ○ Auto                          │ │
│  │                                                            │ │
│  │  Animation Speed:  Slow ━━━○───────── Fast                │ │
│  │                    (for visualizations)                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  🎮 Gameplay                                               │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  Default Team:     [Mixed Team          ▼]                │ │
│  │  Default Scenario: [Early Containment   ▼]                │ │
│  │                                                            │ │
│  │  Auto-play on load:  ☐ Enabled                            │ │
│  │  Show advanced stats: ☑ Enabled                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  🚀 Performance                                            │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  Engine:  ○ JavaScript  ● WASM (faster)                   │ │
│  │           Note: Requires WASM support in browser          │ │
│  │                                                            │ │
│  │  Cache replays:  ☑ Keep last 10 games in browser          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  📊 Data                                                   │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  Storage used: 234 KB / 10 MB                             │ │
│  │  Cached games: 7                                           │ │
│  │                                                            │ │
│  │  [Clear All Data]  [Export All Replays]                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ℹ️ About                                                  │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  Bucket Brigade v2.0                                       │ │
│  │  Research platform for multi-agent cooperation            │ │
│  │                                                            │ │
│  │  [📖 Documentation]  [🔬 Research Paper]  [💻 GitHub]    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│                          [Save Settings]                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎨 Visual Design System

### Colors

```css
/* Light Theme */
--bg-primary: #ffffff;
--bg-secondary: #f5f5f5;
--text-primary: #1a1a1a;
--text-secondary: #666666;
--accent: #ff6b35;  /* Fire orange */
--success: #4caf50;  /* Safe green */
--danger: #f44336;   /* Burning red */
--warning: #ff9800;  /* Warning orange */

/* Dark Theme */
--bg-primary: #1a1a1a;
--bg-secondary: #2a2a2a;
--text-primary: #ffffff;
--text-secondary: #b0b0b0;
/* Accent colors stay the same */
```

### Typography

```css
--font-heading: 'Inter', sans-serif;
--font-body: 'Inter', sans-serif;
--font-mono: 'Fira Code', monospace;

--size-xs: 0.75rem;   /* 12px */
--size-sm: 0.875rem;  /* 14px */
--size-base: 1rem;    /* 16px */
--size-lg: 1.25rem;   /* 20px */
--size-xl: 1.5rem;    /* 24px */
--size-2xl: 2rem;     /* 32px */
```

### Icons

Use simple emoji + text labels:
- 🏠 House
- 🔥 Fire/Burning
- ✅ Safe
- ❌ Ruined
- 👤 Agent
- 💧 Working
- 💤 Resting
- 🎲 Random
- ⚙️ Settings
- 📊 Stats/Analysis

---

## 📱 Responsive Design

### Desktop (1024px+)
- Side-by-side panels
- Full visualization
- All controls visible

### Tablet (768px - 1023px)
- Stacked panels
- Slightly smaller game board
- Collapsible side panels

### Mobile (< 768px)
- Single column layout
- Simplified controls
- Swipeable agent status
- Optimized touch targets

---

## ♿ Accessibility

1. **Keyboard Navigation**
   - Tab through all controls
   - Arrow keys for replay control
   - Space to play/pause

2. **Screen Reader Support**
   - ARIA labels on all interactive elements
   - Live regions for game events
   - Alt text for all icons

3. **Color Contrast**
   - WCAG AA compliant (minimum 4.5:1)
   - Color + icon for house states (not color alone)

4. **Reduced Motion**
   - Respect `prefers-reduced-motion`
   - Option to disable animations

---

## 🔧 Technical Implementation Notes

### State Management

```typescript
// Global app state (React Context or Zustand)
interface AppState {
  // Current game
  currentGame: {
    replay: GameReplay | null;
    currentNight: number;
    isPlaying: boolean;
    speed: number;
  };

  // User preferences
  settings: {
    theme: 'light' | 'dark' | 'auto';
    animationSpeed: number;
    defaultTeam: string;
    defaultScenario: string;
    useWasm: boolean;
    cacheReplays: boolean;
  };

  // Cached data
  recentGames: GameReplay[];  // Last 10
}
```

### Component Hierarchy

```
App
├── Header (navigation)
├── Router
│   ├── Dashboard
│   │   ├── TeamSelector
│   │   ├── ScenarioSelector
│   │   └── RunGameButton
│   │
│   ├── GameReplay
│   │   ├── GameVisualization
│   │   │   ├── Town (circular house layout)
│   │   │   └── AgentLayer (agent positions)
│   │   ├── ReplayControls
│   │   ├── AgentStatusPanel
│   │   ├── TurnDetailsPanel
│   │   └── GameAnalysis (conditional, when done)
│   │
│   └── Settings
│       ├── AppearanceSection
│       ├── GameplaySection
│       ├── PerformanceSection
│       ├── DataSection
│       └── AboutSection
│
└── Footer (links, version)
```

---

## 📦 Removed Components (From Old Design)

### Deleted Pages
- ~~Tournament.tsx~~ - Multi-tournament dashboard
- ~~Rankings.tsx~~ - Global leaderboard
- ~~TeamBuilder.tsx~~ - Complex team builder (simplified to modal)

### Deleted Components
- ~~TournamentRunner.tsx~~ - Batch tournament execution
- ~~TournamentResults.tsx~~ - Multi-game results table
- ~~AgentRadarChart.tsx~~ - Complex agent visualization (maybe keep?)
- ~~AgentStatsDisplay.tsx~~ - Detailed stats panel

### Simplified Components
- **Dashboard** - Was complex router, now simple launcher
- **Settings** - Was extensive config, now basic preferences
- **TeamSelector** - Was complex builder, now presets + simple modal

---

## ✅ Implementation Checklist

### Phase 1: Core Pages
- [ ] Create new simplified Dashboard
- [ ] Add team selector with presets
- [ ] Add scenario selector with descriptions
- [ ] Add "Run Game" button that navigates to replay

### Phase 2: Game Visualization
- [ ] Ensure Town component works with new flow
- [ ] Ensure AgentLayer works with new flow
- [ ] Ensure ReplayControls work
- [ ] Add GameAnalysis component (post-game)

### Phase 3: Analysis Features
- [ ] Calculate individual contributions
- [ ] Generate strategy summaries
- [ ] Identify key moments
- [ ] Add download replay feature

### Phase 4: Settings
- [ ] Implement theme switching
- [ ] Add default preferences
- [ ] Add data management
- [ ] Wire up all settings to app

### Phase 5: Polish
- [ ] Responsive layouts
- [ ] Accessibility audit
- [ ] Performance optimization
- [ ] User testing

---

**Status**: ✅ Design complete, ready for implementation
