# 🔌 Bucket Brigade API Reference

This document describes the data structures and interfaces used in the Bucket Brigade platform.

## 🏗️ Architecture Overview

**Current Implementation**: Client-side only (browser-based)
- No backend API server required
- All computation happens in the browser
- Data stored in browser session storage
- Future: Optional Python API server for advanced features

> ⚠️ **Important**: The backend API endpoints described later in this document are planned for future implementation but do not currently exist. The current version is browser-based only with no server component. See the roadmap in README.md for planned backend features.

## 📊 Core Data Structures

### House States

```typescript
type HouseState = 0 | 1 | 2;
// 0 = SAFE    - House is undamaged
// 1 = BURNING - House is on fire
// 2 = RUINED  - House is destroyed
```

### Scenario Configuration

```typescript
interface Scenario {
  prob_fire_spreads_to_neighbor: number;      // Fire spread probability (0-1)
  prob_solo_agent_extinguishes_fire: number;  // Probability one agent extinguishes fire (0-1)
  prob_house_catches_fire: number;            // Probability of spontaneous ignition (0-1)
  team_reward_house_survives: number;         // Team reward per saved house
  team_penalty_house_burns: number;           // Team penalty per ruined house
  cost_to_work_one_night: number;             // Cost per worker per night
  min_nights: number;                         // Minimum nights before termination
  num_agents: number;                         // Number of agents (4-10)
  reward_own_house_survives: number;          // Individual reward when own house survives
  reward_other_house_survives: number;        // Individual reward when neighbor survives
  penalty_own_house_burns: number;            // Individual penalty when own house burns
  penalty_other_house_burns: number;          // Individual penalty when other house burns
}
```

### Game Night Data

```typescript
interface GameNight {
  night: number;              // Night number (0-based)
  houses: HouseState[];       // Array of 10 house states
  signals: number[];          // Agent signals: 0=REST, 1=WORK
  locations: number[];        // Agent positions: 0-9 (house indices)
  actions: number[][];        // Agent actions: [[house_index, mode_flag], ...]
  rewards: number[];          // Agent rewards for this night
}
```

### Complete Game Replay

```typescript
interface GameReplay {
  scenario: Scenario;
  nights: GameNight[];
}
```

### Agent Interface

```typescript
interface Agent {
  id: number;
  name: string;
  act: (observation: AgentObservation) => number[];
  reset?: () => void;
}
```

### Agent Observation

```typescript
interface AgentObservation {
  signals: number[];          // Other agents' signals
  locations: number[];        // Other agents' locations
  houses: HouseState[];       // Current house states
  last_actions: number[][];   // Previous night actions
  scenario_info: number[];    // Scenario parameters as array
  agent_id: number;           // This agent's ID
  night: number;              // Current night number
}
```

### Agent Action Format

```typescript
// Return value from agent.act()
type AgentAction = [number, number];
// [house_index, mode_flag]
// house_index: 0-9 (which house to target)
// mode_flag: 0=REST, 1=WORK
```

## 🏃‍♂️ Game Execution Flow

### 1. Initialization
```typescript
const game = new BucketBrigadeEngine(scenario);
const agents = [agent1, agent2, ...]; // Array of Agent objects
```

### 2. Game Loop
```typescript
while (!game.isDone()) {
  // Get observations for all agents
  const observations = game.getObservations();

  // Each agent chooses action based on observation
  const actions = agents.map((agent, i) =>
    agent.act(observations[i])
  );

  // Execute one night of gameplay
  game.step(actions);
}
```

### 3. Results
```typescript
const result: GameResult = {
  scenario: game.scenario,
  nights: game.nights,
  final_score: game.getTeamScore(),
  agent_scores: game.getAgentScores()
};
```

## 💾 Data Storage

### Session Storage Keys

```typescript
const STORAGE_KEYS = {
  GAME_REPLAYS: 'bucket_brigade_replays',
  BATCH_RESULTS: 'bucket_brigade_results',
  AGENT_RANKINGS: 'bucket_brigade_rankings',
  UI_SETTINGS: 'bucket_brigade_settings'
} as const;
```

### Storage Format

```typescript
// Game replays
type StoredReplays = GameReplay[];

// Batch results (multiple games)
interface BatchResult {
  id: string;
  timestamp: number;
  scenario: Scenario;
  games: GameResult[];
  summary: {
    average_score: number;
    best_score: number;
    total_games: number;
  };
}

// Agent rankings
interface AgentRanking {
  agent_id: string;
  name: string;
  average_score: number;
  total_games: number;
  win_rate: number;
  last_updated: number;
}
```

## 🔌 Future API Endpoints (Planned)

When a Python backend API is implemented, these endpoints will be available:

### Game Management

```
POST   /api/games
GET    /api/games/:id
DELETE /api/games/:id
```

### Batch Execution

```
POST   /api/batches
GET    /api/batches/:id
GET    /api/batches/:id/status
DELETE /api/batches/:id
```

### Agent Management

```
POST   /api/agents
GET    /api/agents
GET    /api/agents/:id
PUT    /api/agents/:id
DELETE /api/agents/:id
```

### Rankings & Analytics

```
GET    /api/rankings
GET    /api/rankings/:agent_id
GET    /api/analytics/scenarios
GET    /api/analytics/performance
```

### Request/Response Examples

#### Run Single Game

```http
POST /api/games
Content-Type: application/json

{
  "scenario": {
    "prob_fire_spreads_to_neighbor": 0.25,
    "prob_solo_agent_extinguishes_fire": 0.45,
    "prob_house_catches_fire": 0.01,
    "team_reward_house_survives": 100,
    "team_penalty_house_burns": 100,
    "cost_to_work_one_night": 0.5,
    "min_nights": 12,
    "num_agents": 6,
    "reward_own_house_survives": 100.0,
    "reward_other_house_survives": 50.0,
    "penalty_own_house_burns": 100.0,
    "penalty_other_house_burns": 50.0
  },
  "agents": [
    {"id": 1, "name": "Firefighter", "params": [1.0, 0.9, 0.8, 0.7, 0.5, 0.6, 0.1, 0.5, 0.4, 0.8]},
    {"id": 2, "name": "Coordinator", "params": [0.9, 0.6, 0.5, 0.6, 0.8, 1.0, 0.05, 0.7, 0.3, 0.6]}
  ]
}
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "game_id": "game_12345",
  "status": "completed",
  "scenario": { ... },
  "result": {
    "final_score": 241.6,
    "agent_scores": [32.5, 28.0, 45.2, 38.1, 29.8, 41.2],
    "nights_played": 18,
    "houses_saved": 7,
    "houses_ruined": 3
  },
  "replay_url": "/api/games/game_12345/replay"
}
```

#### Get Rankings

```http
GET /api/rankings?limit=10&sort_by=average_score
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "rankings": [
    {
      "agent_id": "agent_001",
      "name": "Firefighter",
      "average_score": 42.3,
      "total_games": 150,
      "win_rate": 0.67,
      "last_updated": 1640995200
    },
    {
      "agent_id": "agent_002",
      "name": "Coordinator",
      "average_score": 38.7,
      "total_games": 120,
      "win_rate": 0.59,
      "last_updated": 1640995100
    }
  ],
  "metadata": {
    "total_agents": 25,
    "last_updated": 1640995200,
    "scenarios_tested": ["trivial_cooperation", "early_containment", "greedy_neighbor"]
  }
}
```

## 🔒 Security Considerations

### Client-Side Execution
- All agent code runs in browser sandbox
- No server-side execution of untrusted code
- Input validation using Zod schemas
- Resource limits prevent infinite loops

### Future API Security
- Rate limiting on all endpoints
- Input sanitization and validation
- CORS configuration for web clients
- Optional authentication for advanced features

## 📈 Performance Characteristics

### Current Browser Implementation

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Single game (50 nights) | ~50ms | Depends on agent complexity |
| Tournament (100 games) | ~5s | Sequential execution |
| Large batch (1000 games) | ~50s | May need worker threads |

### Future Server Implementation

| Operation | Performance | Scaling |
|-----------|-------------|---------|
| Single game | ~5ms | Single CPU core |
| Tournament (100 games) | ~0.5s | Parallel execution |
| Large batch (1000 games) | ~5s | Multi-core scaling |

## 🧪 Testing APIs

### Unit Tests

```bash
# Test data validation
pnpm run test:schemas

# Test game engine
pnpm run test:engine

# Test agent interfaces
pnpm run test:agents
```

### Integration Tests

```bash
# Test complete game flow
pnpm run test:integration

# Test storage operations
pnpm run test:storage

# Test UI interactions
pnpm run test:e2e
```

## 📝 Error Handling

### Error Response Format

```typescript
interface ApiError {
  error: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: number;
  request_id: string;
}
```

### Common Error Codes

```typescript
enum ErrorCode {
  INVALID_SCENARIO = 'INVALID_SCENARIO',
  INVALID_AGENT = 'INVALID_AGENT',
  GAME_TIMEOUT = 'GAME_TIMEOUT',
  STORAGE_ERROR = 'STORAGE_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR'
}
```

---

For implementation details, see the [CLASS_DESIGN.md](docs/development/CLASS_DESIGN.md) document.
For deployment information, see [DEPLOYMENT.md](DEPLOYMENT.md).
