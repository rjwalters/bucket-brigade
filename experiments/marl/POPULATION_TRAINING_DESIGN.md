# Population-Based Multi-Agent Training Architecture

**Goal**: Efficiently train multiple diverse agents simultaneously for the Bucket Brigade game using heterogeneous CPU/GPU compute.

## Core Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CPU: Game Simulator                       │
│  - Single process running Rust-based BucketBrigade envs     │
│  - Manages N parallel games with K agents each              │
│  - Matchmaking: assigns agents to games                     │
│  - Collects trajectories (s, a, r, s', done)                │
│  - Distributes experiences to GPU learners                  │
│  - Maintains shared experience buffer (future)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Experience Queue (multiprocessing)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼────────┐  ┌──────▼────────┐
│ GPU Learner 0  │  │ GPU Learner 1 │  │ GPU Learner N │
│                │  │               │  │               │
│ - Policy Net   │  │ - Policy Net  │  │ - Policy Net  │
│ - PPO Updates  │  │ - PPO Updates │  │ - PPO Updates │
│ - Own exp only │  │ - Own exp only│  │ - Own exp    │
│   (on-policy)  │  │   (on-policy) │  │   (on-policy)│
│                │  │               │  │               │
│ Future:        │  │ Future:       │  │ Future:      │
│ - Sample from  │  │ - Sample from │  │ - Sample from│
│   shared buffer│  │   shared buffer│ │   shared buf │
│   (off-policy) │  │   (off-policy)│  │   (off-policy│
└────────────────┘  └───────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                      Updated Policies
                            │
                            ▼
                    CPU: Policy Repository
```

## Key Design Principles

### 1. **Separation of Concerns**

**CPU Process (Simulator)**:
- **Role**: Environment simulation only
- **Why**: Rust environments are fast, but still CPU-bound
- **Responsibilities**:
  - Run BucketBrigade games with 4 agents each
  - Sample agent policies for each game from population
  - Collect trajectories (state, action, reward, next_state, done)
  - Route experiences to correct GPU learners
  - Coordinate matchmaking (which agents play together)

**GPU Processes (Learners)**:
- **Role**: Policy training only
- **Why**: Neural network training is GPU-bound
- **Responsibilities**:
  - Receive experience batches from CPU
  - Compute policy gradients (forward + backward pass)
  - Update policy parameters (PPO)
  - Send updated policy weights back to CPU
  - Maintain own replay buffer for on-policy learning

### 2. **Communication Pattern**

```
CPU → GPU: Experience tuples (s, a, r, s', done, agent_id)
GPU → CPU: Updated policy weights (periodically)
```

**Design Choices**:
- **Asynchronous**: GPU learners don't block CPU simulator
- **Batched**: CPU sends mini-batches to reduce communication overhead
- **Multiprocessing**: Python multiprocessing.Queue for IPC
- **Shared Memory**: Optional torch.multiprocessing for zero-copy tensor sharing

### 3. **Learning Modes**

#### On-Policy Learning (Phase 1)
Each agent learns exclusively from its own experience:
- **Pro**: Stable, proven (standard PPO)
- **Pro**: No distribution mismatch
- **Con**: Sample inefficient (each agent needs separate data)

```python
# GPU Learner i
for batch in experience_queue[i]:
    if batch.agent_id == i:  # Only own experience
        loss = compute_ppo_loss(batch)
        optimizer.step()
```

#### Off-Policy Learning (Phase 2 - Future)
Agents can learn from other agents' experiences:
- **Pro**: More sample efficient (reuse data)
- **Pro**: Agents learn from diverse behaviors
- **Con**: Distribution mismatch (other agent's policy ≠ own policy)
- **Con**: Requires importance sampling correction

```python
# GPU Learner i
for batch in shared_experience_buffer.sample():
    # Batch contains experiences from any agent
    if batch.agent_id == i:
        # On-policy: use directly
        loss = compute_ppo_loss(batch)
    else:
        # Off-policy: importance sampling
        loss = compute_off_policy_loss(batch, importance_weights)
    optimizer.step()
```

**Importance Sampling**:
```
w(s,a) = π_i(a|s) / π_j(a|s)  # Agent i learning from agent j's data
```

### 4. **Matchmaking Strategy**

**Goal**: Ensure agents play against diverse opponents for robust learning

**Phase 1: Round-Robin**
```python
def create_matches(population, num_games):
    """Each agent plays each other agent equally"""
    for game_id in range(num_games):
        agents = random.sample(population, k=4)
        yield Match(agents=agents, game_id=game_id)
```

**Phase 2: Fitness-Based**
```python
def create_matches(population, num_games):
    """Match agents with similar skill levels"""
    ranked = sorted(population, key=lambda a: a.fitness)
    for game_id in range(num_games):
        # Sample from nearby ranks
        base_rank = random.randint(0, len(ranked) - 4)
        agents = ranked[base_rank:base_rank+4]
        yield Match(agents=agents, game_id=game_id)
```

**Phase 3: Strategic**
```python
def create_matches(population, num_games):
    """Maximize learning signal"""
    for game_id in range(num_games):
        # Mix skill levels deliberately
        best = population.top_k(1)
        worst = population.bottom_k(1)
        middle = population.sample(2)
        yield Match(agents=[best, worst, *middle], game_id=game_id)
```

## Implementation Details

### CPU Process (Simulator)

```python
class GameSimulator:
    def __init__(self, num_games=64, population_size=8):
        # Rust environments for speed
        self.envs = [
            BucketBrigade(scenario, num_agents=4, seed=i)
            for i in range(num_games)
        ]

        # Policy repository (CPU copies)
        self.population = [
            load_policy(f"agent_{i}.pt")
            for i in range(population_size)
        ]

        # Communication queues
        self.experience_queues = [
            multiprocessing.Queue()
            for _ in range(population_size)
        ]

        self.policy_update_queue = multiprocessing.Queue()

    def run_episode(self, env_id):
        """Run one episode, collect experiences, send to learners"""
        env = self.envs[env_id]

        # Matchmaking: assign 4 agents from population
        agent_ids = self.matchmaker.sample_agents(k=4)
        policies = [self.population[i] for i in agent_ids]

        # Run episode
        obs = env.reset()
        done = False
        trajectory = []

        while not done:
            # Each agent selects action with its policy
            actions = []
            for i, agent_id in enumerate(agent_ids):
                with torch.no_grad():
                    action = policies[i](obs[i])
                actions.append(action)

            # Step environment
            next_obs, rewards, done, info = env.step(actions)

            # Record experiences for each agent
            for i, agent_id in enumerate(agent_ids):
                experience = (obs[i], actions[i], rewards[i], next_obs[i], done)
                trajectory.append((agent_id, experience))

            obs = next_obs

        # Send experiences to respective GPU learners
        for agent_id, experience in trajectory:
            self.experience_queues[agent_id].put(experience)

    def update_policies(self):
        """Receive updated policies from GPU learners"""
        while not self.policy_update_queue.empty():
            agent_id, new_weights = self.policy_update_queue.get()
            self.population[agent_id].load_state_dict(new_weights)
```

### GPU Process (Learner)

```python
class PolicyLearner:
    def __init__(self, agent_id, experience_queue, policy_update_queue, device):
        self.agent_id = agent_id
        self.experience_queue = experience_queue
        self.policy_update_queue = policy_update_queue
        self.device = device

        # Policy network
        self.policy = PolicyNetwork(...).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters())

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2

    def train(self):
        """Main training loop"""
        batch = []

        while True:
            # Collect experience batch
            while len(batch) < self.batch_size:
                if not self.experience_queue.empty():
                    experience = self.experience_queue.get()
                    batch.append(experience)

            # Train on batch
            loss = self.compute_ppo_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch.clear()

            # Periodically send updated weights to simulator
            if self.steps % self.update_interval == 0:
                weights = self.policy.state_dict()
                self.policy_update_queue.put((self.agent_id, weights))

    def compute_ppo_loss(self, batch):
        """Standard PPO loss computation"""
        states = torch.stack([exp[0] for exp in batch]).to(self.device)
        actions = torch.stack([exp[1] for exp in batch]).to(self.device)
        returns = self.compute_returns(batch).to(self.device)

        # ... PPO loss calculation ...
        return loss
```

## Advantages

### 1. **Natural Multi-Agent Training**
- Agents play against each other (self-play)
- Emergent cooperation and competition
- Population diversity → richer strategies
- Direct path to Nash equilibrium analysis

### 2. **Efficient Compute Utilization**
- **CPU**: Rust envs keep simulation fast (100x faster than Python)
- **GPU**: Batched training across multiple agents
- **Parallelism**: Simulation and learning happen concurrently
- **Scalability**: Add more GPU learners without changing CPU

### 3. **Flexible Learning**
- **Phase 1**: On-policy (simple, stable)
- **Phase 2**: Add off-policy (sample efficient)
- **Phase 3**: Add population dynamics (selection, mutation)

### 4. **Research Alignment**
- Population = mixed strategy in game theory
- Can compute Nash equilibria over population
- Connects RL training to equilibrium analysis
- Natural extension of existing evolution experiments

## Challenges & Solutions

### Challenge 1: Communication Overhead
**Problem**: Sending experiences CPU → GPU can be slow

**Solutions**:
- Batch experiences (send 256 at a time, not 1)
- Use shared memory (torch.multiprocessing)
- Compress states (only send deltas)
- Prioritize experiences (send high-value first)

### Challenge 2: Policy Staleness
**Problem**: CPU uses old policy while GPU trains new one

**Solutions**:
- Frequent policy updates (every N steps)
- Asynchronous is OK for PPO (robust to staleness)
- Track policy version, discard very old experiences
- Use importance sampling for distribution mismatch

### Challenge 3: Load Balancing
**Problem**: Some agents may learn faster than others

**Solutions**:
- Monitor queue depths, throttle fast learners
- Adaptive batch sizes per agent
- Prioritize slow learners in matchmaking
- Dynamic GPU allocation

### Challenge 4: Exploration
**Problem**: Agents may converge to similar strategies

**Solutions**:
- Diversity bonus in loss function
- Randomize initial weights
- Periodic population resets
- Novelty search objectives

## Phased Implementation

### Phase 1: Single-Policy Baseline (DONE)
- ✅ One agent, one GPU, Python envs
- ✅ Achieved 97% GPU utilization
- ✅ Verified PPO training works

### Phase 2: Population Training (ON-POLICY)
**Goal**: Train 8 agents simultaneously with on-policy learning

**Components**:
```
1. CPU Simulator
   - Rust BucketBrigade environments
   - Round-robin matchmaking
   - Experience distribution

2. GPU Learners (8 processes)
   - Each trains one policy
   - On-policy PPO only
   - Periodic weight updates

3. Coordinator
   - Spawns processes
   - Monitors fitness
   - Logs metrics
```

**Deliverables**:
- `experiments/marl/train_population.py` - Main training script
- `bucket_brigade/training/population_trainer.py` - Population trainer class
- `bucket_brigade/training/game_simulator.py` - CPU simulator
- `bucket_brigade/training/policy_learner.py` - GPU learner

**Metrics**:
- Individual agent fitness (avg reward)
- Population diversity (policy distance)
- GPU utilization per learner
- Training throughput (steps/sec)

### Phase 3: Off-Policy Learning
**Goal**: Enable cross-agent learning from shared buffer

**New Components**:
```
1. Shared Experience Buffer
   - Ring buffer on CPU
   - All agents' experiences
   - Importance weights

2. Modified GPU Learners
   - Sample from shared buffer
   - Compute importance weights
   - Off-policy loss correction
```

**Deliverables**:
- `bucket_brigade/training/experience_buffer.py` - Shared buffer
- Updated `policy_learner.py` with off-policy support
- Hyperparameter tuning for IS weights

### Phase 4: Population Dynamics
**Goal**: Add evolution-inspired mechanisms

**New Components**:
```
1. Selection
   - Periodically remove worst agents
   - Clone best agents

2. Mutation
   - Add noise to weights
   - Randomize some layers

3. Crossover
   - Combine two parent policies
```

## Integration with Existing Research

### Nash Equilibrium Analysis
```python
# After population training
population = load_population("population_v1.pt")

# Compute Nash equilibrium over population
nash_strategy = compute_nash_equilibrium(
    population=population,
    scenario="trivial_cooperation",
    simulations=10000
)

# Nash strategy is a probability distribution over agents
print(f"Best agent mix: {nash_strategy}")
# e.g., [0.3, 0.5, 0.1, 0.1, ...]
```

### Evolution Experiments
```python
# Population training = supervised evolution
# - Evolution: fitness-based selection, random mutations
# - Population training: gradient-based selection, learned mutations

# Can combine both:
# - Use evolution to initialize diverse population
# - Use RL to refine policies
```

## Conclusion

This architecture:
1. ✅ Efficiently uses heterogeneous compute (CPU + GPU)
2. ✅ Naturally fits multi-agent games
3. ✅ Scales to larger populations
4. ✅ Enables both on-policy and off-policy learning
5. ✅ Connects RL training to game theory analysis
6. ✅ Extends existing research infrastructure

**Next Steps**: Implement Phase 2 (on-policy population training)
