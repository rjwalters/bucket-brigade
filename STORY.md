# 🔥 Bucket Brigade: A Multi-Agent Cooperation Challenge

## The Premise

In the quiet town of Riverton, fires are an ever-present threat. The town consists of **10 houses arranged in a circle**, connected by narrow streets that allow flames to spread rapidly between neighbors. Each house can be in one of three states: **Safe** (🏠), **Burning** (🔥), or **Ruined** (💀).

The townsfolk have developed a sophisticated firefighting system, but here's the twist: **not everyone wants to fight fires**. Some are lazy, some are selfish, some are strategic, and some genuinely want to help. The challenge is to coordinate efforts when individual incentives don't always align with the collective good.

## The Game

**Bucket Brigade** is a research platform that simulates this firefighting challenge as a **multi-agent reinforcement learning environment**. Agents must decide whether to **work** (fight fires) or **rest** (save energy) each night, while observing the actions and signals of others.

### Core Mechanics

1. **Fire Dynamics**: Burning houses ignite neighbors with probability β
2. **Agent Actions**: Choose to work on any house or rest
3. **Signaling**: Broadcast intent before acting (honest or deceptive)
4. **Rewards**: Individual costs vs. collective benefits
5. **Termination**: Game ends when all fires out or all houses ruined

### The Cooperation Dilemma

Working costs energy and might not directly benefit you. But if everyone rests, fires spread unchecked and everyone suffers. This creates rich strategic dynamics:

- **Free Riders**: Rest while others work
- **Heroes**: Always work, regardless of cost
- **Strategists**: Work only when necessary
- **Deceivers**: Signal one thing, do another
- **Coordinators**: Work together efficiently

## The Research Challenge

**Bucket Brigade** serves as a **benchmark for multi-agent cooperation research**:

### Agent Learning
- Can reinforcement learning agents discover cooperative strategies?
- How do different reward structures affect emergent behavior?
- Can agents learn to distinguish honest signals from deception?

### Ranking & Evaluation
- How do we fairly compare agents with different strategies?
- Can we rank agents based on their contribution to team performance?
- How does opponent diversity affect learning?

### Human-AI Interaction
- Can humans compete with learned AI agents?
- How do human strategies differ from AI strategies?
- Can we create more intuitive agent behaviors?

## Real-World Relevance

The cooperation challenges in Bucket Brigade mirror real-world problems:

- **Team Coordination**: Emergency response, military operations
- **Resource Allocation**: Disaster relief, supply chain management
- **Trust & Deception**: Social dilemmas, negotiation, security
- **Incentive Design**: Economic systems, governance, policy-making

## Getting Started

### Play in Your Browser
Visit our [interactive tournament](https://rjwalters.github.io/bucket-brigade/) to:
- Write JavaScript agents directly in your browser
- Compete against expert AI strategies
- See real-time rankings and performance metrics

### Research & Development
Clone the repository to:
- Train reinforcement learning agents with PufferLib
- Run large-scale tournaments with diverse agent populations
- Analyze cooperation strategies and ranking methodologies

```bash
git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade

# Install dependencies
npm run install:all

# Run a tournament
npm run dev
```

## The Vision

**Bucket Brigade** aims to be the **go-to platform for multi-agent cooperation research**, providing:

- **Accessible Entry**: Browser-based experimentation for all skill levels
- **Rigorous Research**: Proper evaluation methodologies and benchmarks
- **Open Science**: Reproducible experiments and shared agent strategies
- **Educational Value**: Clear demonstrations of cooperation principles

Whether you're a student learning about multi-agent systems, a researcher developing new algorithms, or just someone interested in the fascinating dynamics of cooperation, **Bucket Brigade** offers an engaging and scientifically valuable challenge.

---

*"In the flickering light of a burning town, we see the eternal dance of cooperation and self-interest. Can we build AI that chooses the bucket brigade over the easy path?"*

🔥🏠⏱️ **Welcome to Bucket Brigade** 🔥🏠⏱️
