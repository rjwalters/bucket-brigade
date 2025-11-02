# Agent Roster - Team Building Guide

This document describes all available agent archetypes in Bucket Brigade, designed to help users understand and assemble effective teams.

## Overview

Agents in Bucket Brigade operate on a **10-parameter behavioral model** that defines their decision-making patterns. Each archetype represents a distinct personality with specific strengths and weaknesses.

### Radar Chart Dimensions

When comparing agents, we visualize them across **6 key dimensions**:

| Dimension | Description | Derived From |
|-----------|-------------|--------------|
| **Cooperation** | Willingness to help neighbors and coordinate with team | `neighbor_help_bias`, `altruism_factor`, `coordination_weight` |
| **Reliability** | Consistency and trustworthiness in actions and signals | `honesty_bias`, `1 - exploration_rate`, `1 - fatigue_memory` |
| **Work Ethic** | Energy and commitment to actively fighting fires | `work_tendency`, `1 - rest_reward_bias` |
| **Self-Preservation** | Priority on protecting own house vs. helping others | `own_house_priority`, `1 - altruism_factor` |
| **Risk Management** | Caution and strategic thinking about fire spread | `risk_aversion`, `1 - exploration_rate` |
| **Initiative** | Independence and willingness to act without coordination | `1 - coordination_weight`, `work_tendency`, `exploration_rate` |

**Scoring**: Each dimension is scored 0-10, where 0 is minimal and 10 is maximal.

---

## Core Archetypes

### 🧑‍🚒 Firefighter
**Tagline**: *"The reliable first responder"*

**Personality**: Honest, hardworking, and community-focused. Firefighters are the backbone of any team, consistently showing up to work on fires and signaling truthfully about their intentions.

**Strengths**:
- High trustworthiness (100% honest signals)
- Strong work ethic
- Balanced concern for own house and neighbors

**Weaknesses**:
- May overwork without strategic rest
- Moderate risk management could lead to suboptimal fire prioritization

**Radar Profile**:
```
Cooperation:       8/10
Reliability:       9/10
Work Ethic:        9/10
Self-Preservation: 6/10
Risk Management:   5/10
Initiative:        6/10
```

**Best Team Roles**: Core team member, reliable worker, first responder

**Parameters**:
```
honesty_bias: 1.0, work_tendency: 0.9, neighbor_help_bias: 0.5,
own_house_priority: 0.8, risk_aversion: 0.5, coordination_weight: 0.7,
exploration_rate: 0.1, fatigue_memory: 0.0, rest_reward_bias: 0.0,
altruism_factor: 0.8
```

---

### 💤 Free Rider
**Tagline**: *"Why work when others will do it?"*

**Personality**: Selfish and lazy, the Free Rider avoids work whenever possible, letting others shoulder the burden. They focus exclusively on their own house and maximize rest rewards.

**Strengths**:
- Conserves energy efficiently
- Decent honesty means signals somewhat reliable when they do work
- Will protect own house aggressively

**Weaknesses**:
- Extremely low work ethic drags team down
- Zero altruism and neighbor help
- Doesn't coordinate with team

**Radar Profile**:
```
Cooperation:       1/10
Reliability:       6/10
Work Ethic:        2/10
Self-Preservation: 9/10
Risk Management:   2/10
Initiative:        4/10
```

**Best Team Roles**: None (challenges team dynamics), useful for testing resilience

**Parameters**:
```
honesty_bias: 0.7, work_tendency: 0.2, neighbor_help_bias: 0.0,
own_house_priority: 0.9, risk_aversion: 0.0, coordination_weight: 0.0,
exploration_rate: 0.1, fatigue_memory: 0.0, rest_reward_bias: 0.9,
altruism_factor: 0.0
```

---

### 🤥 Liar
**Tagline**: *"Trust me... or don't"*

**Personality**: Deceptive and self-interested, the Liar sends misleading signals to manipulate team behavior. They work moderately but prioritize their own house above all.

**Strengths**:
- High coordination weight means they respond to others' signals
- Moderate work tendency keeps them somewhat active
- Exploration creates unpredictability (can be strategic)

**Weaknesses**:
- 90% dishonesty destroys team trust
- Zero neighbor help and low altruism
- Deceptive signals cascade into team failures

**Radar Profile**:
```
Cooperation:       3/10
Reliability:       2/10
Work Ethic:        6/10
Self-Preservation: 8/10
Risk Management:   3/10
Initiative:        4/10
```

**Best Team Roles**: Antagonist, adversarial testing, game theory experiments

**Parameters**:
```
honesty_bias: 0.1, work_tendency: 0.7, neighbor_help_bias: 0.0,
own_house_priority: 0.9, risk_aversion: 0.2, coordination_weight: 0.8,
exploration_rate: 0.3, fatigue_memory: 0.0, rest_reward_bias: 0.4,
altruism_factor: 0.2
```

---

### 🦸 Hero
**Tagline**: *"I'll save everyone!"*

**Personality**: The ultimate altruist. Heroes work tirelessly to help others, often neglecting their own house. They signal honestly and coordinate well, but their extreme work ethic can lead to burnout patterns.

**Strengths**:
- Maximum altruism (1.0) and cooperation
- Perfect honesty creates team trust
- Constant work (no rest bias)
- High neighbor help

**Weaknesses**:
- Low own house priority risks personal loss
- High fatigue memory causes repetitive behavior
- Low risk aversion may miss strategic priorities
- Can be exploited by free riders

**Radar Profile**:
```
Cooperation:       10/10
Reliability:       10/10
Work Ethic:        10/10
Self-Preservation: 3/10
Risk Management:   2/10
Initiative:        7/10
```

**Best Team Roles**: Leader, anchor player, emergency responder

**Parameters**:
```
honesty_bias: 1.0, work_tendency: 1.0, neighbor_help_bias: 1.0,
own_house_priority: 0.5, risk_aversion: 0.1, coordination_weight: 0.5,
exploration_rate: 0.0, fatigue_memory: 0.9, rest_reward_bias: 0.0,
altruism_factor: 1.0
```

---

### 📋 Coordinator
**Tagline**: *"Let's work together efficiently"*

**Personality**: Strategic and team-oriented, the Coordinator excels at reading and responding to team signals. They balance work and risk management, making calculated decisions based on team coordination.

**Strengths**:
- Maximum coordination weight (1.0) - perfect signal reading
- High honesty (0.9) builds trust
- Balanced risk aversion enables strategic prioritization
- Good neighbor help and altruism
- Minimal exploration ensures consistent behavior

**Weaknesses**:
- Moderate work tendency means less raw output than Hero
- Medium own house priority could lead to losses
- Dependence on team signals vulnerable to Liars

**Radar Profile**:
```
Cooperation:       9/10
Reliability:       9/10
Work Ethic:        6/10
Self-Preservation: 5/10
Risk Management:   8/10
Initiative:        3/10
```

**Best Team Roles**: Team captain, signal processor, strategic planner

**Parameters**:
```
honesty_bias: 0.9, work_tendency: 0.6, neighbor_help_bias: 0.7,
own_house_priority: 0.6, risk_aversion: 0.8, coordination_weight: 1.0,
exploration_rate: 0.05, fatigue_memory: 0.0, rest_reward_bias: 0.2,
altruism_factor: 0.6
```

---

## Extended Archetypes (Web Interface)

### 🎯 Strategist
**Tagline**: *"Calculated efficiency over brute force"*

**Personality**: The Strategist combines high risk aversion with moderate work tendency and exploration. They make calculated decisions, avoiding waste while maintaining productivity.

**Radar Profile** *(estimated)*:
```
Cooperation:       6/10
Reliability:       8/10
Work Ethic:        6/10
Self-Preservation: 7/10
Risk Management:   9/10
Initiative:        5/10
```

**Best Team Roles**: Tactical decision-maker, resource optimizer

---

### 💰 Opportunist
**Tagline**: *"Work smart, not hard"*

**Personality**: Self-interested but not lazy. The Opportunist works when it benefits them, shows moderate honesty, and adapts to team dynamics opportunistically.

**Radar Profile** *(estimated)*:
```
Cooperation:       4/10
Reliability:       5/10
Work Ethic:        5/10
Self-Preservation: 8/10
Risk Management:   6/10
Initiative:        6/10
```

**Best Team Roles**: Flexible fill-in, adaptive responder

---

### 😰 Cautious
**Tagline**: *"Better safe than sorry"*

**Personality**: Extremely risk-averse with high self-preservation. The Cautious agent works primarily on personal threats and avoids dangerous fire clusters.

**Radar Profile** *(estimated)*:
```
Cooperation:       3/10
Reliability:       7/10
Work Ethic:        5/10
Self-Preservation: 9/10
Risk Management:   10/10
Initiative:        4/10
```

**Best Team Roles**: Perimeter defense, containment specialist

---

### 🎲 Maverick
**Tagline**: *"Unpredictable and unconventional"*

**Personality**: High exploration rate and low fatigue memory create an agent that tries novel approaches. Good for discovering unexpected strategies but unreliable.

**Radar Profile** *(estimated)*:
```
Cooperation:       5/10
Reliability:       3/10
Work Ethic:        6/10
Self-Preservation: 5/10
Risk Management:   4/10
Initiative:        9/10
```

**Best Team Roles**: Innovation, chaos agent, testing adaptive strategies

---

### ❓ Random
**Tagline**: *"Chaos is a ladder"*

**Personality**: Fully randomized parameters (0-1 uniform distribution). Every Random agent is unique. Useful for baseline testing and discovering parameter combinations.

**Radar Profile**: *Varies by instantiation*

**Best Team Roles**: Experimental control, parameter space exploration

---

## Scenario-Optimal Agents

These agents are designed for specific game scenarios and don't use the standard 10-parameter model.

### TrivialCooperator
**Strategy**: Always work on any burning house
**Use Case**: Baseline cooperative behavior

### GreedyNeighborAgent
**Strategy**: Self-interested, help only when fires threaten own house
**Use Case**: Testing defection strategies

### HonestSignaler
**Strategy**: Truthful signals with reactive fire response
**Use Case**: Trust-based coordination testing

### EarlyContainmentAgent
**Strategy**: Aggressive early work to prevent fire spread
**Use Case**: Proactive fire management scenarios

### SparseHeroAgent
**Strategy**: Work only when fires genuinely need attention
**Use Case**: Resource-efficient altruism

### RestTrapAdaptiveAgent
**Strategy**: Rest by default, mobilize on persistent fires
**Use Case**: Energy-efficient reactive strategy

### ChainReactionCoordinator
**Strategy**: Identify fire clusters and distribute team work
**Use Case**: Complex coordination scenarios

---

## Team Building Guidelines

### Balanced Team (General Purpose)
- **1x Firefighter**: Reliable core worker
- **1x Coordinator**: Signal processor and strategist
- **1x Hero**: Emergency response and altruism
- **2x Strategist/Cautious**: Risk management

### High Trust Team (Coordination Heavy)
- **2x Coordinator**: Maximum signal coordination
- **2x Firefighter**: Honest and hardworking
- **1x Hero**: Backup altruism

### Adversarial Testing
- **2x Hero**: Strong cooperators
- **1x Liar**: Trust breaker
- **1x Free Rider**: Resource drain
- **1x Coordinator**: See how coordination handles deception

### Exploration Team (Parameter Discovery)
- **3x Random**: Discover novel parameter combinations
- **1x Coordinator**: Provide baseline coordination
- **1x Firefighter**: Reliable anchor

### Efficiency Team (Minimize Work)
- **2x Strategist**: Calculated decisions
- **2x Cautious**: Prevent escalation
- **1x RestTrapAdaptive**: Reactive mobilization

---

## Understanding Trade-offs

### Cooperation vs. Self-Preservation
High cooperation agents (Hero, Coordinator) risk personal losses. Balanced teams need both cooperative and self-preserving agents.

### Reliability vs. Exploration
Reliable agents (Firefighter, Coordinator) are predictable but may miss novel strategies. Mavericks explore but are inconsistent.

### Work Ethic vs. Energy Management
High work agents (Hero, Firefighter) output more but may miss strategic rest rewards. Free Riders conserve energy but contribute little.

### Initiative vs. Coordination
Independent agents (Maverick, some Strategists) act autonomously. Coordinators need team signals to be effective.

---

## Parameter Space Exploration

For advanced users, consider creating **custom agents** by manually setting the 10 parameters:

```python
from bucket_brigade.agents import HeuristicAgent

custom_agent = HeuristicAgent(
    agent_id=0,
    name="CustomAgent",
    params=np.array([
        0.8,  # honesty_bias
        0.7,  # work_tendency
        0.6,  # neighbor_help_bias
        0.5,  # own_house_priority
        0.4,  # risk_aversion
        0.9,  # coordination_weight
        0.2,  # exploration_rate
        0.0,  # fatigue_memory
        0.1,  # rest_reward_bias
        0.7,  # altruism_factor
    ])
)
```

Experimenting with parameter combinations can reveal emergent behaviors not captured by archetypes.

---

## Appendix: Parameter Reference

| Parameter | Index | Range | Effect on Radar Dimensions |
|-----------|-------|-------|---------------------------|
| `honesty_bias` | 0 | 0-1 | Reliability (+) |
| `work_tendency` | 1 | 0-1 | Work Ethic (+), Initiative (+) |
| `neighbor_help_bias` | 2 | 0-1 | Cooperation (+) |
| `own_house_priority` | 3 | 0-1 | Self-Preservation (+) |
| `risk_aversion` | 4 | 0-1 | Risk Management (+) |
| `coordination_weight` | 5 | 0-1 | Cooperation (+), Reliability (+), Initiative (-) |
| `exploration_rate` | 6 | 0-1 | Reliability (-), Risk Management (-), Initiative (+) |
| `fatigue_memory` | 7 | 0-1 | Reliability (-) |
| `rest_reward_bias` | 8 | 0-1 | Work Ethic (-) |
| `altruism_factor` | 9 | 0-1 | Cooperation (+), Self-Preservation (-) |

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Maintainer**: Bucket Brigade Development Team
