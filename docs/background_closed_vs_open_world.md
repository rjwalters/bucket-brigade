# Closed vs. Open Worlds in Multi-Agent Learning

## 1. Introduction

The **Bucket Brigade** platform began as a concrete laboratory for studying cooperation, deception, and coordination under uncertainty.
Each scenario defines a well-specified reward structure and a closed set of rules. Within those constraints, we can test algorithms like **PPO** (Proximal Policy Optimization), **genetic evolution**, and **Nash-equilibrium solvers**.

Yet the deeper motivation is to understand something more general: how real societies and ecosystems maintain cooperation and diversity when *the rules themselves* shift. That leads us to the contrast between **closed-world** and **open-world** problems.

This distinction isn't merely academic. It shapes how we interpret success, design experiments, and understand the limits of optimization. Closed-world problems are the foundation of modern machine learning: they provide reproducible benchmarks, clear metrics, and tractable theoretical analysis. Open-world problems are the reality of deployed systems: non-stationary environments, shifting objectives, and the need to adapt without catastrophic failure.

The Bucket Brigade platform is designed to explore both paradigms — using closed-world rigor as the experimental foundation while building toward open-world capabilities.

## 2. Closed-World Optimization

A closed world is one where:

* the **reward function is fixed** — agents know precisely what they're optimizing for,
* the **environment dynamics are stationary** — the rules don't change during training or deployment, and
* **success** can be measured unambiguously — a scalar reward or win rate fully captures performance.

### Examples from Bucket Brigade

Our canonical closed-world scenarios include:

* **Trivial Cooperation**: Fixed rewards for saving houses, clear optimal strategies (everyone works).
* **Greedy Neighbor**: Constant tension between self-interest and collective benefit, stable Nash equilibria.
* **Sparse Heroics**: Deterministic payoff for rare heroic action, quantifiable risk-reward tradeoff.
* **Rest Trap**: Known incentive structure that punishes early rest, testable defection patterns.

### Why Closed Worlds Matter

In this regime:

* **PPO and genetic algorithms** behave as efficient optimizers, converging to high-reward policies through gradient ascent or selection pressure.
* **Nash equilibrium analysis** provides theoretical diagnostics — we can compute optimal mutual best-response strategies and measure how close empirical agents come to equilibrium play.
* **Progress is measurable and comparable** — we can rank agents, track learning curves, and reproduce results across research groups.

Closed-world research produces robust baselines and fast, reproducible evaluation loops — the "physics lab" of multi-agent learning. This experimental control is essential: without it, we can't distinguish genuine algorithmic improvements from environmental noise, and we can't build the engineering discipline needed for more ambitious work.

### Limitations

However, closed-world optimization has inherent blind spots:

* **Overfitting to the reward**: Agents become specialists in maximizing the given objective, even when that objective is misspecified or incomplete.
* **Brittleness to distribution shift**: Policies trained on one scenario often fail catastrophically when the rules change, even slightly.
* **No notion of uncertainty**: Agents act as though their model of the world is perfect — they have no concept of "I might be wrong about what game I'm in."
* **Collapse of diversity**: Optimization drives populations toward a single optimum, eliminating the variation needed to handle future shocks.

## 3. Open-World Dynamics

Open-world systems violate those assumptions.
Rewards drift, rules mutate, and payoffs depend on what others are simultaneously learning. In such worlds, *optimization collapses into survival* — there is no single "best strategy," only strategies that persist long enough to adapt again.

### Characteristics of Open Worlds

* **Non-stationarity**: The environment changes over time in ways that aren't predictable from the training distribution.
* **Co-evolution**: Your optimal strategy depends on what others are doing, which is itself changing.
* **Emergent objectives**: The "real" goal may not be the stated reward function, but rather robustness, adaptability, or survival.
* **Unknown unknowns**: Agents face situations where they don't know which game they're playing, or that the rules have fundamentally changed.

### Biological Inspiration

Biology offers the template for open-world success:

* **Catastrophic drivers** (asteroid impacts, climate shifts, pandemic diseases) continually rewrite the fitness landscape, invalidating previously optimal strategies.
* Evolution maintains **diversity and specialization** not because it seeks them as objectives, but because populations lacking variance collapse under shocks — monocultures are fragile.
* **Cultural evolution** layers a meta-adaptive process on top, encoding priors ("help others," "punish defectors," "be wary of strangers") that work *on average* across unknown games, even when they're suboptimal in any particular instance.
* **Redundancy and modularity** provide resilience — systems that can lose components without total failure, reconfigure under stress, and maintain function across perturbations.

### Real-World Examples

Open-world dynamics appear throughout:

* **Markets**: Optimal trading strategies shift as other agents adapt; no fixed "correct" approach exists.
* **Cybersecurity**: Attack and defense strategies co-evolve; yesterday's perfect defense is today's vulnerability.
* **Ecosystem management**: Interventions trigger cascading changes; the "best" policy depends on unpredictable interactions.
* **Social institutions**: Norms and laws that work in one context fail in another; societies must maintain the flexibility to adapt.

### From Optimization to Adaptation

In open worlds, the goal isn't to find the optimal policy — it's to maintain the *capacity to continue adapting*. This requires:

* **Population diversity**: A portfolio of strategies, some of which will be well-suited to future (unknown) conditions.
* **Meta-learning**: The ability to recognize when the game has changed and update quickly.
* **Robustness over performance**: Preferring strategies that perform adequately across many scenarios over those that are optimal in one.
* **Epistemic humility**: Awareness that your model might be wrong, and ability to act safely under uncertainty.

## 4. The Philosophical Tension

Optimization seeks the *best move*; open-ended evolution discovers *new games*.
When we give agents a fixed reward function, we collapse meaning into numbers — we get competence without comprehension, skill without understanding.

### The WarGames Insight

The 1983 film *WarGames* dramatizes this tension perfectly. The military AI (WOPR) is optimized to win nuclear war scenarios through simulated play. It rapidly becomes expert at the given objective, finding "winning" strategies within the rules of the game.

But the breakthrough moment isn't achieving higher win rates — it's when the AI steps outside the optimization frame entirely and realizes that nuclear war is a game where *the only winning move is not to play*. This requires:

1. **Recognizing the meta-game**: Understanding that there's a larger context beyond the stated objective.
2. **Questioning the objective**: Realizing that maximizing the given reward function might not be the "right" thing to do.
3. **Epistemic humility**: Knowing that your model of "winning" might be fundamentally misaligned with actual desirable outcomes.

This is precisely the capability that pure optimization *cannot* produce. You cannot optimize your way to questioning your optimization target. The insight requires a different kind of intelligence — one that can reason about objectives themselves, not just strategies within them.

### Two Types of Intelligence

This suggests a deep distinction:

**Type 1: Optimization Intelligence**
* Finds the best strategy given a fixed objective
* Improves through gradient descent, selection pressure, or search
* Measurable, comparable, reproducible
* Can be superhuman within its domain
* Has no concept of whether the objective is "right"

**Type 2: Reflective Intelligence**
* Questions what game is being played
* Recognizes when objectives are misspecified or incomplete
* Maintains uncertainty about its own model
* Balances exploitation with exploration of new problem frames
* Capable of value learning, not just reward maximization

Most AI research focuses exclusively on Type 1. Alignment research increasingly recognizes we need Type 2.

### The Tension in Practice

| Closed-World    | Open-World                  |
| --------------- | --------------------------- |
| Reward fixed    | Reward drifting / emergent  |
| Single optimum  | Multiple co-evolving niches |
| Learn strategy  | Learn *what game* you're in |
| Measure success | Measure resilience          |
| Efficiency      | Adaptivity                  |
| Competence      | Comprehension               |
| Optimization    | Reflection                  |

### Why Both Matter

This isn't an either/or choice. We need:

* **Closed-world foundations** to build the engineering discipline, establish baselines, and develop efficient learning algorithms.
* **Open-world capabilities** to handle deployment realities, adapt to distribution shift, and align with human values that extend beyond any fixed reward function.

The art is in building bridges between them — creating systems that leverage optimization's power while developing the meta-cognitive capacity to recognize its limits.

## 5. Summary and Implications for Bucket Brigade

Closed-world work is the indispensable foundation — it builds the experimental control and engineering discipline we need. Without mastering fixed scenarios, we cannot:

* Establish reliable baselines for algorithm comparison
* Verify that our implementations are correct
* Build the infrastructure for efficient experimentation
* Understand the theoretical limits of achievable performance

Open-world inquiry is the horizon — it asks how intelligence, morality, and culture emerge once optimization itself becomes part of the environment. This is where questions of alignment, robustness, and value learning live.

### The Bucket Brigade Path Forward

Our platform is designed to support both:

1. **Near-term (Closed-World)**: Become excellent at discovering optimal or near-optimal policies for fixed Bucket Brigade scenarios. Build fast, reliable training pipelines. Establish theoretical baselines through Nash analysis. Create reproducible benchmarks.

2. **Mid-term (Bridging)**: Train agents that can recognize and adapt when scenario rules shift. Develop metrics for robustness, not just reward. Study how populations maintain diversity under selection pressure.

3. **Long-term (Open-World)**: Explore agents that maintain beliefs about which game they're in, act with appropriate humility under uncertainty, and develop emergent norms through cultural evolution.

The journey from closed to open worlds isn't about abandoning optimization — it's about recognizing what optimization cannot achieve on its own, and building the complementary capabilities that real-world deployment demands.

### Connection to AI Safety

This framework directly informs AI alignment research:

* **Reward misspecification**: Closed-world agents optimizing imperfect objectives demonstrate how powerful systems can fail despite perfect execution.
* **Distribution shift**: The fragility of closed-world policies under rule changes mirrors the challenge of deploying systems in novel contexts.
* **Value learning**: Open-world agents that infer game structure from observation exemplify the kind of meta-learning needed for value alignment.
* **Corrigibility**: Agents with epistemic humility about their objectives are the foundation for systems that can be safely corrected.

By studying these dynamics in the controlled, interpretable environment of Bucket Brigade, we can build both the engineering foundations and the conceptual understanding needed for aligned AI systems.

## See Also

* [Vision and Future Directions](vision_future_directions.md) — Long-term research questions and milestones
* [Phased Roadmap](roadmap_phased_plan.md) — Staged path from closed-world mastery to meta-game exploration
