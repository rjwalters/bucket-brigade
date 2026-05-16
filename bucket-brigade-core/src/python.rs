use crate::agents::HeuristicAgent;
use crate::{AgentObservation, BucketBrigade, Scenario};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Python-compatible Bucket Brigade environment
#[pyclass]
pub struct PyBucketBrigade {
    inner: BucketBrigade,
}

#[pymethods]
impl PyBucketBrigade {
    #[new]
    #[pyo3(signature = (scenario, num_agents, seed=None))]
    fn new(scenario: PyScenario, num_agents: usize, seed: Option<u64>) -> Self {
        Self {
            inner: BucketBrigade::new(scenario.inner, num_agents, seed),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn step(&mut self, py: Python, actions: Vec<Vec<u8>>) -> PyResult<(PyObject, bool, PyObject)> {
        // Issue #235: action is [house, mode, signal] (length 3). For
        // backward compatibility we accept length-2 inputs and default the
        // signal to the mode bit (honest signaling) so existing Python
        // callers that haven't been migrated yet keep working. Length-3
        // inputs pass through unchanged.
        let rust_actions: Vec<[u8; 3]> = actions
            .iter()
            .map(|a| match a.len() {
                2 => [a[0], a[1], a[1]],
                3 => [a[0], a[1], a[2]],
                _ => [a[0], a.get(1).copied().unwrap_or(0), a.get(2).copied().unwrap_or(0)],
            })
            .collect();

        let result = self.inner.step(&rust_actions);

        let rewards = result.rewards.clone().into_py(py);
        let info = result.info.to_string().into_py(py);

        Ok((rewards, result.done, info))
    }

    fn get_observation(&self, agent_id: usize) -> PyAgentObservation {
        let obs = self.inner.get_observation(agent_id);
        PyAgentObservation { inner: obs }
    }

    fn get_current_state(&self) -> PyGameState {
        let state = self.inner.get_current_state();
        PyGameState { inner: state }
    }

    fn get_result(&self) -> PyGameResult {
        let result = self.inner.get_result();
        PyGameResult { inner: result }
    }

    #[getter]
    fn scenario(&self) -> PyScenario {
        PyScenario {
            inner: self.inner.scenario.clone(),
        }
    }
}

/// Python-compatible Scenario
#[pyclass]
#[derive(Clone)]
pub struct PyScenario {
    inner: Scenario,
}

#[pymethods]
impl PyScenario {
    /// Build a Scenario from Python.
    ///
    /// The four ownership reward fields are per-agent vectors of length
    /// ``num_agents`` (issue #198). For backward compatibility we accept
    /// either a Python ``float`` (auto-promoted to ``vec![v; 10]``) or a
    /// ``list[float]`` (passed through). The default vector length when a
    /// scalar is provided matches the Rust `MAX_AGENTS` const (10).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        prob_fire_spreads_to_neighbor,
        prob_solo_agent_extinguishes_fire,
        prob_house_catches_fire,
        team_reward_house_survives,
        team_penalty_house_burns,
        cost_to_work_one_night,
        min_nights,
        reward_own_house_survives,
        reward_other_house_survives,
        penalty_own_house_burns,
        penalty_other_house_burns,
    ))]
    fn new(
        prob_fire_spreads_to_neighbor: f32,
        prob_solo_agent_extinguishes_fire: f32,
        prob_house_catches_fire: f32,
        team_reward_house_survives: f32,
        team_penalty_house_burns: f32,
        cost_to_work_one_night: f32,
        min_nights: u32,
        reward_own_house_survives: PyObject,
        reward_other_house_survives: PyObject,
        penalty_own_house_burns: PyObject,
        penalty_other_house_burns: PyObject,
        py: Python,
    ) -> PyResult<Self> {
        // Scalar -> vec![v; 10] auto-promotion (mirrors the Python
        // ``Scenario.__post_init__`` and the Rust serde deserializer).
        const DEFAULT_LEN: usize = 10;
        fn to_vec(py: Python, obj: PyObject) -> PyResult<Vec<f32>> {
            if let Ok(scalar) = obj.extract::<f32>(py) {
                return Ok(vec![scalar; DEFAULT_LEN]);
            }
            obj.extract::<Vec<f32>>(py)
        }

        let inner = Scenario {
            prob_fire_spreads_to_neighbor,
            prob_solo_agent_extinguishes_fire,
            prob_house_catches_fire,
            team_reward_house_survives,
            team_penalty_house_burns,
            cost_to_work_one_night,
            min_nights,
            reward_own_house_survives: to_vec(py, reward_own_house_survives)?,
            reward_other_house_survives: to_vec(py, reward_other_house_survives)?,
            penalty_own_house_burns: to_vec(py, penalty_own_house_burns)?,
            penalty_other_house_burns: to_vec(py, penalty_other_house_burns)?,
            // Issue #203 spatial-cost fields. The PyScenario constructor
            // doesn't yet accept these (kwargs would break backward
            // compat); to use them from Python pull the scenario out of
            // ``bucket_brigade_core.SCENARIOS["positional_default"]`` (the
            // SCENARIOS dict preserves all fields). Manually-constructed
            // PyScenarios default to the pre-#203 behavior so existing
            // callers are unchanged.
            agent_home_positions: Vec::new(),
            distance_cost_alpha: 0.0,
            distance_metric: "ring_arc".to_string(),
        };
        // Issue #222: route programmatic construction through the allowlist
        // validator. The literal above is safe today but the helper keeps the
        // chokepoint singular for future kwargs additions.
        inner
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn prob_fire_spreads_to_neighbor(&self) -> f32 {
        self.inner.prob_fire_spreads_to_neighbor
    }

    #[getter]
    fn prob_solo_agent_extinguishes_fire(&self) -> f32 {
        self.inner.prob_solo_agent_extinguishes_fire
    }

    #[getter]
    fn prob_house_catches_fire(&self) -> f32 {
        self.inner.prob_house_catches_fire
    }

    #[getter]
    fn team_reward_house_survives(&self) -> f32 {
        self.inner.team_reward_house_survives
    }

    #[getter]
    fn team_penalty_house_burns(&self) -> f32 {
        self.inner.team_penalty_house_burns
    }

    #[getter]
    fn cost_to_work_one_night(&self) -> f32 {
        self.inner.cost_to_work_one_night
    }

    #[getter]
    fn min_nights(&self) -> u32 {
        self.inner.min_nights
    }

    #[getter]
    fn reward_own_house_survives(&self) -> Vec<f32> {
        self.inner.reward_own_house_survives.clone()
    }

    #[getter]
    fn reward_other_house_survives(&self) -> Vec<f32> {
        self.inner.reward_other_house_survives.clone()
    }

    #[getter]
    fn penalty_own_house_burns(&self) -> Vec<f32> {
        self.inner.penalty_own_house_burns.clone()
    }

    #[getter]
    fn penalty_other_house_burns(&self) -> Vec<f32> {
        self.inner.penalty_other_house_burns.clone()
    }

    // Issue #203 spatial-cost field getters. These let Python read the new
    // fields from scenarios pulled out of the SCENARIOS dict (the canonical
    // entry point for ``positional_default``). Empty
    // ``agent_home_positions`` means "engine falls back to the round-robin
    // anchor"; ``distance_cost_alpha == 0.0`` means "no spatial cost term".
    #[getter]
    fn agent_home_positions(&self) -> Vec<u8> {
        self.inner.agent_home_positions.clone()
    }

    #[getter]
    fn distance_cost_alpha(&self) -> f32 {
        self.inner.distance_cost_alpha
    }

    #[getter]
    fn distance_metric(&self) -> String {
        self.inner.distance_metric.clone()
    }
}

/// Python-compatible Agent Observation
#[pyclass]
pub struct PyAgentObservation {
    inner: AgentObservation,
}

#[pymethods]
impl PyAgentObservation {
    #[getter]
    fn signals(&self) -> Vec<u8> {
        self.inner.signals.clone()
    }
    #[getter]
    fn locations(&self) -> Vec<u8> {
        self.inner.locations.clone()
    }
    #[getter]
    fn houses(&self) -> Vec<u8> {
        self.inner.houses.clone()
    }
    #[getter]
    fn last_actions(&self, py: Python) -> PyObject {
        // Issue #235: the engine Action is now [house, mode, signal]
        // (length 3), but this getter keeps the legacy length-2
        // [house, mode] shape so the obs vector width is unchanged and
        // trained policies from PRs #216/#225 stay structurally
        // loadable. The broadcast signal is exposed separately via the
        // ``signals`` getter (length num_agents), so no information is
        // hidden — only deduplicated.
        let actions: Vec<Vec<u8>> = self
            .inner
            .last_actions
            .iter()
            .map(|action| vec![action[0], action[1]])
            .collect();
        actions.into_py(py)
    }

    #[getter]
    fn actions(&self, py: Python) -> PyObject {
        // See ``last_actions`` getter above for the [house, mode] vs
        // [house, mode, signal] decision.
        let actions: Vec<Vec<u8>> = self
            .inner
            .last_actions
            .iter()
            .map(|action| vec![action[0], action[1]])
            .collect();
        actions.into_py(py)
    }
    #[getter]
    fn scenario_info(&self) -> Vec<f32> {
        self.inner.scenario_info.clone()
    }
    #[getter]
    fn agent_id(&self) -> usize {
        self.inner.agent_id
    }
    #[getter]
    fn night(&self) -> u32 {
        self.inner.night
    }
}

/// Python-compatible Game State
#[pyclass]
pub struct PyGameState {
    inner: crate::GameState,
}

#[pymethods]
impl PyGameState {
    #[getter]
    fn houses(&self) -> Vec<u8> {
        self.inner.houses.clone()
    }
    #[getter]
    fn agent_positions(&self) -> Vec<u8> {
        self.inner.agent_positions.clone()
    }
    #[getter]
    fn agent_signals(&self) -> Vec<u8> {
        self.inner.agent_signals.clone()
    }
    #[getter]
    fn night(&self) -> u32 {
        self.inner.night
    }
    #[getter]
    fn done(&self) -> bool {
        self.inner.done
    }
}

/// Python-compatible Game Result
#[pyclass]
pub struct PyGameResult {
    inner: crate::GameResult,
}

#[pymethods]
impl PyGameResult {
    #[getter]
    fn scenario(&self) -> PyScenario {
        PyScenario {
            inner: self.inner.scenario.clone(),
        }
    }
    #[getter]
    fn final_score(&self) -> f32 {
        self.inner.final_score
    }
    #[getter]
    fn agent_scores(&self) -> Vec<f32> {
        self.inner.agent_scores.clone()
    }
    #[getter]
    fn nights(&self, py: Python) -> PyObject {
        // Convert nights to Python objects
        let nights: Vec<PyObject> = self
            .inner
            .nights
            .iter()
            .map(|night| {
                let dict = PyDict::new_bound(py);
                dict.set_item("night", night.night).unwrap();
                dict.set_item("houses", &night.houses).unwrap();
                dict.set_item("signals", &night.signals).unwrap();
                dict.set_item("locations", &night.locations).unwrap();
                dict.set_item(
                    "actions",
                    night
                        .actions
                        .iter()
                        .map(|a| vec![a[0], a[1], a[2]])
                        .collect::<Vec<Vec<u8>>>(),
                )
                .unwrap();
                dict.set_item("rewards", &night.rewards).unwrap();
                dict.into()
            })
            .collect();
        nights.into_py(py)
    }
}

/// Vectorized Bucket Brigade environment for efficient batch processing.
///
/// This class manages multiple independent BucketBrigade environments in Rust,
/// allowing for efficient parallel rollouts without Python loop overhead.
#[pyclass]
pub struct PyVectorEnv {
    envs: Vec<BucketBrigade>,
    num_envs: usize,
}

#[pymethods]
impl PyVectorEnv {
    #[new]
    #[pyo3(signature = (scenario, num_envs, num_agents, seed=None))]
    fn new(scenario: PyScenario, num_envs: usize, num_agents: usize, seed: Option<u64>) -> Self {
        // Create multiple independent environments with different seeds
        let envs = (0..num_envs)
            .map(|i| {
                let env_seed = seed.map(|s| s.wrapping_add(i as u64 * 12345));
                BucketBrigade::new(scenario.inner.clone(), num_agents, env_seed)
            })
            .collect();

        Self { envs, num_envs }
    }

    /// Reset all environments and return batched observations for agent 0
    fn reset(&mut self, py: Python) -> PyResult<PyObject> {
        for env in &mut self.envs {
            env.reset();
        }

        // Get observations for agent 0 from all environments
        self.get_observations_batch(py, 0)
    }

    /// Step all environments with given actions
    ///
    /// Args:
    ///     actions: List of actions, shape (num_envs, 3) where each action is
    ///         [house_id, mode, signal] (issue #235). Length-2 inputs are
    ///         accepted for backward compatibility and default the signal to
    ///         the mode bit (honest signaling).
    ///
    /// Returns:
    ///     Tuple of (observations, rewards, dones, infos) where each is batched
    fn step(
        &mut self,
        py: Python,
        actions: Vec<Vec<u8>>,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
        if actions.len() != self.num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions.len()
            )));
        }

        let mut all_rewards = Vec::with_capacity(self.num_envs);
        let mut all_dones = Vec::with_capacity(self.num_envs);

        // Step each environment
        for (env, action) in self.envs.iter_mut().zip(actions.iter()) {
            // Issue #235: accept length-2 (legacy, honest default) or
            // length-3 (post-#235, explicit signal) action shapes.
            if action.len() != 2 && action.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Each action must have 2 (legacy [house, mode]) or 3 \
                     ([house, mode, signal]) elements",
                ));
            }

            let signal = if action.len() == 3 { action[2] } else { action[1] };
            let rust_action = [action[0], action[1], signal];
            let result = env.step(&[rust_action]);

            all_rewards.push(result.rewards[0]); // Get reward for agent 0
            all_dones.push(result.done);

            // Auto-reset if done
            if result.done {
                env.reset();
            }
        }

        // Get batched observations for agent 0
        let observations = self.get_observations_batch(py, 0)?;

        // Convert rewards and dones to numpy arrays
        use pyo3::types::PyList;
        let rewards = PyList::new_bound(py, &all_rewards).into_py(py);
        let dones = PyList::new_bound(py, &all_dones).into_py(py);
        let infos = PyList::new_bound(py, Vec::<PyObject>::new()).into_py(py);

        Ok((observations, rewards, dones, infos))
    }

    /// Get batched observations for a specific agent across all environments
    fn get_observations_batch(&self, py: Python, agent_id: usize) -> PyResult<PyObject> {
        // Collect observations from all environments
        let observations: Vec<Vec<f32>> = self
            .envs
            .iter()
            .map(|env| {
                let obs = env.get_observation(agent_id);
                // Convert observation to flat f32 vector
                obs_to_vector(&obs)
            })
            .collect();

        // Convert to Python list of lists (will be converted to numpy array in Python)
        let obs_list: Vec<PyObject> = observations
            .iter()
            .map(|obs| obs.clone().into_py(py))
            .collect();

        use pyo3::types::PyList;
        Ok(PyList::new_bound(py, &obs_list).into_py(py))
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }
}

/// Convert AgentObservation to flat f32 vector (matches Python observation space)
fn obs_to_vector(obs: &AgentObservation) -> Vec<f32> {
    let mut vec = Vec::new();

    // Add signals (num_agents elements)
    vec.extend(obs.signals.iter().map(|&x| x as f32));

    // Add locations (num_agents elements)
    vec.extend(obs.locations.iter().map(|&x| x as f32));

    // Add houses (num_agents elements)
    vec.extend(obs.houses.iter().map(|&x| x as f32));

    // Add flattened last_actions (num_agents * 2 elements).
    //
    // Issue #235 note: per the curator's recommendation we keep
    // last_actions at length 2 (house + mode) here even though the
    // underlying Action is now length 3. The signal channel is already
    // exposed via ``obs.signals`` (length num_agents), so duplicating it
    // here would just widen the obs vector without adding information.
    // Keeping the width unchanged means trained policies from PRs #216
    // and #225 still load structurally, though their *semantics* are
    // stale (signals now carry real bits, so the learned filters are
    // operating on a different signal distribution).
    for action in &obs.last_actions {
        vec.push(action[0] as f32);
        vec.push(action[1] as f32);
    }

    // Add scenario_info (11 elements)
    vec.extend(obs.scenario_info.iter());

    // Add agent_id and night
    vec.push(obs.agent_id as f32);
    vec.push(obs.night as f32);

    vec
}

/// Run a complete heuristic episode with arbitrary agent team compositions.
///
/// This is the generalized version that supports heterogeneous teams where
/// each agent can have different strategy parameters.
///
/// # Arguments
/// * `scenario` - Game scenario parameters
/// * `agent_params` - List of 10-parameter vectors, one per agent
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Vector of cumulative rewards for all agents
#[pyfunction]
#[pyo3(signature = (scenario, num_agents, agent_params, seed))]
fn run_heuristic_episode(
    scenario: PyScenario,
    num_agents: usize,
    agent_params: Vec<Vec<f64>>,
    seed: u64,
) -> PyResult<Vec<f64>> {
    // Validate agent count
    if agent_params.len() != num_agents {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} agent parameter vectors for scenario, got {}",
            num_agents,
            agent_params.len()
        )));
    }

    // Convert and validate parameter vectors
    let mut agents = Vec::new();
    for (id, params) in agent_params.iter().enumerate() {
        if params.len() != 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Agent {} has {} parameters, expected 10",
                id,
                params.len()
            )));
        }
        let params_array: [f64; 10] = params
            .as_slice()
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to convert parameters"))?;
        agents.push(HeuristicAgent::new(
            id,
            &format!("agent_{}", id),
            params_array,
        ));
    }

    // Create game
    let mut game = BucketBrigade::new(scenario.inner, num_agents, Some(seed));

    // Create RNG for heuristic decisions
    let mut rng = Pcg64::seed_from_u64(seed);

    // Run episode
    let mut step_count = 0;
    const MAX_STEPS: u32 = 100;

    while step_count < MAX_STEPS {
        // Get observations for all agents
        let observations: Vec<AgentObservation> =
            (0..num_agents).map(|id| game.get_observation(id)).collect();

        // Get actions from heuristic agents (issue #235: 3-element Action)
        let actions: Vec<[u8; 3]> = observations
            .iter()
            .enumerate()
            .map(|(id, obs)| agents[id].select_action(obs, &mut rng))
            .collect();

        // Step the game
        let result = game.step(&actions);

        // Check if done
        if result.done {
            break;
        }

        step_count += 1;
    }

    // Return rewards for all agents
    let final_result = game.get_result();
    Ok(final_result
        .agent_scores
        .iter()
        .map(|&r| r as f64)
        .collect())
}

/// Run a heuristic episode optimized for Nash equilibrium computation.
///
/// This is a specialized version for Nash equilibrium where we have one "focal"
/// agent vs N-1 identical "opponent" agents. Returns only the focal agent's reward.
/// This function wraps the generalized `run_heuristic_episode` for backward compatibility.
///
/// # Arguments
/// * `scenario` - Game scenario parameters
/// * `theta_focal` - Strategy parameters for focal agent (agent 0)
/// * `theta_opponents` - Strategy parameters for opponent agents (agents 1+)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Cumulative reward for the focal agent (agent 0)
#[pyfunction]
#[pyo3(signature = (scenario, num_agents, theta_focal, theta_opponents, seed))]
fn run_heuristic_episode_focal(
    scenario: PyScenario,
    num_agents: usize,
    theta_focal: Vec<f64>,
    theta_opponents: Vec<f64>,
    seed: u64,
) -> PyResult<f64> {
    // Build agent_params vector: focal agent + N-1 opponent agents
    let mut agent_params = vec![theta_focal];
    for _ in 1..num_agents {
        agent_params.push(theta_opponents.clone());
    }

    // Call the generalized function
    let rewards = run_heuristic_episode(scenario, num_agents, agent_params, seed)?;

    // Return only focal agent reward
    Ok(rewards[0])
}

#[pymodule]
fn bucket_brigade_core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyBucketBrigade>()?;
    m.add_class::<PyScenario>()?;
    m.add_class::<PyAgentObservation>()?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameResult>()?;
    m.add_class::<PyVectorEnv>()?;

    // Add functions
    m.add_function(wrap_pyfunction!(run_heuristic_episode, m)?)?;
    m.add_function(wrap_pyfunction!(run_heuristic_episode_focal, m)?)?;

    // Add scenarios
    let scenarios = PyDict::new_bound(m.py());
    for (name, scenario) in crate::SCENARIOS.iter() {
        scenarios.set_item(
            name,
            Py::new(
                m.py(),
                PyScenario {
                    inner: scenario.clone(),
                },
            )?,
        )?;
    }
    m.add("SCENARIOS", scenarios)?;

    Ok(())
}
