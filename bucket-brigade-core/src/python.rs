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
    #[pyo3(signature = (scenario, seed=None))]
    fn new(scenario: PyScenario, seed: Option<u64>) -> Self {
        Self {
            inner: BucketBrigade::new(scenario.inner, seed),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn step(&mut self, py: Python, actions: Vec<Vec<u8>>) -> PyResult<(PyObject, bool, PyObject)> {
        let rust_actions: Vec<[u8; 2]> = actions.iter().map(|a| [a[0], a[1]]).collect();

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
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        prob_fire_spreads_to_neighbor: f32,
        prob_solo_agent_extinguishes_fire: f32,
        prob_house_catches_fire: f32,
        team_reward_house_survives: f32,
        team_penalty_house_burns: f32,
        cost_to_work_one_night: f32,
        min_nights: u32,
        num_agents: usize,
        reward_own_house_survives: f32,
        reward_other_house_survives: f32,
        penalty_own_house_burns: f32,
        penalty_other_house_burns: f32,
    ) -> Self {
        Self {
            inner: Scenario {
                prob_fire_spreads_to_neighbor,
                prob_solo_agent_extinguishes_fire,
                prob_house_catches_fire,
                team_reward_house_survives,
                team_penalty_house_burns,
                cost_to_work_one_night,
                min_nights,
                num_agents,
                reward_own_house_survives,
                reward_other_house_survives,
                penalty_own_house_burns,
                penalty_other_house_burns,
            },
        }
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
    fn num_agents(&self) -> usize {
        self.inner.num_agents
    }

    #[getter]
    fn reward_own_house_survives(&self) -> f32 {
        self.inner.reward_own_house_survives
    }

    #[getter]
    fn reward_other_house_survives(&self) -> f32 {
        self.inner.reward_other_house_survives
    }

    #[getter]
    fn penalty_own_house_burns(&self) -> f32 {
        self.inner.penalty_own_house_burns
    }

    #[getter]
    fn penalty_other_house_burns(&self) -> f32 {
        self.inner.penalty_other_house_burns
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
                        .map(|a| vec![a[0], a[1]])
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
#[pyo3(signature = (scenario, agent_params, seed))]
fn run_heuristic_episode(
    scenario: PyScenario,
    agent_params: Vec<Vec<f64>>,
    seed: u64,
) -> PyResult<Vec<f64>> {
    // Validate agent count
    let num_agents = scenario.num_agents();
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
    let mut game = BucketBrigade::new(scenario.inner, Some(seed));

    // Create RNG for heuristic decisions
    let mut rng = Pcg64::seed_from_u64(seed);

    // Run episode
    let mut step_count = 0;
    const MAX_STEPS: u32 = 100;

    while step_count < MAX_STEPS {
        // Get observations for all agents
        let observations: Vec<AgentObservation> =
            (0..num_agents).map(|id| game.get_observation(id)).collect();

        // Get actions from heuristic agents
        let actions: Vec<[u8; 2]> = observations
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
#[pyo3(signature = (scenario, theta_focal, theta_opponents, seed))]
fn run_heuristic_episode_focal(
    scenario: PyScenario,
    theta_focal: Vec<f64>,
    theta_opponents: Vec<f64>,
    seed: u64,
) -> PyResult<f64> {
    // Build agent_params vector: focal agent + N-1 opponent agents
    let num_agents = scenario.num_agents();
    let mut agent_params = vec![theta_focal];
    for _ in 1..num_agents {
        agent_params.push(theta_opponents.clone());
    }

    // Call the generalized function
    let rewards = run_heuristic_episode(scenario, agent_params, seed)?;

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

    // Add functions
    m.add_function(wrap_pyfunction!(run_heuristic_episode, m)?)?;
    m.add_function(wrap_pyfunction!(run_heuristic_episode_focal, m)?)?;

    // Add scenarios
    let scenarios = PyDict::new_bound(m.py());
    for (name, scenario) in crate::SCENARIOS.entries() {
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
