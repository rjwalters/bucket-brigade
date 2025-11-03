use crate::{AgentObservation, BucketBrigade, Scenario};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

#[pymodule]
fn bucket_brigade_core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyBucketBrigade>()?;
    m.add_class::<PyScenario>()?;
    m.add_class::<PyAgentObservation>()?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameResult>()?;

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
