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
        beta: f32,
        kappa: f32,
        a: f32,
        l: f32,
        c: f32,
        rho_ignite: f32,
        n_min: u32,
        p_spark: f32,
        n_spark: u32,
        num_agents: usize,
    ) -> Self {
        Self {
            inner: Scenario {
                beta,
                kappa,
                a,
                l,
                c,
                rho_ignite,
                n_min,
                p_spark,
                n_spark,
                num_agents,
            },
        }
    }

    #[getter]
    fn beta(&self) -> f32 {
        self.inner.beta
    }
    #[getter]
    fn kappa(&self) -> f32 {
        self.inner.kappa
    }
    #[getter]
    fn a(&self) -> f32 {
        self.inner.a
    }
    #[getter]
    fn l(&self) -> f32 {
        self.inner.l
    }
    #[getter]
    fn c(&self) -> f32 {
        self.inner.c
    }
    #[getter]
    fn rho_ignite(&self) -> f32 {
        self.inner.rho_ignite
    }
    #[getter]
    fn n_min(&self) -> u32 {
        self.inner.n_min
    }
    #[getter]
    fn p_spark(&self) -> f32 {
        self.inner.p_spark
    }
    #[getter]
    fn n_spark(&self) -> u32 {
        self.inner.n_spark
    }
    #[getter]
    fn num_agents(&self) -> usize {
        self.inner.num_agents
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
