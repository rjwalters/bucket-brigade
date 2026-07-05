use super::core::BucketBrigade;
use crate::Action;

impl BucketBrigade {
    pub(super) fn extinguish_fires(&mut self, actions: &[Action]) {
        // Issue #253 / option D of architect proposal #234: dispatch on
        // `scenario.extinguish_mode`. The `"bernoulli"` branch is the
        // pre-#253 code unchanged (kept verbatim so every existing scenario
        // is bit-exact); the `"continuous"` branch accumulates per-worker
        // suppression progress and transitions deterministically when the
        // accumulator reaches 1.0.
        //
        // The reward attribution for the extinguish event is unchanged in
        // both modes — `engine/rewards.rs` keys off the BURNING -> SAFE
        // transition on `prev_houses[h] != 0 && houses[h] == 0`, which
        // fires in continuous mode exactly when the accumulator crosses
        // the threshold. The smoothing benefit option D targets comes
        // from credit assignment via the value function: workers who
        // contributed but didn't trigger the threshold this step still
        // affected the next-step suppression state, so the critic learns
        // their work matters.
        match self.scenario.extinguish_mode.as_str() {
            "bernoulli" => self.extinguish_fires_bernoulli(actions),
            "continuous" => self.extinguish_fires_continuous(actions),
            // `Scenario::validate()` rejects unknown modes at
            // construction time; this defensive panic keeps the dispatch
            // total so a future mode added without engine wiring
            // surfaces a clear error rather than silently no-op'ing.
            other => panic!(
                "Unknown extinguish_mode={:?}; supported: {:?}",
                other,
                crate::scenarios::ALLOWED_EXTINGUISH_MODES
            ),
        }
    }

    /// Pre-#253 Bernoulli extinguish model. Kept verbatim from the original
    /// `extinguish_fires` implementation so default-mode scenarios produce
    /// bit-exact identical RNG streams to the pre-#253 engine.
    fn extinguish_fires_bernoulli(&mut self, actions: &[Action]) {
        let num_houses = self.scenario.num_houses as usize;
        for house_idx in 0..num_houses {
            if self.houses[house_idx] != 1 {
                continue;
            }

            // Count workers at this house
            let workers_here = actions
                .iter()
                .filter(|action| action[0] as usize == house_idx && action[1] == 1)
                .count();

            // Probability of extinguishing - independent probabilities
            let p_extinguish = 1.0
                - (1.0 - self.scenario.prob_solo_agent_extinguishes_fire).powi(workers_here as i32);

            if self.rng.random() < p_extinguish {
                self.houses[house_idx] = 0;
            }
        }
    }

    /// Issue #253: continuous extinguish model. Each work step at a burning
    /// house adds `workers_here * suppression_per_worker` to the per-house
    /// accumulator; the fire transitions BURNING -> SAFE deterministically
    /// when the accumulator reaches 1.0. On transition the accumulator is
    /// zeroed so it's ready for the next ignition cycle.
    ///
    /// Notable difference from the Bernoulli branch: this dispatch does
    /// **not** draw from `self.rng`, so the deterministic RNG stream is
    /// reserved for the spread / spawn phases. That's why parity tests
    /// against the Bernoulli baseline only check expected long-run
    /// behavior, not per-step RNG byte-equality.
    fn extinguish_fires_continuous(&mut self, actions: &[Action]) {
        let num_houses = self.scenario.num_houses as usize;
        let suppression_per_worker = self.scenario.suppression_per_worker;
        for house_idx in 0..num_houses {
            if self.houses[house_idx] != 1 {
                continue;
            }

            let workers_here = actions
                .iter()
                .filter(|action| action[0] as usize == house_idx && action[1] == 1)
                .count();

            if workers_here == 0 {
                continue;
            }

            let increment = workers_here as f32 * suppression_per_worker;
            self.fire_progress[house_idx] += increment;

            if self.fire_progress[house_idx] >= 1.0 {
                self.houses[house_idx] = 0;
                self.fire_progress[house_idx] = 0.0;
            }
        }
    }

    pub(super) fn spread_fires(&mut self) {
        // β-inertness (issue #458): in the default "bernoulli" extinguish
        // mode this phase is a structural no-op. `burn_out_houses` runs
        // before this phase in the step order (`engine/core.rs`) and
        // bernoulli burn-out ruins every still-BURNING house, so the
        // `houses[house_idx] != 1` guard below skips every house:
        // `prob_fire_spreads_to_neighbor` never gates a spread and this
        // phase draws zero RNG (cross-β trajectories are bit-identical
        // under a shared seed — pinned by tests/test_beta_inertness.py).
        // Fire spread is only live in "continuous" extinguish mode
        // (#253), where burn-out returns early and fires persist into
        // this phase. Do NOT "clean up" β as dead code: it is exposed to
        // agents as scenario_info[0] (`engine/observation.rs`).
        //
        // Issue #254: ring length is now `scenario.num_houses` rather than a
        // hardcoded 10. The neighbor wraparound modulo also tracks the
        // scenario value so 2-house and other small-ring topologies wrap
        // correctly. Note that on a 2-house ring the (i-1) and (i+1)
        // neighbors are the same house, so spread is single-step but the
        // probability roll still happens twice — matching the way the
        // 10-ring case handled it for the trivial wrap edge.
        let num_houses = self.scenario.num_houses as usize;
        let mut new_houses = self.houses.clone();

        for house_idx in 0..num_houses {
            if self.houses[house_idx] != 1 {
                continue;
            }

            // Check neighbors (wrap modulo num_houses).
            let neighbors = [
                (house_idx + num_houses - 1) % num_houses,
                (house_idx + 1) % num_houses,
            ];

            for &neighbor in &neighbors {
                if self.houses[neighbor] == 0
                    && self.rng.random() < self.scenario.prob_fire_spreads_to_neighbor
                {
                    new_houses[neighbor] = 1;
                }
            }
        }

        self.houses = new_houses;
    }

    pub(super) fn burn_out_houses(&mut self) {
        // Issue #253: the burn-out semantics are mode-dependent.
        //
        //   * Bernoulli mode (pre-#253 default): every BURNING house that
        //     wasn't extinguished this step transitions to RUINED. Fires
        //     last exactly one step under the Bernoulli outcome — the
        //     per-step coin flip is make-or-break. Structural consequence
        //     (issue #458): because this phase runs before `spread_fires`,
        //     no house is ever BURNING when the spread phase executes, so
        //     `prob_fire_spreads_to_neighbor` is dynamics-inert in this
        //     mode (see the note on `spread_fires` above).
        //
        //   * Continuous mode: fires persist across steps because the
        //     point of the accumulator is to spread suppression credit
        //     over multiple work steps. If burn-out fired every step, the
        //     accumulator would never see more than one step's worth of
        //     suppression and the calibration to the Bernoulli
        //     expectation would be wrong (we'd be modeling
        //     "deterministic extinguish vs guaranteed ruin"). Fires only
        //     leave BURNING via the extinguish path; episodes still
        //     terminate via the existing `min_nights` + all-safe / no-fires
        //     conditions in `check_termination`.
        //
        // This branch is the load-bearing semantic difference between the
        // two modes — every other change (the accumulator field, the
        // dispatch, the scenario knobs) is plumbing.
        if self.scenario.extinguish_mode == "continuous" {
            return;
        }
        // Bernoulli mode: pre-#253 behavior unchanged.
        for (house, progress) in self.houses.iter_mut().zip(self.fire_progress.iter_mut()) {
            if *house == 1 {
                *house = 2;
                *progress = 0.0;
            }
        }
    }

    pub(super) fn spontaneous_ignition(&mut self) {
        let num_houses = self.scenario.num_houses as usize;
        for house_idx in 0..num_houses {
            if self.houses[house_idx] == 0
                && self.rng.random() < self.scenario.prob_house_catches_fire
            {
                self.houses[house_idx] = 1;
                // Issue #253: a fresh ignition gets a fresh accumulator.
                // (It was already 0.0 from the SAFE state, but writing
                // explicitly here keeps the spread-vs-spontaneous code
                // paths symmetric and the invariant
                // "fire_progress[i] != 0 implies houses[i] == BURNING"
                // exact in both directions.)
                self.fire_progress[house_idx] = 0.0;
            }
        }
    }
}
