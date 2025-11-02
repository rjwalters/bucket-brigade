use rand::prelude::*;
use rand_pcg::Pcg64;

/// Deterministic random number generator for reproducible simulations
pub struct DeterministicRng {
    rng: Pcg64,
}

impl DeterministicRng {
    pub fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(42);
        Self {
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    pub fn random(&mut self) -> f32 {
        self.rng.gen::<f32>()
    }

    pub fn randint(&mut self, min: usize, max: usize) -> usize {
        self.rng.gen_range(min..max)
    }

    pub fn choice<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let index = self.randint(0, items.len());
        &items[index]
    }
}
