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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng_default_seed() {
        let mut rng1 = DeterministicRng::new(None);
        let mut rng2 = DeterministicRng::new(None);

        // With same default seed, should produce identical sequences
        for _ in 0..10 {
            assert_eq!(rng1.random(), rng2.random());
        }
    }

    #[test]
    fn test_deterministic_rng_custom_seed() {
        let mut rng1 = DeterministicRng::new(Some(12345));
        let mut rng2 = DeterministicRng::new(Some(12345));

        // Same seed produces same sequence
        for _ in 0..10 {
            assert_eq!(rng1.random(), rng2.random());
        }

        // Different seed produces different sequence
        let val1 = DeterministicRng::new(Some(12345)).random();
        let val2 = DeterministicRng::new(Some(54321)).random();
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_random_range() {
        let mut rng = DeterministicRng::new(Some(42));

        // Test that random() returns values in [0, 1)
        for _ in 0..100 {
            let val = rng.random();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_randint_range() {
        let mut rng = DeterministicRng::new(Some(42));

        // Test that randint() returns values in [min, max)
        for _ in 0..100 {
            let val = rng.randint(5, 15);
            assert!((5..15).contains(&val));
        }
    }

    #[test]
    fn test_randint_single_value() {
        let mut rng = DeterministicRng::new(Some(42));

        // Range [5, 6) should always return 5
        for _ in 0..10 {
            assert_eq!(rng.randint(5, 6), 5);
        }
    }

    #[test]
    fn test_choice() {
        let mut rng = DeterministicRng::new(Some(42));
        let items = vec!["a", "b", "c", "d", "e"];

        // Test that choice returns items from the list
        for _ in 0..50 {
            let chosen = rng.choice(&items);
            assert!(items.contains(chosen));
        }
    }

    #[test]
    fn test_choice_deterministic() {
        let items = vec![1, 2, 3, 4, 5];

        let mut rng1 = DeterministicRng::new(Some(999));
        let mut rng2 = DeterministicRng::new(Some(999));

        // Same seed should make same choices
        for _ in 0..10 {
            assert_eq!(rng1.choice(&items), rng2.choice(&items));
        }
    }

    #[test]
    fn test_choice_coverage() {
        let mut rng = DeterministicRng::new(Some(42));
        let items = vec![1, 2, 3];
        let mut seen = std::collections::HashSet::new();

        // Over many iterations, should see all items
        for _ in 0..100 {
            seen.insert(*rng.choice(&items));
        }

        assert_eq!(seen.len(), 3);
        assert!(seen.contains(&1));
        assert!(seen.contains(&2));
        assert!(seen.contains(&3));
    }
}
