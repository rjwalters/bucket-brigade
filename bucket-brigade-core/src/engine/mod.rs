mod core;
mod observation;
mod phases;
mod rewards;
mod types;

#[cfg(test)]
mod tests;

// Re-export public types
pub use types::{GameResult, GameState, StepResult};

// Re-export main engine struct
pub use core::BucketBrigade;
