#![warn(missing_docs)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![forbid(unsafe_code)]

//! Causal reasoning engine for The Foundry's REASON station (A3).
//!
//! Provides DAG construction, counterfactual testing, and inference
//! to produce causal chains from pattern data.

pub mod counterfactual;
pub mod dag;
pub mod inference;
