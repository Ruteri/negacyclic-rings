#![allow(clippy::needless_range_loop)]

pub mod params;
pub mod poly;

pub use poly::{arithmetic, decomposition, ntt32, ntt64, Residues, Ring32, Ring64, Rns};
