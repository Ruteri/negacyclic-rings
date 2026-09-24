pub mod arithmetic;
pub mod decomposition;
pub mod ntt32;
pub mod ntt64;
pub mod residue_number_system;

pub use ntt32::Ring32;
pub use ntt64::Ring64;
pub use residue_number_system::{ResidueNumberSystem, Residues};
