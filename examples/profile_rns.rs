//! `cargo run --release --example profile_rns -- [fwd|inv|mul|all] [iterations]`
use negacyclic_rings::params::{find_psi32, generate_ring32};
use negacyclic_rings::Rns;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;

const N: usize = 2048;

fn main() {
    let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
    let moduli = [16_760_833u32, 16_736_257];
    let rns = Rns::new(moduli.map(|q| generate_ring32::<N>(q, find_psi32::<N>(q))));
    let a = core::array::from_fn(|limb| core::array::from_fn(|_| rng.gen_range(0..rns.ch[limb].q)));
    let b = core::array::from_fn(|limb| core::array::from_fn(|_| rng.gen_range(0..rns.ch[limb].q)));
    let mut an = a;
    rns.forward(&mut an);
    let phase = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30_000);
    let mut sink = 0u32;

    if phase == "fwd" || phase == "all" {
        for _ in 0..iters {
            let mut value = *black_box(&a);
            rns.forward(&mut value);
            sink ^= value[0][0];
        }
    }
    if phase == "inv" || phase == "all" {
        for _ in 0..iters {
            let mut value = *black_box(&an);
            rns.inverse(&mut value);
            sink ^= value[0][0];
        }
    }
    if phase == "mul" || phase == "all" {
        for _ in 0..iters {
            sink ^= rns.pointwise_mul(black_box(&a), black_box(&b))[0][0];
        }
    }
    println!("{}", black_box(sink));
}
