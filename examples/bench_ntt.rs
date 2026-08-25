//! `cargo run --release --example bench_ntt`
use negacyclic_rings::ntt32;
use negacyclic_rings::ntt64;
use negacyclic_rings::params::{find_psi32, find_psi64, generate_ring32, generate_ring64};
use negacyclic_rings::Rns;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 2048;

fn median_us(reps: usize, iters: usize, mut f: impl FnMut() -> u64) -> f64 {
    let mut sink = 0u64;
    for _ in 0..iters.min(500) {
        sink ^= f();
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..iters {
            sink ^= f();
        }
        samples.push(t.elapsed().as_secs_f64() / iters as f64 * 1e6);
    }
    black_box(sink);
    samples.sort_by(f64::total_cmp);
    samples[reps / 2]
}

fn main() {
    let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
    let q32 = 16_760_833u32;
    let ring32 = generate_ring32::<N>(q32, find_psi32::<N>(q32));
    let q64 = 347_280_875_347_969u64;
    let ring64 = generate_ring64::<N>(q64, find_psi64::<N>(q64));
    let rns = Rns::new([
        generate_ring32::<N>(q32, find_psi32::<N>(q32)),
        generate_ring32::<N>(16_736_257, find_psi32::<N>(16_736_257)),
    ]);
    let a32 = core::array::from_fn(|_| rng.gen_range(0..q32));
    let b32 = core::array::from_fn(|_| rng.gen_range(0..q32));
    let a64 = core::array::from_fn(|_| rng.gen_range(0..q64));
    let residues = [a32, core::array::from_fn(|_| rng.gen_range(0..rns.ch[1].q))];
    let (reps, iters) = (7, 4_000);

    println!(
        "ntt32 fwd={:7.2} pointwise={:7.2}  ntt64 fwd={:7.2}  rns2 fwd={:7.2} us/op",
        median_us(reps, iters, || {
            let mut value = *black_box(&a32);
            ntt32::ntt(&ring32, &mut value);
            value[0] as u64
        }),
        median_us(reps, iters, || {
            ntt32::pointwise_mul(&ring32, black_box(&a32), black_box(&b32))[0] as u64
        }),
        median_us(reps, iters, || {
            let mut value = *black_box(&a64);
            ntt64::ntt(&ring64, &mut value);
            value[0]
        }),
        median_us(reps, iters / 2, || {
            let mut value = *black_box(&residues);
            rns.forward(&mut value);
            value[0][0] as u64
        }),
    );
}
