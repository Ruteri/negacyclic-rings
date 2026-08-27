//! `cargo run --release --example bench_ntt`
use negacyclic_rings::ntt32;
use negacyclic_rings::ntt64;
use negacyclic_rings::params::{find_psi32, find_psi64, generate_ring32, generate_ring64};
use negacyclic_rings::Rns;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;
use std::time::{Duration, Instant};

const N: usize = 2048;

fn median_us(mut f: impl FnMut() -> u64) -> f64 {
    const REPS: usize = 3;
    const TARGET: Duration = Duration::from_millis(150);
    let mut sink = 0u64;
    for _ in 0..64 {
        sink ^= f();
    }
    let mut iters = 1usize;
    loop {
        let started = Instant::now();
        for _ in 0..iters {
            sink ^= f();
        }
        if started.elapsed() >= TARGET || iters >= 1 << 24 {
            break;
        }
        iters *= 2;
    }
    let mut samples = [0.0; REPS];
    for sample in &mut samples {
        let t = Instant::now();
        for _ in 0..iters {
            sink ^= f();
        }
        *sample = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    }
    black_box(sink);
    samples.sort_by(f64::total_cmp);
    samples[REPS / 2]
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
    let signed = core::array::from_fn(|_| rng.gen_range(-(q64 as i64 / 2)..q64 as i64 / 2));
    let input_modulus = 139_301i32;
    let signed_i32 =
        core::array::from_fn(|_| rng.gen_range(-input_modulus / 2..=input_modulus / 2));
    let residues = [a32, core::array::from_fn(|_| rng.gen_range(0..rns.ch[1].q))];
    println!(
        "ntt32 fwd={:7.2} pointwise={:7.2}  ntt64 fwd={:7.2}  rns2 fwd={:7.2} us/op",
        median_us(|| {
            let mut value = *black_box(&a32);
            ntt32::ntt(&ring32, &mut value);
            value[0] as u64
        }),
        median_us(|| { ntt32::pointwise_mul(&ring32, black_box(&a32), black_box(&b32))[0] as u64 }),
        median_us(|| {
            let mut value = *black_box(&a64);
            ntt64::ntt(&ring64, &mut value);
            value[0]
        }),
        median_us(|| {
            let mut value = *black_box(&residues);
            rns.forward(&mut value);
            value[0][0] as u64
        }),
    );
    println!(
        "rns2 reduce-i64={:7.2} us/op",
        median_us(|| {
            let mut output = [[0u32; N]; 2];
            rns.reduce_i64_into(black_box(&signed), &mut output);
            output[0][0] as u64
        }),
    );
    println!(
        "rns2 reduce-i32={:7.2} lift={:7.2} us/op",
        median_us(|| {
            let mut output = [[0u32; N]; 2];
            rns.reduce_centered_i32_into(black_box(&signed_i32), &mut output);
            output[0][0] as u64
        }),
        median_us(|| {
            let mut output = [0i64; N];
            rns.lift_centered_i64_into(black_box(&residues), &mut output);
            output[0] as u64
        }),
    );
}
