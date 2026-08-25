# Repository guide

`negacyclic-rings` implements arithmetic in `Z_q[X]/(X^N + 1)`. It provides const-generic 32-bit and 64-bit rings plus fixed-size, generic RNS chains. AVX2 is selected at runtime on x86_64; NEON is used directly on AArch64.

## Integration

During local development, add the crate by path:

```toml
[dependencies]
negacyclic-rings = { path = "../negacyclic-rings" }
```

Construct parameters at startup when experimenting:

```rust
use negacyclic_rings::params::{find_psi32, generate_ring32};
use negacyclic_rings::{ntt32, Rns};

const N: usize = 2048;
let q = 16_760_833;
let ring = generate_ring32::<N>(q, find_psi32::<N>(q));

let mut p = [0u32; N];
ntt32::ntt(&ring, &mut p);
ntt32::inv_ntt(&ring, &mut p);

let rns = Rns::new([
    generate_ring32::<N>(16_760_833, find_psi32::<N>(16_760_833)),
    generate_ring32::<N>(16_736_257, find_psi32::<N>(16_736_257)),
]);
let mut residues = [[0u32; N]; 2];
rns.forward(&mut residues);
rns.inverse(&mut residues);
```

Inputs and outputs are canonical residues in `[0, q)`. Polynomial multiplication is forward NTT on both operands, `pointwise_mul`, then inverse NTT. Use `pointwise_mac` for a sum of products; it avoids repeated conversion out of Montgomery form.

RNS channels are independent NTT rings. `reduce_coeff` maps an integer to its channels. `lift_coeff` reconstructs into `[0, product)` and `lift_centered` into the centered interval. Channel moduli must be pairwise coprime, and their product must fit `u128`.

## Parameters

`N` must be a power of two and each prime modulus must satisfy `q = 1 mod 2N`. For `Ring32`, require `2q < 2^31`; 24-bit limbs are the preferred RNS configuration. For `Ring64`, require `q < 2^62`.

Runtime generation is convenient but should not be placed on a protocol hot path. Generate checked Rust constants for production:

```sh
python3 scripts/paramgen.py \
  --bits 32 \
  --degree 2048 \
  --modulus 16760833 \
  --name RING \
  --output params.rs
```

Keep protocol-specific names and parameter selections in the consuming protocol repository. This crate owns arithmetic and generic parameter machinery only.

## Correctness checks

Run the full suite:

```sh
cargo test --all-targets
```

The test profile uses `opt-level = 1` while retaining debug assertions. The tests compare NTT multiplication with schoolbook negacyclic multiplication and cover multi-limb RNS, 24-bit NTTs, pointwise multiplication, and MAC. Run tests natively on AArch64 before accepting NEON changes. A cross-build catches intrinsic and target-specific type errors:

```sh
rustup target add aarch64-unknown-linux-gnu
cargo check --target aarch64-unknown-linux-gnu --all-targets
```

CI runs formatting, strict Clippy, release tests on x86_64, and release tests on native AArch64. Enable the local hook as described in `CONTRIBUTING.md`.

## Benchmarking

Run optimized benchmarks on an otherwise idle machine:

```sh
cargo run --release --example bench_ntt
```

The example reports median microseconds per operation for 32-bit NTT, 32-bit pointwise multiplication, 64-bit NTT, and two-limb RNS forward NTT. Compare results only on the same CPU, governor, compiler, and parameter set. Benchmark both x86_64 and AArch64 when changing shared loop structure.

## Profiling

Select one hot operation and enough iterations to dominate startup:

```sh
cargo run --release --example profile_rns -- fwd 30000
cargo run --release --example profile_rns -- inv 30000
cargo run --release --example profile_rns -- mul 30000
```

On Linux, record the optimized binary with the system profiler:

```sh
perf record --call-graph dwarf -- cargo run --release --example profile_rns -- fwd 30000
perf report
```

Inspect generated code after SIMD changes. AArch64 twiddle multiplication must retain `sqrdmulh` followed by `mls`; pointwise Montgomery multiplication should use widening `umull` operations.

## Design constraints

- Keep hot loops concrete and const-generic. Do not add dynamic dispatch, ring traits, boxed abstractions, or code-generating macros.
- Preserve canonical residue and modulus bounds at every public boundary.
- Add scalar-equivalence or schoolbook tests with every new kernel.
- Use `REFERENCES.md` to cross-check optimized kernels and record new implementation sources at pinned revisions.
