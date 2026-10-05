# References

These sources cover the algorithms and implementation techniques used throughout the repository. Each entry names the paper, then points at the concrete code that implements it, so the code can be read as the next step after the paper. Pinned implementation links are retained for cross-checking and later audit.

Sections are filled in as they're reviewed; entries not yet linked to code live in [General](#general) below.

## Residue Number System

- H. L. Garner, [The Residue Number System](https://doi.org/10.1109/TEC.1959.5219515) (RNS) — mixed-radix (Garner) reconstruction for a chain of pairwise-coprime moduli.
  - `Rns<N, CHANNEL_COUNT>` holds one `Ring32<N>` channel per modulus `q_i`. `Rns::new` precomputes the mixed-radix constants the algorithm needs: `prefix_products[i] = ∏_{j<i} q_j` and `prefix_inverses[i] = prefix_products[i]^-1 mod q_i`, via an extended-Euclid inverse. That inverse fails — panicking with a pairwise-coprimality message — if two channel moduli share a factor, which is exactly Garner's pairwise-coprimality requirement.
  - `Rns::lift_coefficient` is Garner's algorithm directly: for each channel `i` in turn it solves `digit = (r[i] - x) * prefix_inverses[i] mod q_i` for the next mixed-radix digit and folds it in as `x += prefix_products[i] * digit`, so `x` only ever grows to fit within `∏ q_j` for the channels seen so far.
  - `Rns::lift_centered` wraps `lift_coefficient` and recenters the result into `(-product/2, product/2]`.
  - `Rns::lift_centered_i64_into` specializes the same two-term recurrence for `CHANNEL_COUNT = 2`, producing an `i64` directly instead of going through `u128`, with AVX2 and NEON kernels that stay colocated with the Montgomery/csub helpers they share with the NTT kernels in `ntt32.rs`.
  - `Rns::reduce_coefficient`, `::reduce_coefficients_i64_into`, and `::reduce_centered_i32_into` are the forward direction, integer to per-channel residues. They are independent `rem_euclid` calls per channel, not part of Garner's algorithm — Garner only governs reconstruction.

## General

Not yet linked to specific code; see the section above for the target format.

- J. M. Pollard, [The Fast Fourier Transform in a Finite Field](https://doi.org/10.1090/S0025-5718-1971-0301966-0), for finite-field Fourier transforms and radix-two NTTs.
- P. Longa and M. Naehrig, [Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography](https://www.microsoft.com/en-us/research/publication/speeding-up-the-number-theoretic-transform-for-faster-ideal-lattice-based-cryptography/), for negacyclic NTTs, signed representatives, and efficient butterfly organization.
- D. Harvey, [Faster Arithmetic for Number-Theoretic Transforms](https://arxiv.org/abs/1205.2926), especially Algorithm 4, for Shoup multiplication with the precomputed companion `floor(w·2^k/q)`.
- P. Barrett, [Implementing the Rivest Shamir and Adleman Public Key Encryption Algorithm on a Standard Digital Signal Processor](https://doi.org/10.1007/3-540-47721-7_24), for reciprocal-based modular reduction.
- P. L. Montgomery, [Modular Multiplication Without Trial Division](https://doi.org/10.1090/S0025-5718-1985-0777282-X), for the Montgomery products used by pointwise multiplication and accumulation.
- Becker et al., [Neon NTT: Faster Dilithium, Kyber, and Saber](https://eprint.iacr.org/2021/986.pdf), for the AArch64 signed Barrett reduction using `SQRDMULH` and `MLS`, and for layered SIMD layouts.
- Becker et al., pinned [butterfly macros](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/macros_common.i), [forward NTT](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/__asm_NTT.S), and [inverse NTT](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/__asm_iNTT.S), for instruction and lane-layout cross-checks.
- CRYSTALS-Dilithium, pinned [scalar NTT](https://github.com/pq-crystals/dilithium/blob/61b51a71701b8ae9f546a1e5d220e1950ed20d06/ref/ntt.c), for transform semantics independent of either SIMD implementation.
- Arm, [Advanced SIMD intrinsic reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html), and Intel, [Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html), for NEON and AVX2 instruction semantics.

The AArch64 implementation reuses the existing Shoup tables to derive signed `2^31/q` companions. When `4q < 2^31`, its outer stages keep values in `[0, 2q)` and canonicalize only at the packed forward tail or final inverse scaling. Larger moduli retain per-butterfly canonicalization. Becker et al. use wider transform-specific lazy signed bounds and reduce still less often. Variable pointwise products use widening Montgomery reduction because both operands vary at runtime.
