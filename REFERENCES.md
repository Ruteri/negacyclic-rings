# References

These sources cover the algorithms and implementation techniques used throughout the repository. Pinned implementation links are retained for cross-checking and later audit.

- J. M. Pollard, [The Fast Fourier Transform in a Finite Field](https://doi.org/10.1090/S0025-5718-1971-0301966-0), for finite-field Fourier transforms and radix-two NTTs.
- P. Longa and M. Naehrig, [Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography](https://www.microsoft.com/en-us/research/publication/speeding-up-the-number-theoretic-transform-for-faster-ideal-lattice-based-cryptography/), for negacyclic NTTs, signed representatives, and efficient butterfly organization.
- D. Harvey, [Faster Arithmetic for Number-Theoretic Transforms](https://arxiv.org/abs/1205.2926), especially Algorithm 4, for Shoup multiplication with the precomputed companion `floor(w·2^k/q)`.
- P. Barrett, [Implementing the Rivest Shamir and Adleman Public Key Encryption Algorithm on a Standard Digital Signal Processor](https://doi.org/10.1007/3-540-47721-7_24), for reciprocal-based modular reduction.
- P. L. Montgomery, [Modular Multiplication Without Trial Division](https://doi.org/10.1090/S0025-5718-1985-0777282-X), for the Montgomery products used by pointwise multiplication and accumulation.
- H. L. Garner, [The Residue Number System](https://doi.org/10.1109/TEC.1959.5219515), for mixed-radix reconstruction of generic RNS chains.
- Becker et al., [Neon NTT: Faster Dilithium, Kyber, and Saber](https://eprint.iacr.org/2021/986.pdf), for the AArch64 signed Barrett reduction using `SQRDMULH` and `MLS`, and for layered SIMD layouts.
- Becker et al., pinned [butterfly macros](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/macros_common.i), [forward NTT](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/__asm_NTT.S), and [inverse NTT](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/__asm_iNTT.S), for instruction and lane-layout cross-checks.
- CRYSTALS-Dilithium, pinned [scalar NTT](https://github.com/pq-crystals/dilithium/blob/61b51a71701b8ae9f546a1e5d220e1950ed20d06/ref/ntt.c), for transform semantics independent of either SIMD implementation.
- Arm, [Advanced SIMD intrinsic reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html), and Intel, [Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html), for NEON and AVX2 instruction semantics.

The AArch64 implementation reuses the existing Shoup tables to derive signed `2^31/q` companions. When `4q < 2^31`, its outer stages keep values in `[0, 2q)` and canonicalize only at the packed forward tail or final inverse scaling. Larger moduli retain per-butterfly canonicalization. Becker et al. use wider transform-specific lazy signed bounds and reduce still less often. Variable pointwise products use widening Montgomery reduction because both operands vary at runtime.
