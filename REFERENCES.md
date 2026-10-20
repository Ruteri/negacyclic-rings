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

## Number Theoretic Transform

- J. M. Pollard, [The Fast Fourier Transform in a Finite Field](https://doi.org/10.1090/S0025-5718-1971-0301966-0) — The Number-Theoretic Transform (NTT): a radix-two Fast Fourier Transform (FFT) over `Z_q`, which needs a primitive root of unity of power-of-two order and so `q = 1 mod 2N`.
- P. Longa and M. Naehrig, [Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography](https://www.microsoft.com/en-us/research/publication/speeding-up-the-number-theoretic-transform-for-faster-ideal-lattice-based-cryptography/) — the transform structure. Section 2.2 multiplies in `Z_q[X]/(X^N + 1)` through a primitive `2N`-th root `psi` and folds the powers of `psi` into the twiddles, so a forward Cooley–Tukey NTT (standard order in, bit-reversed out) followed by an inverse Gentleman–Sande NTT (bit-reversed in, standard out) needs no bit-reversal permutation.
  - `params::find_psi32` / `find_psi64` find `psi` and `generate_ring32` / `generate_ring64` build the paper's tables: `ntt_table[i] = psi^bitrev(i)` is `Psi_rev`, `inv_ntt_table[i] = psi^-bitrev(i)` is `Psi_rev^-1`, and `n_inv` is the final `n^-1` scale. `scripts/paramgen.py` emits the same tables as constants.
  - `ntt_stages_scalar` (in `ntt32.rs` and `ntt64.rs`) is Algorithm 1 line for line: the code's `t` is the paper's `t` before halving and `ht` is the paper's `t`; the twiddle is `ntt_table[m + i]`.
  - `inv_ntt_stages_scalar` followed by `inv_ntt_final_scale_scalar` is Algorithm 2: `hm` is the paper's `h`, `dt` is `2t`, and the twiddle is `inv_ntt_table[hm + i]`.
  - `arithmetic::forward_ntt_i32` / `inverse_ntt_i32` are the same two algorithms with plain `%` reductions: a direct translation of the paper's pseudocode, with no precomputed companions or SIMD.
  - The AVX2 and NEON kernels run the same loops. They switch to in-register shuffles (`ntt_tail` / `inv_ntt_tail`) once the butterfly distance drops below the vector width.
  - The paper's other contribution, the `K-RED` reduction for `q = k·2^m + 1`, is not used here.
- D. Harvey, [Faster Arithmetic for Number-Theoretic Transforms](https://arxiv.org/abs/1205.2926) — Shoup multiplication by a known twiddle `W` with the precomputed companion `W' = floor(W·β/p)`.
  - `generate_ring32` / `generate_ring64` compute every `*_shoup` field as `floor(w·2^32/q)` or `floor(w·2^64/q)`, which is Harvey's `W'` with `β = 2^32` or `2^64`.
  - `shoup_mul` and the AVX2 `shoup_mul_v` are Algorithm 2, lines 5–7: `Q = high(W'·T)`, `Y = (W·T − Q·p) mod β`, then one conditional subtract. By Theorem 1, the value before that subtract lies in `[0, 2p)`.
  - The butterflies keep everything canonical, like Harvey's Algorithm 2 (Shoup's Number Theory Library, NTL). They do not use the redundant `[0, 2p)` / `[0, 4p)` representations of Algorithms 3 and 4. The forward butterfly has Algorithm 4's `(X + WY, X − WY)` shape, but the code reduces after every operation.
  - The crate requires `q < β/4` (`2q < 2^31` for `Ring32`, `q < 2^62` for `Ring64`). That is stricter than Harvey's `p < β/2` because the SIMD reductions use signed comparisons.
- Becker, Hwang, Kannwischer, Yang, and Yang, [Neon NTT: Faster Dilithium, Kyber, and Saber on Cortex-A72 and Apple M1](https://eprint.iacr.org/2021/986.pdf) — AArch64 twiddle multiplication. Section 3.1.2 defines "Barrett multiplication": a Barrett reduction (P. Barrett, [Implementing the Rivest Shamir and Adleman Public Key Encryption Algorithm on a Standard Digital Signal Processor](https://doi.org/10.1007/3-540-47721-7_24)) with the known factor folded into the precomputed quotient approximation.
  - `neon::becker_mul_v` is Algorithm 10: `mul`, `sqrdmulh` with the companion `round_even(w·2^32/q)/2`, then `mls`. The `mls` stays in inline assembly so that the compiler keeps the three-instruction sequence.
  - `neon::centered_twiddle` derives that companion from the Shoup table as `(w_shoup + 1) >> 1`. This equals `round(w·2^31/q)`, Algorithm 10's constant exactly, so no separate table is stored. Twiddles above `q/2` are recentered to `w − q`, with `2^31` subtracted from the companion.
  - Corollary 2 bounds the result by `|a|·q/2^32 + q/2`. Because inputs stay in `[0, 2q)` with `2q < 2^31`, that bound is below `q`, so the single `canonicalize` step afterwards is enough.
  - `ntt_neon_inner` / `inv_ntt_neon_inner` with `LAZY = true` (chosen when `4q < 2^31`) keep butterfly sums and differences in `[0, 2q)`. They canonicalize only before the packed forward tail and at the final `n^-1` scaling. Larger moduli canonicalize after every butterfly. Becker et al. use wider transform-specific signed bounds and reduce still less often.
  - Pinned [butterfly macros](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/macros_common.i), [forward NTT](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/__asm_NTT.S), and [inverse NTT](https://github.com/neon-ntt/neon-ntt/blob/a96c17dbe74ac7675c785a728396e216555c432b/dilithium3/ntt/__asm_iNTT.S) for instruction and lane-layout cross-checks.
- CRYSTALS-Dilithium, pinned [scalar NTT](https://github.com/pq-crystals/dilithium/blob/61b51a71701b8ae9f546a1e5d220e1950ed20d06/ref/ntt.c) — an independent reading of the same transform. `ntt` has the same loop nest as `ntt_stages_scalar`, using `zetas[++k]` and bit-reversed output. It differs in representation only: the twiddles are in Montgomery form, add and subtract are not reduced, and `invntt_tomont` walks `-zetas[--k]` and folds `n^-1` into a Montgomery constant.

## General

Not yet linked to specific code; see the section above for the target format.

- P. L. Montgomery, [Modular Multiplication Without Trial Division](https://doi.org/10.1090/S0025-5718-1985-0777282-X), for the Montgomery products used by pointwise multiplication and accumulation.
- Arm, [Advanced SIMD intrinsic reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html), and Intel, [Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html), for NEON and AVX2 instruction semantics.
