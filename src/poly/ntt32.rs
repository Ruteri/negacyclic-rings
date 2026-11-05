//! 32-bit negacyclic NTT core for single-modulus and RNS arithmetic.
//! Same Shoup/Montgomery scheme as [`super::ntt64`] with R = 2^32, so every
//! product is a single 32x32->64 multiply instead of a u128.
//!
//! Requirements on the modulus: q odd prime, q ≡ 1 (mod 2N), and 2q < 2^31 —
//! the SIMD reductions use signed 32-bit compares, so a butterfly's `[0, 2q)`
//! output must stay non-negative as an i32.

#![cfg_attr(target_arch = "aarch64", allow(dead_code))]

pub struct Ring32<const N: usize> {
    /// The prime modulus: `q = 1 mod 2N`, `2q < 2^31`.
    pub q: u32,
    /// `-(q^-1) mod 2^32`, for Montgomery reduction.
    pub q_inv_neg: u32,
    /// `2^64 mod q`; a Montgomery product with it leaves the Montgomery domain.
    pub r2: u32,
    /// `psi^bitrev(i)`, the forward twiddles (Longa–Naehrig's `Psi_rev`).
    /// Entry 0 is unused.
    pub ntt_table: [u32; N],
    /// `floor(ntt_table[i]·2^32/q)`, Harvey's Shoup companion for each twiddle.
    pub ntt_table_shoup: [u32; N],
    /// `psi^-bitrev(i)`, the inverse twiddles (Longa–Naehrig's `Psi_rev^-1`).
    /// Entry 0 is unused.
    pub inv_ntt_table: [u32; N],
    /// `floor(inv_ntt_table[i]·2^32/q)`, the Shoup companion for each inverse twiddle.
    pub inv_ntt_table_shoup: [u32; N],
    /// `N^-1 mod q`, the final scale of the inverse NTT.
    pub n_inv: u32,
    /// `floor(n_inv·2^32/q)`, its Shoup companion.
    pub n_inv_shoup: u32,
}

// `Residues`/`Rns` live in `super::rns`; pulled in here (and, transitively,
// into the `avx2`/`neon` submodules below via `super::`) because
// `pointwise_mac` and the RNS-specific SIMD kernels are channel-indexed over
// them.
use super::rns::{Residues, Rns};

/// Forward NTT over canonical `[0, q)` values, in place. The output is
/// canonical and in bit-reversed order, which `inv_ntt` and the pointwise
/// functions expect.
pub fn ntt<const N: usize>(ring: &Ring32<N>, coefficients: &mut [u32; N]) {
    assert!(N.is_power_of_two(), "NTT degree must be a power of two");
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        unsafe { avx2::ntt_avx2(ring, coefficients) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if N >= 4 {
        unsafe { neon::ntt_neon(ring, coefficients) };
        return;
    }
    ntt_scalar(ring, coefficients);
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(in crate::poly) fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

fn ntt_scalar<const N: usize>(ring: &Ring32<N>, coefficients: &mut [u32; N]) {
    ntt_stages_scalar(ring, coefficients, 0, N);
}

fn ntt_stages_scalar<const N: usize>(
    ring: &Ring32<N>,
    coefficients: &mut [u32; N],
    start_l: usize,
    mut t: usize,
) {
    let q = ring.q;
    for l in start_l..N.trailing_zeros() as usize {
        let m = 1usize << l;
        let ht = t >> 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < m {
            let w = ring.ntt_table[m + i];
            let ws = ring.ntt_table_shoup[m + i];
            let j2 = j1 + ht;
            let mut j = j1;
            while j < j2 {
                let u = coefficients[j];
                let v = shoup_mul(coefficients[j + ht], w, ws, q);
                coefficients[j] = add_mod(u, v, q);
                coefficients[j + ht] = sub_mod(u, v, q);
                j += 1;
            }
            i += 1;
            j1 += t;
        }
        t = ht;
    }
}

#[inline(always)]
fn shoup_mul(a: u32, w: u32, w_shoup: u32, q: u32) -> u32 {
    // Estimated quotient: may be one less than floor(a·w / q), never more.
    let quotient = ((a as u64 * w_shoup as u64) >> 32) as u32;
    csub(a.wrapping_mul(w).wrapping_sub(quotient.wrapping_mul(q)), q)
}

/// Maps `x` in `[0, 2q)` to `[0, q)`: `x - q` if that is non-negative, else
/// `x`. Branchless: `2q < 2^31` means the difference's sign bit is a valid
/// predicate, so this is a shift and a mask rather than an unpredictable
/// conditional jump.
#[inline(always)]
fn csub(x: u32, q: u32) -> u32 {
    let difference = x.wrapping_sub(q);
    difference.wrapping_add(q & (((difference as i32) >> 31) as u32))
}

/// `(a + b) mod q` for canonical `a, b` in `[0, q)`.
#[inline(always)]
pub fn add_mod(a: u32, b: u32, q: u32) -> u32 {
    csub(a + b, q)
}

/// `(a - b) mod q` for canonical `a, b` in `[0, q)`.
#[inline(always)]
pub fn sub_mod(a: u32, b: u32, q: u32) -> u32 {
    let difference = a.wrapping_sub(b);
    difference.wrapping_add(q & (((difference as i32) >> 31) as u32))
}

/// Inverse NTT over canonical `[0, q)` values, in place (includes the final
/// `N^-1` scaling). The input must be in bit-reversed order, as `ntt` and the
/// pointwise functions produce; the output is in standard order.
pub fn inv_ntt<const N: usize>(ring: &Ring32<N>, coefficients: &mut [u32; N]) {
    assert!(N.is_power_of_two(), "NTT degree must be a power of two");
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        unsafe { avx2::inv_ntt_avx2(ring, coefficients) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if N >= 4 {
        unsafe { neon::inv_ntt_neon(ring, coefficients) };
        return;
    }
    inv_ntt_scalar(ring, coefficients);
}

fn inv_ntt_scalar<const N: usize>(ring: &Ring32<N>, coefficients: &mut [u32; N]) {
    // trailing_zeros returns log2(N), since N is a power of two.
    let stages = N.trailing_zeros() as usize;
    let distance = 1;
    inv_ntt_stages_scalar(ring, coefficients, stages, distance, N / distance);
    inv_ntt_final_scale_scalar(ring, coefficients);
}

fn inv_ntt_stages_scalar<const N: usize>(
    ring: &Ring32<N>,
    coefficients: &mut [u32; N],
    iters: usize,
    mut t: usize,
    mut m: usize,
) -> (usize, usize) {
    let q = ring.q;
    for _ in 0..iters {
        if m <= 1 {
            break;
        }
        let hm = m >> 1;
        let dt = t << 1;
        let mut i = 0usize;
        let mut j1 = 0usize;
        while i < hm {
            let w = ring.inv_ntt_table[hm + i];
            let ws = ring.inv_ntt_table_shoup[hm + i];
            let j2 = j1 + t;
            let mut j = j1;
            while j < j2 {
                let u = coefficients[j];
                let v = coefficients[j + t];
                coefficients[j] = add_mod(u, v, q);
                coefficients[j + t] = shoup_mul(sub_mod(u, v, q), w, ws, q);
                j += 1;
            }
            i += 1;
            j1 += dt;
        }
        t = dt;
        m = hm;
    }
    (t, m)
}

fn inv_ntt_final_scale_scalar<const N: usize>(ring: &Ring32<N>, coefficients: &mut [u32; N]) {
    for coefficient in coefficients.iter_mut() {
        *coefficient = shoup_mul(*coefficient, ring.n_inv, ring.n_inv_shoup, ring.q);
    }
}

/// Pointwise product `lhs[i]·rhs[i] mod q` of two canonical NTT-domain
/// polynomials; apply `inv_ntt` to the result to get their negacyclic product.
#[inline]
pub fn pointwise_mul<const N: usize>(ring: &Ring32<N>, lhs: &[u32; N], rhs: &[u32; N]) -> [u32; N] {
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        let mut out = [0u32; N];
        unsafe { avx2::pointwise_mul_avx2(ring, &mut out, lhs, rhs) };
        return out;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut out = [0u32; N];
        unsafe { neon::pointwise_mul_neon(ring, &mut out, lhs, rhs) };
        return out;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        core::array::from_fn(|i| mul_mod(lhs[i], rhs[i], ring))
    }
}

/// Modular product `a·b mod q`, not in Montgomery form, for canonical inputs.
#[inline(always)]
pub fn mul_mod<const N: usize>(a: u32, b: u32, ring: &Ring32<N>) -> u32 {
    mont_mul(mont_mul(a, b, ring), ring.r2, ring)
}

/// `a·b·2^-32 mod q` (Montgomery-domain product) for canonical inputs.
#[inline(always)]
pub fn mont_mul<const N: usize>(a: u32, b: u32, ring: &Ring32<N>) -> u32 {
    mont_reduce(a as u64 * b as u64, ring)
}

/// Montgomery reduction: `x < q·2^32` → `x·2^-32 mod q`, canonical in `[0, q)`.
#[inline(always)]
pub fn mont_reduce<const N: usize>(x: u64, ring: &Ring32<N>) -> u32 {
    let m = (x as u32).wrapping_mul(ring.q_inv_neg);
    let t = ((x + m as u64 * ring.q as u64) >> 32) as u32;
    csub(t, ring.q)
}

/// `out[i] = Σ_k a[k][i]·b[k][i] (mod q)` for canonical NTT-domain inputs;
/// apply `inv_ntt` to the result to get the sum of the negacyclic products.
pub fn pointwise_dot<const N: usize>(ring: &Ring32<N>, a: &[[u32; N]], b: &[[u32; N]]) -> [u32; N] {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        let mut out = [0u32; N];
        unsafe { avx2::pointwise_dot_avx2(ring, &mut out, a, b) };
        return out;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut out = [0u32; N];
        unsafe { neon::pointwise_dot_neon(ring, &mut out, a, b) };
        return out;
    }
    #[cfg(not(target_arch = "aarch64"))]
    pointwise_dot_scalar(ring, a, b)
}

// Accumulates Montgomery-domain products, converting out of Montgomery form
// once per coefficient at the end: mont(Σ mont(a·b) · r2) = Σ a·b.
fn pointwise_dot_scalar<const N: usize>(
    ring: &Ring32<N>,
    a: &[[u32; N]],
    b: &[[u32; N]],
) -> [u32; N] {
    let mut out = [0u32; N];
    for i in 0..N {
        let mut mont_acc = 0;
        for (av, bv) in a.iter().zip(b) {
            mont_acc = add_mod(mont_acc, mont_mul(av[i], bv[i], ring), ring.q);
        }
        out[i] = mont_mul(mont_acc, ring.r2, ring);
    }
    out
}

/// `acc[i] += Σ_k a[k][channel][i]·b[k][channel][i] (mod q)` for canonical
/// NTT-domain inputs; apply `inv_ntt` to `acc` to get the sum of the
/// negacyclic products. Takes the whole residue arrays and a channel index so
/// the RNS layer needs no per-channel reference vectors.
pub fn pointwise_mac<const N: usize, const CHANNEL_COUNT: usize>(
    ring: &Ring32<N>,
    acc: &mut [u32; N],
    a: &[&Residues<N, CHANNEL_COUNT>],
    b: &[&Residues<N, CHANNEL_COUNT>],
    channel: usize,
) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        unsafe { avx2::pointwise_mac_avx2(ring, acc, a, b, channel) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::pointwise_mac_neon(ring, acc, a, b, channel) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    pointwise_mac_scalar(ring, acc, a, b, channel);
}

// Accumulates Montgomery-domain products, converting out of Montgomery form
// once per coefficient at the end: mont(Σ mont(a·b) · r2) = Σ a·b.
fn pointwise_mac_scalar<const N: usize, const CHANNEL_COUNT: usize>(
    ring: &Ring32<N>,
    acc: &mut [u32; N],
    a: &[&Residues<N, CHANNEL_COUNT>],
    b: &[&Residues<N, CHANNEL_COUNT>],
    channel: usize,
) {
    let q = ring.q;
    let mut mont_acc = [0u32; N];
    for (av, bv) in a.iter().zip(b.iter()) {
        for i in 0..N {
            mont_acc[i] = add_mod(
                mont_acc[i],
                mont_mul(av[channel][i], bv[channel][i], ring),
                q,
            );
        }
    }
    for i in 0..N {
        acc[i] = add_mod(acc[i], mont_mul(mont_acc[i], ring.r2, ring), q);
    }
}

/// `lhs[i] = (lhs[i] + rhs[i]) mod q` for canonical inputs, in place; valid in
/// either the coefficient or the NTT domain.
#[inline]
pub fn add_assign<const N: usize>(ring: &Ring32<N>, lhs: &mut [u32; N], rhs: &[u32; N]) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = add_mod(*x, y, ring.q);
    }
}

/// `lhs[i] = (lhs[i] - rhs[i]) mod q` for canonical inputs, in place; valid in
/// either the coefficient or the NTT domain.
#[inline]
pub fn sub_assign<const N: usize>(ring: &Ring32<N>, lhs: &mut [u32; N], rhs: &[u32; N]) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = sub_mod(*x, y, ring.q);
    }
}

// AVX2 kernels (8 × u32 lanes). Shoup butterflies use `_mm256_mullo_epi32`;
// Montgomery products split the vector into even/odd u64 halves so
// `_mm256_mul_epu32` supplies the 32x32->64 products.
#[cfg(target_arch = "x86_64")]
pub(in crate::poly) mod avx2 {
    use super::{Residues, Ring32, Rns};
    use std::arch::x86_64::*;

    const LANES: usize = 8;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ntt_avx2<const N: usize>(ring: &Ring32<N>, coefficients: &mut [u32; N]) {
        let q_v = _mm256_set1_epi32(ring.q as i32);
        let mut t = N;
        for l in 0..(N.trailing_zeros() as usize - 3) {
            let m = 1usize << l;
            let ht = t >> 1;
            let mut i = 0usize;
            let mut j1 = 0usize;
            while i < m {
                let w_v = _mm256_set1_epi32(ring.ntt_table[m + i] as i32);
                let ws_v = _mm256_set1_epi32(ring.ntt_table_shoup[m + i] as i32);
                let mut j = j1;
                while j < j1 + ht {
                    let u = _mm256_loadu_si256(coefficients.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(coefficients.as_ptr().add(j + ht) as *const __m256i);
                    let v_red = shoup_mul_v(v, w_v, ws_v, q_v);
                    let new_u = add_mod_v(u, v_red, q_v);
                    let new_v = sub_mod_v(u, v_red, q_v);
                    _mm256_storeu_si256(coefficients.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(
                        coefficients.as_mut_ptr().add(j + ht) as *mut __m256i,
                        new_v,
                    );
                    j += LANES;
                }
                i += 1;
                j1 += t;
            }
            t = ht;
        }
        debug_assert_eq!(t, LANES);
        ntt_tail(ring, coefficients, q_v);
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shoup_mul_v(a: __m256i, w_v: __m256i, ws_v: __m256i, q_v: __m256i) -> __m256i {
        // Estimated quotient: may be one less than floor(a·w / q), never more.
        let quotient = mulhi32(a, ws_v);
        let r = _mm256_sub_epi32(
            _mm256_mullo_epi32(a, w_v),
            _mm256_mullo_epi32(quotient, q_v),
        );
        csub32(r, q_v)
    }

    /// High 32 bits of each of the 8 lanewise u32 products.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mulhi32(a: __m256i, b: __m256i) -> __m256i {
        let mask = lo_mask();
        // Even lanes: each product's high half, shifted down into the even word.
        let even = _mm256_srli_epi64(
            _mm256_mul_epu32(_mm256_and_si256(a, mask), _mm256_and_si256(b, mask)),
            32,
        );
        // Odd lanes: each product's high half already sits in the odd word.
        let odd = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
        _mm256_or_si256(even, _mm256_andnot_si256(mask, odd))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lo_mask() -> __m256i {
        _mm256_set1_epi64x(0xFFFF_FFFF)
    }

    /// `if v >= q { v - q } else { v }` for u32 lanes < 2^31.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn csub32(v: __m256i, q_v: __m256i) -> __m256i {
        let difference = _mm256_sub_epi32(v, q_v);
        let negative = _mm256_cmpgt_epi32(_mm256_setzero_si256(), difference);
        _mm256_blendv_epi8(difference, v, negative)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        csub32(_mm256_add_epi32(a, b), q_v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        let difference = _mm256_sub_epi32(a, b);
        let negative = _mm256_cmpgt_epi32(_mm256_setzero_si256(), difference);
        _mm256_add_epi32(difference, _mm256_and_si256(negative, q_v))
    }

    /// Forward stages with `ht = 4, 2, 1`. All three pair elements inside one
    /// aligned group of 8, so a group is loaded once, butterflied in-register
    /// with shuffles, and stored once — no scalar tail.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ntt_tail<const N: usize>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
        q_v: __m256i,
    ) {
        let (ma, mb, mc) = (N / 8, N / 4, N / 2);
        for g in 0..(N / LANES) {
            let ptr = coefficients.as_mut_ptr().add(g * LANES) as *mut __m256i;
            let mut x = _mm256_loadu_si256(ptr as *const __m256i);

            // ht = 4: one twiddle; each 128-bit half copied across the register.
            let w = _mm256_set1_epi32(ring.ntt_table[ma + g] as i32);
            let ws = _mm256_set1_epi32(ring.ntt_table_shoup[ma + g] as i32);
            let u = _mm256_permute2x128_si256::<0x00>(x, x);
            let v = shoup_mul_v(_mm256_permute2x128_si256::<0x11>(x, x), w, ws, q_v);
            x = _mm256_permute2x128_si256::<0x20>(add_mod_v(u, v, q_v), sub_mod_v(u, v, q_v));

            // ht = 2: twiddles per 4-group, operands split by 64-bit halves.
            let (i0, i1) = (mb + 2 * g, mb + 2 * g + 1);
            let w = twiddle_pairs(ring.ntt_table[i0], ring.ntt_table[i1]);
            let ws = twiddle_pairs(ring.ntt_table_shoup[i0], ring.ntt_table_shoup[i1]);
            let u = _mm256_unpacklo_epi64(x, x);
            let v = shoup_mul_v(_mm256_unpackhi_epi64(x, x), w, ws, q_v);
            x = _mm256_unpacklo_epi64(add_mod_v(u, v, q_v), sub_mod_v(u, v, q_v));

            // ht = 1: twiddles per adjacent pair, operands are even/odd lanes.
            let j = mc + 4 * g;
            let w = twiddle_quads(&ring.ntt_table[j..j + 4]);
            let ws = twiddle_quads(&ring.ntt_table_shoup[j..j + 4]);
            let u = _mm256_shuffle_epi32::<0xA0>(x);
            let v = shoup_mul_v(_mm256_shuffle_epi32::<0xF5>(x), w, ws, q_v);
            let s = add_mod_v(u, v, q_v);
            let d = sub_mod_v(u, v, q_v);
            x = _mm256_blend_epi32::<0x55>(_mm256_slli_epi64::<32>(d), s);

            _mm256_storeu_si256(ptr, x);
        }
    }

    /// `w0` in lanes 0,1 and `w1` in lanes 4,5 — the unused lanes are zero, and
    /// a zero twiddle with a zero Shoup companion yields zero.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn twiddle_pairs(w0: u32, w1: u32) -> __m256i {
        _mm256_set_epi32(0, 0, w1 as i32, w1 as i32, 0, 0, w0 as i32, w0 as i32)
    }

    /// `w[k]` in lane `2k`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn twiddle_quads(w: &[u32]) -> __m256i {
        // Reads exactly w[0..4]. A slice rather than &[u32; 4] keeps the callers,
        // which pass 4-element table slices, free of conversions.
        _mm256_set_epi32(
            0,
            w[3] as i32,
            0,
            w[2] as i32,
            0,
            w[1] as i32,
            0,
            w[0] as i32,
        )
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn inv_ntt_avx2<const N: usize>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
    ) {
        let q_v = _mm256_set1_epi32(ring.q as i32);
        inv_ntt_tail(ring, coefficients, q_v);
        let (mut t, mut m) = (LANES, N / LANES);
        while m > 1 {
            let hm = m >> 1;
            let dt = t << 1;
            let mut i = 0usize;
            let mut j1 = 0usize;
            while i < hm {
                let w_v = _mm256_set1_epi32(ring.inv_ntt_table[hm + i] as i32);
                let ws_v = _mm256_set1_epi32(ring.inv_ntt_table_shoup[hm + i] as i32);
                let mut j = j1;
                while j < j1 + t {
                    let u = _mm256_loadu_si256(coefficients.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(coefficients.as_ptr().add(j + t) as *const __m256i);
                    let new_u = add_mod_v(u, v, q_v);
                    let difference = sub_mod_v(u, v, q_v);
                    let new_v = shoup_mul_v(difference, w_v, ws_v, q_v);
                    _mm256_storeu_si256(coefficients.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(
                        coefficients.as_mut_ptr().add(j + t) as *mut __m256i,
                        new_v,
                    );
                    j += LANES;
                }
                i += 1;
                j1 += dt;
            }
            t = dt;
            m = hm;
        }
        let n_inv_v = _mm256_set1_epi32(ring.n_inv as i32);
        let n_inv_s_v = _mm256_set1_epi32(ring.n_inv_shoup as i32);
        for chunk in 0..(N / LANES) {
            let ptr = coefficients.as_mut_ptr().add(chunk * LANES) as *mut __m256i;
            let v = _mm256_loadu_si256(ptr as *const __m256i);
            _mm256_storeu_si256(ptr, shoup_mul_v(v, n_inv_v, n_inv_s_v, q_v));
        }
    }

    /// Inverse stages with `t = 1, 2, 4`, mirroring [`ntt_tail`].
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn inv_ntt_tail<const N: usize>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
        q_v: __m256i,
    ) {
        let (ma, mb, mc) = (N / 2, N / 4, N / 8);
        for g in 0..(N / LANES) {
            let ptr = coefficients.as_mut_ptr().add(g * LANES) as *mut __m256i;
            let mut x = _mm256_loadu_si256(ptr as *const __m256i);

            // t = 1: twiddles per adjacent pair, operands are even/odd lanes.
            let j = ma + 4 * g;
            let w = twiddle_quads(&ring.inv_ntt_table[j..j + 4]);
            let ws = twiddle_quads(&ring.inv_ntt_table_shoup[j..j + 4]);
            let u = _mm256_shuffle_epi32::<0xA0>(x);
            let v = _mm256_shuffle_epi32::<0xF5>(x);
            let new_u = add_mod_v(u, v, q_v);
            let new_v = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_blend_epi32::<0x55>(_mm256_slli_epi64::<32>(new_v), new_u);

            // t = 2: twiddles per 4-group, operands split by 64-bit halves.
            let (i0, i1) = (mb + 2 * g, mb + 2 * g + 1);
            let w = twiddle_pairs(ring.inv_ntt_table[i0], ring.inv_ntt_table[i1]);
            let ws = twiddle_pairs(ring.inv_ntt_table_shoup[i0], ring.inv_ntt_table_shoup[i1]);
            let u = _mm256_unpacklo_epi64(x, x);
            let v = _mm256_unpackhi_epi64(x, x);
            let new_u = add_mod_v(u, v, q_v);
            let new_v = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_unpacklo_epi64(new_u, new_v);

            // t = 4: one twiddle; each 128-bit half copied across the register.
            let w = _mm256_set1_epi32(ring.inv_ntt_table[mc + g] as i32);
            let ws = _mm256_set1_epi32(ring.inv_ntt_table_shoup[mc + g] as i32);
            let u = _mm256_permute2x128_si256::<0x00>(x, x);
            let v = _mm256_permute2x128_si256::<0x11>(x, x);
            let new_u = add_mod_v(u, v, q_v);
            let new_v = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_permute2x128_si256::<0x20>(new_u, new_v);

            _mm256_storeu_si256(ptr, x);
        }
    }

    /// `lhs[i]·rhs[i] mod q` via two Montgomery rounds, the second
    /// against `r2` to leave the Montgomery domain.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pointwise_mul_avx2<const N: usize>(
        ring: &Ring32<N>,
        out: &mut [u32; N],
        lhs: &[u32; N],
        rhs: &[u32; N],
    ) {
        let q64_v = _mm256_set1_epi64x(ring.q as i64);
        let qinv_v = _mm256_set1_epi64x(ring.q_inv_neg as i64);
        let r2_v = _mm256_set1_epi32(ring.r2 as i32);
        for chunk in 0..(N / LANES) {
            let offset = chunk * LANES;
            let a = _mm256_loadu_si256(lhs.as_ptr().add(offset) as *const __m256i);
            let b = _mm256_loadu_si256(rhs.as_ptr().add(offset) as *const __m256i);
            let t = mont_mul_v(a, b, q64_v, qinv_v);
            let c = mont_mul_v(t, r2_v, q64_v, qinv_v);
            _mm256_storeu_si256(out.as_mut_ptr().add(offset) as *mut __m256i, c);
        }
    }

    /// Montgomery product `a·b·2^-32 mod q` across 8 u32 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_v(a: __m256i, b: __m256i, q64_v: __m256i, qinv_v: __m256i) -> __m256i {
        let mask = lo_mask();
        // Even lanes: the low 32 bits of each u64.
        let even = mont_mul_half(
            _mm256_and_si256(a, mask),
            _mm256_and_si256(b, mask),
            q64_v,
            qinv_v,
        );
        // Odd lanes: shifted down from the high 32 bits, shifted back up when combined.
        let odd = mont_mul_half(
            _mm256_srli_epi64(a, 32),
            _mm256_srli_epi64(b, 32),
            q64_v,
            qinv_v,
        );
        _mm256_or_si256(even, _mm256_slli_epi64(odd, 32))
    }

    /// Montgomery product `a·b·2^-32 mod q` on 4 u64 lanes (one half of
    /// `mont_mul_v`): inputs in the low 32 bits of each lane, output likewise,
    /// canonical in `[0, q)` with the high 32 bits zero.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_half(a: __m256i, b: __m256i, q64_v: __m256i, qinv_v: __m256i) -> __m256i {
        let x = _mm256_mul_epu32(a, b);
        mont_reduce_half(x, q64_v, qinv_v)
    }

    // Montgomery reduction per u64 lane: `x < q·2^32` → `x·2^-32 mod q`, canonical in `[0, q)`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mont_reduce_half(x: __m256i, q64_v: __m256i, qinv_v: __m256i) -> __m256i {
        let m = _mm256_and_si256(_mm256_mul_epu32(x, qinv_v), lo_mask());
        let t = _mm256_srli_epi64(_mm256_add_epi64(x, _mm256_mul_epu32(m, q64_v)), 32);
        csub64(t, q64_v)
    }

    /// `if v >= q { v - q } else { v }` on 4 × u64 lanes holding values < 2^31.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn csub64(v: __m256i, q_v: __m256i) -> __m256i {
        let difference = _mm256_sub_epi64(v, q_v);
        let negative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), difference);
        _mm256_blendv_epi8(difference, v, negative)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pointwise_dot_avx2<const N: usize>(
        ring: &Ring32<N>,
        out: &mut [u32; N],
        a: &[[u32; N]],
        b: &[[u32; N]],
    ) {
        let q_v = _mm256_set1_epi32(ring.q as i32);
        let q64_v = _mm256_set1_epi64x(ring.q as i64);
        let qinv_v = _mm256_set1_epi64x(ring.q_inv_neg as i64);
        let r2_v = _mm256_set1_epi32(ring.r2 as i32);
        for chunk in 0..(N / LANES) {
            let offset = chunk * LANES;
            // Montgomery-domain accumulator: Σ_k a_k·b_k·2^-32 mod q.
            let mut mont_acc = _mm256_setzero_si256();
            for k in 0..a.len() {
                let av = _mm256_loadu_si256(a[k].as_ptr().add(offset) as *const __m256i);
                let bv = _mm256_loadu_si256(b[k].as_ptr().add(offset) as *const __m256i);
                let prod = mont_mul_v(av, bv, q64_v, qinv_v);
                mont_acc = add_mod_v(mont_acc, prod, q_v);
            }
            // One conversion out of Montgomery form per chunk: mont(Σ · r2) = Σ.
            let sum = mont_mul_v(mont_acc, r2_v, q64_v, qinv_v);
            _mm256_storeu_si256(out.as_mut_ptr().add(offset) as *mut __m256i, sum);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pointwise_mac_avx2<const N: usize, const CHANNEL_COUNT: usize>(
        ring: &Ring32<N>,
        acc: &mut [u32; N],
        a: &[&super::Residues<N, CHANNEL_COUNT>],
        b: &[&super::Residues<N, CHANNEL_COUNT>],
        channel: usize,
    ) {
        let q_v = _mm256_set1_epi32(ring.q as i32);
        let q64_v = _mm256_set1_epi64x(ring.q as i64);
        let qinv_v = _mm256_set1_epi64x(ring.q_inv_neg as i64);
        let r2_v = _mm256_set1_epi32(ring.r2 as i32);
        for chunk in 0..(N / LANES) {
            let offset = chunk * LANES;
            let acc_ptr = acc.as_mut_ptr().add(offset) as *mut __m256i;
            // Montgomery-domain accumulator: Σ_k a_k·b_k·2^-32 mod q.
            let mut mont_acc = _mm256_setzero_si256();
            for k in 0..a.len() {
                let av = _mm256_loadu_si256(a[k][channel].as_ptr().add(offset) as *const __m256i);
                let bv = _mm256_loadu_si256(b[k][channel].as_ptr().add(offset) as *const __m256i);
                let prod = mont_mul_v(av, bv, q64_v, qinv_v);
                mont_acc = add_mod_v(mont_acc, prod, q_v);
            }
            // One conversion out of Montgomery form per chunk: mont(Σ · r2) = Σ.
            let sum = mont_mul_v(mont_acc, r2_v, q64_v, qinv_v);
            let acc_v = _mm256_loadu_si256(acc_ptr as *const __m256i);
            _mm256_storeu_si256(acc_ptr, add_mod_v(acc_v, sum, q_v));
        }
    }

    #[target_feature(enable = "avx2")]
    pub(in crate::poly) unsafe fn reduce_channel_i64_avx2<const N: usize>(
        ring: &Ring32<N>,
        input: &[i64; N],
        output: &mut [u32; N],
    ) {
        for offset in (0..N).step_by(4) {
            let value = _mm256_loadu_si256(input.as_ptr().add(offset) as *const __m256i);
            let residue = reduce_i64x4(ring, value);
            let mut lanes = [0u64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, residue);
            for lane in 0..4 {
                output[offset + lane] = lanes[lane] as u32;
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn reduce_i64x4<const N: usize>(ring: &Ring32<N>, value: __m256i) -> __m256i {
        let mask = lo_mask();
        let q = _mm256_set1_epi64x(ring.q as i64);
        let q_inv = _mm256_set1_epi64x(ring.q_inv_neg as i64);
        let radix = _mm256_set1_epi64x(((1u64 << 32) % ring.q as u64) as i64);
        let r2 = _mm256_set1_epi64x(ring.r2 as i64);
        let sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), value);
        let magnitude = _mm256_sub_epi64(_mm256_xor_si256(value, sign), sign);
        let low = _mm256_and_si256(magnitude, mask);
        let high = _mm256_srli_epi64::<32>(magnitude);
        // |value| = high·2^32 + low ≡ low + high·(2^32 mod q) (mod q), and the sum is
        // below q·2^32, so mont_reduce_half accepts it; the r2 product undoes its 2^-32.
        let combined = _mm256_add_epi64(low, _mm256_mul_epu32(high, radix));
        let montgomery = mont_reduce_half(combined, q, q_inv);
        let residue = mont_mul_half(montgomery, r2, q, q_inv);
        let negated = csub64(_mm256_sub_epi64(q, residue), q);
        _mm256_blendv_epi8(residue, negated, sign)
    }

    #[target_feature(enable = "avx2")]
    pub(in crate::poly) unsafe fn reduce_centered_i32_avx2<const N: usize>(
        ring: &Ring32<N>,
        input: &[i32; N],
        output: &mut [u32; N],
    ) {
        let q = _mm256_set1_epi32(ring.q as i32);
        for offset in (0..N).step_by(LANES) {
            let value = _mm256_loadu_si256(input.as_ptr().add(offset) as *const __m256i);
            let negative = _mm256_cmpgt_epi32(_mm256_setzero_si256(), value);
            let magnitude = _mm256_abs_epi32(value);
            let residue = _mm256_blendv_epi8(value, _mm256_sub_epi32(q, magnitude), negative);
            _mm256_storeu_si256(output.as_mut_ptr().add(offset) as *mut __m256i, residue);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(in crate::poly) unsafe fn lift_centered_i64_avx2<const N: usize>(
        ring: &Rns<N, 2>,
        input: &Residues<N, 2>,
        output: &mut [i64; N],
    ) {
        let q0 = ring.channels[0].q as u64;
        let q1 = ring.channels[1].q;
        let product = q0 * q1 as u64;
        let q1v = _mm256_set1_epi64x(q1 as i64);
        let q1inv = _mm256_set1_epi64x(ring.channels[1].q_inv_neg as i64);
        let prefix = _mm256_set1_epi32(ring.prefix_inverses[1] as i32);
        let r2 = _mm256_set1_epi32(ring.channels[1].r2 as i32);
        // r0 < q0 < 2·q1, so r0 mod q1 is one conditional subtraction; otherwise
        // the loop below computes it with a scalar `%` per coefficient.
        if q0 < 2 * q1 as u64 {
            let q1_32 = _mm256_set1_epi32(q1 as i32);
            let q0_64 = _mm256_set1_epi64x(q0 as i64);
            for offset in (0..N).step_by(LANES) {
                let r0 = _mm256_loadu_si256(input[0].as_ptr().add(offset) as *const __m256i);
                let r1 = _mm256_loadu_si256(input[1].as_ptr().add(offset) as *const __m256i);
                let reduced_r0 = _mm256_blendv_epi8(
                    r0,
                    _mm256_sub_epi32(r0, q1_32),
                    _mm256_cmpgt_epi32(r0, _mm256_set1_epi32(q1 as i32 - 1)),
                );
                let difference = _mm256_sub_epi32(r1, reduced_r0);
                let delta = _mm256_add_epi32(
                    difference,
                    _mm256_and_si256(
                        _mm256_cmpgt_epi32(_mm256_setzero_si256(), difference),
                        q1_32,
                    ),
                );
                // delta·prefix mod q1, as in the scalar mul_mod: the r2 product cancels the 2^-32.
                let digit = mont_mul_v(mont_mul_v(delta, prefix, q1v, q1inv), r2, q1v, q1inv);
                let even = _mm256_add_epi64(
                    _mm256_and_si256(r0, lo_mask()),
                    _mm256_mul_epu32(digit, q0_64),
                );
                let odd = _mm256_add_epi64(
                    _mm256_srli_epi64::<32>(r0),
                    _mm256_mul_epu32(_mm256_srli_epi64::<32>(digit), q0_64),
                );
                let mut even_lanes = [0u64; 4];
                let mut odd_lanes = [0u64; 4];
                _mm256_storeu_si256(even_lanes.as_mut_ptr() as *mut __m256i, even);
                _mm256_storeu_si256(odd_lanes.as_mut_ptr() as *mut __m256i, odd);
                for lane in 0..4 {
                    for (index, value) in [
                        (2 * lane, even_lanes[lane]),
                        (2 * lane + 1, odd_lanes[lane]),
                    ] {
                        output[offset + index] =
                            value as i64 - (value > product / 2) as i64 * product as i64;
                    }
                }
            }
            return;
        }
        for offset in (0..N).step_by(LANES) {
            let delta: [u32; LANES] = core::array::from_fn(|lane| {
                let i = offset + lane;
                super::sub_mod(input[1][i], input[0][i] % q1, q1)
            });
            let delta = _mm256_loadu_si256(delta.as_ptr() as *const __m256i);
            let digit = mont_mul_v(mont_mul_v(delta, prefix, q1v, q1inv), r2, q1v, q1inv);
            let mut digits = [0u32; LANES];
            _mm256_storeu_si256(digits.as_mut_ptr() as *mut __m256i, digit);
            for lane in 0..LANES {
                let value = input[0][offset + lane] as u64 + q0 * digits[lane] as u64;
                output[offset + lane] =
                    value as i64 - (value > product / 2) as i64 * product as i64;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(in crate::poly) mod neon {
    use super::{mul_mod, Residues, Ring32, Rns};
    use core::arch::{aarch64::*, asm};

    const LANES: usize = 4;

    pub(super) unsafe fn ntt_neon<const N: usize>(ring: &Ring32<N>, coefficients: &mut [u32; N]) {
        // Lazy butterflies keep coefficients in [0, 2q) between stages, so sums stay
        // below 4q; that must fit a signed 32-bit lane. Otherwise reduce fully each stage.
        if (ring.q as u64) * 4 < (1u64 << 31) {
            ntt_neon_inner::<N, true>(ring, coefficients);
        } else {
            ntt_neon_inner::<N, false>(ring, coefficients);
        }
    }

    unsafe fn ntt_neon_inner<const N: usize, const LAZY: bool>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
    ) {
        let q_v = vdupq_n_s32(ring.q as i32);
        let two_q_v = vdupq_n_s32((ring.q * 2) as i32);
        let mut t = N;
        for l in 0..(N.trailing_zeros() as usize - 2) {
            let m = 1usize << l;
            let ht = t >> 1;
            let mut j1 = 0usize;
            for i in 0..m {
                let (w, scaled) =
                    twiddle_v(ring.ntt_table[m + i], ring.ntt_table_shoup[m + i], ring.q);
                let mut j = j1;
                while j < j1 + ht {
                    let u = vreinterpretq_s32_u32(vld1q_u32(coefficients.as_ptr().add(j)));
                    let v = vreinterpretq_s32_u32(vld1q_u32(coefficients.as_ptr().add(j + ht)));
                    let v = becker_mul_v(v, w, scaled, q_v);
                    // Lazy: u < 2q and v < q (becker_mul_v is canonical), so u + v < 3q
                    // and u − v > −q; one correction by 2q puts both in [0, 2q).
                    let (sum, diff) = if LAZY {
                        (
                            sub_if_ge(vaddq_s32(u, v), two_q_v),
                            add_if_negative(vsubq_s32(u, v), two_q_v),
                        )
                    } else {
                        (add_mod_v(u, v, q_v), sub_mod_v(u, v, q_v))
                    };
                    vst1q_u32(coefficients.as_mut_ptr().add(j), vreinterpretq_u32_s32(sum));
                    vst1q_u32(
                        coefficients.as_mut_ptr().add(j + ht),
                        vreinterpretq_u32_s32(diff),
                    );
                    j += LANES;
                }
                j1 += t;
            }
            t = ht;
        }
        debug_assert_eq!(t, LANES);
        // Lazy stages leave coefficients in [0, 2q); the tail needs them in [0, q).
        if LAZY {
            for offset in (0..N).step_by(LANES) {
                let ptr = coefficients.as_mut_ptr().add(offset);
                let v = vreinterpretq_s32_u32(vld1q_u32(ptr));
                vst1q_u32(ptr, vreinterpretq_u32_s32(sub_if_ge(v, q_v)));
            }
        }
        ntt_tail(ring, coefficients, q_v);
    }

    #[inline(always)]
    unsafe fn twiddle_v(w: u32, w_shoup: u32, q: u32) -> (int32x4_t, int32x4_t) {
        let (w, scaled) = centered_twiddle(w, w_shoup, q);
        (vdupq_n_s32(w), vdupq_n_s32(scaled))
    }

    // Becker et al. Algorithm 10's companion round(w·2^31/q), derived from the Shoup
    // table. Twiddles above q/2 become w − q, so the companion fits a signed 32-bit lane.
    #[inline(always)]
    fn centered_twiddle(w: u32, w_shoup: u32, q: u32) -> (i32, i32) {
        let mut centered = w as i64;
        let mut scaled = ((w_shoup as u64 + 1) >> 1) as i64;
        if w > q / 2 {
            centered -= q as i64;
            scaled -= 1i64 << 31;
        }
        (centered as i32, scaled as i32)
    }

    #[inline(always)]
    unsafe fn becker_mul_v(
        a: int32x4_t,
        w_v: int32x4_t,
        scaled_v: int32x4_t,
        q_v: int32x4_t,
    ) -> int32x4_t {
        let quotient = vqrdmulhq_s32(a, scaled_v);
        let mut product = vmulq_s32(a, w_v);
        // Inline asm, not vmlsq_s32, so the compiler keeps a single mls after sqrdmulh.
        asm!(
            "mls {product:v}.4s, {quotient:v}.4s, {q:v}.4s",
            product = inout(vreg) product,
            quotient = in(vreg) quotient,
            q = in(vreg) q_v,
            options(pure, nomem, nostack)
        );
        canonicalize(product, q_v)
    }

    // [−q, 2q) → [0, q): at most one correction by q in each direction.
    #[inline(always)]
    unsafe fn canonicalize(v: int32x4_t, q_v: int32x4_t) -> int32x4_t {
        let zero = vdupq_n_s32(0);
        let add_q = vandq_u32(vcltq_s32(v, zero), vreinterpretq_u32_s32(q_v));
        let v = vaddq_s32(v, vreinterpretq_s32_u32(add_q));
        let sub_q = vandq_u32(vcgeq_s32(v, q_v), vreinterpretq_u32_s32(q_v));
        vsubq_s32(v, vreinterpretq_s32_u32(sub_q))
    }

    #[inline(always)]
    unsafe fn sub_if_ge(v: int32x4_t, bound: int32x4_t) -> int32x4_t {
        let subtract = vandq_u32(vcgeq_s32(v, bound), vreinterpretq_u32_s32(bound));
        vsubq_s32(v, vreinterpretq_s32_u32(subtract))
    }

    #[inline(always)]
    unsafe fn add_if_negative(v: int32x4_t, bound: int32x4_t) -> int32x4_t {
        let add = vandq_u32(vcltq_s32(v, vdupq_n_s32(0)), vreinterpretq_u32_s32(bound));
        vaddq_s32(v, vreinterpretq_s32_u32(add))
    }

    #[inline(always)]
    unsafe fn add_mod_v(a: int32x4_t, b: int32x4_t, q_v: int32x4_t) -> int32x4_t {
        canonicalize(vaddq_s32(a, b), q_v)
    }

    #[inline(always)]
    unsafe fn sub_mod_v(a: int32x4_t, b: int32x4_t, q_v: int32x4_t) -> int32x4_t {
        canonicalize(vsubq_s32(a, b), q_v)
    }

    #[inline]
    unsafe fn ntt_tail<const N: usize>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
        q_v: int32x4_t,
    ) {
        let (m2, m1) = (N / 4, N / 2);
        for g in 0..(N / LANES) {
            let ptr = coefficients.as_mut_ptr().add(g * LANES);
            let mut x = vreinterpretq_s32_u32(vld1q_u32(ptr));

            // ht = 2: one twiddle; each 64-bit half copied across the register.
            let (w, scaled) =
                twiddle_v(ring.ntt_table[m2 + g], ring.ntt_table_shoup[m2 + g], ring.q);
            let u = vcombine_s32(vget_low_s32(x), vget_low_s32(x));
            let v = becker_mul_v(
                vcombine_s32(vget_high_s32(x), vget_high_s32(x)),
                w,
                scaled,
                q_v,
            );
            x = vcombine_s32(
                vget_low_s32(add_mod_v(u, v, q_v)),
                vget_low_s32(sub_mod_v(u, v, q_v)),
            );

            // ht = 1: twiddles per adjacent pair, operands are even/odd lanes.
            let i = m1 + 2 * g;
            let (w, scaled) = twiddle_pairs(
                ring.ntt_table[i],
                ring.ntt_table_shoup[i],
                ring.ntt_table[i + 1],
                ring.ntt_table_shoup[i + 1],
                ring.q,
            );
            let u = vuzp1q_s32(x, x);
            let v = becker_mul_v(vuzp2q_s32(x, x), w, scaled, q_v);
            x = vzip1q_s32(add_mod_v(u, v, q_v), sub_mod_v(u, v, q_v));

            vst1q_u32(ptr, vreinterpretq_u32_s32(x));
        }
    }

    // [w0, w1, w0, w1] and their companions: after the tail's even/odd split, pair k is
    // in lane k; lanes 2 and 3 repeat 0 and 1 and are discarded.
    #[inline(always)]
    unsafe fn twiddle_pairs(
        w0: u32,
        ws0: u32,
        w1: u32,
        ws1: u32,
        q: u32,
    ) -> (int32x4_t, int32x4_t) {
        let (w0, s0) = centered_twiddle(w0, ws0, q);
        let (w1, s1) = centered_twiddle(w1, ws1, q);
        let w = vset_lane_s32::<1>(w1, vdup_n_s32(w0));
        let s = vset_lane_s32::<1>(s1, vdup_n_s32(s0));
        (vcombine_s32(w, w), vcombine_s32(s, s))
    }

    pub(super) unsafe fn inv_ntt_neon<const N: usize>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
    ) {
        // Lazy butterflies keep coefficients in [0, 2q) between stages, so sums stay
        // below 4q; that must fit a signed 32-bit lane. Otherwise reduce fully each stage.
        if (ring.q as u64) * 4 < (1u64 << 31) {
            inv_ntt_neon_inner::<N, true>(ring, coefficients);
        } else {
            inv_ntt_neon_inner::<N, false>(ring, coefficients);
        }
    }

    unsafe fn inv_ntt_neon_inner<const N: usize, const LAZY: bool>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
    ) {
        let q_v = vdupq_n_s32(ring.q as i32);
        let two_q_v = vdupq_n_s32((ring.q * 2) as i32);
        inv_ntt_tail(ring, coefficients, q_v);
        let (mut t, mut m) = (LANES, N / LANES);
        while m > 1 {
            let hm = m >> 1;
            let dt = t << 1;
            let mut j1 = 0usize;
            for i in 0..hm {
                let (w, scaled) = twiddle_v(
                    ring.inv_ntt_table[hm + i],
                    ring.inv_ntt_table_shoup[hm + i],
                    ring.q,
                );
                let mut j = j1;
                while j < j1 + t {
                    let u = vreinterpretq_s32_u32(vld1q_u32(coefficients.as_ptr().add(j)));
                    let v = vreinterpretq_s32_u32(vld1q_u32(coefficients.as_ptr().add(j + t)));
                    let (sum, diff) = if LAZY {
                        (
                            sub_if_ge(vaddq_s32(u, v), two_q_v),
                            add_if_negative(vsubq_s32(u, v), two_q_v),
                        )
                    } else {
                        (add_mod_v(u, v, q_v), sub_mod_v(u, v, q_v))
                    };
                    vst1q_u32(coefficients.as_mut_ptr().add(j), vreinterpretq_u32_s32(sum));
                    let d = becker_mul_v(diff, w, scaled, q_v);
                    vst1q_u32(
                        coefficients.as_mut_ptr().add(j + t),
                        vreinterpretq_u32_s32(d),
                    );
                    j += LANES;
                }
                j1 += dt;
            }
            t = dt;
            m = hm;
        }
        let (n_inv, scaled) = twiddle_v(ring.n_inv, ring.n_inv_shoup, ring.q);
        for offset in (0..N).step_by(LANES) {
            let ptr = coefficients.as_mut_ptr().add(offset);
            let v = vreinterpretq_s32_u32(vld1q_u32(ptr));
            vst1q_u32(
                ptr,
                vreinterpretq_u32_s32(becker_mul_v(v, n_inv, scaled, q_v)),
            );
        }
    }

    #[inline]
    unsafe fn inv_ntt_tail<const N: usize>(
        ring: &Ring32<N>,
        coefficients: &mut [u32; N],
        q_v: int32x4_t,
    ) {
        let (m1, m2) = (N / 2, N / 4);
        for g in 0..(N / LANES) {
            let ptr = coefficients.as_mut_ptr().add(g * LANES);
            let mut x = vreinterpretq_s32_u32(vld1q_u32(ptr));

            let i = m1 + 2 * g;
            let (w, scaled) = twiddle_pairs(
                ring.inv_ntt_table[i],
                ring.inv_ntt_table_shoup[i],
                ring.inv_ntt_table[i + 1],
                ring.inv_ntt_table_shoup[i + 1],
                ring.q,
            );
            let u = vuzp1q_s32(x, x);
            let v = vuzp2q_s32(x, x);
            x = vzip1q_s32(
                add_mod_v(u, v, q_v),
                becker_mul_v(sub_mod_v(u, v, q_v), w, scaled, q_v),
            );

            let (w, scaled) = twiddle_v(
                ring.inv_ntt_table[m2 + g],
                ring.inv_ntt_table_shoup[m2 + g],
                ring.q,
            );
            let u = vcombine_s32(vget_low_s32(x), vget_low_s32(x));
            let v = vcombine_s32(vget_high_s32(x), vget_high_s32(x));
            let sum = add_mod_v(u, v, q_v);
            let diff = becker_mul_v(sub_mod_v(u, v, q_v), w, scaled, q_v);
            x = vcombine_s32(vget_low_s32(sum), vget_low_s32(diff));

            vst1q_u32(ptr, vreinterpretq_u32_s32(x));
        }
    }

    pub(super) unsafe fn pointwise_mul_neon<const N: usize>(
        ring: &Ring32<N>,
        out: &mut [u32; N],
        lhs: &[u32; N],
        rhs: &[u32; N],
    ) {
        let r2 = vdupq_n_u32(ring.r2);
        let vector_end = N / LANES * LANES;
        for offset in (0..vector_end).step_by(LANES) {
            let a = vld1q_u32(lhs.as_ptr().add(offset));
            let b = vld1q_u32(rhs.as_ptr().add(offset));
            let t = mont_mul_v(a, b, ring.q, ring.q_inv_neg);
            vst1q_u32(
                out.as_mut_ptr().add(offset),
                mont_mul_v(t, r2, ring.q, ring.q_inv_neg),
            );
        }
        for i in vector_end..N {
            out[i] = mul_mod(lhs[i], rhs[i], ring);
        }
    }

    #[inline(always)]
    unsafe fn mont_mul_v(a: uint32x4_t, b: uint32x4_t, q: u32, q_inv: u32) -> uint32x4_t {
        let q2 = vdup_n_u32(q);
        let qi2 = vdup_n_u32(q_inv);
        let lo = mont_mul_half(vget_low_u32(a), vget_low_u32(b), q2, qi2);
        let hi = mont_mul_half(vget_high_u32(a), vget_high_u32(b), q2, qi2);
        let v = vcombine_u32(lo, hi);
        vreinterpretq_u32_s32(canonicalize(
            vreinterpretq_s32_u32(v),
            vdupq_n_s32(q as i32),
        ))
    }

    #[inline(always)]
    unsafe fn mont_mul_half(
        a: uint32x2_t,
        b: uint32x2_t,
        q_v: uint32x2_t,
        q_inv_v: uint32x2_t,
    ) -> uint32x2_t {
        let x = vmull_u32(a, b);
        mont_reduce_half(x, q_v, q_inv_v)
    }

    #[inline(always)]
    unsafe fn mont_reduce_half(x: uint64x2_t, q_v: uint32x2_t, q_inv_v: uint32x2_t) -> uint32x2_t {
        let m = vmul_u32(vmovn_u64(x), q_inv_v);
        let t = vshrq_n_u64::<32>(vaddq_u64(x, vmull_u32(m, q_v)));
        vmovn_u64(t)
    }

    pub(super) unsafe fn pointwise_dot_neon<const N: usize>(
        ring: &Ring32<N>,
        out: &mut [u32; N],
        a: &[[u32; N]],
        b: &[[u32; N]],
    ) {
        let q_v = vdupq_n_s32(ring.q as i32);
        let r2 = vdupq_n_u32(ring.r2);
        let vector_end = N / LANES * LANES;
        for offset in (0..vector_end).step_by(LANES) {
            let mut mont_acc = vdupq_n_s32(0);
            for k in 0..a.len() {
                let av = vld1q_u32(a[k].as_ptr().add(offset));
                let bv = vld1q_u32(b[k].as_ptr().add(offset));
                let prod = mont_mul_v(av, bv, ring.q, ring.q_inv_neg);
                mont_acc = add_mod_v(mont_acc, vreinterpretq_s32_u32(prod), q_v);
            }
            vst1q_u32(
                out.as_mut_ptr().add(offset),
                mont_mul_v(vreinterpretq_u32_s32(mont_acc), r2, ring.q, ring.q_inv_neg),
            );
        }
        for i in vector_end..N {
            let mut sum = 0;
            for k in 0..a.len() {
                sum = super::add_mod(sum, mul_mod(a[k][i], b[k][i], ring), ring.q);
            }
            out[i] = sum;
        }
    }

    pub(super) unsafe fn pointwise_mac_neon<const N: usize, const CHANNEL_COUNT: usize>(
        ring: &Ring32<N>,
        acc: &mut [u32; N],
        a: &[&Residues<N, CHANNEL_COUNT>],
        b: &[&Residues<N, CHANNEL_COUNT>],
        channel: usize,
    ) {
        let q_v = vdupq_n_s32(ring.q as i32);
        let r2 = vdupq_n_u32(ring.r2);
        let vector_end = N / LANES * LANES;
        for offset in (0..vector_end).step_by(LANES) {
            let mut mont_acc = vdupq_n_s32(0);
            for k in 0..a.len() {
                let av = vld1q_u32(a[k][channel].as_ptr().add(offset));
                let bv = vld1q_u32(b[k][channel].as_ptr().add(offset));
                let prod = mont_mul_v(av, bv, ring.q, ring.q_inv_neg);
                mont_acc = add_mod_v(mont_acc, vreinterpretq_s32_u32(prod), q_v);
            }
            let sum = mont_mul_v(vreinterpretq_u32_s32(mont_acc), r2, ring.q, ring.q_inv_neg);
            let acc_v = vreinterpretq_s32_u32(vld1q_u32(acc.as_ptr().add(offset)));
            vst1q_u32(
                acc.as_mut_ptr().add(offset),
                vreinterpretq_u32_s32(add_mod_v(acc_v, vreinterpretq_s32_u32(sum), q_v)),
            );
        }
        for i in vector_end..N {
            let mut sum = 0;
            for k in 0..a.len() {
                sum = super::add_mod(
                    sum,
                    mul_mod(a[k][channel][i], b[k][channel][i], ring),
                    ring.q,
                );
            }
            acc[i] = super::add_mod(acc[i], sum, ring.q);
        }
    }

    pub(in crate::poly) unsafe fn reduce_channel_i64_neon<const N: usize>(
        ring: &Ring32<N>,
        input: &[i64; N],
        output: &mut [u32; N],
    ) {
        for offset in (0..N).step_by(2) {
            let value = vld1q_s64(input.as_ptr().add(offset));
            vst1_u32(output.as_mut_ptr().add(offset), reduce_i64x2(ring, value));
        }
    }

    #[inline(always)]
    unsafe fn reduce_i64x2<const N: usize>(ring: &Ring32<N>, value: int64x2_t) -> uint32x2_t {
        let q = vdup_n_u32(ring.q);
        let q_inv = vdup_n_u32(ring.q_inv_neg);
        let radix = vdup_n_u32(((1u64 << 32) % ring.q as u64) as u32);
        let r2 = vdup_n_u32(ring.r2);
        let negative64 = vcltq_s64(value, vdupq_n_s64(0));
        let magnitude = vreinterpretq_u64_s64(vabsq_s64(value));
        let low = vmovn_u64(magnitude);
        let high = vmovn_u64(vshrq_n_u64::<32>(magnitude));
        let combined = vaddq_u64(vmovl_u32(low), vmull_u32(high, radix));
        let montgomery = mont_reduce_half(combined, q, q_inv);
        let residue = mont_mul_half(montgomery, r2, q, q_inv);
        let residue = vsub_u32(residue, vand_u32(vcge_u32(residue, q), q));
        let negated = vsub_u32(q, residue);
        let negated = vsub_u32(negated, vand_u32(vcge_u32(negated, q), q));
        let negative = vmovn_u64(negative64);
        vbsl_u32(negative, negated, residue)
    }

    pub(in crate::poly) unsafe fn reduce_centered_i32_neon<const N: usize>(
        ring: &Ring32<N>,
        input: &[i32; N],
        output: &mut [u32; N],
    ) {
        let q = vdupq_n_u32(ring.q);
        for offset in (0..N).step_by(LANES) {
            let value = vld1q_s32(input.as_ptr().add(offset));
            let negative = vcltq_s32(value, vdupq_n_s32(0));
            let magnitude = vreinterpretq_u32_s32(vabsq_s32(value));
            vst1q_u32(
                output.as_mut_ptr().add(offset),
                vbslq_u32(
                    negative,
                    vsubq_u32(q, magnitude),
                    vreinterpretq_u32_s32(value),
                ),
            );
        }
    }

    pub(in crate::poly) unsafe fn lift_centered_i64_neon<const N: usize>(
        ring: &Rns<N, 2>,
        input: &Residues<N, 2>,
        output: &mut [i64; N],
    ) {
        let q0 = ring.channels[0].q as u64;
        let q1 = ring.channels[1].q;
        let product = q0 * q1 as u64;
        let prefix = vdupq_n_u32(ring.prefix_inverses[1]);
        let r2 = vdupq_n_u32(ring.channels[1].r2);
        if q0 < 2 * q1 as u64 {
            let q1v = vdupq_n_u32(q1);
            for offset in (0..N).step_by(LANES) {
                let r0 = vld1q_u32(input[0].as_ptr().add(offset));
                let r1 = vld1q_u32(input[1].as_ptr().add(offset));
                let reduced_r0 = vsubq_u32(r0, vandq_u32(vcgeq_u32(r0, q1v), q1v));
                let difference = vsubq_u32(r1, reduced_r0);
                let delta = vaddq_u32(difference, vandq_u32(vcltq_u32(r1, reduced_r0), q1v));
                let digit = mont_mul_v(
                    mont_mul_v(delta, prefix, q1, ring.channels[1].q_inv_neg),
                    r2,
                    q1,
                    ring.channels[1].q_inv_neg,
                );
                let low = vaddq_u64(
                    vmovl_u32(vget_low_u32(r0)),
                    vmull_u32(vget_low_u32(digit), vdup_n_u32(q0 as u32)),
                );
                let high = vaddq_u64(
                    vmovl_u32(vget_high_u32(r0)),
                    vmull_u32(vget_high_u32(digit), vdup_n_u32(q0 as u32)),
                );
                let half = vdupq_n_u64(product / 2);
                let product_v = vdupq_n_u64(product);
                let low = vsubq_u64(low, vandq_u64(vcgtq_u64(low, half), product_v));
                let high = vsubq_u64(high, vandq_u64(vcgtq_u64(high, half), product_v));
                vst1q_s64(output.as_mut_ptr().add(offset), vreinterpretq_s64_u64(low));
                vst1q_s64(
                    output.as_mut_ptr().add(offset + 2),
                    vreinterpretq_s64_u64(high),
                );
            }
            return;
        }
        for offset in (0..N).step_by(LANES) {
            let delta: [u32; LANES] = core::array::from_fn(|lane| {
                let i = offset + lane;
                super::sub_mod(input[1][i], input[0][i] % q1, q1)
            });
            let delta = vld1q_u32(delta.as_ptr());
            let digit = mont_mul_v(
                mont_mul_v(delta, prefix, q1, ring.channels[1].q_inv_neg),
                r2,
                q1,
                ring.channels[1].q_inv_neg,
            );
            let mut digits = [0u32; LANES];
            vst1q_u32(digits.as_mut_ptr(), digit);
            for lane in 0..LANES {
                let value = input[0][offset + lane] as u64 + q0 * digits[lane] as u64;
                output[offset + lane] =
                    value as i64 - (value > product / 2) as i64 * product as i64;
            }
        }
    }
}
