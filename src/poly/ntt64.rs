//! Shared 64-bit negacyclic NTT core, used by the KAHE ring (q ≈ 2^48.3), the
//! CS auxiliary convolution ring (p ≈ 2^58) and the digest ring (q ≈ 2^61).
//! Works on `[u64; N]` buffers
//! with values canonical in `[0, q)`.
//!
//! Butterflies use Shoup multiplication: for a precomputed twiddle `w` with
//! companion `w_shoup = floor(w·2^64/q)`,
//!   q̂ = high64(a · w_shoup);  r = a·w − q̂·q  ∈ [0, 2q)
//! computed entirely in wrapping 64-bit arithmetic (the true value fits since
//! r < 2q < 2^63). Pointwise products (no precomputed companion) use
//! Montgomery reduction with R = 2^64.
//!
//! Requirements on the modulus: q odd prime, q < 2^62, q ≡ 1 (mod 2N).

pub struct Ring64<const N: usize> {
    pub q: u64,
    /// `-(q^-1) mod 2^64`.
    pub q_inv_neg: u64,
    /// `2^128 mod q` (Montgomery conversion factor).
    pub r2: u64,
    pub ntt_table: [u64; N],
    pub ntt_table_shoup: [u64; N],
    pub inv_ntt_table: [u64; N],
    pub inv_ntt_table_shoup: [u64; N],
    pub n_inv: u64,
    pub n_inv_shoup: u64,
}

/// `x - q` if that is non-negative, else `x`. Branchless: `q < 2^62` means the
/// difference's sign bit is a valid predicate, so this is a shift and a mask
/// rather than an unpredictable conditional jump.
#[inline(always)]
fn csub_s(x: u64, q: u64) -> u64 {
    let d = x.wrapping_sub(q);
    d.wrapping_add(q & (((d as i64) >> 63) as u64))
}

#[inline(always)]
pub fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    csub_s(a + b, q)
}

#[inline(always)]
pub fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    let d = a.wrapping_sub(b);
    d.wrapping_add(q & (((d as i64) >> 63) as u64))
}

#[inline(always)]
fn shoup_mul(a: u64, w: u64, w_shoup: u64, q: u64) -> u64 {
    let qq = ((a as u128 * w_shoup as u128) >> 64) as u64;
    csub_s(a.wrapping_mul(w).wrapping_sub(qq.wrapping_mul(q)), q)
}

/// Montgomery reduction: `x < q·2^64` → `x·2^-64 mod q`, canonical.
#[inline(always)]
pub fn mont_reduce<const N: usize>(x: u128, ring: &Ring64<N>) -> u64 {
    let m = (x as u64).wrapping_mul(ring.q_inv_neg);
    let t = ((x + m as u128 * ring.q as u128) >> 64) as u64;
    csub_s(t, ring.q)
}

/// `a·b·2^-64 mod q` (Montgomery-domain product).
#[inline(always)]
pub fn mont_mul<const N: usize>(a: u64, b: u64, ring: &Ring64<N>) -> u64 {
    mont_reduce(a as u128 * b as u128, ring)
}

/// Canonical modular product `a·b mod q` for canonical inputs.
#[inline(always)]
pub fn mul_mod<const N: usize>(a: u64, b: u64, ring: &Ring64<N>) -> u64 {
    mont_mul(mont_mul(a, b, ring), ring.r2, ring)
}

#[inline]
pub fn add_assign<const N: usize>(ring: &Ring64<N>, lhs: &mut [u64; N], rhs: &[u64; N]) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = add_mod(*x, y, ring.q);
    }
}

#[inline]
pub fn sub_assign<const N: usize>(ring: &Ring64<N>, lhs: &mut [u64; N], rhs: &[u64; N]) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = sub_mod(*x, y, ring.q);
    }
}

/// Scalar on purpose: x86-64 has no 64x64->128 vector multiply, so the AVX2
/// Montgomery emulation loses to a scalar `mulx` here (measured 2.2x slower).
#[inline]
pub fn pointwise_mul<const N: usize>(ring: &Ring64<N>, lhs: &[u64; N], rhs: &[u64; N]) -> [u64; N] {
    core::array::from_fn(|i| mul_mod(lhs[i], rhs[i], ring))
}

// ---------------------------------------------------------------
// Scalar NTT.
// ---------------------------------------------------------------

/// Forward butterfly stages starting at `start_l` with `t = N >> start_l`.
/// Inputs canonical `[0, q)`.
fn ntt_stages_scalar<const N: usize>(
    ring: &Ring64<N>,
    p: &mut [u64; N],
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
                let u = p[j];
                let v = shoup_mul(p[j + ht], w, ws, q);
                p[j] = add_mod(u, v, q);
                p[j + ht] = sub_mod(u, v, q);
                j += 1;
            }
            i += 1;
            j1 += t;
        }
        t = ht;
    }
}

/// Inverse butterfly stages, `iters` iterations from `(t, m)`. Returns the
/// updated `(t, m)` for resumption. Inputs canonical `[0, q)`.
fn inv_ntt_stages_scalar<const N: usize>(
    ring: &Ring64<N>,
    p: &mut [u64; N],
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
                let u = p[j];
                let v = p[j + t];
                p[j] = add_mod(u, v, q);
                p[j + t] = shoup_mul(sub_mod(u, v, q), w, ws, q);
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

fn inv_ntt_final_scale_scalar<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N]) {
    for e in p.iter_mut() {
        *e = shoup_mul(*e, ring.n_inv, ring.n_inv_shoup, ring.q);
    }
}

fn ntt_scalar<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N]) {
    ntt_stages_scalar(ring, p, 0, N);
}

fn inv_ntt_scalar<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N]) {
    inv_ntt_stages_scalar(ring, p, N.trailing_zeros() as usize, 1, N);
    inv_ntt_final_scale_scalar(ring, p);
}

// ---------------------------------------------------------------
// Public entry points with AVX2 runtime dispatch.
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Forward NTT over canonical `[0, q)` values, in place.
pub fn ntt<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N]) {
    assert!(N.is_power_of_two(), "NTT degree must be a power of two");
    #[cfg(target_arch = "x86_64")]
    if N >= 4 && avx2_available() {
        unsafe { avx2::ntt_avx2(ring, p) };
        return;
    }
    ntt_scalar(ring, p);
}

/// Inverse NTT over canonical `[0, q)` values, in place (includes the final
/// `N^-1` scaling).
pub fn inv_ntt<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N]) {
    assert!(N.is_power_of_two(), "NTT degree must be a power of two");
    #[cfg(target_arch = "x86_64")]
    if N >= 4 && avx2_available() {
        unsafe { avx2::inv_ntt_avx2(ring, p) };
        return;
    }
    inv_ntt_scalar(ring, p);
}

// ---------------------------------------------------------------
// Pointwise multiply-accumulate (NTT-domain inner products).
// Accumulates Montgomery-domain products, converting to canonical once per
// coefficient at the end: mont(Σ mont(a·b) · r2) = Σ a·b.
// ---------------------------------------------------------------

fn pointwise_mac_scalar<const N: usize>(
    ring: &Ring64<N>,
    acc: &mut [u64; N],
    a: &[&[u64; N]],
    b: &[&[u64; N]],
) {
    let q = ring.q;
    let mut mont_acc = [0u64; N];
    for (av, bv) in a.iter().zip(b.iter()) {
        for i in 0..N {
            mont_acc[i] = add_mod(mont_acc[i], mont_mul(av[i], bv[i], ring), q);
        }
    }
    for i in 0..N {
        acc[i] = add_mod(acc[i], mont_mul(mont_acc[i], ring.r2, ring), q);
    }
}

/// `acc[i] += Σ_k a[k][i]·b[k][i] (mod q)` for canonical inputs.
pub fn pointwise_mac<const N: usize>(
    ring: &Ring64<N>,
    acc: &mut [u64; N],
    a: &[&[u64; N]],
    b: &[&[u64; N]],
) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if N >= 4 && avx2_available() {
        unsafe { avx2::pointwise_mac_avx2(ring, acc, a, b) };
        return;
    }
    pointwise_mac_scalar(ring, acc, a, b);
}

// ---------------------------------------------------------------
// AVX2 kernels (4 × u64 lanes). 64×64 products are decomposed into
// `_mm256_mul_epu32` partial products; all values stay < 2^62 so signed
// 64-bit lane compares are safe for conditional subtracts.
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Ring64;
    use std::arch::x86_64::*;

    /// High 64 bits of the 128-bit product of two u64 vectors.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn umulhi64(x: __m256i, y: __m256i) -> __m256i {
        let mask = _mm256_set1_epi64x(0xFFFF_FFFF);
        let x0 = _mm256_and_si256(x, mask);
        let x1 = _mm256_srli_epi64(x, 32);
        let y0 = _mm256_and_si256(y, mask);
        let y1 = _mm256_srli_epi64(y, 32);
        let p00 = _mm256_mul_epu32(x0, y0);
        let p01 = _mm256_mul_epu32(x0, y1);
        let p10 = _mm256_mul_epu32(x1, y0);
        let p11 = _mm256_mul_epu32(x1, y1);
        // mid1 = p01 + (p00 >> 32): ≤ (2^64 − 2^33) + 2^32 — no overflow.
        let mid1 = _mm256_add_epi64(p01, _mm256_srli_epi64(p00, 32));
        // mid2 = p10 + lo32(mid1): same bound — no overflow.
        let mid2 = _mm256_add_epi64(p10, _mm256_and_si256(mid1, mask));
        let hi = _mm256_add_epi64(p11, _mm256_srli_epi64(mid1, 32));
        _mm256_add_epi64(hi, _mm256_srli_epi64(mid2, 32))
    }

    /// Low 64 bits of the product of two u64 vectors (wrapping).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mullo64(x: __m256i, y: __m256i) -> __m256i {
        let mask = _mm256_set1_epi64x(0xFFFF_FFFF);
        let x0 = _mm256_and_si256(x, mask);
        let x1 = _mm256_srli_epi64(x, 32);
        let y0 = _mm256_and_si256(y, mask);
        let y1 = _mm256_srli_epi64(y, 32);
        let p00 = _mm256_mul_epu32(x0, y0);
        let cross = _mm256_add_epi64(_mm256_mul_epu32(x0, y1), _mm256_mul_epu32(x1, y0));
        _mm256_add_epi64(p00, _mm256_slli_epi64(cross, 32))
    }

    /// `if v >= q { v - q } else { v }` for lanes < 2^63 (signed compare safe).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn csub(v: __m256i, q_v: __m256i) -> __m256i {
        let d = _mm256_sub_epi64(v, q_v);
        let lt = _mm256_cmpgt_epi64(_mm256_setzero_si256(), d);
        _mm256_blendv_epi8(d, v, lt)
    }

    /// Shoup multiply by a broadcast twiddle: `a·w mod q`, canonical output.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shoup_mul_v(a: __m256i, w_v: __m256i, ws_v: __m256i, q_v: __m256i) -> __m256i {
        let qq = umulhi64(a, ws_v);
        let r = _mm256_sub_epi64(mullo64(a, w_v), mullo64(qq, q_v));
        csub(r, q_v)
    }

    /// `(a + b) mod q`, canonical inputs.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        csub(_mm256_add_epi64(a, b), q_v)
    }

    /// `(a - b) mod q`, canonical inputs.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        let d = _mm256_sub_epi64(a, b);
        let neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), d);
        _mm256_add_epi64(d, _mm256_and_si256(neg, q_v))
    }

    /// Montgomery product `a·b·2^-64 mod q`, canonical output in `[0, q)`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_v(a: __m256i, b: __m256i, q_v: __m256i, qinv_v: __m256i) -> __m256i {
        let lo = mullo64(a, b);
        let hi = umulhi64(a, b);
        let m = mullo64(lo, qinv_v);
        let mhi = umulhi64(m, q_v);
        // (x + m·q) >> 64 = hi + mhi + (lo != 0): low words cancel exactly.
        let lo_nz = _mm256_cmpeq_epi64(lo, _mm256_setzero_si256());
        let carry = _mm256_add_epi64(_mm256_set1_epi64x(1), lo_nz); // 1 if lo≠0 else 0
        let t = _mm256_add_epi64(_mm256_add_epi64(hi, mhi), carry);
        csub(t, q_v)
    }

    /// `w0` in lane 0 and `w1` in lane 2 — the unused lanes are zero, and a zero
    /// twiddle with a zero Shoup companion yields zero.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn twiddle_pair(w0: u64, w1: u64) -> __m256i {
        _mm256_set_epi64x(0, w1 as i64, 0, w0 as i64)
    }

    /// Forward stages with `ht = 2, 1`. Both pair elements inside one aligned
    /// group of 4, so a group is loaded once and butterflied in-register.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ntt_tail<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N], q_v: __m256i) {
        let (ma, mb) = (N / 4, N / 2);
        for g in 0..(N / 4) {
            let ptr = p.as_mut_ptr().add(g * 4) as *mut __m256i;
            let mut x = _mm256_loadu_si256(ptr as *const __m256i);

            // ht = 2: one twiddle, halves swapped across the 128-bit lanes.
            let w = _mm256_set1_epi64x(ring.ntt_table[ma + g] as i64);
            let ws = _mm256_set1_epi64x(ring.ntt_table_shoup[ma + g] as i64);
            let u = _mm256_permute2x128_si256::<0x00>(x, x);
            let v = shoup_mul_v(_mm256_permute2x128_si256::<0x11>(x, x), w, ws, q_v);
            x = _mm256_permute2x128_si256::<0x20>(add_mod_v(u, v, q_v), sub_mod_v(u, v, q_v));

            // ht = 1: twiddles per adjacent pair, operands are even/odd lanes.
            let (i0, i1) = (mb + 2 * g, mb + 2 * g + 1);
            let w = twiddle_pair(ring.ntt_table[i0], ring.ntt_table[i1]);
            let ws = twiddle_pair(ring.ntt_table_shoup[i0], ring.ntt_table_shoup[i1]);
            let u = _mm256_unpacklo_epi64(x, x);
            let v = shoup_mul_v(_mm256_unpackhi_epi64(x, x), w, ws, q_v);
            x = _mm256_unpacklo_epi64(add_mod_v(u, v, q_v), sub_mod_v(u, v, q_v));

            _mm256_storeu_si256(ptr, x);
        }
    }

    /// Inverse stages with `t = 1, 2`, mirroring [`ntt_tail`].
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn inv_ntt_tail<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N], q_v: __m256i) {
        let (ma, mb) = (N / 2, N / 4);
        for g in 0..(N / 4) {
            let ptr = p.as_mut_ptr().add(g * 4) as *mut __m256i;
            let mut x = _mm256_loadu_si256(ptr as *const __m256i);

            // t = 1
            let (i0, i1) = (ma + 2 * g, ma + 2 * g + 1);
            let w = twiddle_pair(ring.inv_ntt_table[i0], ring.inv_ntt_table[i1]);
            let ws = twiddle_pair(ring.inv_ntt_table_shoup[i0], ring.inv_ntt_table_shoup[i1]);
            let u = _mm256_unpacklo_epi64(x, x);
            let v = _mm256_unpackhi_epi64(x, x);
            let s = add_mod_v(u, v, q_v);
            let d = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_unpacklo_epi64(s, d);

            // t = 2
            let w = _mm256_set1_epi64x(ring.inv_ntt_table[mb + g] as i64);
            let ws = _mm256_set1_epi64x(ring.inv_ntt_table_shoup[mb + g] as i64);
            let u = _mm256_permute2x128_si256::<0x00>(x, x);
            let v = _mm256_permute2x128_si256::<0x11>(x, x);
            let s = add_mod_v(u, v, q_v);
            let d = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_permute2x128_si256::<0x20>(s, d);

            _mm256_storeu_si256(ptr, x);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ntt_avx2<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N]) {
        let q_v = _mm256_set1_epi64x(ring.q as i64);
        let mut t = N;
        for l in 0..(N.trailing_zeros() as usize - 2) {
            let m = 1usize << l;
            let ht = t >> 1;
            let mut i = 0usize;
            let mut j1 = 0usize;
            while i < m {
                let w_v = _mm256_set1_epi64x(ring.ntt_table[m + i] as i64);
                let ws_v = _mm256_set1_epi64x(ring.ntt_table_shoup[m + i] as i64);
                let mut j = j1;
                while j < j1 + ht {
                    let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(p.as_ptr().add(j + ht) as *const __m256i);
                    let v_red = shoup_mul_v(v, w_v, ws_v, q_v);
                    let new_u = add_mod_v(u, v_red, q_v);
                    let new_v = sub_mod_v(u, v_red, q_v);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j + ht) as *mut __m256i, new_v);
                    j += 4;
                }
                i += 1;
                j1 += t;
            }
            t = ht;
        }
        debug_assert_eq!(t, 4);
        ntt_tail(ring, p, q_v);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn inv_ntt_avx2<const N: usize>(ring: &Ring64<N>, p: &mut [u64; N]) {
        let q_v = _mm256_set1_epi64x(ring.q as i64);
        inv_ntt_tail(ring, p, q_v);
        let (mut t, mut m) = (4usize, N / 4);
        while m > 1 {
            let hm = m >> 1;
            let dt = t << 1;
            let mut i = 0usize;
            let mut j1 = 0usize;
            while i < hm {
                let w_v = _mm256_set1_epi64x(ring.inv_ntt_table[hm + i] as i64);
                let ws_v = _mm256_set1_epi64x(ring.inv_ntt_table_shoup[hm + i] as i64);
                let mut j = j1;
                while j < j1 + t {
                    let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(p.as_ptr().add(j + t) as *const __m256i);
                    let new_u = add_mod_v(u, v, q_v);
                    let d = sub_mod_v(u, v, q_v);
                    let new_v = shoup_mul_v(d, w_v, ws_v, q_v);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j + t) as *mut __m256i, new_v);
                    j += 4;
                }
                i += 1;
                j1 += dt;
            }
            t = dt;
            m = hm;
        }
        let n_inv_v = _mm256_set1_epi64x(ring.n_inv as i64);
        let n_inv_s_v = _mm256_set1_epi64x(ring.n_inv_shoup as i64);
        for chunk in 0..(N / 4) {
            let ptr = p.as_mut_ptr().add(chunk * 4) as *mut __m256i;
            let v = _mm256_loadu_si256(ptr as *const __m256i);
            _mm256_storeu_si256(ptr, shoup_mul_v(v, n_inv_v, n_inv_s_v, q_v));
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pointwise_mac_avx2<const N: usize>(
        ring: &Ring64<N>,
        acc: &mut [u64; N],
        a: &[&[u64; N]],
        b: &[&[u64; N]],
    ) {
        let q_v = _mm256_set1_epi64x(ring.q as i64);
        let qinv_v = _mm256_set1_epi64x(ring.q_inv_neg as i64);
        let r2_v = _mm256_set1_epi64x(ring.r2 as i64);
        for chunk in 0..(N / 4) {
            let off = chunk * 4;
            let acc_ptr = acc.as_mut_ptr().add(off) as *mut __m256i;
            // Montgomery-domain accumulator: Σ_k a_k·b_k·2^-64 mod q.
            let mut mont_acc = _mm256_setzero_si256();
            for k in 0..a.len() {
                let av = _mm256_loadu_si256(a[k].as_ptr().add(off) as *const __m256i);
                let bv = _mm256_loadu_si256(b[k].as_ptr().add(off) as *const __m256i);
                let prod = mont_mul_v(av, bv, q_v, qinv_v);
                mont_acc = add_mod_v(mont_acc, prod, q_v);
            }
            // One conversion to canonical per chunk: mont(Σ · r2) = Σ.
            let canon = mont_mul_v(mont_acc, r2_v, q_v, qinv_v);
            let acc_v = _mm256_loadu_si256(acc_ptr as *const __m256i);
            _mm256_storeu_si256(acc_ptr, add_mod_v(acc_v, canon, q_v));
        }
    }
}
