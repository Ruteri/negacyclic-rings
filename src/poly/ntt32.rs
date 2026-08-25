//! 32-bit negacyclic NTT core for single-modulus and RNS arithmetic.
//! Same Shoup/Montgomery scheme as [`super::ntt64`] with R = 2^32, so every
//! product is a single 32x32->64 multiply instead of a u128.
//!
//! Requirements on the modulus: q odd prime, q ≡ 1 (mod 2N), and 2q < 2^31 —
//! the SIMD reductions use signed 32-bit compares, so a butterfly's `[0, 2q)`
//! output must stay non-negative as an i32.

#![cfg_attr(target_arch = "aarch64", allow(dead_code))]

pub struct Ring32<const N: usize> {
    pub q: u32,
    /// `-(q^-1) mod 2^32`.
    pub q_inv_neg: u32,
    /// `2^64 mod q`.
    pub r2: u32,
    pub ntt_table: [u32; N],
    pub ntt_table_shoup: [u32; N],
    pub inv_ntt_table: [u32; N],
    pub inv_ntt_table_shoup: [u32; N],
    pub n_inv: u32,
    pub n_inv_shoup: u32,
}

/// `x - q` if that is non-negative, else `x`. Branchless: `2q < 2^31` means the
/// difference's sign bit is a valid predicate, so this is a shift and a mask
/// rather than an unpredictable conditional jump.
#[inline(always)]
fn csub(x: u32, q: u32) -> u32 {
    let d = x.wrapping_sub(q);
    d.wrapping_add(q & (((d as i32) >> 31) as u32))
}

#[inline(always)]
pub fn add_mod(a: u32, b: u32, q: u32) -> u32 {
    csub(a + b, q)
}

#[inline(always)]
pub fn sub_mod(a: u32, b: u32, q: u32) -> u32 {
    let d = a.wrapping_sub(b);
    d.wrapping_add(q & (((d as i32) >> 31) as u32))
}

#[inline(always)]
fn shoup_mul(a: u32, w: u32, w_shoup: u32, q: u32) -> u32 {
    let qq = ((a as u64 * w_shoup as u64) >> 32) as u32;
    csub(a.wrapping_mul(w).wrapping_sub(qq.wrapping_mul(q)), q)
}

/// Montgomery reduction: `x < q·2^32` → `x·2^-32 mod q`, canonical.
#[inline(always)]
pub fn mont_reduce<const N: usize>(x: u64, ring: &Ring32<N>) -> u32 {
    let m = (x as u32).wrapping_mul(ring.q_inv_neg);
    let t = ((x + m as u64 * ring.q as u64) >> 32) as u32;
    csub(t, ring.q)
}

/// `a·b·2^-32 mod q` (Montgomery-domain product).
#[inline(always)]
pub fn mont_mul<const N: usize>(a: u32, b: u32, ring: &Ring32<N>) -> u32 {
    mont_reduce(a as u64 * b as u64, ring)
}

/// Canonical modular product `a·b mod q` for canonical inputs.
#[inline(always)]
pub fn mul_mod<const N: usize>(a: u32, b: u32, ring: &Ring32<N>) -> u32 {
    mont_mul(mont_mul(a, b, ring), ring.r2, ring)
}

#[inline]
pub fn add_assign<const N: usize>(ring: &Ring32<N>, lhs: &mut [u32; N], rhs: &[u32; N]) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = add_mod(*x, y, ring.q);
    }
}

#[inline]
pub fn sub_assign<const N: usize>(ring: &Ring32<N>, lhs: &mut [u32; N], rhs: &[u32; N]) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = sub_mod(*x, y, ring.q);
    }
}

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

/// A fixed-size RNS chain. Arithmetic runs independently per channel and
/// reconstruction uses mixed-radix Garner lifting.
pub struct Rns<const N: usize, const LIMBS: usize> {
    pub ch: [Ring32<N>; LIMBS],
    pub sample_threshold: [u32; LIMBS],
    /// Product of channels preceding `i`; entry zero is one.
    pub prefix_products: [u128; LIMBS],
    /// `prefix_products[i]^-1 mod ch[i].q`; entry zero is unused.
    pub prefix_inverses: [u32; LIMBS],
    pub product: u128,
}

pub type Residues<const N: usize, const LIMBS: usize> = [[u32; N]; LIMBS];

impl<const N: usize, const LIMBS: usize> Rns<N, LIMBS> {
    pub fn new(ch: [Ring32<N>; LIMBS]) -> Self {
        assert!(LIMBS > 0, "RNS needs at least one channel");
        let mut prefix_products = [1u128; LIMBS];
        let mut prefix_inverses = [0u32; LIMBS];
        let mut product = 1u128;
        for i in 0..LIMBS {
            let q = ch[i].q;
            assert!(q > 1, "RNS channel modulus must exceed one");
            if i != 0 {
                prefix_products[i] = product;
                prefix_inverses[i] = inverse_mod((product % q as u128) as u32, q)
                    .expect("RNS channel moduli must be pairwise coprime");
            }
            product = product
                .checked_mul(q as u128)
                .expect("RNS product exceeds u128");
        }
        let sample_threshold =
            core::array::from_fn(|i| (((1u64 << 32) / ch[i].q as u64) * ch[i].q as u64) as u32);
        Self {
            ch,
            sample_threshold,
            prefix_products,
            prefix_inverses,
            product,
        }
    }

    #[inline]
    pub fn reduce_coeff(&self, x: i128) -> [u32; LIMBS] {
        core::array::from_fn(|i| x.rem_euclid(self.ch[i].q as i128) as u32)
    }

    /// Garner lift into `[0, product)`.
    #[inline]
    pub fn lift_coeff(&self, r: [u32; LIMBS]) -> u128 {
        assert!(LIMBS > 0, "RNS needs at least one channel");
        let mut x = r[0] as u128;
        for i in 1..LIMBS {
            let q = self.ch[i].q as u128;
            let delta = (r[i] as u128 + q - x % q) % q;
            let digit = delta * self.prefix_inverses[i] as u128 % q;
            x += self.prefix_products[i] * digit;
        }
        x
    }

    /// Garner lift into the centered interval `(-product/2, product/2]`.
    #[inline]
    pub fn lift_centered(&self, r: [u32; LIMBS]) -> i128 {
        assert!(
            self.product <= i128::MAX as u128,
            "centered lift exceeds i128"
        );
        let x = self.lift_coeff(r);
        if x > self.product / 2 {
            x as i128 - self.product as i128
        } else {
            x as i128
        }
    }

    pub fn forward(&self, res: &mut Residues<N, LIMBS>) {
        for (c, ring) in res.iter_mut().zip(&self.ch) {
            ntt(ring, c);
        }
    }

    pub fn inverse(&self, res: &mut Residues<N, LIMBS>) {
        for (c, ring) in res.iter_mut().zip(&self.ch) {
            inv_ntt(ring, c);
        }
    }

    pub fn add_assign(&self, lhs: &mut Residues<N, LIMBS>, rhs: &Residues<N, LIMBS>) {
        for ((l, r), ring) in lhs.iter_mut().zip(rhs).zip(&self.ch) {
            add_assign(ring, l, r);
        }
    }

    pub fn sub_assign(&self, lhs: &mut Residues<N, LIMBS>, rhs: &Residues<N, LIMBS>) {
        for ((l, r), ring) in lhs.iter_mut().zip(rhs).zip(&self.ch) {
            sub_assign(ring, l, r);
        }
    }

    pub fn pointwise_mul(
        &self,
        lhs: &Residues<N, LIMBS>,
        rhs: &Residues<N, LIMBS>,
    ) -> Residues<N, LIMBS> {
        core::array::from_fn(|i| pointwise_mul(&self.ch[i], &lhs[i], &rhs[i]))
    }

    pub fn pointwise_mac(
        &self,
        acc: &mut Residues<N, LIMBS>,
        a: &[&Residues<N, LIMBS>],
        b: &[&Residues<N, LIMBS>],
    ) {
        debug_assert_eq!(a.len(), b.len());
        for (i, ring) in self.ch.iter().enumerate() {
            pointwise_mac(ring, &mut acc[i], a, b, i);
        }
    }

    /// Uniform over `Z_product` — independent uniform residues, by CRT.
    pub fn rand<R: rand::Rng>(&self, rng: &mut R) -> Residues<N, LIMBS> {
        core::array::from_fn(|i| {
            core::array::from_fn(|_| {
                let mut v = rng.next_u32();
                while v >= self.sample_threshold[i] {
                    v = rng.next_u32();
                }
                v % self.ch[i].q
            })
        })
    }
}

fn inverse_mod(value: u32, modulus: u32) -> Option<u32> {
    let (mut old_r, mut r) = (value as i64, modulus as i64);
    let (mut old_s, mut s) = (1i64, 0i64);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }
    (old_r == 1).then(|| old_s.rem_euclid(modulus as i64) as u32)
}

// ---------------------------------------------------------------
// Scalar NTT.
// ---------------------------------------------------------------

fn ntt_stages_scalar<const N: usize>(
    ring: &Ring32<N>,
    p: &mut [u32; N],
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

fn inv_ntt_stages_scalar<const N: usize>(
    ring: &Ring32<N>,
    p: &mut [u32; N],
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

fn inv_ntt_final_scale_scalar<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
    for e in p.iter_mut() {
        *e = shoup_mul(*e, ring.n_inv, ring.n_inv_shoup, ring.q);
    }
}

fn ntt_scalar<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
    ntt_stages_scalar(ring, p, 0, N);
}

fn inv_ntt_scalar<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
    inv_ntt_stages_scalar(ring, p, N.trailing_zeros() as usize, 1, N);
    inv_ntt_final_scale_scalar(ring, p);
}

// ---------------------------------------------------------------
// Public entry points with architecture-specific SIMD dispatch.
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Forward NTT over canonical `[0, q)` values, in place.
pub fn ntt<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
    assert!(N.is_power_of_two(), "NTT degree must be a power of two");
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        unsafe { avx2::ntt_avx2(ring, p) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if N >= 4 {
        unsafe { neon::ntt_neon(ring, p) };
        return;
    }
    ntt_scalar(ring, p);
}

/// Inverse NTT over canonical `[0, q)` values, in place (includes the final
/// `N^-1` scaling).
pub fn inv_ntt<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
    assert!(N.is_power_of_two(), "NTT degree must be a power of two");
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        unsafe { avx2::inv_ntt_avx2(ring, p) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if N >= 4 {
        unsafe { neon::inv_ntt_neon(ring, p) };
        return;
    }
    inv_ntt_scalar(ring, p);
}

fn pointwise_mac_scalar<const N: usize, const LIMBS: usize>(
    ring: &Ring32<N>,
    acc: &mut [u32; N],
    a: &[&Residues<N, LIMBS>],
    b: &[&Residues<N, LIMBS>],
    ch: usize,
) {
    let q = ring.q;
    let mut mont_acc = [0u32; N];
    for (av, bv) in a.iter().zip(b.iter()) {
        for i in 0..N {
            mont_acc[i] = add_mod(mont_acc[i], mont_mul(av[ch][i], bv[ch][i], ring), q);
        }
    }
    for i in 0..N {
        acc[i] = add_mod(acc[i], mont_mul(mont_acc[i], ring.r2, ring), q);
    }
}

/// `acc[i] += Σ_k a[k][ch][i]·b[k][ch][i] (mod q)` for canonical inputs. Takes
/// the whole residue arrays and a channel index so the RNS layer needs no
/// per-channel reference vectors.
pub fn pointwise_mac<const N: usize, const LIMBS: usize>(
    ring: &Ring32<N>,
    acc: &mut [u32; N],
    a: &[&Residues<N, LIMBS>],
    b: &[&Residues<N, LIMBS>],
    ch: usize,
) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && avx2_available() {
        unsafe { avx2::pointwise_mac_avx2(ring, acc, a, b, ch) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::pointwise_mac_neon(ring, acc, a, b, ch) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    pointwise_mac_scalar(ring, acc, a, b, ch);
}

// ---------------------------------------------------------------
// AVX2 kernels (8 × u32 lanes). Shoup butterflies use `_mm256_mullo_epi32`;
// Montgomery products split the vector into even/odd u64 halves so
// `_mm256_mul_epu32` supplies the 32x32->64 products.
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Ring32;
    use std::arch::x86_64::*;

    const LANES: usize = 8;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lo_mask() -> __m256i {
        _mm256_set1_epi64x(0xFFFF_FFFF)
    }

    /// High 32 bits of each of the 8 lanewise u32 products.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mulhi32(a: __m256i, b: __m256i) -> __m256i {
        let mask = lo_mask();
        let pe = _mm256_srli_epi64(
            _mm256_mul_epu32(_mm256_and_si256(a, mask), _mm256_and_si256(b, mask)),
            32,
        );
        let po = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
        // pe holds the even-lane high words; po's high half already sits in the
        // odd lane position.
        _mm256_or_si256(pe, _mm256_andnot_si256(mask, po))
    }

    /// `if v >= q { v - q } else { v }` for u32 lanes < 2^31.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn csub32(v: __m256i, q_v: __m256i) -> __m256i {
        let d = _mm256_sub_epi32(v, q_v);
        let lt = _mm256_cmpgt_epi32(_mm256_setzero_si256(), d);
        _mm256_blendv_epi8(d, v, lt)
    }

    /// Same, on 4 × u64 lanes holding values < 2^31.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn csub64(v: __m256i, q_v: __m256i) -> __m256i {
        let d = _mm256_sub_epi64(v, q_v);
        let lt = _mm256_cmpgt_epi64(_mm256_setzero_si256(), d);
        _mm256_blendv_epi8(d, v, lt)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shoup_mul_v(a: __m256i, w_v: __m256i, ws_v: __m256i, q_v: __m256i) -> __m256i {
        let qq = mulhi32(a, ws_v);
        let r = _mm256_sub_epi32(_mm256_mullo_epi32(a, w_v), _mm256_mullo_epi32(qq, q_v));
        csub32(r, q_v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        csub32(_mm256_add_epi32(a, b), q_v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_mod_v(a: __m256i, b: __m256i, q_v: __m256i) -> __m256i {
        let d = _mm256_sub_epi32(a, b);
        let neg = _mm256_cmpgt_epi32(_mm256_setzero_si256(), d);
        _mm256_add_epi32(d, _mm256_and_si256(neg, q_v))
    }

    /// Montgomery product on one u64 half: inputs are u32 values in the low
    /// 32 bits of each u64 lane, output likewise, canonical in `[0, q)`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_half(a: __m256i, b: __m256i, q64_v: __m256i, qinv_v: __m256i) -> __m256i {
        let x = _mm256_mul_epu32(a, b);
        let m = _mm256_and_si256(_mm256_mul_epu32(x, qinv_v), lo_mask());
        let t = _mm256_srli_epi64(_mm256_add_epi64(x, _mm256_mul_epu32(m, q64_v)), 32);
        csub64(t, q64_v)
    }

    /// Montgomery product `a·b·2^-32 mod q` across 8 u32 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mont_mul_v(a: __m256i, b: __m256i, q64_v: __m256i, qinv_v: __m256i) -> __m256i {
        let mask = lo_mask();
        let re = mont_mul_half(
            _mm256_and_si256(a, mask),
            _mm256_and_si256(b, mask),
            q64_v,
            qinv_v,
        );
        let ro = mont_mul_half(
            _mm256_srli_epi64(a, 32),
            _mm256_srli_epi64(b, 32),
            q64_v,
            qinv_v,
        );
        _mm256_or_si256(re, _mm256_slli_epi64(ro, 32))
    }

    /// Forward stages with `ht = 4, 2, 1`. All three pair elements inside one
    /// aligned group of 8, so a group is loaded once, butterflied in-register
    /// with shuffles, and stored once — no scalar tail.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ntt_tail<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N], q_v: __m256i) {
        let (ma, mb, mc) = (N / 8, N / 4, N / 2);
        for g in 0..(N / 8) {
            let ptr = p.as_mut_ptr().add(g * 8) as *mut __m256i;
            let mut x = _mm256_loadu_si256(ptr as *const __m256i);

            // ht = 4: one twiddle, halves swapped across the 128-bit lanes.
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

    /// Inverse stages with `t = 1, 2, 4`, mirroring [`ntt_tail`].
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn inv_ntt_tail<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N], q_v: __m256i) {
        let (ma, mb, mc) = (N / 2, N / 4, N / 8);
        for g in 0..(N / 8) {
            let ptr = p.as_mut_ptr().add(g * 8) as *mut __m256i;
            let mut x = _mm256_loadu_si256(ptr as *const __m256i);

            // t = 1
            let j = ma + 4 * g;
            let w = twiddle_quads(&ring.inv_ntt_table[j..j + 4]);
            let ws = twiddle_quads(&ring.inv_ntt_table_shoup[j..j + 4]);
            let u = _mm256_shuffle_epi32::<0xA0>(x);
            let v = _mm256_shuffle_epi32::<0xF5>(x);
            let s = add_mod_v(u, v, q_v);
            let d = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_blend_epi32::<0x55>(_mm256_slli_epi64::<32>(d), s);

            // t = 2
            let (i0, i1) = (mb + 2 * g, mb + 2 * g + 1);
            let w = twiddle_pairs(ring.inv_ntt_table[i0], ring.inv_ntt_table[i1]);
            let ws = twiddle_pairs(ring.inv_ntt_table_shoup[i0], ring.inv_ntt_table_shoup[i1]);
            let u = _mm256_unpacklo_epi64(x, x);
            let v = _mm256_unpackhi_epi64(x, x);
            let s = add_mod_v(u, v, q_v);
            let d = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_unpacklo_epi64(s, d);

            // t = 4
            let w = _mm256_set1_epi32(ring.inv_ntt_table[mc + g] as i32);
            let ws = _mm256_set1_epi32(ring.inv_ntt_table_shoup[mc + g] as i32);
            let u = _mm256_permute2x128_si256::<0x00>(x, x);
            let v = _mm256_permute2x128_si256::<0x11>(x, x);
            let s = add_mod_v(u, v, q_v);
            let d = shoup_mul_v(sub_mod_v(u, v, q_v), w, ws, q_v);
            x = _mm256_permute2x128_si256::<0x20>(s, d);

            _mm256_storeu_si256(ptr, x);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ntt_avx2<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
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
                    let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(p.as_ptr().add(j + ht) as *const __m256i);
                    let v_red = shoup_mul_v(v, w_v, ws_v, q_v);
                    let new_u = add_mod_v(u, v_red, q_v);
                    let new_v = sub_mod_v(u, v_red, q_v);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j + ht) as *mut __m256i, new_v);
                    j += LANES;
                }
                i += 1;
                j1 += t;
            }
            t = ht;
        }
        debug_assert_eq!(t, 8);
        ntt_tail(ring, p, q_v);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn inv_ntt_avx2<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
        let q_v = _mm256_set1_epi32(ring.q as i32);
        inv_ntt_tail(ring, p, q_v);
        let (mut t, mut m) = (8usize, N / 8);
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
                    let u = _mm256_loadu_si256(p.as_ptr().add(j) as *const __m256i);
                    let v = _mm256_loadu_si256(p.as_ptr().add(j + t) as *const __m256i);
                    let new_u = add_mod_v(u, v, q_v);
                    let d = sub_mod_v(u, v, q_v);
                    let new_v = shoup_mul_v(d, w_v, ws_v, q_v);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j) as *mut __m256i, new_u);
                    _mm256_storeu_si256(p.as_mut_ptr().add(j + t) as *mut __m256i, new_v);
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
            let ptr = p.as_mut_ptr().add(chunk * LANES) as *mut __m256i;
            let v = _mm256_loadu_si256(ptr as *const __m256i);
            _mm256_storeu_si256(ptr, shoup_mul_v(v, n_inv_v, n_inv_s_v, q_v));
        }
    }

    /// Canonical `lhs[i]·rhs[i] mod q`: two Montgomery rounds, the second
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
            let off = chunk * LANES;
            let a = _mm256_loadu_si256(lhs.as_ptr().add(off) as *const __m256i);
            let b = _mm256_loadu_si256(rhs.as_ptr().add(off) as *const __m256i);
            let t = mont_mul_v(a, b, q64_v, qinv_v);
            let c = mont_mul_v(t, r2_v, q64_v, qinv_v);
            _mm256_storeu_si256(out.as_mut_ptr().add(off) as *mut __m256i, c);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn pointwise_mac_avx2<const N: usize, const LIMBS: usize>(
        ring: &Ring32<N>,
        acc: &mut [u32; N],
        a: &[&super::Residues<N, LIMBS>],
        b: &[&super::Residues<N, LIMBS>],
        ch: usize,
    ) {
        let q_v = _mm256_set1_epi32(ring.q as i32);
        let q64_v = _mm256_set1_epi64x(ring.q as i64);
        let qinv_v = _mm256_set1_epi64x(ring.q_inv_neg as i64);
        let r2_v = _mm256_set1_epi32(ring.r2 as i32);
        for chunk in 0..(N / LANES) {
            let off = chunk * LANES;
            let acc_ptr = acc.as_mut_ptr().add(off) as *mut __m256i;
            let mut mont_acc = _mm256_setzero_si256();
            for k in 0..a.len() {
                let av = _mm256_loadu_si256(a[k][ch].as_ptr().add(off) as *const __m256i);
                let bv = _mm256_loadu_si256(b[k][ch].as_ptr().add(off) as *const __m256i);
                let prod = mont_mul_v(av, bv, q64_v, qinv_v);
                mont_acc = add_mod_v(mont_acc, prod, q_v);
            }
            let canon = mont_mul_v(mont_acc, r2_v, q64_v, qinv_v);
            let acc_v = _mm256_loadu_si256(acc_ptr as *const __m256i);
            _mm256_storeu_si256(acc_ptr, add_mod_v(acc_v, canon, q_v));
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{mul_mod, Residues, Ring32};
    use core::arch::{aarch64::*, asm};

    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn canonicalize(v: int32x4_t, q_v: int32x4_t) -> int32x4_t {
        let zero = vdupq_n_s32(0);
        let add_q = vandq_u32(vcltq_s32(v, zero), vreinterpretq_u32_s32(q_v));
        let v = vaddq_s32(v, vreinterpretq_s32_u32(add_q));
        let sub_q = vandq_u32(vcgeq_s32(v, q_v), vreinterpretq_u32_s32(q_v));
        vsubq_s32(v, vreinterpretq_s32_u32(sub_q))
    }

    #[inline(always)]
    unsafe fn add_mod_v(a: int32x4_t, b: int32x4_t, q_v: int32x4_t) -> int32x4_t {
        canonicalize(vaddq_s32(a, b), q_v)
    }

    #[inline(always)]
    unsafe fn sub_mod_v(a: int32x4_t, b: int32x4_t, q_v: int32x4_t) -> int32x4_t {
        canonicalize(vsubq_s32(a, b), q_v)
    }

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
    unsafe fn twiddle_v(w: u32, w_shoup: u32, q: u32) -> (int32x4_t, int32x4_t) {
        let (w, scaled) = centered_twiddle(w, w_shoup, q);
        (vdupq_n_s32(w), vdupq_n_s32(scaled))
    }

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

    #[inline(always)]
    unsafe fn becker_mul_v(
        a: int32x4_t,
        w_v: int32x4_t,
        scaled_v: int32x4_t,
        q_v: int32x4_t,
    ) -> int32x4_t {
        let quotient = vqrdmulhq_s32(a, scaled_v);
        let mut product = vmulq_s32(a, w_v);
        asm!(
            "mls {product:v}.4s, {quotient:v}.4s, {q:v}.4s",
            product = inout(vreg) product,
            quotient = in(vreg) quotient,
            q = in(vreg) q_v,
            options(pure, nomem, nostack)
        );
        canonicalize(product, q_v)
    }

    #[inline(always)]
    unsafe fn mont_mul_half(
        a: uint32x2_t,
        b: uint32x2_t,
        q_v: uint32x2_t,
        q_inv_v: uint32x2_t,
    ) -> uint32x2_t {
        let x = vmull_u32(a, b);
        let m = vmul_u32(vmovn_u64(x), q_inv_v);
        let t = vshrq_n_u64::<32>(vaddq_u64(x, vmull_u32(m, q_v)));
        vmovn_u64(t)
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

    #[inline]
    unsafe fn ntt_tail<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N], q_v: int32x4_t) {
        let (m2, m1) = (N / 4, N / 2);
        for g in 0..(N / LANES) {
            let ptr = p.as_mut_ptr().add(g * LANES);
            let mut x = vreinterpretq_s32_u32(vld1q_u32(ptr));

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

    #[inline]
    unsafe fn inv_ntt_tail<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N], q_v: int32x4_t) {
        let (m1, m2) = (N / 2, N / 4);
        for g in 0..(N / LANES) {
            let ptr = p.as_mut_ptr().add(g * LANES);
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

    pub(super) unsafe fn ntt_neon<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
        let q_v = vdupq_n_s32(ring.q as i32);
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
                    let u = vreinterpretq_s32_u32(vld1q_u32(p.as_ptr().add(j)));
                    let v = vreinterpretq_s32_u32(vld1q_u32(p.as_ptr().add(j + ht)));
                    let v = becker_mul_v(v, w, scaled, q_v);
                    vst1q_u32(
                        p.as_mut_ptr().add(j),
                        vreinterpretq_u32_s32(add_mod_v(u, v, q_v)),
                    );
                    vst1q_u32(
                        p.as_mut_ptr().add(j + ht),
                        vreinterpretq_u32_s32(sub_mod_v(u, v, q_v)),
                    );
                    j += LANES;
                }
                j1 += t;
            }
            t = ht;
        }
        debug_assert_eq!(t, LANES);
        ntt_tail(ring, p, q_v);
    }

    pub(super) unsafe fn inv_ntt_neon<const N: usize>(ring: &Ring32<N>, p: &mut [u32; N]) {
        let q_v = vdupq_n_s32(ring.q as i32);
        inv_ntt_tail(ring, p, q_v);
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
                    let u = vreinterpretq_s32_u32(vld1q_u32(p.as_ptr().add(j)));
                    let v = vreinterpretq_s32_u32(vld1q_u32(p.as_ptr().add(j + t)));
                    vst1q_u32(
                        p.as_mut_ptr().add(j),
                        vreinterpretq_u32_s32(add_mod_v(u, v, q_v)),
                    );
                    let d = becker_mul_v(sub_mod_v(u, v, q_v), w, scaled, q_v);
                    vst1q_u32(p.as_mut_ptr().add(j + t), vreinterpretq_u32_s32(d));
                    j += LANES;
                }
                j1 += dt;
            }
            t = dt;
            m = hm;
        }
        let (n_inv, scaled) = twiddle_v(ring.n_inv, ring.n_inv_shoup, ring.q);
        for off in (0..N).step_by(LANES) {
            let ptr = p.as_mut_ptr().add(off);
            let v = vreinterpretq_s32_u32(vld1q_u32(ptr));
            vst1q_u32(
                ptr,
                vreinterpretq_u32_s32(becker_mul_v(v, n_inv, scaled, q_v)),
            );
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
        for off in (0..vector_end).step_by(LANES) {
            let a = vld1q_u32(lhs.as_ptr().add(off));
            let b = vld1q_u32(rhs.as_ptr().add(off));
            let t = mont_mul_v(a, b, ring.q, ring.q_inv_neg);
            vst1q_u32(
                out.as_mut_ptr().add(off),
                mont_mul_v(t, r2, ring.q, ring.q_inv_neg),
            );
        }
        for i in vector_end..N {
            out[i] = mul_mod(lhs[i], rhs[i], ring);
        }
    }

    pub(super) unsafe fn pointwise_mac_neon<const N: usize, const LIMBS: usize>(
        ring: &Ring32<N>,
        acc: &mut [u32; N],
        a: &[&Residues<N, LIMBS>],
        b: &[&Residues<N, LIMBS>],
        ch: usize,
    ) {
        let q_v = vdupq_n_s32(ring.q as i32);
        let r2 = vdupq_n_u32(ring.r2);
        let vector_end = N / LANES * LANES;
        for off in (0..vector_end).step_by(LANES) {
            let mut mont_acc = vdupq_n_s32(0);
            for k in 0..a.len() {
                let av = vld1q_u32(a[k][ch].as_ptr().add(off));
                let bv = vld1q_u32(b[k][ch].as_ptr().add(off));
                let prod = mont_mul_v(av, bv, ring.q, ring.q_inv_neg);
                mont_acc = add_mod_v(mont_acc, vreinterpretq_s32_u32(prod), q_v);
            }
            let canon = mont_mul_v(vreinterpretq_u32_s32(mont_acc), r2, ring.q, ring.q_inv_neg);
            let acc_v = vreinterpretq_s32_u32(vld1q_u32(acc.as_ptr().add(off)));
            vst1q_u32(
                acc.as_mut_ptr().add(off),
                vreinterpretq_u32_s32(add_mod_v(acc_v, vreinterpretq_s32_u32(canon), q_v)),
            );
        }
        for i in vector_end..N {
            let mut sum = 0;
            for k in 0..a.len() {
                sum = super::add_mod(sum, mul_mod(a[k][ch][i], b[k][ch][i], ring), ring.q);
            }
            acc[i] = super::add_mod(acc[i], sum, ring.q);
        }
    }
}
