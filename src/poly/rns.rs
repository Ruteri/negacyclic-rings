//! Fixed-size RNS chains over [`Ring32`] channels.
//!
//! Each channel is an independent `Ring32<N>` (its own NTT-friendly prime);
//! [`Rns`] runs per-channel arithmetic and, for reconstruction, implements
//! Garner's mixed-radix algorithm. See `REFERENCES.md`, Residue Number
//! System, for the paper this maps to and a per-function breakdown.

#[cfg(target_arch = "aarch64")]
use super::ntt32::neon;
use super::ntt32::{
    add_assign, inv_ntt, mul_mod, ntt, pointwise_mac, pointwise_mul, sub_assign, sub_mod, Ring32,
};
#[cfg(target_arch = "x86_64")]
use super::ntt32::{avx2, avx2_available};

/// A fixed-size RNS chain. Arithmetic runs independently per channel and
/// reconstruction uses mixed-radix Garner lifting.
pub struct Rns<const N: usize, const CHANNEL_COUNT: usize> {
    /// One ring per channel, each its own prime modulus `q_i` and sharing the
    /// same polynomial degree `N`.
    pub channels: [Ring32<N>; CHANNEL_COUNT],
    /// Rejection-sampling cutoff for `rand`: the largest multiple of
    /// `channels[i].q` that is `<= 2^32`, so drawing a uniform `u32` below it
    /// and reducing mod `q` gives an unbiased residue.
    sample_threshold: [u32; CHANNEL_COUNT],
    /// Running product of the channels before `i`, i.e.
    /// `channels[0].q * ... * channels[i-1].q`; entry zero is one. The
    /// "place value" channel `i`'s digit is multiplied by during Garner
    /// reconstruction (`lift_coefficient`).
    prefix_products: [u128; CHANNEL_COUNT],
    /// `prefix_products[i]^-1 mod channels[i].q`; entry zero is unused.
    /// Precomputed here, at construction, so `lift_coefficient` never has to
    /// compute a modular inverse itself.
    pub prefix_inverses: [u32; CHANNEL_COUNT],
    /// The full modulus this chain represents: `∏ channels[i].q`.
    pub product: u128,
}

/// One `[u32; N]` array of canonical residues per channel.
pub type Residues<const N: usize, const CHANNEL_COUNT: usize> = [[u32; N]; CHANNEL_COUNT];

impl<const N: usize, const CHANNEL_COUNT: usize> Rns<N, CHANNEL_COUNT> {
    /// Builds a chain from its channels, precomputing the Garner mixed-radix
    /// constants (`prefix_products`/`prefix_inverses`) and the rejection-
    /// sampling threshold `rand` needs. Panics if `CHANNEL_COUNT` is `0`,
    /// any channel's modulus isn't greater than `1`, the moduli aren't
    /// pairwise coprime, or their product doesn't fit in `u128`.
    pub fn new(channels: [Ring32<N>; CHANNEL_COUNT]) -> Self {
        assert!(CHANNEL_COUNT > 0, "RNS needs at least one channel");
        let mut prefix_products = [1u128; CHANNEL_COUNT];
        let mut prefix_inverses = [0u32; CHANNEL_COUNT];
        let mut product = 1u128;
        for i in 0..CHANNEL_COUNT {
            let q = channels[i].q;
            assert!(q > 1, "RNS channel modulus must exceed one");
            if i != 0 {
                // `product` is still "product of channels before `i`" here
                // — this channel's own modulus is folded in below. Reduce
                // it mod `q` before inverting, since `inverse_mod` (an
                // extended-Euclid inverse) takes `u32` operands.
                prefix_products[i] = product;
                prefix_inverses[i] = inverse_mod((product % q as u128) as u32, q)
                    .expect("RNS channel moduli must be pairwise coprime");
            }
            // Fold this channel's modulus into the running product, for the
            // next iteration.
            product = product
                .checked_mul(q as u128)
                .expect("RNS product exceeds u128");
        }
        // One past `u32::MAX`: the exclusive upper bound of the range
        // `rand`'s `next_u32()` draws from.
        let u32_range = 1u64 << 32;
        let sample_threshold = core::array::from_fn(|i| {
            // The cast to `u32` never truncates since every channel modulus is odd.
            ((u32_range / channels[i].q as u64) * channels[i].q as u64) as u32
        });
        Self {
            channels,
            sample_threshold,
            prefix_products,
            prefix_inverses,
            product,
        }
    }
}

fn inverse_mod(value: u32, modulus: u32) -> Option<u32> {
    let (mut previous_remainder, mut remainder) = (value as i64, modulus as i64);
    let (mut previous_coefficient, mut coefficient) = (1i64, 0i64);
    while remainder != 0 {
        let quotient = previous_remainder / remainder;
        (previous_remainder, remainder) = (remainder, previous_remainder - quotient * remainder);
        (previous_coefficient, coefficient) =
            (coefficient, previous_coefficient - quotient * coefficient);
    }
    (previous_remainder == 1).then(|| previous_coefficient.rem_euclid(modulus as i64) as u32)
}

impl<const N: usize, const CHANNEL_COUNT: usize> Rns<N, CHANNEL_COUNT> {
    /// Reduce a signed coefficient into its canonical residue for every
    /// channel.
    #[inline]
    pub fn reduce_coefficient(&self, x: i128) -> [u32; CHANNEL_COUNT] {
        // `rem_euclid` rather than `%`, so the result is always in
        // `[0, q_i)`, even for negative `x`.
        core::array::from_fn(|i| x.rem_euclid(self.channels[i].q as i128) as u32)
    }

    /// Reduce signed coefficients into canonical residues for every channel.
    pub fn reduce_coefficients_i64_into(
        &self,
        input: &[i64; N],
        output: &mut Residues<N, CHANNEL_COUNT>,
    ) {
        for (ring, channel) in self.channels.iter().zip(output.iter_mut()) {
            reduce_channel_i64(ring, input, channel);
        }
    }
}

fn reduce_channel_i64<const N: usize>(ring: &Ring32<N>, input: &[i64; N], output: &mut [u32; N]) {
    #[cfg(target_arch = "x86_64")]
    if N >= 4 && N.is_multiple_of(4) && avx2_available() {
        unsafe { avx2::reduce_channel_i64_avx2(ring, input, output) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if N >= 2 && N.is_multiple_of(2) {
        unsafe { neon::reduce_channel_i64_neon(ring, input, output) };
        return;
    }
    reduce_channel_i64_scalar(ring, input, output);
}

fn reduce_channel_i64_scalar<const N: usize>(
    ring: &Ring32<N>,
    input: &[i64; N],
    output: &mut [u32; N],
) {
    for (out, &value) in output.iter_mut().zip(input) {
        *out = value.rem_euclid(ring.q as i64) as u32;
    }
}

impl<const N: usize, const CHANNEL_COUNT: usize> Rns<N, CHANNEL_COUNT> {
    /// Reduce centered coefficients into every RNS channel.
    pub fn reduce_centered_i32_into(
        &self,
        input: &[i32; N],
        output: &mut Residues<N, CHANNEL_COUNT>,
    ) {
        debug_assert!(input.iter().all(|value| self
            .channels
            .iter()
            .all(|ring| value.unsigned_abs() < ring.q)));
        for (ring, channel) in self.channels.iter().zip(output.iter_mut()) {
            reduce_centered_i32_into(ring, input, channel);
        }
    }
}

fn reduce_centered_i32_into<const N: usize>(
    ring: &Ring32<N>,
    input: &[i32; N],
    output: &mut [u32; N],
) {
    debug_assert!(input.iter().all(|value| value.unsigned_abs() < ring.q));
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && N.is_multiple_of(8) && avx2_available() {
        unsafe { avx2::reduce_centered_i32_avx2(ring, input, output) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if N >= 4 && N.is_multiple_of(4) {
        unsafe { neon::reduce_centered_i32_neon(ring, input, output) };
        return;
    }
    reduce_channel_centered_i32_scalar(ring, input, output);
}

fn reduce_channel_centered_i32_scalar<const N: usize>(
    ring: &Ring32<N>,
    input: &[i32; N],
    output: &mut [u32; N],
) {
    for (out, &value) in output.iter_mut().zip(input) {
        let negative = 0u32.wrapping_sub((value < 0) as u32);
        *out = (value as u32 & !negative) | ((ring.q - value.unsigned_abs()) & negative);
    }
}

impl<const N: usize, const CHANNEL_COUNT: usize> Rns<N, CHANNEL_COUNT> {
    /// Garner lift into `[0, product)`.
    #[inline]
    pub fn lift_coefficient(&self, r: [u32; CHANNEL_COUNT]) -> u128 {
        assert!(CHANNEL_COUNT > 0, "RNS needs at least one channel");
        let mut x = r[0] as u128;
        for i in 1..CHANNEL_COUNT {
            let q = self.channels[i].q as u128;
            let delta = (r[i] as u128 + q - x % q) % q;
            let digit = delta * self.prefix_inverses[i] as u128 % q;
            x += self.prefix_products[i] * digit;
        }
        x
    }

    /// Garner lift into the centered interval `(-product/2, product/2]`.
    #[inline]
    pub fn lift_centered(&self, r: [u32; CHANNEL_COUNT]) -> i128 {
        assert!(
            self.product <= i128::MAX as u128,
            "centered lift exceeds i128"
        );
        let x = self.lift_coefficient(r);
        if x > self.product / 2 {
            x as i128 - self.product as i128
        } else {
            x as i128
        }
    }

    /// Forward NTT, in place. Each channel is just its own `Ring32<N>`'s NTT
    /// run on its own slice, with no interaction between channels.
    pub fn forward(&self, res: &mut Residues<N, CHANNEL_COUNT>) {
        for (c, ring) in res.iter_mut().zip(&self.channels) {
            ntt(ring, c);
        }
    }

    /// Inverse NTT, in place.
    pub fn inverse(&self, res: &mut Residues<N, CHANNEL_COUNT>) {
        for (c, ring) in res.iter_mut().zip(&self.channels) {
            inv_ntt(ring, c);
        }
    }

    pub fn add_assign(
        &self,
        left_hand_side: &mut Residues<N, CHANNEL_COUNT>,
        right_hand_side: &Residues<N, CHANNEL_COUNT>,
    ) {
        for ((l, r), ring) in left_hand_side
            .iter_mut()
            .zip(right_hand_side)
            .zip(&self.channels)
        {
            add_assign(ring, l, r);
        }
    }

    /// `left_hand_side -= right_hand_side`.
    pub fn sub_assign(
        &self,
        left_hand_side: &mut Residues<N, CHANNEL_COUNT>,
        right_hand_side: &Residues<N, CHANNEL_COUNT>,
    ) {
        for ((l, r), ring) in left_hand_side
            .iter_mut()
            .zip(right_hand_side)
            .zip(&self.channels)
        {
            sub_assign(ring, l, r);
        }
    }

    /// Element-by-element product, channel by channel: no cross terms
    /// between coefficients, unlike a full polynomial convolution. Valid
    /// once both sides have gone through `forward` — that's what turns
    /// convolution into a per-index product in the first place.
    pub fn pointwise_mul(
        &self,
        left_hand_side: &Residues<N, CHANNEL_COUNT>,
        right_hand_side: &Residues<N, CHANNEL_COUNT>,
    ) -> Residues<N, CHANNEL_COUNT> {
        core::array::from_fn(|i| {
            pointwise_mul(&self.channels[i], &left_hand_side[i], &right_hand_side[i])
        })
    }

    pub fn pointwise_mul_accumulate(
        &self,
        accumulator: &mut Residues<N, CHANNEL_COUNT>,
        left_hand_sides: &[&Residues<N, CHANNEL_COUNT>],
        right_hand_sides: &[&Residues<N, CHANNEL_COUNT>],
    ) {
        debug_assert_eq!(left_hand_sides.len(), right_hand_sides.len());
        for (i, ring) in self.channels.iter().enumerate() {
            pointwise_mac(
                ring,
                &mut accumulator[i],
                left_hand_sides,
                right_hand_sides,
                i,
            );
        }
    }

    /// Draws a fresh, uniform-random `Residues`: one independent, uniform
    /// residue per coefficient, in every channel, via rejection sampling
    /// against `sample_threshold` (see `new()` for why that avoids bias).
    ///
    /// The Chinese Remainder Theorem is a bijection between `Z_product` and
    /// the product of the per-channel residue rings, so sampling every
    /// channel independently and uniformly is equivalent to sampling one
    /// uniform integer in `[0, product)` and reducing it.
    pub fn rand<RandomNumberGenerator: rand::Rng>(
        &self,
        rng: &mut RandomNumberGenerator,
    ) -> Residues<N, CHANNEL_COUNT> {
        core::array::from_fn(|i| {
            core::array::from_fn(|_| {
                let mut candidate = rng.next_u32();
                while candidate >= self.sample_threshold[i] {
                    candidate = rng.next_u32();
                }
                candidate % self.channels[i].q
            })
        })
    }
}

impl<const N: usize> Rns<N, 2> {
    /// Reconstruct canonical two-channel residues into centered `i64`
    /// values.
    pub fn lift_centered_i64_into(&self, input: &Residues<N, 2>, output: &mut [i64; N]) {
        assert!(self.product <= i64::MAX as u128, "RNS product exceeds i64");
        debug_assert!(input[0].iter().all(|&x| x < self.channels[0].q));
        debug_assert!(input[1].iter().all(|&x| x < self.channels[1].q));
        lift_two_channel_centered_i64(self, input, output);
    }
}

fn lift_two_channel_centered_i64<const N: usize>(
    ring: &Rns<N, 2>,
    input: &Residues<N, 2>,
    output: &mut [i64; N],
) {
    #[cfg(target_arch = "x86_64")]
    if N >= 8 && N.is_multiple_of(8) && avx2_available() {
        unsafe { avx2::lift_centered_i64_avx2(ring, input, output) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if N >= 4 && N.is_multiple_of(4) {
        unsafe { neon::lift_centered_i64_neon(ring, input, output) };
        return;
    }
    lift_two_channel_centered_i64_scalar(ring, input, output);
}

fn lift_two_channel_centered_i64_scalar<const N: usize>(
    ring: &Rns<N, 2>,
    input: &Residues<N, 2>,
    output: &mut [i64; N],
) {
    let q0 = ring.channels[0].q;
    let q1 = ring.channels[1].q;
    let product = q0 as u64 * q1 as u64;
    for (i, out) in output.iter_mut().enumerate() {
        let r0_mod_q1 = input[0][i] % q1;
        let delta = sub_mod(input[1][i], r0_mod_q1, q1);
        let digit = mul_mod(delta, ring.prefix_inverses[1], &ring.channels[1]);
        let value = input[0][i] as u64 + q0 as u64 * digit as u64;
        *out = value as i64 - (value > product / 2) as i64 * product as i64;
    }
}
