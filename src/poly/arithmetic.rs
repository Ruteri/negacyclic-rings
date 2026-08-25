use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::fmt;

#[inline]
pub fn lift_i32(a: i32, modulus: i32) -> i32 {
    (a % modulus + modulus) % modulus
}

#[inline]
pub fn normalize_i32(mut a: i32, modulus: i32) -> i32 {
    a %= modulus;
    if a > modulus / 2 {
        a -= modulus;
    }
    if a < -modulus / 2 {
        a += modulus;
    }
    a
}

#[inline]
pub fn eq_i32<const N: usize>(a: &[i32; N], b: &[i32; N], modulus: i32) -> bool {
    a.iter()
        .zip(b)
        .all(|(&x, &y)| lift_i32(x, modulus) == lift_i32(y, modulus))
}

#[inline]
pub fn add_assign_i32<const N: usize>(lhs: &mut [i32; N], rhs: &[i32; N], modulus: i32) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = ((*x as i64 + y as i64) % modulus as i64) as i32;
    }
}

#[inline]
pub fn sub_assign_i32<const N: usize>(lhs: &mut [i32; N], rhs: &[i32; N], modulus: i32) {
    for (x, &y) in lhs.iter_mut().zip(rhs) {
        *x = ((*x as i64 - y as i64) % modulus as i64) as i32;
    }
}

pub fn fmt_i32<const N: usize>(coeffs: &[i32; N], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
        f,
        "{:#16x} {:#16x} {:#16x} {:#16x}",
        coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    )
}

pub fn schoolbook_i32<const N: usize>(a: &[i32; N], b: &[i32; N], modulus: i32) -> [i32; N] {
    let mut buf = vec![0i64; N * 2];
    let q = modulus as i64;
    for i in 0..N {
        for j in 0..N {
            buf[i + j] += a[i] as i64 * b[j] as i64 % q;
            buf[i + j] %= q;
        }
    }
    core::array::from_fn(|i| lift_i32(((buf[i] - buf[i + N]) % q) as i32, modulus))
}

pub fn rand_poly_i32<const N: usize, R: Rng>(
    rng: &mut R,
    modulus: i32,
    modulus_over_two: i32,
    threshold: u32,
) -> [i32; N] {
    core::array::from_fn(|_| {
        let mut value = rng.next_u32();
        while value >= threshold {
            value = rng.next_u32();
        }
        (value % modulus as u32) as i32 - modulus_over_two
    })
}

pub fn rand_balanced_ternary_i32<const N: usize, R: Rng>(
    rng: &mut R,
    half_weight: usize,
) -> [i32; N] {
    assert!(N.is_power_of_two());
    let log_n = N.trailing_zeros() as usize;
    let mut coeffs = [0; N];
    let mut value = rng.next_u32();
    let mut chunks = 0;
    for sign in [1, -1] {
        let mut count = 0usize;
        while count < half_weight {
            let index = (value & (N as u32 - 1)) as usize;
            value >>= log_n;
            chunks += 1;
            if chunks == 32 / log_n {
                value = rng.next_u32();
                chunks = 0;
            }
            if coeffs[index] == 0 {
                coeffs[index] = sign;
                count += 1;
            }
        }
    }
    coeffs
}

pub fn rand_binary_i32<const N: usize, R: Rng>(rng: &mut R) -> [i32; N] {
    let mut coeffs = [0; N];
    for chunk in coeffs.as_chunks_mut::<32>().0 {
        let mut value = rng.next_u32();
        for coefficient in chunk {
            *coefficient = (value & 1) as i32;
            value >>= 1;
        }
    }
    coeffs
}

pub fn rand_ternary_i32<const N: usize, R: Rng>(rng: &mut R, weight: usize) -> [i32; N] {
    assert!(N.is_power_of_two());
    let log_n = N.trailing_zeros() as usize;
    let mut coeffs = [0; N];
    let mut count = 0;
    while count < weight {
        let value = rng.next_u32() as usize;
        let index = value % N;
        if coeffs[index] == 0 {
            coeffs[index] = if (value >> log_n) & 1 == 1 { 1 } else { -1 };
            count += 1;
        }
    }
    coeffs
}

pub fn rand_mod_p_i32<const N: usize, R: Rng>(rng: &mut R, p: u32) -> [i32; N] {
    let modulus = p
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .expect("centered modulus exceeds u32");
    let threshold = ((1u64 << 32) / modulus as u64) * modulus as u64;
    core::array::from_fn(|_| {
        let mut value = rng.next_u32();
        while value as u64 >= threshold {
            value = rng.next_u32();
        }
        (value % modulus) as i32 - p as i32
    })
}

pub fn from_hash_message_i32<const N: usize>(msg: &[u8], weight: usize) -> [i32; N] {
    let seed: [u8; 32] = Sha256::digest(msg).into();
    let mut rng = rand_chacha::ChaCha20Rng::from_seed(seed);
    rand_ternary_i32(&mut rng, weight)
}

pub fn digest_i32<const N: usize>(coeffs: &[i32; N]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &coefficient in coeffs {
        hasher.update([
            (coefficient & 0xff) as u8,
            ((coefficient >> 8) & 0xff) as u8,
        ]);
    }
    hasher.finalize().into()
}

pub fn infinity_norm_i32<const N: usize>(coeffs: &[i32; N]) -> u32 {
    coeffs.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0)
}

pub fn is_ternary_i32<const N: usize>(coeffs: &[i32; N]) -> bool {
    coeffs.iter().all(|&x| matches!(x, -1..=1))
}

pub fn lift_assign_i32<const N: usize>(coeffs: &mut [i32; N], modulus: i32) {
    for x in coeffs {
        *x = lift_i32(*x, modulus);
    }
}

pub fn normalize_assign_i32<const N: usize>(coeffs: &mut [i32; N], modulus: i32) {
    for x in coeffs {
        *x = normalize_i32(*x, modulus);
    }
}

#[inline]
pub fn forward_ntt_i32<const N: usize>(p: &mut [i32; N], modulus: i32, table: &[i32; N]) {
    let q = modulus as i64;
    let mut t = N;
    for l in 0..N.trailing_zeros() as usize {
        let m = 1 << l;
        let half_t = t >> 1;
        for i in 0..m {
            let twiddle = table[m + i] as i64;
            let start = i * t;
            for j in start..start + half_t {
                let u = p[j] as i64;
                let v = p[j + half_t] as i64 * twiddle % q;
                p[j] = ((u + v) % q) as i32;
                p[j + half_t] = ((u + q - v) % q) as i32;
            }
        }
        t = half_t;
    }
}

#[inline]
pub fn inverse_ntt_i32<const N: usize>(
    p: &mut [i32; N],
    modulus: i32,
    inverse_table: &[i32; N],
    inverse_n: i32,
) {
    let q = modulus as i64;
    let mut t = 1;
    let mut m = N;
    while m > 1 {
        let half_m = m >> 1;
        let double_t = t << 1;
        for i in 0..half_m {
            let twiddle = inverse_table[half_m + i] as i64;
            let start = i * double_t;
            for j in start..start + t {
                let u = p[j] as i64;
                let v = p[j + t] as i64;
                p[j] = ((u + v) % q) as i32;
                p[j + t] = ((u + q - v) * twiddle % q) as i32;
            }
        }
        t = double_t;
        m = half_m;
    }
    for x in p {
        *x = (*x as i64 * inverse_n as i64 % q) as i32;
    }
}
