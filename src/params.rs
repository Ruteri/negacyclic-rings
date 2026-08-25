use crate::ntt32::Ring32;
use crate::ntt64::Ring64;

fn pow_mod(mut base: u128, mut exponent: u128, modulus: u128) -> u128 {
    let mut result = 1u128;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    result
}

fn reverse_bits(value: usize, bits: u32) -> usize {
    value.reverse_bits() >> (usize::BITS - bits)
}

fn validate_root<const N: usize>(modulus: u128, psi: u128) {
    assert!(N.is_power_of_two(), "NTT degree must be a power of two");
    assert_eq!((modulus - 1) % (2 * N) as u128, 0);
    assert_eq!(pow_mod(psi, N as u128, modulus), modulus - 1);
    assert_eq!(pow_mod(psi, (2 * N) as u128, modulus), 1);
}

pub fn find_psi32<const N: usize>(modulus: u32) -> u32 {
    assert_eq!((modulus as usize - 1) % (2 * N), 0);
    let exponent = (modulus as usize - 1) / (2 * N);
    for base in 2..modulus {
        let psi = pow_mod(base as u128, exponent as u128, modulus as u128) as u32;
        if pow_mod(psi as u128, N as u128, modulus as u128) == modulus as u128 - 1 {
            return psi;
        }
    }
    panic!("no primitive 2N-th root found");
}

pub fn find_psi64<const N: usize>(modulus: u64) -> u64 {
    assert_eq!((modulus as u128 - 1) % (2 * N) as u128, 0);
    let exponent = (modulus as u128 - 1) / (2 * N) as u128;
    for base in 2..modulus {
        let psi = pow_mod(base as u128, exponent, modulus as u128) as u64;
        if pow_mod(psi as u128, N as u128, modulus as u128) == modulus as u128 - 1 {
            return psi;
        }
    }
    panic!("no primitive 2N-th root found");
}

pub fn generate_ring32<const N: usize>(modulus: u32, psi: u32) -> Ring32<N> {
    assert!(modulus % 2 == 1);
    assert!(2 * (modulus as u64) < 1 << 31);
    validate_root::<N>(modulus as u128, psi as u128);

    let bits = N.trailing_zeros();
    let psi_inv = pow_mod(psi as u128, modulus as u128 - 2, modulus as u128) as u32;
    let ntt_table = core::array::from_fn(|i| {
        pow_mod(psi as u128, reverse_bits(i, bits) as u128, modulus as u128) as u32
    });
    let inv_ntt_table = core::array::from_fn(|i| {
        pow_mod(
            psi_inv as u128,
            reverse_bits(i, bits) as u128,
            modulus as u128,
        ) as u32
    });
    let ntt_table_shoup =
        core::array::from_fn(|i| (((ntt_table[i] as u64) << 32) / modulus as u64) as u32);
    let inv_ntt_table_shoup =
        core::array::from_fn(|i| (((inv_ntt_table[i] as u64) << 32) / modulus as u64) as u32);
    let n_inv = pow_mod(N as u128, modulus as u128 - 2, modulus as u128) as u32;
    let mut q_inv = 1u32;
    for _ in 0..5 {
        q_inv = q_inv.wrapping_mul(2u32.wrapping_sub(modulus.wrapping_mul(q_inv)));
    }

    Ring32 {
        q: modulus,
        q_inv_neg: q_inv.wrapping_neg(),
        r2: pow_mod(2, 64, modulus as u128) as u32,
        ntt_table,
        ntt_table_shoup,
        inv_ntt_table,
        inv_ntt_table_shoup,
        n_inv,
        n_inv_shoup: (((n_inv as u64) << 32) / modulus as u64) as u32,
    }
}

pub fn generate_ring64<const N: usize>(modulus: u64, psi: u64) -> Ring64<N> {
    assert!(modulus % 2 == 1);
    assert!(modulus < 1 << 62);
    validate_root::<N>(modulus as u128, psi as u128);

    let bits = N.trailing_zeros();
    let psi_inv = pow_mod(psi as u128, modulus as u128 - 2, modulus as u128) as u64;
    let ntt_table = core::array::from_fn(|i| {
        pow_mod(psi as u128, reverse_bits(i, bits) as u128, modulus as u128) as u64
    });
    let inv_ntt_table = core::array::from_fn(|i| {
        pow_mod(
            psi_inv as u128,
            reverse_bits(i, bits) as u128,
            modulus as u128,
        ) as u64
    });
    let ntt_table_shoup =
        core::array::from_fn(|i| (((ntt_table[i] as u128) << 64) / modulus as u128) as u64);
    let inv_ntt_table_shoup =
        core::array::from_fn(|i| (((inv_ntt_table[i] as u128) << 64) / modulus as u128) as u64);
    let n_inv = pow_mod(N as u128, modulus as u128 - 2, modulus as u128) as u64;
    let mut q_inv = 1u64;
    for _ in 0..6 {
        q_inv = q_inv.wrapping_mul(2u64.wrapping_sub(modulus.wrapping_mul(q_inv)));
    }

    Ring64 {
        q: modulus,
        q_inv_neg: q_inv.wrapping_neg(),
        r2: pow_mod(2, 128, modulus as u128) as u64,
        ntt_table,
        ntt_table_shoup,
        inv_ntt_table,
        inv_ntt_table_shoup,
        n_inv,
        n_inv_shoup: (((n_inv as u128) << 64) / modulus as u128) as u64,
    }
}
