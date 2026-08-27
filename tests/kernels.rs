use negacyclic_rings::ntt32;
use negacyclic_rings::ntt64;
use negacyclic_rings::params::{find_psi32, find_psi64, generate_ring32, generate_ring64};
use negacyclic_rings::Rns;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 64;
const Q: u32 = 12_289;

fn schoolbook(a: &[u32; N], b: &[u32; N], q: u32) -> [u32; N] {
    let mut out = [0i128; N];
    for i in 0..N {
        for j in 0..N {
            let product = a[i] as i128 * b[j] as i128;
            if i + j < N {
                out[i + j] += product;
            } else {
                out[i + j - N] -= product;
            }
        }
    }
    core::array::from_fn(|i| out[i].rem_euclid(q as i128) as u32)
}

#[test]
fn ntt32_roundtrip_and_product() {
    let ring = generate_ring32::<N>(Q, find_psi32::<N>(Q));
    let mut rng = ChaCha20Rng::from_seed([1; 32]);
    let a = core::array::from_fn(|_| rng.gen_range(0..Q));
    let b = core::array::from_fn(|_| rng.gen_range(0..Q));
    let mut an = a;
    let mut bn = b;
    ntt32::ntt(&ring, &mut an);
    ntt32::ntt(&ring, &mut bn);
    let mut product = ntt32::pointwise_mul(&ring, &an, &bn);
    ntt32::inv_ntt(&ring, &mut product);
    assert_eq!(product, schoolbook(&a, &b, Q));
    ntt32::inv_ntt(&ring, &mut an);
    assert_eq!(an, a);
}

#[test]
fn ntt32_pointwise_dot_matches_separate_products() {
    let ring = generate_ring32::<N>(Q, find_psi32::<N>(Q));
    let mut rng = ChaCha20Rng::from_seed([5; 32]);
    let a: [[u32; N]; 3] = core::array::from_fn(|_| core::array::from_fn(|_| rng.gen_range(0..Q)));
    let b: [[u32; N]; 3] = core::array::from_fn(|_| core::array::from_fn(|_| rng.gen_range(0..Q)));
    let dot = ntt32::pointwise_dot(&ring, &a, &b);
    let expected = core::array::from_fn(|i| {
        a.iter().zip(&b).fold(0, |sum, (x, y)| {
            ntt32::add_mod(sum, ntt32::mul_mod(x[i], y[i], &ring), Q)
        })
    });
    assert_eq!(dot, expected);
}

#[test]
fn ntt64_roundtrip_and_product() {
    let q = Q as u64;
    let ring = generate_ring64::<N>(q, find_psi64::<N>(q));
    let mut rng = ChaCha20Rng::from_seed([2; 32]);
    let a = core::array::from_fn(|_| rng.gen_range(0..q));
    let b = core::array::from_fn(|_| rng.gen_range(0..q));
    let mut an = a;
    let mut bn = b;
    ntt64::ntt(&ring, &mut an);
    ntt64::ntt(&ring, &mut bn);
    let mut product = ntt64::pointwise_mul(&ring, &an, &bn);
    ntt64::inv_ntt(&ring, &mut product);
    let a32 = a.map(|x| x as u32);
    let b32 = b.map(|x| x as u32);
    assert_eq!(product.map(|x| x as u32), schoolbook(&a32, &b32, Q));
    ntt64::inv_ntt(&ring, &mut an);
    assert_eq!(an, a);
}

#[test]
fn three_limb_rns_roundtrip_and_ntt() {
    let moduli = [12_289u32, 7_681, 3_329];
    let rns = Rns::new(moduli.map(|q| generate_ring32::<N>(q, find_psi32::<N>(q))));
    for value in [0i128, 1, 12_288, 91_337, rns.product as i128 / 2] {
        assert_eq!(rns.lift_coeff(rns.reduce_coeff(value)), value as u128);
    }
    assert_eq!(rns.lift_centered(rns.reduce_coeff(-123_456)), -123_456);

    let mut rng = ChaCha20Rng::from_seed([3; 32]);
    let mut residues =
        core::array::from_fn(|limb| core::array::from_fn(|_| rng.gen_range(0..rns.ch[limb].q)));
    let original = residues;
    rns.forward(&mut residues);
    rns.inverse(&mut residues);
    assert_eq!(residues, original);
}

#[test]
fn rns_24_bit_ntt_pointwise_and_mac() {
    let moduli = [16_760_833u32, 16_736_257];
    let rns = Rns::new(moduli.map(|q| generate_ring32::<N>(q, find_psi32::<N>(q))));
    let mut rng = ChaCha20Rng::from_seed([4; 32]);
    let a = core::array::from_fn(|limb| core::array::from_fn(|_| rng.gen_range(0..rns.ch[limb].q)));
    let b = core::array::from_fn(|limb| core::array::from_fn(|_| rng.gen_range(0..rns.ch[limb].q)));
    let mut an = a;
    let mut bn = b;
    rns.forward(&mut an);
    rns.forward(&mut bn);

    let mut product = rns.pointwise_mul(&an, &bn);
    rns.inverse(&mut product);
    for limb in 0..2 {
        assert_eq!(product[limb], schoolbook(&a[limb], &b[limb], moduli[limb]));
    }

    let mut mac = [[0u32; N]; 2];
    rns.pointwise_mac(&mut mac, &[&an], &[&bn]);
    rns.inverse(&mut mac);
    assert_eq!(mac, product);
}

#[test]
fn bulk_signed_reduction_matches_scalar() {
    let moduli = [16_760_833u32, 16_736_257, 12_289];
    let rns = Rns::new(moduli.map(|q| generate_ring32::<N>(q, find_psi32::<N>(q))));
    let cases = [
        i64::MIN,
        i64::MIN + 1,
        -280_513_608_622_080,
        -1,
        0,
        1,
        280_513_608_622_080,
        i64::MAX,
    ];
    let input = core::array::from_fn(|i| cases[i % cases.len()]);
    let mut residues = [[0u32; N]; 3];
    rns.reduce_i64_into(&input, &mut residues);
    for i in 0..N {
        let expected = rns.reduce_coeff(input[i] as i128);
        assert_eq!(core::array::from_fn(|limb| residues[limb][i]), expected);
    }
}

#[test]
fn bulk_centered_i32_reduction_matches_scalar() {
    let moduli = [16_760_833u32, 16_736_257];
    let rns = Rns::new(moduli.map(|q| generate_ring32::<N>(q, find_psi32::<N>(q))));
    let input_modulus = 139_301i32;
    let cases = [-input_modulus / 2, -1, 0, 1, input_modulus / 2];
    let input = core::array::from_fn(|i| cases[i % cases.len()]);
    let mut residues = [[0u32; N]; 2];
    rns.reduce_centered_i32_into(&input, &mut residues);
    for i in 0..N {
        assert_eq!(
            [residues[0][i], residues[1][i]],
            rns.reduce_coeff(input[i] as i128)
        );
    }
}

#[test]
fn bulk_two_limb_centered_lift_matches_garner() {
    for moduli in [[16_760_833u32, 16_736_257], [16_760_833, 40_961]] {
        let rns = Rns::new(moduli.map(|q| generate_ring32::<N>(q, find_psi32::<N>(q))));
        let mut rng = ChaCha20Rng::from_seed([6; 32]);
        let residues =
            core::array::from_fn(|limb| core::array::from_fn(|_| rng.gen_range(0..rns.ch[limb].q)));
        let mut lifted = [0i64; N];
        rns.lift_centered_i64_into(&residues, &mut lifted);
        for i in 0..N {
            assert_eq!(
                lifted[i] as i128,
                rns.lift_centered([residues[0][i], residues[1][i]])
            );
        }
    }
}
