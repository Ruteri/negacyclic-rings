#!/usr/bin/env python3
from paramgen import main

raise SystemExit(main())

"""Generate compile-time-const-selected NTT parameter tables for N in {512,1024,2048}.

Selection is driven by `crate::N` (a hand-edited const in param.rs), NOT cargo
features. Each table is emitted as three named consts (NAME_512/1024/2048) plus a
selector `NAME` chosen at const-eval time. N=512 data is taken verbatim from the
pristine originals so the shipped 512 parameter set is unchanged.
"""
import re, sympy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src/poly"                            # destination (in place)
R16 = 1 << 16
R32 = 1 << 32

def revbits(i, n):
    r = 0
    for _ in range(n):
        r = (r << 1) | (i & 1); i >>= 1
    return r
def find_psi(q, twoN):
    assert (q - 1) % twoN == 0, (q, twoN)
    psi = pow(sympy.primitive_root(q), (q - 1) // twoN, q)
    assert pow(psi, twoN, q) == 1 and pow(psi, twoN // 2, q) == q - 1
    return psi
def center(t, q):
    t %= q
    return t - q if t > q // 2 else t
def one_over_n(q, N): return pow(N, -1, q)
def mod_over_2(q):    return (q - 1) // 2
def sample_threshold(q): return ((1 << 32) // q) * q
def q_inv_pos_u16(q): return pow(q, -1, R16)
def q_inv_neg_u32(q): return (-pow(q, -1, R32)) % R32
def r2_mod_q(q):      return pow(2, 64, q)
def n_inv_mont_i16(q, N): return center(one_over_n(q, N) * R16, q)
def n_inv_mont_i32(q, N): return center(one_over_n(q, N) * R32, q)

PRIMES = {
    512:  {'HOTS': 3168257, 'HVC': 25601, 'KAHE': 347280875347969, 'CS': 139301},
    1024: {'HOTS': 3168257, 'HVC': 18433, 'KAHE': 347280875347969, 'CS': 139301},
    2048: {'HOTS': 3174401, 'HVC': 40961, 'KAHE': 347280875347969, 'CS': 139301},
}
# CS q = 139301 is NOT NTT-friendly (≡ 37 mod 4096). CsPoly multiplication is
# exact integer negacyclic convolution over this auxiliary prime (≡ 1 mod 4096),
# reduced mod q afterwards. p/2 ≈ 2^57 ≫ any convolution bound in the protocol.
CS_AUX = 288230376151748609
# Digest ring: the Ajtai hash of an aggregated KAHE ciphertext. Inputs are the
# *unreduced* integer sum Σ ct_i, so a collision has ‖·‖∞ ≤ ρ·q_kahe ≈ 2^56.5 at
# ρ = 300; SIS needs the modulus comfortably above that, hence 2^61 (ratio 2^4.5)
# rather than reusing CS_AUX at 2^58 (ratio 2^1.5, and ρ ≤ 207).
# ≡ 1 mod 4096 and < 2^62 as ntt64 requires.
DGT = 2305843009213616129
R64 = 1 << 64
# RNS chains (feature = "rns"): all ≡ 1 mod 8192 and < 2^30, so 2q < 2^31 keeps
# ntt32's signed-compare AVX2 kernels in range. Each replaces a 64-bit modulus so
# NTT products fit 32x32->64.
#   KAHE:   product 2^47.995, carries t = 2^35
#   CS_AUX: product 2^60, P/2 clears the 2^55.5 kappa=5 dot bound by 2^3.5
#   DGT:    product 2^60, q/2 clears rho*q_kahe = 2^56.2 by 2^2.8
KAHE_RNS = [16760833, 16736257]
CS_AUX_RNS = [1073692673, 1073668097]
DGT_RNS = [1073651713, 1073643521]

# ---- verify formulas reproduce existing N=512 constants ----
def check(label, got, want):
    assert got == want, f"FORMULA MISMATCH {label}: got {got} want {want}"
print("Verifying formulas vs existing N=512 constants...")
q = 25601
for l,g,w in [("HVC_ONE_OVER_N",one_over_n(q,512),25551),("HVC_MOD2",mod_over_2(q),12800),
              ("HVC_THR",sample_threshold(q),4294951765),("QINVP",q_inv_pos_u16(q),39937),
              ("NINV16",n_inv_mont_i16(q,512),128)]: check(l,g,w)
q = 3168257
for l,g,w in [("HOTS_INV",one_over_n(q,512),3162069),("HOTS_M2",mod_over_2(q),1584128),
              ("HOTS_THR",sample_threshold(q),4292988235)]: check(l,g,w)
q = 1073738753
for l,g,w in [("KAHE_INV",one_over_n(q,512),1071641607),("KAHE_M2",mod_over_2(q),536869376),
              ("KAHE_THR",sample_threshold(q),4294955012),("NINV32",n_inv_mont_i32(q,512),8388608),
              ("QINVN",q_inv_neg_u32(q),2138043391),("R2",r2_mod_q(q),150896656)]: check(l,g,w)
print("  all N=512 formula checks passed.\n")

# ---- table generators (N in {1024,2048}) ----
def fwd_table(q, N, mont=1):
    psi = find_psi(q, 2*N); L = N.bit_length()-1
    return [center(pow(psi, revbits(i,L), q)*mont, q) for i in range(N)]
def inv_table(q, N, mont=1):
    psi = find_psi(q, 2*N); pv = pow(psi,-1,q); L = N.bit_length()-1
    return [center(pow(pv, revbits(i,L), q)*mont, q) for i in range(N)]

# ---- 64-bit ring helpers (canonical [0, q) values + Shoup companions) ----
def fwd_table_u64(q, N):
    psi = find_psi(q, 2*N); L = N.bit_length()-1
    return [pow(psi, revbits(i,L), q) for i in range(N)]
def inv_table_u64(q, N):
    psi = find_psi(q, 2*N); pv = pow(psi,-1,q); L = N.bit_length()-1
    return [pow(pv, revbits(i,L), q) for i in range(N)]
def shoup_of(vals, q): return [v * R64 // q for v in vals]
def q_inv_neg_u64(q): return (-pow(q, -1, R64)) % R64
def r2_mod_q_u64(q):  return pow(2, 128, q)
def sample_threshold_u64(q): return (R64 // q) * q

# ---- 32-bit ring helpers (canonical [0, q) values + Shoup companions) ----
# Same Shoup/Montgomery scheme as the 64-bit core with R = 2^32 instead of 2^64,
# so every product is a single 32x32->64 multiply. Requires q < 2^31.
def shoup_of_u32(vals, q): return [v * R32 // q for v in vals]
def q_inv_neg_u32_(q): return (-pow(q, -1, R32)) % R32
def r2_mod_q_u32(q):  return pow(2, 64, q)
def sample_threshold_u32(q): return (R32 // q) * q

def fmt_named_array(name, ty, vals):
    body = ", ".join(str(v) for v in vals)
    return f"pub(crate) const {name}: [{ty}; {len(vals)}] = [\n    {body},\n];\n"

def array_block(name, ty, genfn):
    """NAME_512/1024/2048 arrays + an N-selected slice `NAME`."""
    return "\n".join([
        fmt_named_array(f"{name}_512", ty, genfn(512)),
        fmt_named_array(f"{name}_1024", ty, genfn(1024)),
        fmt_named_array(f"{name}_2048", ty, genfn(2048)),
        f"pub(crate) const {name}: &[{ty}] = crate::param::pick_slice_{ty}"
        f"(&{name}_512, &{name}_1024, &{name}_2048);\n",
    ])

def scalar_block(name, ty, genfn):
    return (f"pub(crate) const {name}: {ty} = match crate::N {{ "
            f"512 => {genfn(512)}, 1024 => {genfn(1024)}, 2048 => {genfn(2048)}, "
            f'_ => panic!("unsupported N") }};\n')

def write_param_file(rel, header, arrays, scalars=()):
    parts = [header.rstrip() + "\n"]
    for name, ty, genfn in arrays:
        parts.append(array_block(name, ty, genfn))
    for name, ty, genfn in scalars:
        parts.append(scalar_block(name, ty, genfn))
    open(SRC / rel, "w").write("\n".join(parts))
    print(f"  {rel}: {[a[0] for a in arrays] + [s[0] for s in scalars]}")

GEN = ("// @generated by scripts/gen_ntt_n.py — do not edit by hand.\n"
       "// Tables/constants for the N selected in param.rs (512/1024/2048).\n")

print("Generating tables (all N from PRIMES):")
write_param_file("hots/ntt_param.rs",
    GEN + "//! HOTS NTT twiddle tables. Entry i = psi^bit_reverse(i, log2 N) mod q (centered).",
    [("NTT_TABLE", "i32", lambda N: fwd_table(PRIMES[N]['HOTS'], N)),
     ("INV_NTT_TABLE", "i32", lambda N: inv_table(PRIMES[N]['HOTS'], N))])
write_param_file("hvc/hvc_ntt_param.rs",
    GEN + "//! HVC NTT twiddle tables. Entry i = psi^bit_reverse(i, log2 N) mod q (centered).",
    [("NTT_TABLE", "i32", lambda N: fwd_table(PRIMES[N]['HVC'], N)),
     ("INV_NTT_TABLE", "i32", lambda N: inv_table(PRIMES[N]['HVC'], N))])
write_param_file("hvc/hvc_mont_i16_param.rs",
    GEN + "//! HVC Montgomery-form (R=2^16) twiddle tables for the AVX2 i16 NTT.",
    [("NTT_TABLE_MONT_I16", "i16", lambda N: fwd_table(PRIMES[N]['HVC'], N, R16)),
     ("INV_NTT_TABLE_MONT_I16", "i16", lambda N: inv_table(PRIMES[N]['HVC'], N, R16))],
    [("N_INV_MONT_I16", "i16", lambda N: n_inv_mont_i16(PRIMES[N]['HVC'], N)),
     ("Q_INV_POS_U16", "u16", lambda N: q_inv_pos_u16(PRIMES[N]['HVC']))])
def ntt64_param_file(rel, ring_doc, q):
    """u64 NTT tables (canonical) + Shoup companions + Montgomery scalars for a
    64-bit ring with the same modulus at every N."""
    header = (GEN + ring_doc + "\n"
              f"//! q = {q} (same at every N). Tables canonical in [0, q); the\n"
              "//! _SHOUP companions hold floor(w·2^64/q) for Shoup multiplication.")
    write_param_file(rel, header,
        [("NTT_TABLE", "u64", lambda N: fwd_table_u64(q, N)),
         ("NTT_TABLE_SHOUP", "u64", lambda N: shoup_of(fwd_table_u64(q, N), q)),
         ("INV_NTT_TABLE", "u64", lambda N: inv_table_u64(q, N)),
         ("INV_NTT_TABLE_SHOUP", "u64", lambda N: shoup_of(inv_table_u64(q, N), q))],
        [("N_INV", "u64", lambda N: one_over_n(q, N)),
         ("N_INV_SHOUP", "u64", lambda N: one_over_n(q, N) * R64 // q)])
    with open(SRC / rel, "a") as f:
        f.write(f"\npub(crate) const MODULUS: u64 = {q};\n")
        f.write(f"pub(crate) const Q_INV_NEG64: u64 = {q_inv_neg_u64(q)};\n")
        f.write(f"pub(crate) const R2_MOD_Q: u64 = {r2_mod_q_u64(q)};\n")

ntt64_param_file("kahe/kahe_ntt64_param.rs",
    "//! KAHE 64-bit NTT tables. Entry i = psi^bit_reverse(i, log2 N) mod q.",
    PRIMES[2048]['KAHE'])
ntt64_param_file("cs/cs_aux_param.rs",
    "//! CS auxiliary-prime NTT tables (exact convolution ring; CS q itself is\n"
    "//! not NTT-friendly). Entry i = psi^bit_reverse(i, log2 N) mod p_aux.",
    CS_AUX)
ntt64_param_file("digest/dgt_param.rs",
    "//! Digest-ring NTT tables (Ajtai hash over aggregated KAHE ciphertexts;\n"
    "//! q chosen above the ρ·q_kahe collision bound). Entry i =\n"
    "//! psi^bit_reverse(i, log2 N) mod q.",
    DGT)

def ntt32_param_file(rel, ring_doc, primes):
    """u32 NTT tables + Shoup companions + Montgomery scalars, one set per RNS
    channel, plus the CRT constants for the product modulus."""
    header = (GEN + ring_doc + "\n"
              f"//! Channel moduli {primes}, product {primes[0]*primes[1]}.\n"
              "//! Tables canonical in [0, q); _SHOUP companions hold\n"
              "//! floor(w·2^32/q).")
    arrays, scalars = [], []
    for c, q in enumerate(primes):
        arrays += [
            (f"NTT_TABLE_C{c}", "u32", lambda N, q=q: fwd_table_u64(q, N)),
            (f"NTT_TABLE_SHOUP_C{c}", "u32", lambda N, q=q: shoup_of_u32(fwd_table_u64(q, N), q)),
            (f"INV_NTT_TABLE_C{c}", "u32", lambda N, q=q: inv_table_u64(q, N)),
            (f"INV_NTT_TABLE_SHOUP_C{c}", "u32", lambda N, q=q: shoup_of_u32(inv_table_u64(q, N), q)),
        ]
        scalars += [
            (f"N_INV_C{c}", "u32", lambda N, q=q: one_over_n(q, N)),
            (f"N_INV_SHOUP_C{c}", "u32", lambda N, q=q: one_over_n(q, N) * R32 // q),
        ]
    write_param_file(rel, header, arrays, scalars)
    q0, q1 = primes
    with open(SRC / rel, "a") as f:
        f.write(f"\npub(crate) const MODULI: [u32; {len(primes)}] = "
                f"[{', '.join(str(q) for q in primes)}];\n")
        f.write(f"pub(crate) const Q_INV_NEG32: [u32; {len(primes)}] = "
                f"[{', '.join(str(q_inv_neg_u32_(q)) for q in primes)}];\n")
        f.write(f"pub(crate) const R2_MOD_Q: [u32; {len(primes)}] = "
                f"[{', '.join(str(r2_mod_q_u32(q)) for q in primes)}];\n")
        f.write(f"pub(crate) const SAMPLE_THRESHOLD: [u32; {len(primes)}] = "
                f"[{', '.join(str(sample_threshold_u32(q)) for q in primes)}];\n")
        # Garner: x = r0 + q0·((r1 - r0)·Q0_INV_MOD_Q1 mod q1), then centered.
        f.write(f"pub(crate) const Q0_INV_MOD_Q1: u32 = {pow(q0, -1, q1)};\n")
        f.write(f"pub(crate) const Q_PRODUCT: i64 = {q0 * q1};\n")
        f.write("const _: () = assert!(\n")
        f.write("    2 * (MODULI[0] as u64) < (1 << 31) && 2 * (MODULI[1] as u64) < (1 << 31),\n")
        f.write('    "ntt32 AVX2 csub32 uses signed compares: needs 2q < 2^31",\n')
        f.write(");\n")

ntt32_param_file("kahe/kahe_rns_param.rs",
    "//! KAHE RNS NTT tables. Entry i = psi^bit_reverse(i, log2 N) mod q.",
    KAHE_RNS)
ntt32_param_file("cs/cs_aux_rns_param.rs",
    "//! CS auxiliary-prime RNS NTT tables (exact integer convolution ring).\n"
    "//! Entry i = psi^bit_reverse(i, log2 N) mod q.",
    CS_AUX_RNS)
ntt32_param_file("digest/dgt_rns_param.rs",
    "//! Digest-ring RNS NTT tables (Ajtai hash over aggregated KAHE\n"
    "//! ciphertexts). Entry i = psi^bit_reverse(i, log2 N) mod q.",
    DGT_RNS)

# ---- rewrite param.rs ring constants (script owns the magic numbers) ----
# Per-ring const types: KAHE is a 64-bit ring; the others fit i32.
RING_TYPES = {
    'HOTS': ("i32", "sel_i32", "u32", "sel_u32", sample_threshold),
    'HVC':  ("i32", "sel_i32", "u32", "sel_u32", sample_threshold),
    'CS':   ("i32", "sel_i32", "u32", "sel_u32", sample_threshold),
    'KAHE': ("i64", "sel_i64", "u64", "sel_u64", sample_threshold_u64),
}
def rewrite_param_rs():
    path = ROOT / "src/param.rs"
    src = open(path).read()
    def sub(name, ty, fn, vals):
        nonlocal src
        new = f"pub const {name}: {ty} = {fn}({vals[0]}, {vals[1]}, {vals[2]});"
        src, n = re.subn(rf'pub const {name}: {ty} = [^;]+;', new, src, count=1)
        if n == 0: raise SystemExit(f"param.rs: {name} ({ty}) not found")
    for ring in ('HOTS', 'HVC', 'KAHE', 'CS'):
        ity, isel, uty, usel, thr = RING_TYPES[ring]
        qs = [PRIMES[N][ring] for N in (512, 1024, 2048)]
        sub(f"{ring}_MODULUS", ity, isel, qs)
        sub(f"{ring}_ONE_OVER_N", ity, isel, [one_over_n(qs[i], n) for i, n in enumerate((512,1024,2048))])
        sub(f"{ring}_MODULUS_OVER_TWO", ity, isel, [mod_over_2(q) for q in qs])
        sub(f"{ring}_SAMPLE_THRESHOLD", uty, usel, [thr(q) for q in qs])
    new = f"pub const KAHE_RNS_PRIMES: [i32; {len(KAHE_RNS)}] = {KAHE_RNS};"
    src, n = re.subn(r'pub const KAHE_RNS_PRIMES: \[i32; \d+\] = [^;]+;', new, src, count=1)
    if n == 0: raise SystemExit("param.rs: KAHE_RNS_PRIMES not found")
    open(path, "w").write(src)
    print("  rewrote src/param.rs ring constants")

print("Rewriting source-level magic numbers:")
rewrite_param_rs()

print("\n==== 64-bit ring scalars ====")
for label, q in [("KAHE", PRIMES[2048]["KAHE"]), ("CS_AUX", CS_AUX), ("DGT", DGT)]:
    print(f"{label}: q={q} qinv_neg64={q_inv_neg_u64(q)} r2={r2_mod_q_u64(q)} thr={sample_threshold_u64(q)}")
print("\nDONE")
