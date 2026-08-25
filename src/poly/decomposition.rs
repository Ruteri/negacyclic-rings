use super::arithmetic::normalize_i32;

pub fn decompose_i32<const N: usize, const WIDTH: usize>(
    input: &[i32; N],
    source_modulus: Option<i32>,
    base: i32,
    centered_digits: bool,
) -> [[i32; N]; WIDTH] {
    let mut remaining = *input;
    if let Some(modulus) = source_modulus {
        for value in &mut remaining {
            *value = normalize_i32(*value, modulus);
        }
    }
    let mut digits = [[0; N]; WIDTH];
    for digit in &mut digits {
        for (output, value) in digit.iter_mut().zip(&mut remaining) {
            *output = if centered_digits {
                normalize_i32(*value, base)
            } else {
                *value % base
            };
            *value = (*value - *output) / base;
        }
    }
    digits
}

pub fn recompose_i32<const N: usize, const WIDTH: usize>(
    digits: &[[i32; N]; WIDTH],
    base: i32,
    digit_modulus: Option<i32>,
    target_modulus: Option<i32>,
) -> [i32; N] {
    let mut result = digits[WIDTH - 1];
    if let Some(modulus) = digit_modulus {
        for value in &mut result {
            *value = normalize_i32(*value, modulus);
        }
    }
    for digit in digits.iter().rev().skip(1) {
        for (value, &next) in result.iter_mut().zip(digit) {
            let next = digit_modulus.map_or(next, |modulus| normalize_i32(next, modulus));
            *value = *value * base + next;
        }
    }
    if let Some(modulus) = target_modulus {
        for value in &mut result {
            *value = normalize_i32(*value, modulus);
        }
    }
    result
}
