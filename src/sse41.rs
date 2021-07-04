#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{
    counter_high, counter_low, CVBytes, CVWords, IncrementCounter, BLOCK_LEN, IV, MSG_SCHEDULE,
    OUT_LEN,
};
use arrayref::{array_mut_ref, array_ref, mut_array_refs};

pub const DEGREE: usize = 4;

#[inline(always)]
unsafe fn loadu(src: *const u8) -> __m128i {
    // This is an unaligned load, so the pointer cast is allowed.
    _mm_loadu_si128(src as *const __m128i)
}

#[inline(always)]
unsafe fn storeu(src: __m128i, dest: *mut u8) {
    // This is an unaligned store, so the pointer cast is allowed.
    _mm_storeu_si128(dest as *mut __m128i, src)
}

#[inline(always)]
unsafe fn add(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi32(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(a, b)
}

#[inline(always)]
unsafe fn set1(x: u32) -> __m128i {
    _mm_set1_epi32(x as i32)
}

#[inline(always)]
unsafe fn set4(a: u32, b: u32, c: u32, d: u32) -> __m128i {
    _mm_setr_epi32(a as i32, b as i32, c as i32, d as i32)
}

// These rotations are the "simple version". For the "complicated version", see
// https://github.com/sneves/blake2-avx2/blob/b3723921f668df09ece52dcd225a36d4a4eea1d9/blake2s-common.h#L63-L66.
// For a discussion of the tradeoffs, see
// https://github.com/sneves/blake2-avx2/pull/5. In short:
// - This version performs better on modern x86 chips, Skylake and later.
// - LLVM is able to optimize this version to AVX-512 rotation instructions
//   when those are enabled.

#[inline(always)]
unsafe fn rot16(a: __m128i) -> __m128i {
    _mm_or_si128(_mm_srli_epi32(a, 16), _mm_slli_epi32(a, 32 - 16))
}

#[inline(always)]
unsafe fn rot12(a: __m128i) -> __m128i {
    _mm_or_si128(_mm_srli_epi32(a, 12), _mm_slli_epi32(a, 32 - 12))
}

#[inline(always)]
unsafe fn rot8(a: __m128i) -> __m128i {
    _mm_or_si128(_mm_srli_epi32(a, 8), _mm_slli_epi32(a, 32 - 8))
}

#[inline(always)]
unsafe fn rot7(a: __m128i) -> __m128i {
    _mm_or_si128(_mm_srli_epi32(a, 7), _mm_slli_epi32(a, 32 - 7))
}

#[inline(always)]
unsafe fn g1(
    row1: &mut __m128i,
    row2: &mut __m128i,
    row3: &mut __m128i,
    row4: &mut __m128i,
    m: __m128i,
) {
    *row1 = add(add(*row1, m), *row2);
    *row4 = xor(*row4, *row1);
    *row4 = rot16(*row4);
    *row3 = add(*row3, *row4);
    *row2 = xor(*row2, *row3);
    *row2 = rot12(*row2);
}

#[inline(always)]
unsafe fn g2(
    row1: &mut __m128i,
    row2: &mut __m128i,
    row3: &mut __m128i,
    row4: &mut __m128i,
    m: __m128i,
) {
    *row1 = add(add(*row1, m), *row2);
    *row4 = xor(*row4, *row1);
    *row4 = rot8(*row4);
    *row3 = add(*row3, *row4);
    *row2 = xor(*row2, *row3);
    *row2 = rot7(*row2);
}

// Adapted from https://github.com/rust-lang-nursery/stdsimd/pull/479.
macro_rules! _MM_SHUFFLE {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

// Note the optimization here of leaving row2 as the unrotated row, rather than
// row1. All the message loads below are adjusted to compensate for this. See
// discussion at https://github.com/sneves/blake2-avx2/pull/4
#[inline(always)]
unsafe fn diagonalize(row1: &mut __m128i, row3: &mut __m128i, row4: &mut __m128i) {
    *row1 = _mm_shuffle_epi32(*row1, _MM_SHUFFLE!(2, 1, 0, 3));
    *row4 = _mm_shuffle_epi32(*row4, _MM_SHUFFLE!(1, 0, 3, 2));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(0, 3, 2, 1));
}

#[inline(always)]
unsafe fn undiagonalize(row1: &mut __m128i, row3: &mut __m128i, row4: &mut __m128i) {
    *row1 = _mm_shuffle_epi32(*row1, _MM_SHUFFLE!(0, 3, 2, 1));
    *row4 = _mm_shuffle_epi32(*row4, _MM_SHUFFLE!(1, 0, 3, 2));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(2, 1, 0, 3));
}

#[inline(always)]
unsafe fn compress_pre(
    cv: &CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) -> [__m128i; 4] {
    let row1 = &mut loadu(cv.as_ptr().add(0) as *const u8);
    let row2 = &mut loadu(cv.as_ptr().add(4) as *const u8);
    let row3 = &mut set4(IV[0], IV[1], IV[2], IV[3]);
    let row4 = &mut set4(
        counter_low(counter),
        counter_high(counter),
        block_len as u32,
        flags as u32,
    );

    let m0 = loadu(block.as_ptr().add(0 * 4 * DEGREE));
    let m1 = loadu(block.as_ptr().add(1 * 4 * DEGREE));
    let m2 = loadu(block.as_ptr().add(2 * 4 * DEGREE));
    let m3 = loadu(block.as_ptr().add(3 * 4 * DEGREE));

    // round 1
    let buf = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(m0),
        _mm_castsi128_ps(m1),
        _MM_SHUFFLE!(2, 0, 2, 0),
    ));
    g1(row1, row2, row3, row4, buf);
    let buf = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(m0),
        _mm_castsi128_ps(m1),
        _MM_SHUFFLE!(3, 1, 3, 1),
    ));
    g2(row1, row2, row3, row4, buf);
    diagonalize(row1, row3, row4);
    let t0 = _mm_shuffle_epi32(m2, _MM_SHUFFLE!(3, 2, 0, 1));
    let t1 = _mm_shuffle_epi32(m3, _MM_SHUFFLE!(0, 1, 3, 2));
    let buf = _mm_blend_epi16(t0, t1, 0xC3);
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_blend_epi16(t0, t1, 0x3C);
    let buf = _mm_shuffle_epi32(t0, _MM_SHUFFLE!(2, 3, 0, 1));
    g2(row1, row2, row3, row4, buf);
    undiagonalize(row1, row3, row4);

    // round 2
    let t0 = _mm_blend_epi16(m1, m2, 0x0C);
    let t1 = _mm_slli_si128(m3, 4);
    let t2 = _mm_blend_epi16(t0, t1, 0xF0);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 1, 0, 3));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_shuffle_epi32(m2, _MM_SHUFFLE!(0, 0, 2, 0));
    let t1 = _mm_blend_epi16(m1, m3, 0xC0);
    let t2 = _mm_blend_epi16(t0, t1, 0xF0);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 3, 0, 1));
    g2(row1, row2, row3, row4, buf);
    diagonalize(row1, row3, row4);
    let t0 = _mm_slli_si128(m1, 4);
    let t1 = _mm_blend_epi16(m2, t0, 0x30);
    let t2 = _mm_blend_epi16(m0, t1, 0xF0);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(3, 0, 1, 2));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_unpackhi_epi32(m0, m1);
    let t1 = _mm_slli_si128(m3, 4);
    let t2 = _mm_blend_epi16(t0, t1, 0x0C);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(3, 0, 1, 2));
    g2(row1, row2, row3, row4, buf);
    undiagonalize(row1, row3, row4);

    // round 3
    let t0 = _mm_unpackhi_epi32(m2, m3);
    let t1 = _mm_blend_epi16(m3, m1, 0x0C);
    let t2 = _mm_blend_epi16(t0, t1, 0x0F);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(3, 1, 0, 2));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_unpacklo_epi32(m2, m0);
    let t1 = _mm_blend_epi16(t0, m0, 0xF0);
    let t2 = _mm_slli_si128(m3, 8);
    let buf = _mm_blend_epi16(t1, t2, 0xC0);
    g2(row1, row2, row3, row4, buf);
    diagonalize(row1, row3, row4);
    let t0 = _mm_blend_epi16(m0, m2, 0x3C);
    let t1 = _mm_srli_si128(m1, 12);
    let t2 = _mm_blend_epi16(t0, t1, 0x03);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(0, 3, 2, 1));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_slli_si128(m3, 4);
    let t1 = _mm_blend_epi16(m0, m1, 0x33);
    let t2 = _mm_blend_epi16(t1, t0, 0xC0);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(1, 2, 3, 0));
    g2(row1, row2, row3, row4, buf);
    undiagonalize(row1, row3, row4);

    // round 4
    let t0 = _mm_unpackhi_epi32(m0, m1);
    let t1 = _mm_unpackhi_epi32(t0, m2);
    let t2 = _mm_blend_epi16(t1, m3, 0x0C);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(3, 1, 0, 2));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_slli_si128(m2, 8);
    let t1 = _mm_blend_epi16(m3, m0, 0x0C);
    let t2 = _mm_blend_epi16(t1, t0, 0xC0);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 0, 1, 3));
    g2(row1, row2, row3, row4, buf);
    diagonalize(row1, row3, row4);
    let t0 = _mm_blend_epi16(m0, m1, 0x0F);
    let t1 = _mm_blend_epi16(t0, m3, 0xC0);
    let buf = _mm_shuffle_epi32(t1, _MM_SHUFFLE!(0, 1, 2, 3));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_alignr_epi8(m0, m1, 4);
    let buf = _mm_blend_epi16(t0, m2, 0x33);
    g2(row1, row2, row3, row4, buf);
    undiagonalize(row1, row3, row4);

    // round 5
    let t0 = _mm_unpacklo_epi64(m1, m2);
    let t1 = _mm_unpackhi_epi64(m0, m2);
    let t2 = _mm_blend_epi16(t0, t1, 0x33);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 0, 1, 3));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_unpackhi_epi64(m1, m3);
    let t1 = _mm_unpacklo_epi64(m0, m1);
    let buf = _mm_blend_epi16(t0, t1, 0x33);
    g2(row1, row2, row3, row4, buf);
    diagonalize(row1, row3, row4);
    let t0 = _mm_unpackhi_epi64(m3, m1);
    let t1 = _mm_unpackhi_epi64(m2, m0);
    let t2 = _mm_blend_epi16(t1, t0, 0x33);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 1, 0, 3));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_blend_epi16(m0, m2, 0x03);
    let t1 = _mm_slli_si128(t0, 8);
    let t2 = _mm_blend_epi16(t1, m3, 0x0F);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 0, 3, 1));
    g2(row1, row2, row3, row4, buf);
    undiagonalize(row1, row3, row4);

    // round 6
    let t0 = _mm_unpackhi_epi32(m0, m1);
    let t1 = _mm_unpacklo_epi32(m0, m2);
    let buf = _mm_unpacklo_epi64(t0, t1);
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_srli_si128(m2, 4);
    let t1 = _mm_blend_epi16(m0, m3, 0x03);
    let buf = _mm_blend_epi16(t1, t0, 0x3C);
    g2(row1, row2, row3, row4, buf);
    diagonalize(row1, row3, row4);
    let t0 = _mm_blend_epi16(m1, m0, 0x0C);
    let t1 = _mm_srli_si128(m3, 4);
    let t2 = _mm_blend_epi16(t0, t1, 0x30);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 3, 0, 1));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_unpacklo_epi64(m2, m1);
    let t1 = _mm_shuffle_epi32(m3, _MM_SHUFFLE!(2, 0, 1, 0));
    let t2 = _mm_srli_si128(t0, 4);
    let buf = _mm_blend_epi16(t1, t2, 0x33);
    g2(row1, row2, row3, row4, buf);
    undiagonalize(row1, row3, row4);

    // round 7
    let t0 = _mm_slli_si128(m1, 12);
    let t1 = _mm_blend_epi16(m0, m3, 0x33);
    let buf = _mm_blend_epi16(t1, t0, 0xC0);
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_blend_epi16(m3, m2, 0x30);
    let t1 = _mm_srli_si128(m1, 4);
    let t2 = _mm_blend_epi16(t0, t1, 0x03);
    let buf = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 1, 3, 0));
    g2(row1, row2, row3, row4, buf);
    diagonalize(row1, row3, row4);
    let t0 = _mm_unpacklo_epi64(m0, m2);
    let t1 = _mm_srli_si128(m1, 4);
    let buf = _mm_shuffle_epi32(_mm_blend_epi16(t0, t1, 0x0C), _MM_SHUFFLE!(3, 1, 0, 2));
    g1(row1, row2, row3, row4, buf);
    let t0 = _mm_unpackhi_epi32(m1, m2);
    let t1 = _mm_unpackhi_epi64(m0, t0);
    let buf = _mm_shuffle_epi32(t1, _MM_SHUFFLE!(0, 1, 2, 3));
    g2(row1, row2, row3, row4, buf);
    undiagonalize(row1, row3, row4);

    [*row1, *row2, *row3, *row4]
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn compress_in_place(
    cv: &mut CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    let [row1, row2, row3, row4] = compress_pre(cv, block, block_len, counter, flags);
    storeu(xor(row1, row3), cv.as_mut_ptr().add(0) as *mut u8);
    storeu(xor(row2, row4), cv.as_mut_ptr().add(4) as *mut u8);
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn compress_xof(
    cv: &CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) -> [u8; 64] {
    let [mut row1, mut row2, mut row3, mut row4] =
        compress_pre(cv, block, block_len, counter, flags);
    row1 = xor(row1, row3);
    row2 = xor(row2, row4);
    row3 = xor(row3, loadu(cv.as_ptr().add(0) as *const u8));
    row4 = xor(row4, loadu(cv.as_ptr().add(4) as *const u8));
    core::mem::transmute([row1, row2, row3, row4])
}

#[inline(always)]
unsafe fn round(v: &mut [__m128i; 16], m: &[__m128i; 16], r: usize) {
    v[0] = add(v[0], m[MSG_SCHEDULE[r][0] as usize]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][2] as usize]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][4] as usize]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][6] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[15] = rot16(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot12(v[4]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[0] = add(v[0], m[MSG_SCHEDULE[r][1] as usize]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][3] as usize]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][5] as usize]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][7] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[15] = rot8(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot7(v[4]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);

    v[0] = add(v[0], m[MSG_SCHEDULE[r][8] as usize]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][10] as usize]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][12] as usize]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][14] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot16(v[15]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[4] = rot12(v[4]);
    v[0] = add(v[0], m[MSG_SCHEDULE[r][9] as usize]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][11] as usize]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][13] as usize]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][15] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot8(v[15]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);
    v[4] = rot7(v[4]);
}

#[inline(always)]
unsafe fn transpose_vecs(vecs: &mut [__m128i; DEGREE]) {
    // Interleave 32-bit lates. The low unpack is lanes 00/11 and the high is
    // 22/33. Note that this doesn't split the vector into two lanes, as the
    // AVX2 counterparts do.
    let ab_01 = _mm_unpacklo_epi32(vecs[0], vecs[1]);
    let ab_23 = _mm_unpackhi_epi32(vecs[0], vecs[1]);
    let cd_01 = _mm_unpacklo_epi32(vecs[2], vecs[3]);
    let cd_23 = _mm_unpackhi_epi32(vecs[2], vecs[3]);

    // Interleave 64-bit lanes.
    let abcd_0 = _mm_unpacklo_epi64(ab_01, cd_01);
    let abcd_1 = _mm_unpackhi_epi64(ab_01, cd_01);
    let abcd_2 = _mm_unpacklo_epi64(ab_23, cd_23);
    let abcd_3 = _mm_unpackhi_epi64(ab_23, cd_23);

    vecs[0] = abcd_0;
    vecs[1] = abcd_1;
    vecs[2] = abcd_2;
    vecs[3] = abcd_3;
}

#[inline(always)]
unsafe fn transpose_msg_vecs(inputs: &[*const u8; DEGREE], block_offset: usize) -> [__m128i; 16] {
    let mut vecs = [
        loadu(inputs[0].add(block_offset + 0 * 4 * DEGREE)),
        loadu(inputs[1].add(block_offset + 0 * 4 * DEGREE)),
        loadu(inputs[2].add(block_offset + 0 * 4 * DEGREE)),
        loadu(inputs[3].add(block_offset + 0 * 4 * DEGREE)),
        loadu(inputs[0].add(block_offset + 1 * 4 * DEGREE)),
        loadu(inputs[1].add(block_offset + 1 * 4 * DEGREE)),
        loadu(inputs[2].add(block_offset + 1 * 4 * DEGREE)),
        loadu(inputs[3].add(block_offset + 1 * 4 * DEGREE)),
        loadu(inputs[0].add(block_offset + 2 * 4 * DEGREE)),
        loadu(inputs[1].add(block_offset + 2 * 4 * DEGREE)),
        loadu(inputs[2].add(block_offset + 2 * 4 * DEGREE)),
        loadu(inputs[3].add(block_offset + 2 * 4 * DEGREE)),
        loadu(inputs[0].add(block_offset + 3 * 4 * DEGREE)),
        loadu(inputs[1].add(block_offset + 3 * 4 * DEGREE)),
        loadu(inputs[2].add(block_offset + 3 * 4 * DEGREE)),
        loadu(inputs[3].add(block_offset + 3 * 4 * DEGREE)),
    ];
    let squares = mut_array_refs!(&mut vecs, DEGREE, DEGREE, DEGREE, DEGREE);
    transpose_vecs(squares.0);
    transpose_vecs(squares.1);
    transpose_vecs(squares.2);
    transpose_vecs(squares.3);
    vecs
}

#[inline(always)]
unsafe fn load_counters(counter: u64, increment_counter: IncrementCounter) -> (__m128i, __m128i) {
    let mask = if increment_counter.yes() { !0 } else { 0 };
    (
        set4(
            counter_low(counter + (mask & 0)),
            counter_low(counter + (mask & 1)),
            counter_low(counter + (mask & 2)),
            counter_low(counter + (mask & 3)),
        ),
        set4(
            counter_high(counter + (mask & 0)),
            counter_high(counter + (mask & 1)),
            counter_high(counter + (mask & 2)),
            counter_high(counter + (mask & 3)),
        ),
    )
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn hash4(
    inputs: &[*const u8; DEGREE],
    blocks: usize,
    key: &CVWords,
    counter: u64,
    increment_counter: IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8; DEGREE * OUT_LEN],
) {
    let mut h_vecs = [
        set1(key[0]),
        set1(key[1]),
        set1(key[2]),
        set1(key[3]),
        set1(key[4]),
        set1(key[5]),
        set1(key[6]),
        set1(key[7]),
    ];
    let (counter_low_vec, counter_high_vec) = load_counters(counter, increment_counter);
    let mut block_flags = flags | flags_start;

    for block in 0..blocks {
        if block + 1 == blocks {
            block_flags |= flags_end;
        }
        let block_len_vec = set1(BLOCK_LEN as u32); // full blocks only
        let block_flags_vec = set1(block_flags as u32);
        let msg_vecs = transpose_msg_vecs(inputs, block * BLOCK_LEN);

        // The transposed compression function. Note that inlining this
        // manually here improves compile times by a lot, compared to factoring
        // it out into its own function and making it #[inline(always)]. Just
        // guessing, it might have something to do with loop unrolling.
        let mut v = [
            h_vecs[0],
            h_vecs[1],
            h_vecs[2],
            h_vecs[3],
            h_vecs[4],
            h_vecs[5],
            h_vecs[6],
            h_vecs[7],
            set1(IV[0]),
            set1(IV[1]),
            set1(IV[2]),
            set1(IV[3]),
            counter_low_vec,
            counter_high_vec,
            block_len_vec,
            block_flags_vec,
        ];
        round(&mut v, &msg_vecs, 0);
        round(&mut v, &msg_vecs, 1);
        round(&mut v, &msg_vecs, 2);
        round(&mut v, &msg_vecs, 3);
        round(&mut v, &msg_vecs, 4);
        round(&mut v, &msg_vecs, 5);
        round(&mut v, &msg_vecs, 6);
        h_vecs[0] = xor(v[0], v[8]);
        h_vecs[1] = xor(v[1], v[9]);
        h_vecs[2] = xor(v[2], v[10]);
        h_vecs[3] = xor(v[3], v[11]);
        h_vecs[4] = xor(v[4], v[12]);
        h_vecs[5] = xor(v[5], v[13]);
        h_vecs[6] = xor(v[6], v[14]);
        h_vecs[7] = xor(v[7], v[15]);

        block_flags = flags;
    }

    let squares = mut_array_refs!(&mut h_vecs, DEGREE, DEGREE);
    transpose_vecs(squares.0);
    transpose_vecs(squares.1);
    // The first four vecs now contain the first half of each output, and the
    // second four vecs contain the second half of each output.
    storeu(h_vecs[0], out.as_mut_ptr().add(0 * 4 * DEGREE));
    storeu(h_vecs[4], out.as_mut_ptr().add(1 * 4 * DEGREE));
    storeu(h_vecs[1], out.as_mut_ptr().add(2 * 4 * DEGREE));
    storeu(h_vecs[5], out.as_mut_ptr().add(3 * 4 * DEGREE));
    storeu(h_vecs[2], out.as_mut_ptr().add(4 * 4 * DEGREE));
    storeu(h_vecs[6], out.as_mut_ptr().add(5 * 4 * DEGREE));
    storeu(h_vecs[3], out.as_mut_ptr().add(6 * 4 * DEGREE));
    storeu(h_vecs[7], out.as_mut_ptr().add(7 * 4 * DEGREE));
}

#[target_feature(enable = "sse4.1")]
unsafe fn hash1<A: arrayvec::Array<Item = u8>>(
    input: &A,
    key: &CVWords,
    counter: u64,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut CVBytes,
) {
    debug_assert_eq!(A::CAPACITY % BLOCK_LEN, 0, "uneven blocks");
    let mut cv = *key;
    let mut block_flags = flags | flags_start;
    let mut slice = input.as_slice();
    while slice.len() >= BLOCK_LEN {
        if slice.len() == BLOCK_LEN {
            block_flags |= flags_end;
        }
        compress_in_place(
            &mut cv,
            array_ref!(slice, 0, BLOCK_LEN),
            BLOCK_LEN as u8,
            counter,
            block_flags,
        );
        block_flags = flags;
        slice = &slice[BLOCK_LEN..];
    }
    *out = core::mem::transmute(cv); // x86 is little-endian
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn hash_many<A: arrayvec::Array<Item = u8>>(
    mut inputs: &[&A],
    key: &CVWords,
    mut counter: u64,
    increment_counter: IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    mut out: &mut [u8],
) {
    debug_assert!(out.len() >= inputs.len() * OUT_LEN, "out too short");
    while inputs.len() >= DEGREE && out.len() >= DEGREE * OUT_LEN {
        // Safe because the layout of arrays is guaranteed, and because the
        // `blocks` count is determined statically from the argument type.
        let input_ptrs: &[*const u8; DEGREE] = &*(inputs.as_ptr() as *const [*const u8; DEGREE]);
        let blocks = A::CAPACITY / BLOCK_LEN;
        hash4(
            input_ptrs,
            blocks,
            key,
            counter,
            increment_counter,
            flags,
            flags_start,
            flags_end,
            array_mut_ref!(out, 0, DEGREE * OUT_LEN),
        );
        if increment_counter.yes() {
            counter += DEGREE as u64;
        }
        inputs = &inputs[DEGREE..];
        out = &mut out[DEGREE * OUT_LEN..];
    }
    for (&input, output) in inputs.iter().zip(out.chunks_exact_mut(OUT_LEN)) {
        hash1(
            input,
            key,
            counter,
            flags,
            flags_start,
            flags_end,
            array_mut_ref!(output, 0, OUT_LEN),
        );
        if increment_counter.yes() {
            counter += 1;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_transpose() {
        if !crate::platform::sse41_detected() {
            return;
        }

        #[target_feature(enable = "sse4.1")]
        unsafe fn transpose_wrapper(vecs: &mut [__m128i; DEGREE]) {
            transpose_vecs(vecs);
        }

        let mut matrix = [[0 as u32; DEGREE]; DEGREE];
        for i in 0..DEGREE {
            for j in 0..DEGREE {
                matrix[i][j] = (i * DEGREE + j) as u32;
            }
        }

        unsafe {
            let mut vecs: [__m128i; DEGREE] = core::mem::transmute(matrix);
            transpose_wrapper(&mut vecs);
            matrix = core::mem::transmute(vecs);
        }

        for i in 0..DEGREE {
            for j in 0..DEGREE {
                // Reversed indexes from above.
                assert_eq!(matrix[j][i], (i * DEGREE + j) as u32);
            }
        }
    }

    #[test]
    fn test_compress() {
        if !crate::platform::sse41_detected() {
            return;
        }
        crate::test::test_compress_fn(compress_in_place, compress_xof);
    }

    #[test]
    fn test_hash_many() {
        if !crate::platform::sse41_detected() {
            return;
        }
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }
}
