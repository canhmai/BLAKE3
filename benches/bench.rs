use arrayref::array_ref;
use arrayvec::ArrayVec;
use bencher::{benchmark_group, benchmark_main, Bencher};
use blake3::platform::MAX_SIMD_DEGREE;
use blake3::{BLOCK_LEN, CHUNK_LEN, OUT_LEN};
use rand::prelude::*;

const KIB: usize = 1024;

// This struct randomizes two things:
// 1. The actual bytes of input.
// 2. The page offset the input starts at.
pub struct RandomInput {
    buf: Vec<u8>,
    len: usize,
    offsets: Vec<usize>,
    offset_index: usize,
}

impl RandomInput {
    pub fn new(b: &mut Bencher, len: usize) -> Self {
        b.bytes += len as u64;
        let page_size: usize = page_size::get();
        let mut buf = vec![0u8; len + page_size];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut buf);
        let mut offsets: Vec<usize> = (0..page_size).collect();
        offsets.shuffle(&mut rng);
        Self {
            buf,
            len,
            offsets,
            offset_index: 0,
        }
    }

    pub fn get(&mut self) -> &[u8] {
        let offset = self.offsets[self.offset_index];
        self.offset_index += 1;
        if self.offset_index >= self.offsets.len() {
            self.offset_index = 0;
        }
        &self.buf[offset..][..self.len]
    }
}

fn dummy(_: &mut Bencher) {}

type CompressInPlaceFn =
    unsafe fn(cv: &mut [u32; 8], block: &[u8; BLOCK_LEN], block_len: u8, counter: u64, flags: u8);

fn bench_single_compression_fn(b: &mut Bencher, f: CompressInPlaceFn) {
    let mut state = [1u32; 8];
    let mut r = RandomInput::new(b, 64);
    let input = array_ref!(r.get(), 0, 64);
    unsafe {
        b.iter(|| f(&mut state, input, 64 as u8, 0, 0));
    }
}

fn bench_single_compression_portable(b: &mut Bencher) {
    bench_single_compression_fn(b, blake3::portable::compress_in_place);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_single_compression_sse41(b: &mut Bencher) {
    if !blake3::platform::sse41_detected() {
        return;
    }
    bench_single_compression_fn(b, blake3::sse41::compress_in_place);
}

#[cfg(feature = "c_avx512")]
fn bench_single_compression_avx512(b: &mut Bencher) {
    if !blake3::platform::avx512_detected() {
        return;
    }
    bench_single_compression_fn(b, blake3::c_avx512::compress_in_place);
}

type HashManyFn<A> = unsafe fn(
    inputs: &[&A],
    key: &[u32; 8],
    counter: u64,
    increment_counter: blake3::IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8],
);

fn bench_many_chunks_fn(b: &mut Bencher, f: HashManyFn<[u8; CHUNK_LEN]>, degree: usize) {
    let mut inputs = Vec::new();
    for _ in 0..degree {
        inputs.push(RandomInput::new(b, CHUNK_LEN));
    }
    unsafe {
        b.iter(|| {
            let input_arrays: ArrayVec<[&[u8; CHUNK_LEN]; MAX_SIMD_DEGREE]> = inputs
                .iter_mut()
                .take(degree)
                .map(|i| array_ref!(i.get(), 0, CHUNK_LEN))
                .collect();
            let mut out = [0; MAX_SIMD_DEGREE * OUT_LEN];
            f(
                &input_arrays[..],
                &[0; 8],
                0,
                blake3::IncrementCounter::Yes,
                0,
                0,
                0,
                &mut out,
            );
        });
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_many_chunks_sse41(b: &mut Bencher) {
    if !blake3::platform::sse41_detected() {
        return;
    }
    bench_many_chunks_fn(b, blake3::sse41::hash_many, blake3::sse41::DEGREE);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_many_chunks_avx2(b: &mut Bencher) {
    if !blake3::platform::avx2_detected() {
        return;
    }
    bench_many_chunks_fn(b, blake3::avx2::hash_many, blake3::avx2::DEGREE);
}

#[cfg(feature = "c_avx512")]
fn bench_many_chunks_avx512(b: &mut Bencher) {
    if !blake3::platform::avx512_detected() {
        return;
    }
    bench_many_chunks_fn(b, blake3::c_avx512::hash_many, blake3::c_avx512::DEGREE);
}

#[cfg(feature = "c_neon")]
fn bench_many_chunks_neon(b: &mut Bencher) {
    // When "c_neon" is on, NEON support is assumed.
    bench_many_chunks_fn(b, blake3::c_neon::hash_many, blake3::c_neon::DEGREE);
}

// TODO: When we get const generics we can unify this with the chunks code.
fn bench_many_parents_fn(b: &mut Bencher, f: HashManyFn<[u8; BLOCK_LEN]>, degree: usize) {
    let mut inputs = Vec::new();
    for _ in 0..degree {
        inputs.push(RandomInput::new(b, BLOCK_LEN));
    }
    unsafe {
        b.iter(|| {
            let input_arrays: ArrayVec<[&[u8; BLOCK_LEN]; MAX_SIMD_DEGREE]> = inputs
                .iter_mut()
                .take(degree)
                .map(|i| array_ref!(i.get(), 0, BLOCK_LEN))
                .collect();
            let mut out = [0; MAX_SIMD_DEGREE * OUT_LEN];
            f(
                &input_arrays[..],
                &[0; 8],
                0,
                blake3::IncrementCounter::No,
                0,
                0,
                0,
                &mut out,
            );
        });
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_many_parents_sse41(b: &mut Bencher) {
    if !blake3::platform::sse41_detected() {
        return;
    }
    bench_many_parents_fn(b, blake3::sse41::hash_many, blake3::sse41::DEGREE);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_many_parents_avx2(b: &mut Bencher) {
    if !blake3::platform::avx2_detected() {
        return;
    }
    bench_many_parents_fn(b, blake3::avx2::hash_many, blake3::avx2::DEGREE);
}

#[cfg(feature = "c_avx512")]
fn bench_many_parents_avx512(b: &mut Bencher) {
    if !blake3::platform::avx512_detected() {
        return;
    }
    bench_many_parents_fn(b, blake3::c_avx512::hash_many, blake3::c_avx512::DEGREE);
}

#[cfg(feature = "c_neon")]
fn bench_many_parents_neon(b: &mut Bencher) {
    // When "c_neon" is on, NEON support is assumed.
    bench_many_parents_fn(b, blake3::c_neon::hash_many, blake3::c_neon::DEGREE);
}

benchmark_group!(portable, bench_single_compression_portable);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
benchmark_group!(
    x86,
    bench_single_compression_sse41,
    bench_many_chunks_sse41,
    bench_many_chunks_avx2,
    bench_many_parents_sse41,
    bench_many_parents_avx2
);
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
benchmark_group!(x86, dummy);

#[cfg(feature = "c_avx512")]
benchmark_group!(
    avx512,
    bench_single_compression_avx512,
    bench_many_chunks_avx512,
    bench_many_parents_avx512
);
#[cfg(not(feature = "c_avx512"))]
benchmark_group!(avx512, dummy);

#[cfg(feature = "c_neon")]
benchmark_group!(neon, bench_many_chunks_neon, bench_many_parents_neon);
#[cfg(not(feature = "c_neon"))]
benchmark_group!(neon, dummy);

fn bench_atonce(b: &mut Bencher, len: usize) {
    let mut input = RandomInput::new(b, len);
    b.iter(|| blake3::hash(input.get()));
}

fn bench_atonce_0001_block(b: &mut Bencher) {
    bench_atonce(b, BLOCK_LEN);
}

fn bench_atonce_0001_kib(b: &mut Bencher) {
    bench_atonce(b, 1 * KIB);
}

fn bench_atonce_0002_kib(b: &mut Bencher) {
    bench_atonce(b, 2 * KIB);
}

fn bench_atonce_0004_kib(b: &mut Bencher) {
    bench_atonce(b, 4 * KIB);
}

fn bench_atonce_0008_kib(b: &mut Bencher) {
    bench_atonce(b, 8 * KIB);
}

fn bench_atonce_0016_kib(b: &mut Bencher) {
    bench_atonce(b, 16 * KIB);
}

fn bench_atonce_0032_kib(b: &mut Bencher) {
    bench_atonce(b, 32 * KIB);
}

fn bench_atonce_0064_kib(b: &mut Bencher) {
    bench_atonce(b, 64 * KIB);
}

fn bench_atonce_0128_kib(b: &mut Bencher) {
    bench_atonce(b, 128 * KIB);
}

fn bench_atonce_0256_kib(b: &mut Bencher) {
    bench_atonce(b, 256 * KIB);
}

fn bench_atonce_0512_kib(b: &mut Bencher) {
    bench_atonce(b, 512 * KIB);
}

fn bench_atonce_1024_kib(b: &mut Bencher) {
    bench_atonce(b, 1024 * KIB);
}

benchmark_group!(
    atonce,
    bench_atonce_0001_block,
    bench_atonce_0001_kib,
    bench_atonce_0002_kib,
    bench_atonce_0004_kib,
    bench_atonce_0008_kib,
    bench_atonce_0016_kib,
    bench_atonce_0032_kib,
    bench_atonce_0064_kib,
    bench_atonce_0128_kib,
    bench_atonce_0256_kib,
    bench_atonce_0512_kib,
    bench_atonce_1024_kib
);

fn bench_incremental(b: &mut Bencher, len: usize) {
    let mut input = RandomInput::new(b, len);
    b.iter(|| blake3::Hasher::new().update(input.get()).finalize());
}

fn bench_incremental_0001_block(b: &mut Bencher) {
    bench_incremental(b, BLOCK_LEN);
}

fn bench_incremental_0001_kib(b: &mut Bencher) {
    bench_incremental(b, 1 * KIB);
}

fn bench_incremental_0002_kib(b: &mut Bencher) {
    bench_incremental(b, 2 * KIB);
}

fn bench_incremental_0004_kib(b: &mut Bencher) {
    bench_incremental(b, 4 * KIB);
}

fn bench_incremental_0008_kib(b: &mut Bencher) {
    bench_incremental(b, 8 * KIB);
}

fn bench_incremental_0016_kib(b: &mut Bencher) {
    bench_incremental(b, 16 * KIB);
}

fn bench_incremental_0032_kib(b: &mut Bencher) {
    bench_incremental(b, 32 * KIB);
}

fn bench_incremental_0064_kib(b: &mut Bencher) {
    bench_incremental(b, 64 * KIB);
}

fn bench_incremental_0128_kib(b: &mut Bencher) {
    bench_incremental(b, 128 * KIB);
}

fn bench_incremental_0256_kib(b: &mut Bencher) {
    bench_incremental(b, 256 * KIB);
}

fn bench_incremental_0512_kib(b: &mut Bencher) {
    bench_incremental(b, 512 * KIB);
}

fn bench_incremental_1024_kib(b: &mut Bencher) {
    bench_incremental(b, 1024 * KIB);
}

benchmark_group!(
    incremental,
    bench_incremental_0001_block,
    bench_incremental_0001_kib,
    bench_incremental_0002_kib,
    bench_incremental_0004_kib,
    bench_incremental_0008_kib,
    bench_incremental_0016_kib,
    bench_incremental_0032_kib,
    bench_incremental_0064_kib,
    bench_incremental_0128_kib,
    bench_incremental_0256_kib,
    bench_incremental_0512_kib,
    bench_incremental_1024_kib
);

fn bench_reference(b: &mut Bencher, len: usize) {
    let mut input = RandomInput::new(b, len);
    b.iter(|| {
        let mut hasher = reference_impl::Hasher::new();
        hasher.update(input.get());
        let mut out = [0; 32];
        hasher.finalize(&mut out);
        out
    });
}

fn bench_reference_0001_block(b: &mut Bencher) {
    bench_reference(b, BLOCK_LEN);
}

fn bench_reference_0001_kib(b: &mut Bencher) {
    bench_reference(b, 1 * KIB);
}

fn bench_reference_0002_kib(b: &mut Bencher) {
    bench_reference(b, 2 * KIB);
}

fn bench_reference_0004_kib(b: &mut Bencher) {
    bench_reference(b, 4 * KIB);
}

fn bench_reference_0008_kib(b: &mut Bencher) {
    bench_reference(b, 8 * KIB);
}

fn bench_reference_0016_kib(b: &mut Bencher) {
    bench_reference(b, 16 * KIB);
}

fn bench_reference_0032_kib(b: &mut Bencher) {
    bench_reference(b, 32 * KIB);
}

fn bench_reference_0064_kib(b: &mut Bencher) {
    bench_reference(b, 64 * KIB);
}

fn bench_reference_0128_kib(b: &mut Bencher) {
    bench_reference(b, 128 * KIB);
}

fn bench_reference_0256_kib(b: &mut Bencher) {
    bench_reference(b, 256 * KIB);
}

fn bench_reference_0512_kib(b: &mut Bencher) {
    bench_reference(b, 512 * KIB);
}

fn bench_reference_1024_kib(b: &mut Bencher) {
    bench_reference(b, 1024 * KIB);
}

benchmark_group!(
    reference,
    bench_reference_0001_block,
    bench_reference_0001_kib,
    bench_reference_0002_kib,
    bench_reference_0004_kib,
    bench_reference_0008_kib,
    bench_reference_0016_kib,
    bench_reference_0032_kib,
    bench_reference_0064_kib,
    bench_reference_0128_kib,
    bench_reference_0256_kib,
    bench_reference_0512_kib,
    bench_reference_1024_kib
);

benchmark_main!(portable, x86, avx512, neon, atonce, incremental, reference);
