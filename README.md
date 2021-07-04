# BLAKE3

BLAKE3 is a cryptographic hash function that is

1. Faster than MD5, SHA1, SHA2, SHA3, and even BLAKE2
1. Highly parallelizable: The more data and the more cores, the faster it
goes. (For you specialists reading this: this is because it is actually a
MerkleTree under the hood.)
1. Capable of streaming and random-access verification. (Again: the magic of Merkle Trees.)
1. Carefully engineered to be simple and safe to use, with no "modes" or tweaks required.

The complete specifications and design rationale are available as a
[PDF](https://github.com/BLAKE3-team/BLAKE3-specs/raw/master/blake3.pdf) and its
[LaTeX source](https://github.com/BLAKE3-team/BLAKE3-specs/).

This repository provides the reference implementation of BLAKE3, which
is coded in Rust and includes portable as well as optimized code for
various instructions sets.

The [`b3sum` sub-crate](./b3sum) provides a command line interface.
SSE4.1 and AVX2 implementations are provided in Rust, enabled by
default, with dynamic CPU feature detection. AVX-512 and NEON
implementation are available via C FFI, controlled by the `c_avx512` and
`c_neon` features. Rayon-based multi-threading is controlled by the
`rayon` feature.

Eventually docs will be published on docs.rs. For now, you can build and
view the docs locally with `cargo doc --open`.

The following graph shows BLAKE3's multi-threaded throughput on an Intel
Kany Lake processor:

![benchmarks](media/speed.png)

BLAKE3 was designed by:

* [@oconnor63 ](https://github.com/oconnor63) (Jack O'Connor)
* [@sneves](https://github.com/sneves) (Samuel Neves)
* [@veorq](https://github.com/veorq) (Jean-Philippe Aumasson)
* [@zookozcash](https://github.com/zookozcash) (Zooko)

*WARNING*: BLAKE3 is not a password-hash function, because it's designed
to be fast, whereas password hashing should not be fast.
If you hash passwords to store the hashes or if you derive keys from password, we recommend
[Argon2](https://github.com/P-H-C/phc-winner-argon2).

## Usage

TODO

## History

BLAKE3 is essentially an adapted version of [BLAKE2](https://blake2.net)
using the tree [Bao](https://github.com/oconnor663/baokeshed) tree mode.

BLAKE2 is an established cryptographic hash function, for example
supported by OpenSSL, and used in countless applications.
Bao is a tree hashing mode satisfying the requirements for provably
secure tree hashing.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md)

## Licensing

The source code in the present repository is dual-licensed under Apache
2 and CC0 licences.



