# warpcore

**Hashing at the Speed of Light on modern CUDA-accelerators**

## Introduction
warpcore is a framework for creating purpose-built high-throughput hashing data structures on CUDA-accelerators.

This library provides the following data structures:
- [`warpcore::SingleValueHashTable`](include/single_value_hash_table.cuh): Stores a *set* of key/value pairs.
- [`warpcore::HashSet`](include/hash_set.cuh): Stores a *set* of keys.
- [`warpcore::CountingHashTable`](include/counting_hash_table.cuh): Keeps track of the number of occurrences of each inserted key.
- [`warpcore::BloomFilter`](include/bloom_filter.cuh): Pattern-blocked bloom filter for approximate membership queries.
- [`warpcore::MultiValueHashTable`](include/multi_value_hash_table.cuh): Stores a *multi-set* of key/value pairs.
- [`warpcore::BucketListHashTable`](include/bucket_list_hash_table.cuh): Alternative variant of `warpcore::MultiValueHashTable`.

warpcore supports key types `std::uint32_t` and `std::uint64_t` together with any trivially copyable value type. In order to be adaptable to a wide range of possible usecases, we provide a multitude of combinable modules such as [hash functions](include/hashers.cuh), [probing schemes](include/probing_schemes.cuh), and [data layouts](include/storage.cuh) (visit the [documentation](https://sleeepyjack.github.io/warpcore/) for further information).

This will also be the core library for future releases of our multi-GPU hashing library [warpdrive](https://ieeexplore.ieee.org/document/8425198/).

## System Requirements
- [CUDA-capable device](https://developer.nvidia.com/cuda-gpus) with architecture >= 6.0 (Pascal or higher)
- [NVIDIA CUDA toolkit/compiler](https://developer.nvidia.com/cuda-toolkit) version >= 10.2
- [GNU g++](https://gcc.gnu.org/) version >= 7.3 and compatible with your CUDA version
- ISO-C++14 standard

**Note:** Suppport for host compilers other than GNU g++ are planned for future releases.

## Installation
warpcore is a *header-only library*.

In order to use this library in your project, clone the repository to where you need it
```console
git clone https://github.com/sleeepyjack/warpcore
```
and build the required submodules
```console
cd warpcore
git submodule update --init --recursive
```
You can then include this library in your C++/CUDA project by absolute path
```cpp
#include "<PATH_TO_WARPCORE_REPO>/include/warpcore.cuh"
```
or by adding the repository path at compilation
```console
nvcc ... -I<PATH_TO_WARPCORE_REPO>/include ...
```
and include it without the explicit path
```cpp
#include <warpcore.cuh>
```

Take a look at our [examples](examples/README.md) to learn more about how to use warpcore in your project.

## [Documentation](docs/index.html)

## Where to go from here?
Take a look at the [examples](examples/README.md), test your own system performance using the [benchmark suite](benchmark/README.md) and be sure everything works fine by running the [test suite](test/README.md).
***
warpcore Copyright (C) 2018-2020 [Daniel Jünger](https://github.com/sleeepyjack)

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under certain
conditions. See the file [LICENSE](LICENSE.txt) for details.

[repository]: https://github.com/sleeepyjack/warpcore


