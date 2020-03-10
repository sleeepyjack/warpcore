#ifndef WARPCORE_CONFIG_CUH
#define WARPCORE_CONFIG_CUH

#include <cstdint>
#include <algorithm>
#include <assert.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../ext/cudahelpers/cuda_helpers.cuh"
#include "../ext/cub/cub/cub.cuh"
#include "primes.hpp"

namespace warpcore
{

// compilation constraints
#if GCC_VERSION < 50400
    #error gcc/g++ version >= 5.4.0 required
#endif

#if CUDART_VERSION < 10010
    #error CUDA version >= 10.1 required
#endif

namespace cg = cooperative_groups;

using index_t = std::uint64_t;
using status_base_t = std::uint32_t;

namespace detail
{

HOSTQUALIFIER INLINEQUALIFIER
index_t get_valid_capacity(index_t min_capacity, index_t cg_size) noexcept
{
    const auto x = SDIV(min_capacity, cg_size);
    const auto y =
        std::lower_bound(primes.begin(), primes.end(), x);
    return (y == primes.end()) ? 0 : (*y) * cg_size;
}

} // namespace detail

} // namespace warpcore

#include "tags.cuh"
#include "checks.cuh"

#endif /* WARPCORE_CONFIG_CUH */