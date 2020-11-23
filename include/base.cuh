#ifndef WARPCORE_BASE_CUH
#define WARPCORE_BASE_CUH

#include <cstdint>
#include <algorithm>
#include <assert.h>
#include <limits>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../ext/hpc_helpers/include/cuda_helpers.cuh"
#include "../ext/hpc_helpers/include/packed_types.cuh"
#include <cub/cub.cuh>

#include "primes.hpp"

namespace warpcore
{

// compilation constraints
#if GCC_VERSION < 60300
    #error g++ version >= 6.3.0 required
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


#ifdef __CUDACC_DEBUG__
    #ifndef WARPCORE_BLOCKSIZE
    #define WARPCORE_BLOCKSIZE 128
    #endif
#else 
    #ifndef WARPCORE_BLOCKSIZE
    #define WARPCORE_BLOCKSIZE MAXBLOCKSIZE // MAXBLOCKSIZE defined in cuda_helpers
    #endif
#endif


#include "tags.cuh"
#include "checks.cuh"
#include "status.cuh"
#include "hashers.cuh"
#include "probing_schemes.cuh"
#include "storage.cuh"
#include "defaults.cuh"
#include "gpu_engine.cuh"

#endif /* WARPCORE_BASE_CUH */
