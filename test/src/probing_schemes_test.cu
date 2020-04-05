#include "catch.hpp"
#include "base.cuh"

namespace wcps = warpcore::probing_schemes;
namespace wchs = warpcore::hashers;

TEMPLATE_TEST_CASE(
    "Cycle-free Probing",
    "[probing][cyclefree][template]",
    (wcps::DoubleHashing<wchs::MurmurHash<std::uint32_t>, wchs::MurmurHash<std::uint32_t>,  1>),
    (wcps::DoubleHashing<wchs::MurmurHash<std::uint32_t>, wchs::MurmurHash<std::uint32_t>,  2>),
    (wcps::DoubleHashing<wchs::MurmurHash<std::uint32_t>, wchs::MurmurHash<std::uint32_t>,  4>),
    (wcps::DoubleHashing<wchs::MurmurHash<std::uint32_t>, wchs::MurmurHash<std::uint32_t>,  8>),
    (wcps::DoubleHashing<wchs::MurmurHash<std::uint32_t>, wchs::MurmurHash<std::uint32_t>, 16>),
    (wcps::DoubleHashing<wchs::MurmurHash<std::uint32_t>, wchs::MurmurHash<std::uint32_t>, 32>),
    (wcps::LinearProbing<wchs::MurmurHash<std::uint32_t>,  1>),
    (wcps::LinearProbing<wchs::MurmurHash<std::uint32_t>,  2>),
    (wcps::LinearProbing<wchs::MurmurHash<std::uint32_t>,  4>),
    (wcps::LinearProbing<wchs::MurmurHash<std::uint32_t>,  8>),
    (wcps::LinearProbing<wchs::MurmurHash<std::uint32_t>, 16>),
    (wcps::LinearProbing<wchs::MurmurHash<std::uint32_t>, 32>))
{
    namespace cg = cooperative_groups;

    using probing_t = TestType;
    using key_t = typename probing_t::key_type;
    using index_t = warpcore::index_t;
    static constexpr index_t cg_size = probing_t::cg_size();

    CHECK(warpcore::checks::is_cycle_free_probing_scheme<probing_t>());

    const index_t min_capacity = GENERATE(as<index_t>{}, 12345, 42424, 6969);
    const index_t valid_capacity =
        warpcore::detail::get_valid_capacity(min_capacity, probing_t::cg_size());
    const index_t key = GENERATE(as<key_t>{}, 1, 42, 69, 1337);
    const key_t seed = GENERATE(as<key_t>{}, 5, 42);

    CAPTURE(min_capacity, valid_capacity, key, seed);

    index_t * probes = nullptr;
    cudaMallocManaged(&probes, sizeof(index_t) * (valid_capacity + cg_size));
    REQUIRE(cudaPeekAtLastError() == cudaSuccess);

    lambda_kernel
    <<<1, cg_size>>>([=] DEVICEQUALIFIER
    {
        const auto group = cg::tiled_partition<cg_size>(cg::this_thread_block());

        probing_t iter(valid_capacity, valid_capacity, group);

        index_t out_offset = group.thread_rank();
        for(index_t i = iter.begin(key, seed); i != iter.end(); i = iter.next())
        {
            probes[out_offset] = i;
            out_offset += group.size();
        }
    });

    cudaDeviceSynchronize();
    REQUIRE(cudaPeekAtLastError() == cudaSuccess);

    auto begin = probes;
    auto end = probes + valid_capacity;
    std::sort(begin, end);
    auto end_unique = std::unique(begin, end);

    bool has_no_overlap = (end == end_unique);
    CHECK(has_no_overlap);

    cudaFree(probes);
    cudaDeviceSynchronize();
    REQUIRE(cudaPeekAtLastError() == cudaSuccess);

}