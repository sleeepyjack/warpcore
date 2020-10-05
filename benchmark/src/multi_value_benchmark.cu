#include "common.cuh"
#include "warpcore.cuh"
#include "../../ext/hpc_helpers/include/io_helpers.h"
#include <chrono>
#include <iostream>
#include <vector>

template<class HashTable, int BucketSize>
HOSTQUALIFIER INLINEQUALIFIER
void multi_value_benchmark(
    const typename HashTable::key_type * keys_d,
    const uint64_t max_keys,
    std::vector<uint64_t> input_sizes,
    std::vector<float> load_factors,
    bool print_headers = true,
    uint8_t iters = 5,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100))
{
    using index_t = typename HashTable::index_type;
    using key_t = typename HashTable::key_type;
    using value_t = typename HashTable::value_type;

    const uint64_t max_unique_size = num_unique(keys_d, max_keys);

    key_t* unique_keys_d = nullptr;
    cudaMalloc(&unique_keys_d, sizeof(key_t)*max_unique_size); CUERR
    value_t* values_d = nullptr;
    cudaMalloc(&values_d, sizeof(value_t)*max_keys); CUERR
    index_t * offsets_d = nullptr;
    cudaMalloc(&offsets_d, sizeof(index_t)*(max_keys+1)); CUERR

    cudaMemset(values_d, 1, sizeof(value_t)*max_keys); CUERR

    const auto max_input_size =
        *std::max_element(input_sizes.begin(), input_sizes.end());
    const auto min_load_factor =
        *std::min_element(load_factors.begin(), load_factors.end());

    if(max_input_size > max_keys)
    {
        std::cerr << "Maximum input size exceeded." << std::endl;
        exit(1);
    }

    if(!sufficient_memory_oa<HashTable>(max_input_size / min_load_factor))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    for(auto size : input_sizes)
    {
        for(auto load : load_factors)
        {
            const std::uint64_t capacity = float(size) / BucketSize / load;

            HashTable hash_table(capacity);

            Output<key_t,value_t> output;
            output.sample_size = size;
            output.key_capacity = hash_table.capacity();

            output.insert_ms = benchmark_insert(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);

            output.query_ms = benchmark_query_multi(
                hash_table, unique_keys_d, offsets_d, values_d,
                iters, thermal_backoff);

            output.key_load_factor = hash_table.load_factor();
            output.density = output.key_load_factor;
            output.status = hash_table.pop_status();

            if(print_headers)
                output.print_with_headers();
            else
                output.print_without_headers();
        }
    }

    cudaFree(unique_keys_d); CUERR
    cudaFree(values_d); CUERR
    cudaFree(offsets_d); CUERR
}

int main(int argc, char* argv[])
{
    using namespace warpcore;

    using key_t = std::uint32_t;
    using value_t = std::uint32_t;

    const uint64_t max_keys = 1UL << 27;

    uint64_t dev_id = 0;
    if(argc > 2) dev_id = std::atoi(argv[2]);
    cudaSetDevice(dev_id); CUERR

    key_t * keys_d = nullptr;
    if(argc > 1)
        keys_d = load_keys<key_t>(argv[1], max_keys);
    else
        keys_d = generate_keys<key_t>(max_keys, 8);

    using mv_hash_table_t = MultiValueHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, value_t>>;

    // using mb_hash_table_t = MultiBucketHashTable<
    //     key_t,
    //     value_t,
    //     defaults::empty_key<key_t>(),
    //     defaults::tombstone_key<key_t>(),
    //     defaults::empty_key<value_t>(),
    //     defaults::probing_scheme_t<key_t, 8>,
    //     storage::key_value::AoSStore<key_t, ArrayBucket<value_t,2>>>;

    multi_value_benchmark<mv_hash_table_t, 1>(
        keys_d, max_keys,
        {max_keys},
        {0.8});

    // multi_value_benchmark<mb_hash_table_t, mb_hash_table_t::bucket_size()>(
    //     keys_d, max_keys,
    //     {max_keys},
    //     {0.8});

    cudaFree(keys_d); CUERR
}