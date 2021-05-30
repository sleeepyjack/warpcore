#include <warpcore/multi_bucket_hash_table.cuh>
#include "../include/benchmark_common.cuh"

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void multi_bucket_benchmark(
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

    key_t* query_keys_d = nullptr;
    cudaMalloc(&query_keys_d, sizeof(key_t)*max_keys); CUERR
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

    const float bucket_factor =
        float(sizeof(key_t) + sizeof(value_t)) /
             (sizeof(key_t) + sizeof(value_t)*HashTable::bucket_size());

    if(!sufficient_memory_oa<HashTable>(max_input_size * bucket_factor / min_load_factor))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    for(auto size : input_sizes)
    {
        for(auto load : load_factors)
        {
            const std::uint64_t capacity = size * bucket_factor / load;

            HashTable hash_table(capacity);

            Output<key_t,value_t> output;
            output.sample_size = size;
            output.key_capacity = hash_table.capacity();
            output.value_capacity = hash_table.value_capacity();

            output.insert_ms = benchmark_insert(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);

            // std::cerr << "keys in table: " << hash_table.num_keys() << '\n';

            // auto key_set = hash_table.get_key_set();
            // std::cerr << "keys in set: " << key_set.size() << '\n';

            output.query_ms = benchmark_query_multi(
                hash_table, query_keys_d, size,
                offsets_d, values_d,
                iters, thermal_backoff);

            // output.query_ms = benchmark_query_unique(
            //     hash_table, query_keys_d, offsets_d, values_d,
            //     iters, thermal_backoff);

            output.key_load_factor = hash_table.key_load_factor();
            output.value_load_factor = hash_table.value_load_factor();
            output.density = hash_table.storage_density();
            output.relative_density = hash_table.relative_storage_density();
            output.status = hash_table.pop_status();

            if(print_headers)
                output.print_with_headers();
            else
                output.print_without_headers();
        }
    }

    cudaFree(query_keys_d); CUERR
    cudaFree(values_d); CUERR
    cudaFree(offsets_d); CUERR
}

int main(int argc, char* argv[])
{
    using namespace warpcore;

    using key_t = std::uint32_t;
    using value_t = std::uint32_t;

    const uint64_t max_keys = 1UL << 27;

    const bool print_headers = true;

    uint64_t dev_id = 0;
    if(argc > 2) dev_id = std::atoi(argv[2]);
    cudaSetDevice(dev_id); CUERR

    key_t * keys_d = nullptr;
    if(argc > 1)
        keys_d = load_keys<key_t>(argv[1], max_keys);
    else
        keys_d = generate_keys<key_t>(max_keys, 8);

    using mb1_hash_table_t = MultiBucketHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::empty_key<value_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, ArrayBucket<value_t,1>>>;

    using mb2_hash_table_t = MultiBucketHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::empty_key<value_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, ArrayBucket<value_t,2>>>;

    using mb4_hash_table_t = MultiBucketHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::empty_key<value_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, ArrayBucket<value_t,4>>>;

    using mb8_hash_table_t = MultiBucketHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::empty_key<value_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, ArrayBucket<value_t,8>>>;

    multi_bucket_benchmark<mb1_hash_table_t>(
        keys_d, max_keys,
        {max_keys},
        {0.8},
        print_headers);

    multi_bucket_benchmark<mb2_hash_table_t>(
        keys_d, max_keys,
        {max_keys},
        {0.8},
        print_headers);

    multi_bucket_benchmark<mb4_hash_table_t>(
        keys_d, max_keys,
        {max_keys},
        {0.8},
        print_headers);

    multi_bucket_benchmark<mb8_hash_table_t>(
        keys_d, max_keys,
        {max_keys},
        {0.8},
        print_headers);

    cudaFree(keys_d); CUERR
}
