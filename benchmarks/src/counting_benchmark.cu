#include "../include/benchmark_common.cuh"

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void counting_benchmark(
    const typename HashTable::key_type * keys_d,
    const uint64_t max_keys,
    std::vector<uint64_t> input_sizes,
    std::vector<float> load_factors,
    bool print_headers = true,
    uint8_t iters = 5,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100))
{
    using key_t = typename HashTable::key_type;
    using count_t = typename HashTable::value_type;

    count_t* counts_d = nullptr;
    cudaMalloc(&counts_d, sizeof(count_t)*max_keys); CUERR

    const auto max_input_size =
        *std::max_element(input_sizes.begin(), input_sizes.end());
    const auto min_load_factor =
        *std::min_element(load_factors.begin(), load_factors.end());

    if(max_input_size > max_keys)
    {
        std::cerr << "Maximum input size exceeded." << std::endl;
        exit(1);
    }

    const uint64_t max_unique_size = num_unique(keys_d, max_input_size);

    if(!sufficient_memory_oa<HashTable>(max_unique_size / min_load_factor))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    for(auto size : input_sizes)
    {
        for(auto load : load_factors)
        {
            const uint64_t unique_size = num_unique(keys_d, size);
            const uint64_t capacity = unique_size/load;

            HashTable hash_table(capacity);

            Output<key_t> output;
            output.sample_size = size;
            output.key_capacity = hash_table.capacity();

            output.insert_ms = benchmark_insert(
                hash_table, keys_d, size,
                iters, thermal_backoff);

            output.query_ms = benchmark_query(
                hash_table, keys_d, counts_d, size,
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

    cudaFree(counts_d); CUERR
}

int main(int argc, char* argv[])
{
    using key_t = uint32_t;
    using count_t = uint32_t;

    const uint64_t max_keys = 1UL << 28;

    const bool print_headers = true;

    uint64_t dev_id = 0;
    if(argc > 2) dev_id = std::atoi(argv[2]);
    cudaSetDevice(dev_id); CUERR

    key_t * keys_d = nullptr;
    if(argc > 1)
        keys_d = load_keys<key_t>(argv[1], max_keys);
    else
        keys_d = generate_keys<key_t>(max_keys, 8);

    using hash_table_t = warpcore::CountingHashTable<key_t, count_t>;

    counting_benchmark<hash_table_t>(
        keys_d, max_keys, {max_keys}, {0.9}, print_headers);

    cudaFree(keys_d); CUERR
}