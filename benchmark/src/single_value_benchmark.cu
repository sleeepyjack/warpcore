#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <thread>
#include <chrono>
#include "warpcore.cuh"
#include "../../ext/hpc_helpers/include/io_helpers.h"

template<class Key, class Value>
bool sufficient_memory(size_t size, float load, float headroom_factor = 1.1)
{
    const size_t capacity = size/load;
    const size_t key_val_bytes = sizeof(Key)+sizeof(Value);
    const size_t table_bytes = key_val_bytes*capacity;
    const size_t io_bytes = key_val_bytes*size;
    const size_t total_bytes = (table_bytes+io_bytes)*headroom_factor;

    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total); CUERR

    return (total_bytes <= bytes_free);
}

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void single_value_benchmark(
    const std::vector<typename HashTable::key_type>& keys,
    uint64_t dev_id = 0,
    bool print_headers = true,
    std::vector<uint64_t> input_sizes = {(1UL<<27)},
    std::vector<float> load_factors = {0.8},
    uint8_t iters = 5,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100))
{
    cudaSetDevice(dev_id); CUERR

    using key_t = typename HashTable::key_type;
    using value_t = typename HashTable::value_type;

    const auto max_input_size =
        *std::max_element(input_sizes.begin(), input_sizes.end());
    const auto min_load_factor =
        *std::min_element(load_factors.begin(), load_factors.end());

    if(max_input_size > keys.size())
    {
        std::cerr << "Maximum input size exceeded." << std::endl;
        exit(1);
    }

    if(!sufficient_memory<key_t, value_t>(max_input_size, min_load_factor))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    key_t* keys_d = nullptr;
    cudaMalloc(&keys_d, sizeof(key_t)*max_input_size); CUERR
    value_t* values_d = nullptr;
    cudaMalloc(&values_d, sizeof(value_t)*max_input_size); CUERR
    cudaMemcpy(keys_d, keys.data(), sizeof(key_t)*max_input_size, H2D); CUERR

    for(auto size : input_sizes)
    {
        for(auto load : load_factors)
        {
            const std::uint64_t capacity = size / load;

            HashTable hash_table(capacity);

            std::vector<float> insert_times(iters);
            for(uint64_t i = 0; i < iters; i++)
            {
                hash_table.init();
                cudaEvent_t insert_start, insert_stop;
                float t;
                cudaEventCreate(&insert_start);
                cudaEventCreate(&insert_stop);
                cudaEventRecord(insert_start, 0);
                hash_table.insert(keys_d, values_d, size);
                cudaEventRecord(insert_stop, 0);
                cudaEventSynchronize(insert_stop);
                cudaEventElapsedTime(&t, insert_start, insert_stop);
                cudaDeviceSynchronize(); CUERR
                insert_times[i] = t;
                std::this_thread::sleep_for (thermal_backoff);
            }
            const float insert_time =
                *std::min_element(insert_times.begin(), insert_times.end());

            std::vector<float> query_times(iters);
            for(uint64_t i = 0; i < iters; i++)
            {
                cudaEvent_t query_start, query_stop;
                float t;
                cudaEventCreate(&query_start);
                cudaEventCreate(&query_stop);
                cudaEventRecord(query_start, 0);
                hash_table.retrieve(keys_d, size, values_d);
                cudaEventRecord(query_stop, 0);
                cudaEventSynchronize(query_stop);
                cudaEventElapsedTime(&t, query_start, query_stop);
                cudaDeviceSynchronize(); CUERR
                query_times[i] = t;
                std::this_thread::sleep_for(thermal_backoff);
            }
            const float query_time =
                *std::min_element(query_times.begin(), query_times.end());

            const uint64_t total_bytes = (sizeof(key_t) + sizeof(value_t))*size;
            uint64_t ips = size/(insert_time/1000);
            uint64_t qps = size/(query_time/1000);
            float itp = helpers::B2GB(total_bytes) / (insert_time/1000);
            float qtp = helpers::B2GB(total_bytes) / (query_time/1000);
            float actual_load = hash_table.load_factor();
            warpcore::Status status = hash_table.pop_status();

            if(print_headers)
            {
                const char d = ' ';

                std::cout << "N=" << size << std::fixed
                    << d << "C=" << capacity
                    << d << "bits_key=" << sizeof(key_t)*CHAR_BIT
                    << d << "bits_value=" << sizeof(value_t)*CHAR_BIT
                    << d << "mb_keys=" << uint64_t(helpers::B2MB(sizeof(key_t)*size))
                    << d << "mb_values=" << uint64_t(helpers::B2MB(sizeof(value_t)*size))
                    << d << "load=" << actual_load
                    << d << "insert_ms=" << insert_time
                    << d << "query_ms=" << query_time
                    << d << "IPS=" << ips
                    << d << "QPS=" << qps
                    << d << "insert_GB/s=" << itp
                    << d << "query_GB/s=" << qtp
                    << d << "status=" << status << std::endl;
            }
            else
            {
                const char d = ' ';

                std::cout << std::fixed
                    << size
                    << d  << capacity
                    << d  << sizeof(key_t)*CHAR_BIT
                    << d  << sizeof(value_t)*CHAR_BIT
                    << d  << uint64_t(helpers::B2MB(sizeof(key_t)*size))
                    << d  << uint64_t(helpers::B2MB(sizeof(value_t)*size))
                    << d  << actual_load
                    << d  << insert_time
                    << d  << query_time
                    << d  << ips
                    << d << qps
                    << d << itp
                    << d << qtp
                    << d << status << std::endl;
            }
        }
    }

    cudaFree(keys_d); CUERR
    cudaFree(values_d); CUERR
}

int main(int argc, char* argv[])
{
    using namespace warpcore;

    using key_t = std::uint32_t;
    using value_t = std::uint32_t;

    using hash_table_t = SingleValueHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, value_t>>;

    const uint64_t max_keys = 1UL << 28;
    uint64_t dev_id = 0;
    std::vector<key_t> keys;

    if(argc > 2) dev_id = std::atoi(argv[2]);

    if(argc > 1)
    {
        keys = helpers::load_binary<key_t>(argv[1], max_keys);
    }
    else
    {
        keys.resize(max_keys);

        key_t * keys_d = nullptr;
        cudaMalloc(&keys_d, sizeof(key_t) * max_keys); CUERR

        helpers::lambda_kernel
        <<<SDIV(max_keys, 1024), 1024>>>
        ([=] DEVICEQUALIFIER
        {
            const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

            if(tid < max_keys)
            {
                keys_d[tid] = tid + 1;
            }
        });

        cudaMemcpy(keys.data(), keys_d, sizeof(key_t) * max_keys, D2H); CUERR

        cudaFree(keys_d); CUERR
    }

    single_value_benchmark<hash_table_t>(keys, dev_id, true);
}