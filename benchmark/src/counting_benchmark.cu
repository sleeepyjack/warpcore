#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include "warpcore.cuh"
#include "../../ext/hpc_helpers/include/io_helpers.h"

template<class T>
uint64_t num_unique(const std::vector<T>& v) noexcept
{
    T * keys_d = nullptr;
    cudaMalloc(&keys_d, sizeof(T) * v.size()); CUERR
    cudaMemcpy(keys_d, v.data(), sizeof(T) * v.size(), H2D); CUERR

    auto set = warpcore::HashSet<T>(v.size());

    set.insert(keys_d, v.size());

    cudaFree(keys_d);

    return set.size();
}

template<
    class Key,
    class Count,
    class Table>
HOSTQUALIFIER INLINEQUALIFIER
void counting_benchmark(
    std::vector<Key> keys_h,
    float load,
    uint8_t dev_id = 0,
    bool print_headers = true,
    uint8_t iters = 5)
{
    cudaSetDevice(dev_id); CUERR

    uint64_t size = num_unique(keys_h);
    uint64_t capacity = size/load;

    Key* keys_d = nullptr; cudaMalloc(&keys_d, sizeof(Key)*keys_h.size()); CUERR
    Count* counts_d = nullptr; cudaMalloc(&counts_d, sizeof(Count)*keys_h.size()); CUERR

    cudaMemcpy(keys_d, keys_h.data(), sizeof(Key)*keys_h.size(), H2D); CUERR

    Table hash_table(capacity);

    float insert_time = 0.0;
    for(uint8_t i = 0; i < iters; i++)
    {
        hash_table.init();
        cudaEvent_t insert_start, insert_stop;
        float t;
        cudaEventCreate(&insert_start);
        cudaEventCreate(&insert_stop);
        cudaEventRecord(insert_start, 0);
        hash_table.insert(keys_d, keys_h.size());
        cudaEventRecord(insert_stop, 0);
        cudaEventSynchronize(insert_stop);
        cudaEventElapsedTime(&t, insert_start, insert_stop);
        cudaDeviceSynchronize(); CUERR
        insert_time += t;
    }
    insert_time /= iters;

    float query_time = 0.0;
    for(uint8_t i = 0; i < iters; i++)
    {
        cudaEvent_t query_start, query_stop;
        float t;
        cudaEventCreate(&query_start);
        cudaEventCreate(&query_stop);
        cudaEventRecord(query_start, 0);
        hash_table.retrieve(keys_d, keys_h.size(), counts_d);
        cudaEventRecord(query_stop, 0);
        cudaEventSynchronize(query_stop);
        cudaEventElapsedTime(&t, query_start, query_stop);
        cudaDeviceSynchronize(); CUERR
        query_time += t;
    }
    query_time /= iters;

    uint64_t ips = keys_h.size()/insert_time*1000;
    uint64_t qps = keys_h.size()/query_time*1000;
    float itp = helpers::B2GB(sizeof(key_t)*keys_h.size()) / (insert_time/1000);
    float qtp = helpers::B2GB(sizeof(key_t)*keys_h.size()) / (query_time/1000);
    float actual_load = float(hash_table.size())/float(capacity);

    if(print_headers)
    {
        const char d = ' ';

        std::cout << std::fixed
            << "N=" << keys_h.size()
            << d << "C=" << capacity
            << d << "bits_key=" << sizeof(Key)*8
            << d << "bits_count=" << sizeof(Count)*8
            << d << "load=" << actual_load
            << d << "IPS=" << ips
            << d << "QPS=" << qps
            << d << "insert_GB/s=" << itp
            << d << "query_GB/s=" << qtp
            << d << "status=" << hash_table.pop_status() << std::endl;
    }
    else
    {
        const char d = ' ';

        std::cout << std::fixed
                  << keys_h.size()
                  << d << capacity
                  << d << sizeof(Key)*8
                  << d << sizeof(Count)*8
                  << d << actual_load
                  << d << ips
                  << d << qps
                  << d << itp
                  << d << qtp
                  << d << hash_table.pop_status() << std::endl;
    }

    cudaFree(keys_d); CUERR
    cudaFree(counts_d); CUERR
}

int main(int argc, char* argv[])
{
    using key_t = uint32_t;
    using count_t = uint32_t;
    using hash_table_t = warpcore::CountingHashTable<key_t, count_t>;

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
                // 8 values per key
                keys_d[tid] = (tid % (max_keys / 8)) + 1;
            }
        });

        cudaMemcpy(keys.data(), keys_d, sizeof(key_t) * max_keys, D2H); CUERR

        cudaFree(keys_d); CUERR
    }

    counting_benchmark<key_t, count_t, hash_table_t>(keys, 0.9, dev_id);
}