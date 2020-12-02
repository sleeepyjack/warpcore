

#ifndef BENCHMARK_COMMON_CUH
#define BENCHMARK_COMMON_CUH

#include "warpcore.cuh"
#include "../../ext/hpc_helpers/include/cuda_helpers.cuh"
#include "../../ext/hpc_helpers/include/io_helpers.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

uint64_t memory_partition(float factor = 0.4)
{
    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total); CUERR

    return bytes_free * factor;
}

template<class T>
uint64_t num_unique(const T* keys_d, const uint64_t size) noexcept
{
    auto set = warpcore::HashSet<T>(size);

    set.insert(keys_d, size);

    return set.size();
}

template<class key_t>
key_t * generate_keys(
    const uint64_t max_keys = 1UL << 27,
    const uint32_t multiplicity = 1)
{
    key_t * keys_d = nullptr;
    cudaMalloc(&keys_d, sizeof(key_t) * max_keys); CUERR

    helpers::lambda_kernel
    <<<SDIV(max_keys, 1024), 1024>>>
    ([=] DEVICEQUALIFIER
    {
        const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if(tid < max_keys)
        {
            keys_d[tid] = (tid % (max_keys / multiplicity)) + 1;
        }
    });

    cudaDeviceSynchronize(); CUERR

    return keys_d;
}

template<class key_t>
key_t * load_keys(
    const char* file_name,
    const uint64_t max_keys = 1UL << 27)
{
    std::vector<key_t> keys = helpers::load_binary<key_t>(file_name, max_keys);

    key_t * keys_d = nullptr;
    cudaMalloc(&keys_d, sizeof(key_t) * max_keys); CUERR

    cudaMemcpy(keys_d, keys.data(), sizeof(key_t) * max_keys, H2D); CUERR

    return keys_d;
}

template<class HashTable>
bool sufficient_memory_oa(size_t target_capacity, float headroom_factor = 1.1)
{
    using key_t = typename HashTable::key_type;
    using value_t = typename HashTable::value_type;

    const size_t capacity =
        warpcore::detail::get_valid_capacity(target_capacity, HashTable::cg_size());
    const size_t key_val_bytes = sizeof(key_t) + sizeof(value_t);
    const size_t table_bytes = key_val_bytes*capacity;
    const size_t total_bytes = table_bytes*headroom_factor;

    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total); CUERR

    return (total_bytes <= bytes_free);
}

template<class HashTable>
float benchmark_insert(
    HashTable& hash_table,
    const typename HashTable::key_type * keys_d,
    const uint64_t size,
    const uint8_t iters,
    const std::chrono::milliseconds thermal_backoff)
{
    std::vector<float> insert_times(iters);
    for(uint64_t i = 0; i < iters; i++)
    {
        hash_table.init();
        cudaEvent_t insert_start, insert_stop;
        float t;
        cudaEventCreate(&insert_start);
        cudaEventCreate(&insert_stop);
        cudaEventRecord(insert_start, 0);
        hash_table.insert(keys_d, size);
        cudaEventRecord(insert_stop, 0);
        cudaEventSynchronize(insert_stop);
        cudaEventElapsedTime(&t, insert_start, insert_stop);
        cudaDeviceSynchronize(); CUERR
        insert_times[i] = t;
        std::this_thread::sleep_for (thermal_backoff);
    }
    return *std::min_element(insert_times.begin(), insert_times.end());
}

template<class HashTable>
float benchmark_insert(
    HashTable& hash_table,
    const typename HashTable::key_type * keys_d,
    const typename HashTable::value_type * values_d,
    const uint64_t size,
    const uint8_t iters,
    const std::chrono::milliseconds thermal_backoff)
{
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
    return *std::min_element(insert_times.begin(), insert_times.end());
}

template<class HashTable>
float benchmark_query(
    HashTable& hash_table,
    const typename HashTable::key_type * keys_d,
    typename HashTable::value_type * values_d,
    const uint64_t size,
    const uint8_t iters,
    const std::chrono::milliseconds thermal_backoff)
{
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
    return *std::min_element(query_times.begin(), query_times.end());
}

template<class HashTable>
float benchmark_query_multi(
    HashTable& hash_table,
    typename HashTable::key_type * keys_d,
    const uint64_t size,
    typename HashTable::index_type * offsets_d,
    typename HashTable::value_type * values_d,
    const uint8_t iters,
    const std::chrono::milliseconds thermal_backoff)
{
    using index_t = typename HashTable::index_type;

    helpers::lambda_kernel
    <<<SDIV(size, 1024), 1024>>>
    ([=] DEVICEQUALIFIER
    {
        const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if(tid < size)
        {
            keys_d[tid] = tid + 1;
        }
    });
    cudaDeviceSynchronize(); CUERR

    index_t value_size_out = 0;

    std::vector<float> query_times(iters);
    for(uint64_t i = 0; i < iters; i++)
    {
        cudaEvent_t query_start, query_stop;
        float t;
        cudaEventCreate(&query_start);
        cudaEventCreate(&query_stop);
        cudaEventRecord(query_start, 0);
        hash_table.retrieve(
            keys_d,
            size,
            offsets_d,
            offsets_d+1,
            values_d,
            value_size_out);
        cudaEventRecord(query_stop, 0);
        cudaEventSynchronize(query_stop);
        cudaEventElapsedTime(&t, query_start, query_stop);
        cudaDeviceSynchronize(); CUERR
        query_times[i] = t;
        std::this_thread::sleep_for(thermal_backoff);
    }
    return *std::min_element(query_times.begin(), query_times.end());
}

template<class HashTable>
float benchmark_query_unique(
    HashTable& hash_table,
    typename HashTable::key_type * unique_keys_d,
    typename HashTable::index_type * offsets_d,
    typename HashTable::value_type * values_d,
    const uint8_t iters,
    const std::chrono::milliseconds thermal_backoff)
{
    using index_t = typename HashTable::index_type;

    index_t key_size_out = 0;
    index_t value_size_out = 0;

    hash_table.retrieve_all_keys(unique_keys_d, key_size_out); CUERR

    std::vector<float> query_times(iters);
    for(uint64_t i = 0; i < iters; i++)
    {
        cudaEvent_t query_start, query_stop;
        float t;
        cudaEventCreate(&query_start);
        cudaEventCreate(&query_stop);
        cudaEventRecord(query_start, 0);
        hash_table.retrieve(
            unique_keys_d,
            key_size_out,
            offsets_d,
            offsets_d+1,
            values_d,
            value_size_out);
        cudaEventRecord(query_stop, 0);
        cudaEventSynchronize(query_stop);
        cudaEventElapsedTime(&t, query_start, query_stop);
        cudaDeviceSynchronize(); CUERR
        query_times[i] = t;
        std::this_thread::sleep_for(thermal_backoff);
    }
    return *std::min_element(query_times.begin(), query_times.end());
}

class NoValue {};

template<class key_t, class value_t = NoValue>
struct Output {
// private:
    // enum mode { Set, KeyValue, MultiValue };
    // mode mode_ = Set;

public:
    char d = ' ';
    uint64_t sample_size = 0; // num_keys?
    uint64_t key_capacity = 0;
    uint64_t value_capacity = 0;
    float key_load_factor = 0;
    float value_load_factor = 0;
    float density = 0;
    float relative_density = 0;
    float insert_ms = 0;
    float query_ms = 0;

    // void mode_set() { mode_ = Set; }
    // void mode_key_value() { mode_ = KeyValue; }
    // void mode_multi_value() { mode_ = MultiValue; }

    uint32_t bits_key() const
    {
        return sizeof(key_t)*CHAR_BIT;
    }
    uint32_t bits_value() const
    {
        if(std::is_same<value_t, NoValue>::value)
            return 0;
        else
            return sizeof(value_t)*CHAR_BIT;
    }
    float mb_keys() const {
        return helpers::B2MB(sizeof(key_t)*sample_size);
    }
    float mb_values() const
    {
        if(std::is_same<value_t, NoValue>::value)
            return 0;
        else
            return helpers::B2MB(sizeof(value_t)*sample_size);
    }
    float mb_total() const {
        if(std::is_same<value_t, NoValue>::value)
            return mb_keys();
        else
            return mb_keys() + mb_values();
    }

    float insert_s() const { return insert_ms/1000; }
    float query_s() const { return query_ms/1000; }
    uint64_t inserts_per_second() const { return sample_size/insert_s(); }
    uint64_t queries_per_second() const { return sample_size/query_s(); }
    float insert_troughput_gbps() const { return mb_total()/1024 / insert_s(); }
    float query_troughput_gbps() const { return mb_total()/1024 / query_s(); }

    warpcore::Status status = warpcore::Status::none();

    void print_with_headers() const noexcept
    {
        std::cout << std::fixed
            << "sample_size=" << sample_size
            << d << "key_capacity=" << key_capacity
            << d << "value_capacity=" << value_capacity
            << d << "bits_key=" << bits_key()
            << d << "bits_value=" << bits_value()
            << d << "mb_keys=" << mb_keys()
            << d << "mb_values=" << mb_values()
            << d << "key_load=" << key_load_factor
            << d << "value_load=" << value_load_factor
            << d << "density=" << density
            << d << "relative_density=" << relative_density
            << d << "insert_ms=" << insert_ms
            << d << "query_ms=" << query_ms
            << d << "IPS=" << inserts_per_second()
            << d << "QPS=" << queries_per_second()
            << d << "insert_GB/s=" << insert_troughput_gbps()
            << d << "query_GB/s=" << query_troughput_gbps()
            << d << "status=" << status
            << std::endl;
    }

    void print_without_headers() const noexcept
    {
        std::cout << std::fixed
            << sample_size
            << d << key_capacity
            << d << value_capacity
            << d << bits_key()
            << d << bits_value()
            << d << mb_keys()
            << d << mb_values()
            << d << key_load_factor
            << d << value_load_factor
            << d << density
            << d << relative_density
            << d << insert_ms
            << d << query_ms
            << d << inserts_per_second()
            << d << queries_per_second()
            << d << insert_troughput_gbps()
            << d << query_troughput_gbps()
            << d << status
            << std::endl;
    }
};

#endif