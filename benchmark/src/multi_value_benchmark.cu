#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>
#include <tuple>
#include <thread>
#include <chrono>
#include "warpcore.cuh"

template<class Key, class Value>
bool sufficient_memory(
    size_t input_size,
    size_t key_store_capacity,
    size_t value_store_capacity,
    float headroom_factor = 1.1)
{
    const size_t key_handle_bytes = sizeof(Key)+sizeof(uint64_t);
    const size_t table_bytes = key_handle_bytes*key_store_capacity;
    const size_t value_bytes = std::min(sizeof(Value), sizeof(uint64_t));
    const size_t value_store_bytes = value_bytes * value_store_capacity;
    const size_t io_bytes = (sizeof(Key)+sizeof(Value)+sizeof(uint64_t))*input_size;
    const size_t total_bytes = (table_bytes+value_store_bytes+io_bytes)*headroom_factor;

    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total); CUERR

    return (total_bytes <= bytes_free);
}

uint64_t memory_partition(float factor = 0.4)
{
    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total); CUERR

    return bytes_free * factor;
}

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

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void multi_value_benchmark(
    const std::vector<typename HashTable::key_type>& keys,
    uint64_t key_store_capacity,
    uint64_t value_store_capacity,
    std::vector<uint64_t> input_sizes = {(1UL<<27)},
    std::vector<std::tuple<float, uint64_t, uint64_t>> slab_list_configs = {{1.1, 1, 0}},
    typename HashTable::key_type seed = warpcore::defaults::seed<key_t>(),
    uint64_t dev_id = 0,
    bool print_headers = true,
    uint8_t iters = 3,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100))
{
    using index_t = typename HashTable::index_type;
    cudaSetDevice(dev_id); CUERR

    using key_t = typename HashTable::key_type;
    using value_t = typename HashTable::value_type;

    const auto max_input_size =
        *std::max_element(input_sizes.begin(), input_sizes.end());

    if(max_input_size > keys.size())
    {
        std::cerr << "Maximum input size exceeded." << std::endl;
        exit(1);
    }

    if(!sufficient_memory<key_t, value_t>(
            max_input_size, key_store_capacity, value_store_capacity))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    key_t* keys_d = nullptr;
    cudaMalloc(&keys_d, sizeof(key_t)*max_input_size); CUERR
    key_t* unique_keys_d = nullptr;
    cudaMalloc(&unique_keys_d, sizeof(key_t)*max_input_size); CUERR
    value_t* values_d = nullptr;
    cudaMalloc(&values_d, sizeof(value_t)*max_input_size); CUERR
    index_t * offsets_d = nullptr;
    cudaMalloc(&offsets_d, sizeof(index_t)*max_input_size); CUERR
    void * temp_d = nullptr;
    cudaMemcpy(keys_d, keys.data(), sizeof(key_t)*max_input_size, H2D); CUERR

    for(const auto& size : input_sizes)
    {
        for(const auto& slab_list_config : slab_list_configs)
        {
            const float slab_grow_factor = std::get<0>(slab_list_config);
            const index_t min_slab_size = std::get<1>(slab_list_config);
            const index_t max_slab_size = std::get<2>(slab_list_config);

            HashTable hash_table(
                key_store_capacity,
                value_store_capacity,
                seed,
                slab_grow_factor,
                min_slab_size);

            std::vector<float> insert_times;
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
                insert_times.push_back(t);
                std::this_thread::sleep_for(thermal_backoff);
            }
            const float insert_time =
                *std::min_element(insert_times.begin(), insert_times.end());

            index_t key_size_out = 0;
            index_t value_size_out = 0;
            std::size_t temp_bytes = 0;

            hash_table.retrieve_all_keys(unique_keys_d, key_size_out);

            /*
            key_size_out = size;
            lambda_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
                [=] DEVICEQUALIFIER
                {
                    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

                    if(tid >= size) return;
                    unique_keys_d[tid] = tid + 1;
                });
            cudaDeviceSynchronize(); CUERR
            */
            hash_table.retrieve(
                unique_keys_d,
                key_size_out,
                offsets_d,
                values_d,
                value_size_out,
                temp_d,
                temp_bytes);
            cudaMalloc(&temp_d, temp_bytes); CUERR
            cudaDeviceSynchronize(); CUERR

            std::vector<float> query_times;
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
                    values_d,
                    value_size_out,
                    temp_d,
                    temp_bytes);
                cudaEventRecord(query_stop, 0);
                cudaEventSynchronize(query_stop);
                cudaEventElapsedTime(&t, query_start, query_stop);
                cudaDeviceSynchronize(); CUERR
                query_times.push_back(t);
                std::this_thread::sleep_for(thermal_backoff);
            }
            const float query_time =
                *std::min_element(query_times.begin(), query_times.end());

            const uint64_t total_bytes = (sizeof(key_t) + sizeof(value_t))*size;
            uint64_t ips = size/(insert_time/1000);
            uint64_t qps = size/(query_time/1000);
            float itp = B2GB(total_bytes) / (insert_time/1000);
            float qtp = B2GB(total_bytes) / (query_time/1000);
            float key_load = hash_table.key_load_factor();
            float value_load = hash_table.value_load_factor();
            float density = hash_table.storage_density();
            float relative_density = hash_table.relative_storage_density();
            warpcore::Status status = hash_table.pop_status();

            if(print_headers)
            {
                const char d = ' ';

                std::cout << "N=" << size << std::fixed
                    << d << "key_capacity=" << key_store_capacity
                    << d << "value_capacity=" << value_store_capacity
                    << d << "bits_key=" << sizeof(key_t)*CHAR_BIT
                    << d << "bits_value=" << sizeof(value_t)*CHAR_BIT
                    << d << "mb_keys=" << uint64_t(B2MB(sizeof(key_t)*size))
                    << d << "mb_values=" << uint64_t(B2MB(sizeof(value_t)*size))
                    << d << "grow_factor=" << slab_grow_factor
                    << d << "min_slab_size=" << min_slab_size
                    << d << "max_slab_size=" << max_slab_size
                    << d << "key_load=" << key_load
                    << d << "value_load=" << value_load
                    << d << "density=" << density
                    << d << "relative_density=" << relative_density
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
                    << d << key_store_capacity
                    << d << value_store_capacity
                    << d << sizeof(key_t)*CHAR_BIT
                    << d << sizeof(value_t)*CHAR_BIT
                    << d << uint64_t(B2MB(sizeof(key_t)*size))
                    << d << uint64_t(B2MB(sizeof(value_t)*size))
                    << d << slab_grow_factor
                    << d << min_slab_size
                    << d << max_slab_size
                    << d << key_load
                    << d << value_load
                    << d << density
                    << d << relative_density
                    << d << insert_time
                    << d << query_time
                    << d << ips
                    << d << qps
                    << d << itp
                    << d << qtp
                    << d << status << std::endl;
            }

            cudaFree(temp_d); CUERR
            temp_d = nullptr;
        }
    }

    cudaFree(keys_d); CUERR
    cudaFree(unique_keys_d); CUERR
    cudaFree(values_d); CUERR
    cudaFree(offsets_d); CUERR

    cudaDeviceSynchronize(); CUERR
}

int main(int argc, char* argv[])
{
    using namespace warpcore;

    using key_t = std::uint64_t;
    using value_t = std::uint64_t;

    using hash_table_t = MultiValueHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        storage::multi_value::DynamicSlabListStore<value_t, 29, 18, 15>,
        defaults::probing_scheme_t<key_t, 8>>;


    const uint64_t max_keys = 1UL << 27;
    uint64_t dev_id = 0;
    std::vector<key_t> keys;

    if(argc > 2) dev_id = std::atoi(argv[2]);

    if(argc > 1)
    {
        keys = load_binary<key_t>(argv[1], max_keys);
    }
    else
    {
        keys.resize(max_keys);

        key_t * keys_d = nullptr;
        cudaMalloc(&keys_d, sizeof(key_t) * max_keys); CUERR

        lambda_kernel
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

    std::cout << "unique_keys: " <<  num_unique(keys) << "\tvalues: " << keys.size() << std::endl;
    multi_value_benchmark<hash_table_t>(
        keys,
        num_unique(keys) / 0.90,
        keys.size() / 0.60,
        {max_keys},
        {{1.1, 1, 0}},
        0x5ad0ded,
        dev_id);

    cudaDeviceSynchronize(); CUERR
}
