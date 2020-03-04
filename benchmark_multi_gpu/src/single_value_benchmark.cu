#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <thread>
#include <chrono>
#include "warpcore.cuh"
#include "hashers.cuh"

#include "gossip.cuh"
#include "plan_parser.hpp"

template<class Key, class Value>
bool sufficient_memory(
    const std::vector<uint16_t> dev_ids,
    size_t size, float load,
    float multi_split_overhead_factor,
    float headroom_factor)
{
    const size_t size_per_gpu = size/dev_ids.size();
    const size_t capacity = size_per_gpu/load;
    const size_t key_val_bytes = sizeof(Key)+sizeof(Value);
    const size_t table_bytes = key_val_bytes*capacity;
    const size_t io_bytes = key_val_bytes*size_per_gpu*(1+multi_split_overhead_factor);
    const size_t total_bytes = (table_bytes+io_bytes)*headroom_factor;

    bool sufficient = true;
    for(const auto& dev_id : dev_ids) {
        cudaSetDevice(dev_id); CUERR

        size_t bytes_free, bytes_total;
        cudaMemGetInfo(&bytes_free, &bytes_total); CUERR

        sufficient &= (total_bytes <= bytes_free);
    }
    return sufficient;
}

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void single_value_benchmark(
    const std::vector<typename HashTable::key_type>& keys,
    const std::vector<uint16_t> dev_ids,
    gossip::transfer_plan_t transfer_plan,
    bool print_headers = true,
    std::vector<uint64_t> input_sizes = {(1UL<<27)},
    std::vector<float> load_factors = {0.8},
    uint8_t iters = 5,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100),
    float multi_split_overhead_factor = 1.5f)
{
    auto context = std::make_unique< gossip::context_t >(dev_ids);
    auto all2all = std::make_unique< gossip::all2all_t >(*context, transfer_plan);
    auto multisplit = std::make_unique< gossip::multisplit_t >(*context);


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

    float headroom_factor = 1.1;
    if(!sufficient_memory<key_t, value_t>(dev_ids, max_input_size, min_load_factor,
                                          multi_split_overhead_factor, headroom_factor))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    const uint32_t num_gpus = dev_ids.size();
    const uint64_t max_input_size_per_gpu = max_input_size / num_gpus;
    const uint64_t max_overhead_size_per_gpu = max_input_size_per_gpu * multi_split_overhead_factor;

    std::vector<key_t*> keys_d(num_gpus, nullptr);
    std::vector<key_t*> keys_split_d(num_gpus, nullptr);
    std::vector<key_t*> keys_transfer_d(num_gpus, nullptr);
    std::vector<value_t*> values_d(num_gpus, nullptr);
    std::vector<value_t*> values_split_d(num_gpus, nullptr);
    std::vector<value_t*> values_transfer_d(num_gpus, nullptr);

    for(uint32_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(dev_ids[i]); CUERR
        cudaMalloc(&keys_d[i], sizeof(key_t)*max_overhead_size_per_gpu); CUERR
        cudaMalloc(&keys_split_d[i], sizeof(key_t)*max_overhead_size_per_gpu); CUERR
        cudaMalloc(&keys_transfer_d[i], sizeof(key_t)*max_overhead_size_per_gpu); CUERR
        cudaMalloc(&values_d[i], sizeof(value_t)*max_overhead_size_per_gpu); CUERR
        cudaMalloc(&values_split_d[i], sizeof(value_t)*max_overhead_size_per_gpu); CUERR
        cudaMalloc(&values_transfer_d[i], sizeof(value_t)*max_overhead_size_per_gpu); CUERR

        cudaMemcpy(keys_d[i], keys.data()+i*max_input_size_per_gpu,
                   sizeof(key_t)*max_input_size_per_gpu, H2D); CUERR
    }

    for(auto size : input_sizes)
    {
        for(auto load : load_factors)
        {
            const std::uint64_t size_per_gpu = size / num_gpus;
            const std::uint64_t capacity = size_per_gpu / load;
            std::vector<uint64_t> actual_sizes(num_gpus, size_per_gpu);
            std::vector<uint64_t> overhead_sizes(num_gpus, size_per_gpu*multi_split_overhead_factor);
            std::vector<size_t> lengths(num_gpus);

            std::vector<HashTable> hash_table;
            for(uint32_t i = 0; i < num_gpus; ++i) {
                cudaSetDevice(dev_ids[i]); CUERR
                hash_table.emplace_back(capacity);
            }

            std::vector<float> multisplit_times(iters);
            std::vector<float> all2all_times(iters);
            std::vector<float> insert_times(iters);
            for(uint64_t iter = 0; iter < iters; iter++)
            {
                for(uint32_t i = 0; i < num_gpus; ++i) {
                    cudaSetDevice(dev_ids[i]); CUERR
                    hash_table[i].init();
                }

                std::vector<std::vector<size_t>> partition_table(num_gpus, std::vector<size_t>(num_gpus));

                auto part_hash = [=] DEVICEQUALIFIER (const key_t& x){
                    return (x % num_gpus);
                };

                auto multisplit_start = std::chrono::system_clock::now();
                multisplit->execAsync(
                    keys_d, values_d, actual_sizes,
                    keys_split_d, values_split_d, overhead_sizes,
                    partition_table, part_hash);
                multisplit->sync();
                auto multisplit_stop = std::chrono::system_clock::now();
                multisplit_times[iter] = std::chrono::duration<float>(multisplit_stop-multisplit_start).count();

                // for (uint32_t src = 0; src < num_gpus; src++) {
                //     for (uint32_t trg = 0; trg < num_gpus; trg++)
                //         std::cout << partition_table[src][trg] << ' ';
                //     std::cout << '\n';
                // }

                auto all2all_start = std::chrono::system_clock::now();
                all2all->execAsync(
                    keys_split_d, actual_sizes,
                    keys_transfer_d, overhead_sizes,
                    partition_table);
                all2all->execAsync(
                    values_split_d, actual_sizes,
                    values_transfer_d, overhead_sizes,
                    partition_table);
                all2all->sync();
                auto all2all_stop = std::chrono::system_clock::now();
                all2all_times[iter] = std::chrono::duration<float>(all2all_stop-all2all_start).count();

                for (uint32_t trg = 0; trg < num_gpus; trg++) {
                    lengths[trg] = 0;
                    for (uint32_t src = 0; src < num_gpus; src++)
                        lengths[trg] += partition_table[src][trg];
                }

                auto insert_start = std::chrono::system_clock::now();
                for(uint32_t i = 0; i < num_gpus; ++i) {
                    cudaSetDevice(dev_ids[i]); CUERR
                    hash_table[i].insert(keys_transfer_d[i], values_transfer_d[i], lengths[i]);
                }
                for(uint32_t i = 0; i < num_gpus; ++i) {
                    cudaSetDevice(dev_ids[i]); CUERR
                    cudaDeviceSynchronize(); CUERR
                }
                auto insert_stop = std::chrono::system_clock::now();
                insert_times[iter] = std::chrono::duration<float>(insert_stop-insert_start).count();
                std::this_thread::sleep_for (thermal_backoff);
            }
            const float insert_time = *std::min_element(insert_times.begin(), insert_times.end());
            const float multisplit_time = *std::min_element(multisplit_times.begin(), multisplit_times.end());
            const float all2all_time = *std::min_element(all2all_times.begin(), all2all_times.end());

            std::vector<float> query_times(iters);
            for(uint64_t iter = 0; iter < iters; iter++)
            {
                auto query_start = std::chrono::system_clock::now();
                for(uint32_t i = 0; i < num_gpus; ++i) {
                    cudaSetDevice(dev_ids[i]); CUERR
                    hash_table[i].retrieve(keys_transfer_d[i], lengths[i], values_transfer_d[i]);
                }
                for(uint32_t i = 0; i < num_gpus; ++i) {
                    cudaSetDevice(dev_ids[i]); CUERR
                    cudaDeviceSynchronize(); CUERR
                }
                auto query_stop = std::chrono::system_clock::now();
                query_times[iter] = std::chrono::duration<float>(query_stop-query_start).count();
                std::this_thread::sleep_for(thermal_backoff);
            }
            const float query_time =
                *std::min_element(query_times.begin(), query_times.end());

            const uint64_t total_bytes = (sizeof(key_t) + sizeof(value_t))*size;
            uint64_t ips = size/(insert_time);
            uint64_t qps = size/(query_time);
            float itp = helpers::B2GB(total_bytes) / (insert_time);
            float qtp = helpers::B2GB(total_bytes) / (query_time);

            std::vector<float> actual_load;
            std::vector<warpcore::Status> status;
            for(uint32_t i = 0; i < num_gpus; ++i) {
                actual_load.emplace_back(hash_table[i].load_factor());
                status.emplace_back(hash_table[i].pop_status());
            }
            auto minmax_actual_load =
                std::minmax_element(actual_load.begin(), actual_load.end());

            if(print_headers)
            {
                const char d = ' ';

                std::cout << "N=" << size << std::fixed
                    << d << "C=" << capacity
                    << d << "bits_key=" << sizeof(key_t)*CHAR_BIT
                    << d << "bits_value=" << sizeof(value_t)*CHAR_BIT
                    << d << "mb_keys=" << uint64_t(helpers::B2MB(sizeof(key_t)*size))
                    << d << "mb_values=" << uint64_t(helpers::B2MB(sizeof(value_t)*size))
                    << d << "num_gpus=" << num_gpus
                    << d << "load=[" << *minmax_actual_load.first
                    << d  << *minmax_actual_load.second << ']'
                    << d << "multisplit_ms=" << multisplit_time*1000
                    << d << "all2all_ms=" << all2all_time*1000
                    << d << "insert_ms=" << insert_time*1000
                    << d << "query_ms=" << query_time*1000
                    << d << "IPS=" << ips
                    << d << "QPS=" << qps
                    << d << "insert_GB/s=" << itp
                    << d << "query_GB/s=" << qtp
                    << d << "status=";
                for(uint32_t i = 0; i < num_gpus; ++i) {
                    std::cout << status[i];
                }
                std::cout << std::endl;
            }
            else
            {
                const char d = ' ';

                std::cout << std::fixed
                    << size
                    << d << capacity
                    << d << sizeof(key_t)*CHAR_BIT
                    << d << sizeof(value_t)*CHAR_BIT
                    << d << uint64_t(helpers::B2MB(sizeof(key_t)*size))
                    << d << uint64_t(helpers::B2MB(sizeof(value_t)*size))
                    << d << num_gpus
                    << d << *minmax_actual_load.first
                    << d << *minmax_actual_load.second
                    << d << multisplit_time*1000
                    << d << all2all_time*1000
                    << d << insert_time*1000
                    << d << query_time*1000
                    << d << ips
                    << d << qps
                    << d << itp
                    << d << qtp;
                for(uint32_t i = 0; i < num_gpus; ++i) {
                    std::cout << d << status[i];
                }
                std::cout << std::endl;
            }
        }
    }

    for(uint32_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(dev_ids[i]); CUERR
        cudaFree(keys_d[i]); CUERR
        cudaFree(keys_split_d[i]); CUERR
        cudaFree(keys_transfer_d[i]); CUERR
        cudaFree(values_d[i]); CUERR
        cudaFree(values_split_d[i]); CUERR
        cudaFree(values_transfer_d[i]); CUERR
    }
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

    uint16_t num_gpus = 2;
    if(argc > 1)
        num_gpus = std::stoi(argv[1]);

    gossip::transfer_plan_t transfer_plan = gossip::all2all::default_plan(num_gpus);;
    if(argc > 2)
        transfer_plan = parse_plan(argv[2]);
    gossip::all2all::verify_plan(transfer_plan);

    const uint64_t max_keys = 1UL << 28;
    std::vector<key_t> keys(max_keys);
    key_t * keys_d = nullptr;
    cudaMalloc(&keys_d, sizeof(key_t) * max_keys); CUERR

    lambda_kernel
    <<<SDIV(max_keys, 1024), 1024>>>
    ([=] DEVICEQUALIFIER
    {
        const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if(tid < max_keys)
        {
            keys_d[tid] = warpcore::hashers::MurmurHash<std::uint32_t>::hash(tid + 1);
        }
    });

    cudaMemcpy(keys.data(), keys_d, sizeof(key_t) * max_keys, D2H); CUERR

    cudaFree(keys_d); CUERR

    std::vector<uint16_t> dev_ids(num_gpus);
    std::iota(dev_ids.begin(), dev_ids.end(), 0);

    single_value_benchmark<hash_table_t>(keys, dev_ids, transfer_plan, true);
}