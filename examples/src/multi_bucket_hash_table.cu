#include <iostream>
#include <algorithm>
#include <random>
#include <warpcore/multi_bucket_hash_table.cuh>
#include <helpers/timers.cuh>

int main ()
{
    using namespace warpcore;

    // key/value types
    using key_t = std::uint32_t;
    using value_t = std::uint64_t;

    // configure the hash table
    using hash_table_t = MultiBucketHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::empty_key<value_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, ArrayBucket<value_t,4>>>;
    using status_t = typename hash_table_t::status_type;
    using status_handler_t = typename status_handlers::ReturnStatus;

    const index_t size_unique_keys = 1UL << 5;
    const index_t size_values_per_key = 1UL << 15;
    const index_t max_values_per_key = 21;
    const index_t size = size_unique_keys * size_values_per_key;
    const index_t max_buckets = size_unique_keys * SDIV(max_values_per_key,4);
    const float load_factor = 0.95;

    helpers::GpuTimer init_table_timer("init_table");
    hash_table_t hash_table(max_buckets / load_factor, defaults::seed<key_t>(), max_values_per_key);
    init_table_timer.print();
    cudaDeviceSynchronize(); CUERR

    helpers::GpuTimer init_data_timer("init_data");
    key_t * keys_unique_h = nullptr;
    cudaMallocHost(&keys_unique_h, sizeof(key_t) * size_unique_keys); CUERR
    key_t * keys_unique_d = nullptr;
    cudaMalloc(&keys_unique_d, sizeof(key_t) * size_unique_keys); CUERR

    key_t * keys_in_h = nullptr;
    cudaMallocHost(&keys_in_h, sizeof(key_t) * size); CUERR
    key_t * keys_in_d = nullptr;
    cudaMalloc(&keys_in_d, sizeof(key_t) * size); CUERR

    value_t * values_in_h = nullptr;
    cudaMallocHost(&values_in_h, sizeof(value_t) * size); CUERR
    value_t * values_in_d = nullptr;
    cudaMalloc(&values_in_d, sizeof(value_t) * size); CUERR

    index_t * offsets_out_d = nullptr;
    cudaMalloc(&offsets_out_d, sizeof(index_t) * (size_unique_keys+1)); CUERR

    value_t * values_out_h = nullptr;
    cudaMallocHost(&values_out_h, sizeof(value_t) * size); CUERR
    value_t * values_out_d = nullptr;
    cudaMalloc(&values_out_d, sizeof(value_t) * size); CUERR

    status_t * status_h = nullptr;
    cudaMallocHost(&status_h, sizeof(status_t) * size); CUERR
    status_t * status_d = nullptr;
    cudaMalloc(&status_d, sizeof(status_t) * size); CUERR

    #pragma omp parallel for
    for(index_t i = 0; i < size_unique_keys; ++i)
    {
        keys_unique_h[i] = i + 1;

        for(index_t j = 0; j < size_values_per_key; ++j)
        {
            keys_in_h[i * size_values_per_key + j] = i + 1;
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(keys_in_h, keys_in_h + size, g);

    #pragma omp parallel for
    for(index_t i= 0; i < size; ++i)
    {
        values_in_h[i] = keys_in_h[i];
        status_h[i] = status_t::none();
    }

    cudaMemcpy(keys_in_d, keys_in_h, sizeof(key_t)*size, H2D); CUERR
    cudaMemcpy(values_in_d, values_in_h, sizeof(value_t)*size, H2D); CUERR
    cudaMemcpy(status_d, status_h, sizeof(status_t)*size, H2D); CUERR

    cudaMemset(values_out_d, 0, sizeof(value_t)*size); CUERR
    cudaMemset(offsets_out_d, 0, sizeof(index_t)*(size_unique_keys+1)); CUERR
    init_data_timer.print();

    const size_t batch_size = 1UL << 15;
    const size_t full_batches = size / batch_size;
    const size_t last_batch_size = size % batch_size;

    helpers::GpuTimer insert_timer("insert");
    for(size_t b=0; b<full_batches; ++b)
    {
        hash_table.insert<status_handler_t>(
            keys_in_d + b*batch_size,
            values_in_d + b*batch_size,
            batch_size,
            0,
            defaults::probing_length(),
            status_d + b*batch_size);
    }
    if(last_batch_size)
    {
        hash_table.insert<status_handler_t>(
            keys_in_d + full_batches*batch_size,
            values_in_d + full_batches*batch_size,
            last_batch_size,
            0,
            defaults::probing_length(),
            status_d + full_batches*batch_size);
    }
    insert_timer.print_throughput((sizeof(key_t)+sizeof(value_t)), size);
    cudaDeviceSynchronize(); CUERR

    cudaMemcpy(status_h, status_d, sizeof(status_t)*size, D2H); CUERR

    std::cout << "table errors " << hash_table.peek_status().get_errors() << std::endl;
    index_t errors = 0;
    for(index_t i = 0; i < size; ++i)
    {
        status_h[i] -= status_t::duplicate_key();

        if(status_h[i].has_any_errors())
        {
            if(errors++ < 10)
                std::cout <<
                    "STATUS: i " << i <<
                    " key " << keys_in_h[i] <<
                    " status " << status_h[i] << std::endl;
        }
    }
    if(errors >= 10)
    {
        std::cout << "...\n" << "total errors " << errors << std::endl;
    }

    std::cout << "num pairs " << size << std::endl;
    std::cout << "table size " << hash_table.size() << std::endl;
    std::cout << "key capacity " << hash_table.capacity() << std::endl;
    std::cout << "load factor " << hash_table.key_load_factor() << std::endl;
    std::cout << "expected unique keys " << size_unique_keys << std::endl;
    std::cout << "actual unique keys " << hash_table.num_keys() << std::endl;
    std::cout << "values per key " << size_values_per_key << std::endl;
    std::cout << "total values " << size << std::endl;
    std::cout << "capped values " << size_unique_keys*max_values_per_key << std::endl;

    #pragma omp parallel for
    for(index_t i= 0; i < size; ++i)
    {
        status_h[i] = status_t::none();
    }

    cudaMemcpy(status_d, status_h, sizeof(status_t)*size, H2D); CUERR

    index_t value_size = 0;

    index_t count_unique_keys = 0;
    hash_table.retrieve_all_keys(
        keys_unique_d,
        count_unique_keys);
    cudaDeviceSynchronize(); CUERR

    std::cout << "retrieved keys " << count_unique_keys << std::endl;

    helpers::GpuTimer retrieve_timer("retrieve");
    hash_table.retrieve<status_handler_t>(
        keys_unique_d,
        count_unique_keys,
        offsets_out_d,
        offsets_out_d+1,
        values_out_d,
        value_size,
        0,
        defaults::probing_length(),
        status_d);
    retrieve_timer.print_throughput((sizeof(key_t)+sizeof(value_t)), size);

    cudaDeviceSynchronize(); CUERR

    std::cout << "retrieved values " << value_size << std::endl;

    helpers::lambda_kernel<<<SDIV(size_unique_keys, 1024), 1024>>>([=] DEVICEQUALIFIER
    {
        const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if(tid < size_unique_keys)
        {
            const auto key = keys_unique_d[tid];
            const auto lower = offsets_out_d[tid];
            const auto upper = offsets_out_d[tid + 1];

            if(upper - lower != max_values_per_key)
            {
                printf("ERROR size values %llu (expected %llu)\n", upper - lower, max_values_per_key);
            }

            for (index_t i = lower; i < upper; i++)
            {
                if(values_out_d[i] != key)
                {
                    printf("ERROR expected %u got %llu\n", key, values_out_d[i]);
                }
            }
        }
    });

    cudaDeviceSynchronize(); CUERR

    cudaMemcpy(status_h, status_d, sizeof(status_t)*size, D2H); CUERR

    std::cout << "table errors " << hash_table.peek_status().get_errors() << std::endl;

    errors = 0;
    for(index_t i = 0; i < size; ++i)
    {

        if(status_h[i].has_any_errors())
        {
            if(errors++ < 10)
                std::cout <<
                "STATUS: i " << i <<
                " key " << keys_in_h[i] <<
                " status " << status_h[i] << std::endl;
        }
    }
    if(errors >= 10)
    {
        std::cout << "...\n" << "total errors " << errors << std::endl;
    }

    cudaFreeHost(keys_unique_h);
    cudaFreeHost(keys_in_h);
    cudaFreeHost(values_in_h);
    cudaFreeHost(values_out_h);
    cudaFreeHost(status_h);
    cudaFree(keys_unique_d);
    cudaFree(keys_in_d);
    cudaFree(values_in_d);
    cudaFree(offsets_out_d);
    cudaFree(values_out_d);
    cudaFree(status_d);

    cudaDeviceSynchronize(); CUERR
}
