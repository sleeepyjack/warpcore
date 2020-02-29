#include <iostream>
#include <algorithm>
#include <random>
#include "multi_value_hash_table.cuh"

int main ()
{
    using namespace warpcore;

    using key_t = std::uint64_t;
    using value_t = std::uint64_t;

    using hash_table_t = MultiValueHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        storage::multi_value::DynamicSlabListStore<value_t>,
        defaults::probing_scheme_t<key_t, 8>>;
    using status_t = typename hash_table_t::status_type;
    using status_handler_t = typename status_handlers::ReturnStatus;

    const index_t size_unique_keys = 1UL << 20;
    const index_t size_values_per_key = 4;
    const index_t size = size_unique_keys * size_values_per_key;
    const float key_load_factor = 0.95;
    const float value_load_factor = 0.4;

    const index_t key_capacity = float(size_unique_keys) / key_load_factor;
    const index_t value_capacity = float(size) / value_load_factor;

    TIMERSTART(init_table)
    hash_table_t hash_table(key_capacity, value_capacity);
    TIMERSTOP(init_table); CUERR
    std::cout << hash_table.peek_status() << std::endl;

    TIMERSTART(init_data)
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

    index_t * offsets_out_h = nullptr;
    cudaMallocHost(&offsets_out_h, sizeof(index_t) * size_unique_keys); CUERR
    index_t * offsets_out_d = nullptr;
    cudaMalloc(&offsets_out_d, sizeof(index_t) * size_unique_keys); CUERR

    value_t * values_out_h = nullptr;
    cudaMallocHost(&values_out_h, sizeof(value_t) * size); CUERR
    value_t * values_out_d = nullptr;
    cudaMalloc(&values_out_d, sizeof(value_t) * size); CUERR

    status_t * status_h = nullptr;
    cudaMallocHost(&status_h, sizeof(status_t) * size); CUERR
    status_t * status_d = nullptr;
    cudaMalloc(&status_d, sizeof(status_t) * size); CUERR

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

    cudaMemcpy(keys_unique_d, keys_unique_h, sizeof(key_t)*size_unique_keys, H2D); CUERR
    cudaMemcpy(keys_in_d, keys_in_h, sizeof(key_t)*size, H2D); CUERR
    cudaMemcpy(values_in_d, values_in_h, sizeof(value_t)*size, H2D); CUERR
    cudaMemset(values_out_d, 0, sizeof(value_t)*size); CUERR
    cudaMemset(offsets_out_d, 0, sizeof(index_t)*size_unique_keys); CUERR
    cudaMemcpy(status_d, status_h, sizeof(status_t)*size, H2D); CUERR
    TIMERSTOP(init_data); CUERR

    THROUGHPUTSTART(insert)
    hash_table.insert<status_handler_t>(
        keys_in_d,
        values_in_d,
        size,
        0,
        defaults::probing_length(),
        status_d);
    THROUGHPUTSTOP(insert, (sizeof(key_t)+sizeof(value_t)), size); CUERR
    cudaDeviceSynchronize(); CUERR

    cudaMemcpy(status_h, status_d, sizeof(status_t)*size, D2H); CUERR

    std::cout << "table status " << hash_table.peek_status() << std::endl;
    index_t errors = 0;
    for(index_t i = 0; i < size; ++i)
    {

        if(status_h[i].has_any())
        {
            if(errors++ < 10)
                std::cout << "STATUS: i " << i << " key " << keys_in_h[i] << " status " << status_h[i] << std::endl;
        }
    }
    if(errors >= 10)
    {
        std::cout << "...\n" << "total errors " << errors << std::endl;
    }

    std::cout << "capacity keys " << hash_table.key_capacity() << std::endl;
    std::cout << "capacity values " << hash_table.value_capacity() << std::endl;
    std::cout << "unique keys " << size_unique_keys << std::endl;
    std::cout << "values per key " << size_values_per_key << std::endl;
    std::cout << "total values " << size << std::endl;
    std::cout << "unique keys in table " << hash_table.size_keys() << std::endl;
    std::cout << "total values in table " << hash_table.size_values() << std::endl;
    std::cout << "density " << hash_table.storage_density() << std::endl;

    #pragma omp parallel for
    for(index_t i= 0; i < size; ++i)
    {
        status_h[i] = status_t::none();
    }

    cudaMemcpy(status_d, status_h, sizeof(status_t)*size, H2D); CUERR

    void * temp_d = nullptr;
    index_t temp_bytes = 0;
    index_t value_size = 0;

    THROUGHPUTSTART(retrieve_dummy)
    hash_table.retrieve<status_handler_t>(
        keys_unique_d,
        size_unique_keys,
        offsets_out_d,
        values_out_d,
        value_size,
        temp_d,
        temp_bytes,
        0,
        defaults::probing_length(),
        status_d);
    THROUGHPUTSTOP(retrieve_dummy, (sizeof(key_t)+sizeof(value_t)), size); CUERR
    cudaDeviceSynchronize(); CUERR

    cudaMalloc(&temp_d, temp_bytes); CUERR

    THROUGHPUTSTART(retrieve)
    hash_table.retrieve<status_handler_t>(
        keys_unique_d,
        size_unique_keys,
        offsets_out_d,
        values_out_d,
        value_size,
        temp_d,
        temp_bytes,
        0,
        defaults::probing_length(),
        status_d);
    THROUGHPUTSTOP(retrieve, (sizeof(key_t)+sizeof(value_t)), size); CUERR
    cudaDeviceSynchronize(); CUERR

    std::cout << "temp_bytes: " << B2KB(temp_bytes) << "KB\tvalue_size: " << value_size << std::endl;

    lambda_kernel<<<SDIV(size_unique_keys, 1024), 1024>>>([=] DEVICEQUALIFIER
    {
        const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if(tid < size_unique_keys)
        {
            const auto key = keys_unique_d[tid];
            const auto lower = (tid == 0) ? 0 : offsets_out_d[tid - 1];
            const auto upper = offsets_out_d[tid];

            if(upper - lower != size_values_per_key)
            {
                printf("ERROR size values %llu\n", upper - lower);
            }

            for (index_t i = lower; i < upper; i++)
            {
                if(values_out_d[i] != key)
                {
                    printf("ERROR expected %llu got %llu\n", key, values_out_d[i]);
                }
            }
        }
    });

    cudaDeviceSynchronize(); CUERR

    cudaMemcpy(status_h, status_d, sizeof(status_t)*size, D2H); CUERR

    std::cout << "table status " << hash_table.peek_status() << std::endl;

    errors = 0;
    for(index_t i = 0; i < size; ++i)
    {

        if(status_h[i].has_any())
        {
            if(errors++ < 10)
                std::cout << "STATUS: i " << i << " key " << keys_in_h[i] << " status " << status_h[i] << std::endl;
        }
    }
    if(errors >= 10)
    {
        std::cout << "...\n" << "total errors " << errors << std::endl;
    }

    cudaFreeHost(keys_unique_h);
    cudaFreeHost(keys_in_h);
    cudaFreeHost(values_in_h);
    cudaFreeHost(offsets_out_h);
    cudaFreeHost(values_out_h);
    cudaFreeHost(status_h);
    cudaFree(keys_unique_d);
    cudaFree(keys_in_d);
    cudaFree(values_in_d);
    cudaFree(offsets_out_d);
    cudaFree(values_out_d);
    cudaFree(status_d);
    cudaFree(temp_d);

    cudaDeviceSynchronize(); CUERR
}
