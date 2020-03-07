#include "catch.hpp"
#include "single_value_hash_table.cuh"

TEMPLATE_TEST_CASE_SIG(
    "SingleValueHashTable",
    "[singlevalue][hashtable][template]",
    ((class Key, class Value), Key, Value),
    (std::uint32_t, std::uint32_t),
    (std::uint32_t, std::uint64_t),
    (std::uint64_t, std::uint32_t),
    (std::uint64_t, double))
{
    using namespace warpcore;

    using probing_scheme_t = defaults::probing_scheme_t<Key, 8>;

    using hash_table_t =
        SingleValueHashTable<
            Key, Value,
            defaults::empty_key<Key>(),
            defaults::tombstone_key<Key>(),
            probing_scheme_t,
            defaults::table_storage_t<Key, Value>,
            defaults::temp_memory_bytes()>;

    const index_t min_capacity = GENERATE(as<index_t>{}, 12345, 4242424, 696969);
    const index_t valid_capacity =
        get_valid_capacity(min_capacity, probing_scheme_t::cg_size());
    const float load = GENERATE(as<float>{}, 0.5, 0.7, 0.8);
    const Key seed = GENERATE(as<Key>{}, 5, 42);
    const index_t n = float(valid_capacity) * load;

    CAPTURE(min_capacity, valid_capacity, load, seed, n);

    hash_table_t hash_table(min_capacity); CUERR
    REQUIRE(hash_table.peek_status() == Status::none());
    REQUIRE(cudaPeekAtLastError() == cudaSuccess);

    SECTION("state after object creation")
    {
        CHECK(hash_table.size() == 0);
        CHECK(hash_table.capacity() == valid_capacity);
        CHECK(hash_table.capacity() >= min_capacity);
        CHECK(hash_table.peek_status() == Status::none());
        CHECK(cudaPeekAtLastError() == cudaSuccess);
    }

    Key* keys_in_d = nullptr;
    cudaMalloc(&keys_in_d, sizeof(Key)*n);
    Key* keys_out_d = nullptr;
    cudaMalloc(&keys_out_d, sizeof(Key)*n);
    Value* values_in_d = nullptr;
    cudaMalloc(&values_in_d, sizeof(Value)*n);
    Value* values_out_d = nullptr;
    cudaMalloc(&values_out_d, sizeof(Value)*n);

    // generate pseudo-random unique keys and values
    lambda_kernel
    <<<SDIV(n, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
    ([=] DEVICEQUALIFIER () mutable
    {
        const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        Key i = tid;
        if(tid < n)
        {
            Key out;
            do
            {
                out = hashers::MurmurHash<Key>::hash(i+seed);
                i += n;
            }
            while(!hash_table.is_valid_key(out));

            keys_in_d[tid] = out;
            values_in_d[tid] = out;
        }
    });
    CHECK(cudaPeekAtLastError() == cudaSuccess);

    hash_table.insert(keys_in_d, values_in_d, n);

    CHECK(hash_table.size() == n);
    CHECK(hash_table.peek_status() == Status::none());
    CHECK(cudaPeekAtLastError() == cudaSuccess); CUERR

    cudaMemset(values_out_d, 0, sizeof(Value)*n); CUERR

    hash_table.retrieve(keys_in_d, n, values_out_d);

    CHECK(hash_table.peek_status() == Status::none());
    CHECK(cudaPeekAtLastError() == cudaSuccess); CUERR

    SECTION("retrieve inserted values")
    {
        cudaMemset(values_out_d, 0, sizeof(Value)*n);

        hash_table.retrieve(keys_in_d, n, values_out_d);

        CHECK(hash_table.pop_status() == Status::none());

        index_t errors_h = 0;
        index_t * errors_d = nullptr;
        cudaMalloc(&errors_d, sizeof(index_t));
        cudaMemset(errors_d, 0, sizeof(index_t));

        lambda_kernel
        <<<SDIV(n, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < n)
            {
                if(values_out_d[tid] != Value(keys_in_d[tid]))
                {
                    atomicAdd(errors_d, 1);
                }
            }
        });

        cudaMemcpy(&errors_h, errors_d, sizeof(index_t), D2H);

        cudaFree(errors_d);

        CHECK(errors_h == 0);
        CHECK(cudaPeekAtLastError() == cudaSuccess);
    }

    SECTION("retrieve all")
    {
        // TODO
    }

    SECTION("erase key/value pairs")
    {
        // TODO
    }

    SECTION("for each")
    {
        std::uint64_t * num_elems = nullptr;
        cudaMallocManaged(&num_elems, sizeof(std::uint64_t));
        *num_elems = 0;

        hash_table.for_each(
            [=] DEVICEQUALIFIER (Key key, const Value& value)
            {
                if(hash_table_t::is_valid_key(key))
                {
                    atomicAdd(num_elems, 1);
                }
            });

        CHECK(*num_elems == hash_table.size());
    }

    cudaFree(keys_in_d);
    cudaFree(keys_out_d);
    cudaFree(values_in_d);
    cudaFree(values_out_d);

    CHECK(cudaGetLastError() == cudaSuccess);
}