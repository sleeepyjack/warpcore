#include "catch.hpp"
#include "base.cuh"

TEMPLATE_TEST_CASE_SIG(
    "SoAStore",
    "[storage][soa][template]",
    ((class Key, class Value), Key, Value),
    (std::uint32_t, std::uint32_t),
    (std::uint32_t, float))
{
    using namespace warpcore;

    using storage_t = storage::key_value::SoAStore<Key, Value>;

    const index_t capacity = GENERATE(as<index_t>{}, 123456, 42424242, 69696969);

    SECTION("constructor")
    {
        storage_t st = storage_t(capacity); CUERR

        CHECK(st.status() == Status::none());

        CHECK(st.capacity() == capacity);
    }

    SECTION("set and get pairs")
    {
        storage_t st = storage_t(capacity); CUERR

        // set pairs
        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                auto&& p = st[tid];
                p.key = Key(tid);
                p.value = Value(tid);
            }
        });

        index_t * errors = nullptr;
        cudaMallocManaged(&errors, sizeof(index_t)); CUERR
        *errors = 0;

        // get pairs
        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != Key(tid) || st[tid].value != Value(tid))
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*errors == 0);

        cudaFree(errors); CUERR
    }

    SECTION("init keys")
    {
        storage_t st = storage_t(capacity); CUERR

        const Key key = 42;

        st.init_keys(key);

        index_t * errors = nullptr;
        cudaMallocManaged(&errors, sizeof(index_t)); CUERR
        *errors = 0;

        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*errors == 0);

        cudaFree(errors); CUERR
    }

    SECTION("init pairs")
    {
        storage_t st = storage_t(capacity); CUERR

        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(key, value);

        index_t * errors = nullptr;
        cudaMallocManaged(&errors, sizeof(index_t)); CUERR
        *errors = 0;

        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key || st[tid].value != value)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*errors == 0);

        cudaFree(errors); CUERR
    }

    SECTION("CUDA atomics")
    {
        storage_t st = storage_t(1); CUERR

        const Key init = 0;
        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(init, value);

        bool * error = nullptr;
        cudaMallocManaged(&error, sizeof(bool)); CUERR
        *error = false;

        lambda_kernel
        <<<1, 1>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            atomicCAS(&st[0].key, init, key);
            *error = (st[0].key == key && st[0].value == value) ? false : true;
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*error == false);

        cudaFree(error); CUERR
    }
}

TEMPLATE_TEST_CASE_SIG(
    "AoSStore",
    "[storage][aos][template]",
    ((class Key, class Value), Key, Value),
    (std::uint32_t, std::uint32_t),
    (std::uint32_t, float))
{
    using namespace warpcore;

    using storage_t = storage::key_value::AoSStore<Key, Value>;

    const index_t capacity = GENERATE(as<index_t>{}, 123456, 42424242, 69696969);

    SECTION("constructor")
    {
        storage_t st = storage_t(capacity); CUERR

        CHECK(st.status() == Status::none());

        CHECK(st.capacity() == capacity);
    }

    SECTION("set and get pairs")
    {
        storage_t st = storage_t(capacity); CUERR

        // set pairs
        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                auto&& p = st[tid];
                p.key = Key(tid);
                p.value = Value(tid);
            }
        });

        index_t * errors = nullptr;
        cudaMallocManaged(&errors, sizeof(index_t)); CUERR
        *errors = 0;

        // get pairs
        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != Key(tid) || st[tid].value != Value(tid))
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*errors == 0);

        cudaFree(errors); CUERR
    }

    SECTION("init keys")
    {
        storage_t st = storage_t(capacity); CUERR

        const Key key = 42;

        st.init_keys(key);

        index_t * errors = nullptr;
        cudaMallocManaged(&errors, sizeof(index_t)); CUERR
        *errors = 0;

        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*errors == 0);

        cudaFree(errors); CUERR
    }

    SECTION("init pairs")
    {
        storage_t st = storage_t(capacity); CUERR

        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(key, value);

        index_t * errors = nullptr;
        cudaMallocManaged(&errors, sizeof(index_t)); CUERR
        *errors = 0;

        lambda_kernel
        <<<SDIV(capacity, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < capacity)
            {
                if(st[tid].key != key || st[tid].value != value)
                {
                    atomicAdd(errors, 1);
                }
            }
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*errors == 0);

        cudaFree(errors); CUERR
    }

    SECTION("CUDA atomics")
    {
        storage_t st = storage_t(1); CUERR

        const Key init = 0;
        const Key key = 42;
        const Value value = 1337;

        st.init_pairs(init, value);

        bool * error = nullptr;
        cudaMallocManaged(&error, sizeof(bool)); CUERR
        *error = false;

        lambda_kernel
        <<<1, 1>>>
        ([=] DEVICEQUALIFIER () mutable
        {
            atomicCAS(&st[0].key, init, key);
            *error = (st[0].key == key && st[0].value == value) ? false : true;
        });
        cudaDeviceSynchronize(); CUERR

        CHECK(*error == false);

        cudaFree(error); CUERR
    }
}