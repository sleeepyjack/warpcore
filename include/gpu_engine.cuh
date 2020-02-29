#ifndef WARPCORE_GPU_ENGINE_CUH
#define WARPCORE_GPU_ENGINE_CUH

#include "config.cuh"
#include "status.cuh"

namespace warpcore
{

/*! \brief CUDA kernels
 */
namespace kernels
{

template<
    class T,
    T Val = 0>
GLOBALQUALIFIER
void memset(
    T * arr,
    index_t size)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= size) return;
    arr[tid] = Val;
}

template<class Core>
GLOBALQUALIFIER
void insert(
    typename Core::key_type * keys_in,
    index_t size_in,
    Core core)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        core.insert(keys_in[gid], group);
    }
}

template<class Core, class StatusHandler>
GLOBALQUALIFIER
void insert(
    typename Core::key_type * keys_in,
    index_t size_in,
    index_t probing_length,
    Core core,
    typename StatusHandler::base_type * status_out)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        const auto status =
            core.insert(keys_in[gid], group, probing_length);

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler>
GLOBALQUALIFIER
void insert(
    typename Core::key_type * keys_in,
    typename Core::value_type * values_in,
    index_t size_in,
    index_t probing_length,
    Core core,
    typename StatusHandler::base_type * status_out)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        const auto status =
            core.insert(keys_in[gid], values_in[gid], group, probing_length);

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core>
GLOBALQUALIFIER
void retrieve(
    typename Core::key_type * keys_in,
    index_t size_in,
    typename Core::value_type * values_out,
    Core core)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        typename Core::value_type value = core.retrieve(keys_in[gid], group);

        if(group.thread_rank() == 0)
        {
            values_out[gid] = value;
        }
    }
}

template<class Core, class StatusHandler>
GLOBALQUALIFIER
void retrieve(
    typename Core::key_type * keys_in,
    index_t size_in,
    typename Core::value_type * values_out,
    index_t probing_length,
    Core core,
    typename StatusHandler::base_type * status_out)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        typename Core::value_type value_out;

        const auto status =
            core.retrieve(keys_in[gid], value_out, group, probing_length);

        if(group.thread_rank() == 0)
        {
            if(!status.has_any())
            {
                values_out[gid] = value_out;
            }

            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler>
GLOBALQUALIFIER
void retrieve(
    typename Core::key_type * keys_in,
    index_t size_in,
    typename Core::key_type * keys_out,
    typename Core::value_type * values_out,
    index_t * size_out,
    index_t probing_length,
    Core core,
    typename StatusHandler::base_type * status_out)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        const typename Core::key_type key_in = keys_in[gid];
        typename Core::value_type value_out;

        const auto status =
            core.retrieve(key_in, value_out, group, probing_length);

        if(group.thread_rank() == 0)
        {
            if(!status.has_any())
            {
                const auto i = atomicAggInc(size_out);
                keys_out[i] = key_in;
                values_out[i] = value_out;
            }

            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler>
GLOBALQUALIFIER
void retrieve(
    typename Core::key_type * keys_in,
    index_t size_in,
    index_t * offsets_in,
    typename Core::value_type * values_out,
    index_t probing_length,
    Core core,
    typename StatusHandler::base_type * status_out)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    using status_type = typename Core::status_type;

    if(gid < size_in)
    {
        const typename Core::key_type key_in = keys_in[gid];
        status_type status = status_type::unknown_error();
        index_t size_values;

        if(gid == 0)
        {
            status = core.retrieve(
                key_in,
                values_out,
                size_values,
                group,
                probing_length);
        }
        else
        {
            status = core.retrieve(
                key_in,
                values_out + offsets_in[gid - 1],
                size_values,
                group,
                probing_length);
        }

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler>
GLOBALQUALIFIER
void erase(
    typename Core::key_type * keys_in,
    index_t size_in,
    index_t probing_length,
    Core core,
    typename StatusHandler::base_type * status_out)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        const auto status =
            core.erase(keys_in[gid], group, probing_length);

        if(group.thread_rank() == 0)
        {
            StatusHandler::handle(status, status_out, gid);
        }
    }
}

template<class Core, class StatusHandler>
GLOBALQUALIFIER
void size_values(
    typename Core::key_type * keys_in,
    index_t size_in,
    index_t * sizes_out,
    index_t probing_length,
    Core core,
    typename StatusHandler::base_type * status_out)
{
    const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const index_t gid = tid / Core::cg_size();
    const auto group =
        cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

    if(gid < size_in)
    {
        index_t size = 0;

        const auto status =
            core.size_values(keys_in[gid], size, group, probing_length);

        if(group.thread_rank() == 0)
        {
            sizes_out[gid] = size;
            //StatusHandler::handle(status, status_out, gid); //!
        }
    }
}

} // namespace kernels

} // namespace warpcore

#endif /* WARPCORE_GPU_ENGINE_CUH */