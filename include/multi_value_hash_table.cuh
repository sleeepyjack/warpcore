#ifndef WARPCORE_MULTI_VALUE_HASH_TABLE_CUH
#define WARPCORE_MULTI_VALUE_HASH_TABLE_CUH

#include "single_value_hash_table.cuh"

namespace warpcore
{

/*! \brief multi-value hash table
 * \tparam Key key type (\c std::uint32_t or \c std::uint64_t)
 * \tparam Value value type
 * \tparam EmptyKey key which represents an empty slot
 * \tparam TombstoneKey key which represents an erased slot
 * \tparam ValueStore storage class from \c warpcore::storage::multi_value
 * \tparam ProbingScheme probing scheme from \c warpcore::probing_schemes
 */
template<
    class Key,
    class Value,
    Key   EmptyKey = defaults::empty_key<Key>(),
    Key   TombstoneKey = defaults::tombstone_key<Key>(),
    class ValueStore = defaults::value_storage_t<Value>,
    class ProbingScheme = defaults::probing_scheme_t<Key, 8>>
class MultiValueHashTable
{
    static_assert(
        checks::is_value_storage<ValueStore>(),
        "not a valid storage type");

public:
    // TODO why public?
    using handle_type = typename ValueStore::handle_type;

private:
    using hash_table_type = SingleValueHashTable<
        Key,
        handle_type,
        EmptyKey,
        TombstoneKey,
        ProbingScheme>;

    using value_store_type = ValueStore;

public:
    using key_type = Key;
    using value_type = Value;
    using index_type = index_t;
    using status_type = Status;

    /*! \brief get empty key
     * \return empty key
     */
     HOSTDEVICEQUALIFIER INLINEQUALIFIER
     static constexpr key_type empty_key() noexcept
     {
         return EmptyKey;
     }

     /*! \brief get tombstone key
      * \return tombstone key
      */
     HOSTDEVICEQUALIFIER INLINEQUALIFIER
     static constexpr key_type tombstone_key() noexcept
     {
         return TombstoneKey;
     }

    /*! \brief checks if \c key is equal to \c (EmptyKey||TombstoneKey)
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_key(key_type key) noexcept
    {
        return (key != empty_key() && key != tombstone_key());
    }

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept
    {
        return hash_table_type::cg_size();
    }

    /*! \brief constructor
    * \param[in] key_capacity guaranteed number of key slots in the hash table
    * \param[in] value_capacity total number of value slots
    * \param[in] seed random seed
    */
    template<
        class T = value_store_type,
        class = std::enable_if_t<
            std::is_same<typename T::tag, tags::static_value_storage>::value>>
    HOSTQUALIFIER
    explicit MultiValueHashTable(
        index_type key_capacity,
        index_type value_capacity,
        key_type seed = defaults::seed<key_type>(),
        bool no_init = false) noexcept :
        hash_table_(key_capacity, seed, true),
        value_store_(value_capacity),
        is_copy_(false)
    {
        hash_table_.join_status(value_store_.status());

        if(!no_init) init();
    }

    /*! \brief constructor
    * \param[in] key_capacity guaranteed number of key slots in the hash table
    * \param[in] value_capacity total number of value slots
    * \param[in] seed random seed
    * \param[in] grow_factor slab grow factor for \c warpcore::storage::multi_value::DynamicSlabListStore
    * \param[in] min_slab_size initial size of value slabs for \c warpcore::storage::multi_value::DynamicSlabListStore
    * \param[in] max_slab_size slab size of \c warpcore::storage::multi_value::DynamicSlabListStore after which no more growth occurs
    */
    template<
        class T = value_store_type,
        class = std::enable_if_t<
            std::is_same<typename T::tag, tags::dynamic_value_storage>::value>>
    HOSTQUALIFIER
    explicit MultiValueHashTable(
        index_type key_capacity,
        index_type value_capacity,
        key_type seed = defaults::seed<key_type>(),
        float grow_factor = 1.1,
        index_type min_slab_size = 1,
        index_type max_slab_size = handle_type::max_slab_size(),
        bool no_init = false) noexcept :
        hash_table_(key_capacity, seed, true),
        value_store_(value_capacity, grow_factor, min_slab_size, max_slab_size)
    {
        hash_table_.join_status(value_store_.status());

        if(!no_init) init();
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    MultiValueHashTable(const MultiValueHashTable& o) noexcept :
        hash_table_(o.hash_table_),
        value_store_(o.value_store_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    MultiValueHashTable(MultiValueHashTable&& o) noexcept :
        hash_table_(std::move(o.hash_table_)),
        value_store_(std::move(o.value_store_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    /*! \brief destructor
     */
     HOSTQUALIFIER INLINEQUALIFIER
     ~MultiValueHashTable() noexcept
     {}

     /*! \brief re-initialize the hash table
    * \param stream CUDA stream in which this operation is executed
    */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(cudaStream_t stream = 0) noexcept
    {
        const auto status = hash_table_.peek_status(stream);

        if(!status.has_not_initialized())
        {
            hash_table_.init(stream);
            value_store_.init(stream);
            hash_table_.table_.init_values(
                ValueStore::uninitialized_handle(), stream);
        }
    }

    /*
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type insert(
        key_type key_in,
        const value_type& value_in,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length,
        index_type max_values) noexcept
    {
        status_type status = status_type::unknown_error();

        handle_type * handle_ptr =
            hash_table_.insert_impl(key_in, status, group, probing_length);

        if(handle_ptr != nullptr)
        {
            status_type append_status = Status::unknown_error();

            if(group.thread_rank() == 0 &&
                value_store_.size(*handle_ptr) < max_values)
            {
                append_status = value_store_.append(*handle_ptr, value_in);

                if(append_status.has_any())
                {
                    hash_table_.status_->atomic_join(append_status);
                }
            }

            status += append_status.group_shuffle(group, 0);
        }

        return status - status_type::duplicate_key();
    }
    */

    /*! \brief inserts a key/value pair into the hash table
     * \param[in] key_in key to insert into the hash table
     * \param[in] value_in value that corresponds to \c key_in
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type insert(
        key_type key_in,
        const value_type& value_in,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) noexcept
    {
        status_type status = status_type::unknown_error();

        handle_type * handle_ptr =
            hash_table_.insert_impl(key_in, status, group, probing_length);

        if(handle_ptr != nullptr)
        {
            status_type append_status = Status::unknown_error();

            if(group.thread_rank() == 0)
            {
                append_status = value_store_.append(*handle_ptr, value_in);

                if(append_status.has_any())
                {
                    hash_table_.status_->atomic_join(append_status);
                }
            }

            status += append_status.group_shuffle(group, 0);
        }

        return status - status_type::duplicate_key();
    }

    /*
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void insert(
        key_type * keys_in,
        value_type * values_in,
        index_type size_in,
        index_type probing_length,
        cudaStream_t stream = 0,
        typename StatusHandler::base_type * status_out = nullptr) noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!hash_table_.is_initialized_) return;

        kernels::insert<MultiValueHashTable, StatusHandler>
        <<<SDIV(size_in * cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
        (keys_in, values_in, size_in, probing_length, *this, status_out);
    }
    */

    /*! \brief insert a set of keys into the hash table
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to insert into the hash table
     * \param[in] values_in corresponds values to \c keys_in
     * \param[in] size_in number of keys to insert
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information per key
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void insert(
        key_type * keys_in,
        value_type * values_in,
        index_type size_in,
        cudaStream_t stream = 0,
        index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * status_out = nullptr) noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!hash_table_.is_initialized_) return;

        static constexpr index_type block_size = 1024;
        static constexpr index_type groups_per_block = block_size / cg_size();
        static constexpr index_type smem_status_size =
            std::is_same<StatusHandler, status_handlers::ReturnNothing>::value ?
            1 : groups_per_block;

        lambda_kernel
        <<<SDIV(size_in * cg_size(), block_size), block_size, 0, stream>>>
        ([=, *this] DEVICEQUALIFIER () mutable
        {
            const index_type  tid = blockDim.x * blockIdx.x + threadIdx.x;
            const index_type btid = threadIdx.x;
            const index_type  gid = tid / cg_size();
            const index_type bgid = gid % groups_per_block;
            const auto block = cg::this_thread_block();
            const auto group = cg::tiled_partition<cg_size()>(block);

            __shared__ handle_type * handles[groups_per_block];
            __shared__ status_type status[smem_status_size];

            if(gid < size_in)
            {
                status_type probing_status = status_type::unknown_error();

                handles[bgid] = hash_table_.insert_impl(
                    keys_in[gid],
                    probing_status,
                    group,
                    probing_length);

                if(!std::is_same<
                    StatusHandler,
                    status_handlers::ReturnNothing>::value &&
                    group.thread_rank() == 0)
                {
                    status[bgid] = probing_status;
                }

                block.sync();


                if(btid < groups_per_block && handles[btid] != nullptr)
                {
                    const index_type block_offset =
                        blockIdx.x * groups_per_block;

                    const status_type append_status = value_store_.append(
                        *(handles[btid]),
                        values_in[block_offset + btid]);


                    if(append_status.has_any())
                    {
                        hash_table_.status_->atomic_join(append_status);
                    }

                    // TODO not zero-cost
                    if(!std::is_same<
                        StatusHandler,
                        status_handlers::ReturnNothing>::value)
                    {
                        StatusHandler::handle(
                            status[btid]+append_status-status_type::duplicate_key(),
                            status_out,
                            block_offset + btid);
                    }
                }
            }

        });

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief retrieves a key from the hash table
     * \param[in] key_in key to retrieve from the hash table
     * \param[out] values_out pointer to storage fo the retrieved values
     * \param[out] size_out number of values retrieved
     * \param[in] group cooperative group
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type retrieve(
        key_type key_in,
        value_type * values_out,
        index_type& size_out,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) const noexcept
    {
        handle_type handle;

        status_type status =
            hash_table_.retrieve(key_in, handle, group, probing_length);

        if(!status.has_any())
        {
            value_store_.for_each(
                handle,
                [=] DEVICEQUALIFIER (
                    const value_type& value,
                    index_type offset)
                {
                    values_out[offset] = value;
                },
                group);

            size_out = value_store_.size(handle);
        }
        else
        {
            size_out = 0;
        }

        return status - status_type::duplicate_key();
    }

    /*! \brief retrieve a set of keys from the hash table
     * \tparam StatusHandler handles returned status per key (see \c status_handlers)
     * \param[in] keys_in pointer to keys to retrieve from the hash table
     * \param[in] size_in number of keys to retrieve
     * \param[out] offsets_out
     * \param[out] values_out retrieved values of keys in \c key_in
     * \param[out] value_size_out total number of values retrieved by this operation
     * \param[in] temp pointer to auxillary device memory required by this operation
     * \param[out] temp_bytes size of required auxillary memory in bytes
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \param[out] status_out status information (per key)
     * \note if \c temp==nullptr||values_out==nullptr then only \c temp_bytes and \c value_size_out will be computed
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve(
        key_type * keys_in,
        index_type size_in,
        index_type * offsets_out,
        value_type * values_out,
        index_type& value_size_out,
        void * temp,
        index_t& temp_bytes,
        cudaStream_t stream = 0,
        index_type probing_length = defaults::probing_length(),
        typename StatusHandler::base_type * status_out = nullptr) const noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!hash_table_.is_initialized_) return;

        size_values(keys_in, size_in, offsets_out, probing_length, stream);

        cub::DeviceScan::InclusiveSum(
            temp, temp_bytes, offsets_out, offsets_out, size_in, stream);

        cudaMemcpyAsync(
            &value_size_out,
            offsets_out + size_in - 1,
            sizeof(index_type),
            D2H,
            stream);

        if(temp != nullptr && values_out != nullptr)
        {
            kernels::retrieve<MultiValueHashTable, StatusHandler>
            <<<SDIV(size_in * cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            (keys_in, size_in, offsets_out, values_out, probing_length, *this, status_out);
        }

        if(stream == 0 || temp == nullptr || values_out == nullptr)
        {
            cudaStreamSynchronize(stream);
        }
    }

    // TODO host retrieve which also returns the set of unique keys

    /*! \brief applies a funtion over all values of a corresponding key
     * \tparam Func type of map i.e. CUDA device lambda
     * \param[in] key_in key to retrieve
     * \param[in] stream CUDA stream in which this operation is executed in
     * \param[in] size of shared memory to reserve for this execution
     * \param[in] f map to apply
     */
    template<class Func>
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type for_each_value(
        key_type key_in,
        const cg::thread_block_tile<cg_size()>& group,
        Func f,
        index_type probing_length = defaults::probing_length()) const noexcept
    {
        handle_type handle;

        status_type status =
            hash_table_.retrieve(key_in, handle, group, probing_length);

        if(!status.has_any())
        {
            value_store_.for_each(handle, group, f);
        }

        return status - status_type::duplicate_key();
    }

    // TODO host function for_each_value

    /*! \brief retrieves the set of all keys stored inside the hash table
     * \param[out] keys_out pointer to the retrieved keys
     * \param[out] size_out number of retrieved keys
     * \param[in] stream CUDA stream in which this operation is executed in
     * \note if \c keys_out==nullptr then only \c size_out will be computed
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void retrieve_all_keys(
        key_type * keys_out,
        index_type& size_out,
        cudaStream_t stream = 0) noexcept
    {
        if(keys_out == nullptr)
        {
            size_out = hash_table_.size(stream);
        }
        else
        {
            index_type * key_count = hash_table_.temp_.get();
            cudaMemsetAsync(key_count, 0, sizeof(index_type), stream);

            hash_table_.for_each(
            [=] DEVICEQUALIFIER (key_type key, const auto&)
            {
                keys_out[atomicAggInc(key_count)] = key;
            }, stream);

            cudaMemcpyAsync(
                &size_out, key_count, sizeof(index_type), D2H, stream);
        }

        if(stream == 0 || keys_out == nullptr)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief get load factor of the key store
     * \param stream CUDA stream in which this operation is executed in
     * \return load factor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float key_load_factor(cudaStream_t stream = 0) noexcept
    {
        return hash_table_.load_factor(stream);
    }

    /*! \brief get load factor of the value store
     * \param stream CUDA stream in which this operation is executed in
     * \return load factor
     */
     HOSTQUALIFIER INLINEQUALIFIER
     float value_load_factor(cudaStream_t stream = 0) const noexcept
     {
         return value_store_.load_factor(stream);
     }

    /*! \brief get the the total number of bytes occupied by this data structure
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_total() noexcept
    {
        const float bytes_hash_table =
            hash_table_.capacity() * (sizeof(key_type) + sizeof(handle_type));
        const float bytes_value_store =
            value_store_.capacity() * sizeof(typename ValueStore::slab_type);

        return bytes_hash_table + bytes_value_store;
    }

    /*! \brief get the the number of bytes in this data structure occupied by keys
     * \param stream CUDA stream in which this operation is executed in
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_keys(cudaStream_t stream = 0) noexcept
    {
        return size_keys(stream) * sizeof(key_type);
    }

    /*! \brief get the the number of bytes in this data structure occupied by values
     * \param stream CUDA stream in which this operation is executed in
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_values(cudaStream_t stream = 0) noexcept
    {
        return size_values(stream) * sizeof(value_type);
    }

    /*! \brief get the the number of bytes in this data structure occupied by actual information
     * \param stream CUDA stream in which this operation is executed in
     * \return bytes
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type bytes_payload(cudaStream_t stream = 0) noexcept
    {
        return bytes_keys(stream) + bytes_values(stream);
    }

    /*! \brief current storage density of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float storage_density(cudaStream_t stream = 0) noexcept
    {
        return float(bytes_payload(stream)) / float(bytes_total());
    }

    /*! \brief current relative storage density of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return storage density
     */
    HOSTQUALIFIER INLINEQUALIFIER
    float relative_storage_density(cudaStream_t stream = 0) noexcept
    {
        const float bytes_hash_table =
            hash_table_.capacity() * (sizeof(key_type) + sizeof(handle_type));
        const float bytes_value_store =
            value_store_.bytes_occupied(stream);

        return float(bytes_payload(stream)) / (bytes_value_store + bytes_hash_table);
    }

    /*! \brief indicates if the hash table is properly initialized
     * \return \c true iff the hash table is properly initialized
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_initialized() const noexcept
    {
        return hash_table_.is_initialized();
    }

    /*! \brief get the status of the hash table
     * \param stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type peek_status(cudaStream_t stream = 0) const noexcept
    {
        return hash_table_.peek_status(stream) - status_type::duplicate_key();
    }

    /*! \brief get and reset the status of the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return the status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type pop_status(cudaStream_t stream = 0) noexcept
    {
        return hash_table_.pop_status(stream) - status_type::duplicate_key();
    }

    /*! \brief get the key capacity of the hash table
     * \return number of key slots in the hash table
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type key_capacity() const noexcept
    {
        return hash_table_.capacity();
    }

    /*! \brief get the maximum value capacity of the hash table
     * \return maximum value capacity
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type value_capacity() const noexcept
    {
        return value_store_.capacity();
    }

    /*! \brief number of keys stored inside the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return number of keys inside the hash table
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type size_keys(cudaStream_t stream = 0) noexcept
    {
        return hash_table_.size(stream);
    }

    /*! \brief get number of values to a corresponding key inside the hash table
     * \param[in] key_in key to probe
     * \param[out] size_out number of values
     * \param[in] group cooperative group this operation is executed in
     * \param[in] probing_length maximum number of probing attempts
     * \return status (per thread)
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type size_values(
        key_type key_in,
        index_type& size_out,
        const cg::thread_block_tile<cg_size()>& group,
        index_type probing_length = defaults::probing_length()) const noexcept
    {
        handle_type handle;

        status_type status =
            hash_table_.retrieve(key_in, handle, group, probing_length);

        status -= status_type::duplicate_key();
        size_out = (!status.has_any()) ? value_store_.size(handle) : 0;

        return status - status_type::duplicate_key();
    }

    /*! \brief get number of values to a corresponding set of keys inside the hash table
     * \param[in] keys_in keys to probe
     * \param[in] size_in input size
     * \param[out] sizes_out number of values per key
     * \param[in] probing_length maximum number of probing attempts
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    template<class StatusHandler = defaults::status_handler_t>
    HOSTQUALIFIER INLINEQUALIFIER
    void size_values(
        key_type * keys_in,
        index_type size_in,
        index_type * sizes_out,
        index_type probing_length = defaults::probing_length(),
        cudaStream_t stream = 0,
        typename StatusHandler::base_type * status_out = nullptr) const noexcept
    {
        static_assert(
            checks::is_status_handler<StatusHandler>(),
            "not a valid status handler type");

        if(!hash_table_.is_initialized_) return;

        kernels::size_values<MultiValueHashTable, StatusHandler>
        <<<SDIV(size_in * cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
        (keys_in, size_in, sizes_out, probing_length, *this, status_out);

        if(stream == 0)
        {
            cudaStreamSynchronize(stream);
        }
    }

    /*! \brief get number of values inside the hash table
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return total number of values
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type size_values(cudaStream_t stream = 0) noexcept
    {
        index_type * tmp = hash_table_.temp_.get();

        cudaMemsetAsync(tmp, 0, sizeof(index_type), stream);

        hash_table_.for_each(
            [=, *this] DEVICEQUALIFIER (key_type, const handle_type& handle)
            {
                atomicAdd(tmp, value_store_.size(handle));
            },
            stream);

        index_type out = 0;

        cudaMemcpyAsync(&out, tmp, sizeof(index_type), D2H, stream);

        cudaStreamSynchronize(stream);

        return out;
    }

    /*! \brief indicates if this object is a shallow copy
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_copy() const noexcept
    {
        return is_copy_;
    }

private:
    hash_table_type hash_table_; //< storage class for keys
    value_store_type value_store_; //< multi-value storage class
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class MultiValueHashTable

} // namespace warpcore

#endif /* WARPCORE_MULTI_VALUE_HASH_TABLE_CUH */